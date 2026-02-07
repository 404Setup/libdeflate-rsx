use crate::common::*;
use std::cmp::min;

const NUM_SYMBOL_BITS: usize = 10;
const SYMBOL_MASK: u32 = (1 << NUM_SYMBOL_BITS) - 1;
const FREQ_MASK: u32 = !SYMBOL_MASK;

fn heapify_subtree(a: &mut [u32], length: usize, subtree_idx: usize) {
    let v = a[subtree_idx];
    let mut parent_idx = subtree_idx;
    while parent_idx * 2 <= length {
        let mut child_idx = parent_idx * 2;
        if child_idx < length && a[child_idx + 1] > a[child_idx] {
            child_idx += 1;
        }
        if v >= a[child_idx] {
            break;
        }
        a[parent_idx] = a[child_idx];
        parent_idx = child_idx;
    }
    a[parent_idx] = v;
}

fn heap_sort(a: &mut [u32]) {
    let length = a.len();
    if length < 2 {
        return;
    }
    let mut heap = vec![0u32; length + 1];
    heap[1..].copy_from_slice(a);
    for i in (1..=length / 2).rev() {
        heapify_subtree(&mut heap, length, i);
    }
    let mut curr_len = length;
    while curr_len >= 2 {
        heap.swap(1, curr_len);
        curr_len -= 1;
        heapify_subtree(&mut heap, curr_len, 1);
    }
    a.copy_from_slice(&heap[1..]);
}

fn sort_symbols(num_syms: usize, freqs: &[u32], lens: &mut [u8], symout: &mut [u32]) -> usize {
    let mut counters = vec![0u32; num_syms];
    for sym in 0..num_syms {
        counters[min(freqs[sym] as usize, num_syms - 1)] += 1;
    }
    let mut current_pos = 0;
    for i in 1..num_syms {
        let count = counters[i];
        counters[i] = current_pos;
        current_pos += count;
    }
    let num_used_syms = current_pos as usize;
    for sym in 0..num_syms {
        let freq = freqs[sym];
        if freq != 0 {
            let pos = &mut counters[min(freq as usize, num_syms - 1)];
            symout[*pos as usize] = (sym as u32) | (freq << NUM_SYMBOL_BITS);
            *pos += 1;
        } else {
            lens[sym] = 0;
        }
    }
    let heap_start = counters[num_syms - 2] as usize;
    let heap_end = counters[num_syms - 1] as usize;
    if heap_end > heap_start {
        heap_sort(&mut symout[heap_start..heap_end]);
    }
    num_used_syms
}

fn build_tree(a: &mut [u32], sym_count: usize) {
    let last_idx = sym_count - 1;
    let mut i = 0;
    let mut b = 0;
    let mut e = 0;
    while e < last_idx {
        let new_freq;
        if i + 1 <= last_idx && (b == e || (a[i + 1] & FREQ_MASK) <= (a[b] & FREQ_MASK)) {
            new_freq = (a[i] & FREQ_MASK) + (a[i + 1] & FREQ_MASK);
            i += 2;
        } else if b + 2 <= e && (i > last_idx || (a[b + 1] & FREQ_MASK) < (a[i] & FREQ_MASK)) {
            new_freq = (a[b] & FREQ_MASK) + (a[b + 1] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            a[b + 1] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b + 1] & SYMBOL_MASK);
            b += 2;
        } else {
            new_freq = (a[i] & FREQ_MASK) + (a[b] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            i += 1;
            b += 1;
        }
        a[e] = new_freq | (a[e] & SYMBOL_MASK);
        e += 1;
    }
}

fn compute_length_counts(
    a: &mut [u32],
    root_idx: usize,
    len_counts: &mut [u32],
    max_codeword_len: usize,
) {
    for l in 0..=max_codeword_len {
        len_counts[l] = 0;
    }
    len_counts[1] = 2;
    a[root_idx] &= SYMBOL_MASK;
    for node in (0..root_idx).rev() {
        let parent = (a[node] >> NUM_SYMBOL_BITS) as usize;
        let parent_depth = a[parent] >> NUM_SYMBOL_BITS;
        let mut depth = parent_depth + 1;
        a[node] = (a[node] & SYMBOL_MASK) | (depth << NUM_SYMBOL_BITS);
        if depth as usize >= max_codeword_len {
            depth = (max_codeword_len - 1) as u32;
            while len_counts[depth as usize] == 0 {
                depth -= 1;
            }
        }
        len_counts[depth as usize] -= 1;
        len_counts[depth as usize + 1] += 2;
    }
}

const BITREVERSE_TAB: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];

fn reverse_codeword(codeword: u32, len: u8) -> u32 {
    let rev = ((BITREVERSE_TAB[(codeword & 0xFF) as usize] as u32) << 8)
        | (BITREVERSE_TAB[((codeword >> 8) & 0xFF) as usize] as u32);
    rev >> (16 - len)
}

fn gen_codewords(
    a: &mut [u32],
    lens: &mut [u8],
    len_counts: &[u32],
    max_codeword_len: usize,
    num_syms: usize,
) {
    let mut next_codewords = [0u32; DEFLATE_MAX_CODEWORD_LEN + 1];
    let mut i = 0;
    for len in (1..=max_codeword_len).rev() {
        let mut count = len_counts[len];
        while count > 0 {
            lens[(a[i] & SYMBOL_MASK) as usize] = len as u8;
            i += 1;
            count -= 1;
        }
    }
    next_codewords[0] = 0;
    next_codewords[1] = 0;
    for len in 2..=max_codeword_len {
        next_codewords[len] = (next_codewords[len - 1] + len_counts[len - 1]) << 1;
    }
    for sym in 0..num_syms {
        if lens[sym] != 0 {
            a[sym] = reverse_codeword(next_codewords[lens[sym] as usize], lens[sym]);
            next_codewords[lens[sym] as usize] += 1;
        }
    }
}

pub fn make_huffman_code(
    num_syms: usize,
    max_codeword_len: usize,
    freqs: &[u32],
    lens: &mut [u8],
    codewords: &mut [u32],
) {
    let num_used_syms = sort_symbols(num_syms, freqs, lens, codewords);
    if num_used_syms < 2 {
        let sym = if num_used_syms != 0 {
            codewords[0] & SYMBOL_MASK
        } else {
            0
        };
        let nonzero_idx = if sym != 0 { sym as usize } else { 1 };
        codewords[0] = 0;
        lens[0] = 1;
        codewords[nonzero_idx] = 1;
        lens[nonzero_idx] = 1;
        return;
    }
    build_tree(codewords, num_used_syms);
    let mut len_counts = [0u32; DEFLATE_MAX_CODEWORD_LEN + 1];
    compute_length_counts(
        codewords,
        num_used_syms - 2,
        &mut len_counts,
        max_codeword_len,
    );
    gen_codewords(codewords, lens, &len_counts, max_codeword_len, num_syms);
}
