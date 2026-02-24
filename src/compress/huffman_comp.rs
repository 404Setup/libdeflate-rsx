use crate::common::*;
use std::cmp::min;

const NUM_SYMBOL_BITS: usize = 10;
const SYMBOL_MASK: u32 = (1 << NUM_SYMBOL_BITS) - 1;
const FREQ_MASK: u32 = !SYMBOL_MASK;

fn sort_symbols(num_syms: usize, freqs: &[u32], lens: &mut [u8], symout: &mut [u32]) -> usize {
    let mut counters = [0u32; DEFLATE_MAX_NUM_SYMS];
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
        symout[heap_start..heap_end].sort_unstable();
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
        if i < last_idx && (b == e || (a[i + 1] & FREQ_MASK) <= (a[b] & FREQ_MASK)) {
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

fn reverse_codeword(codeword: u32, len: u8) -> u32 {
    (codeword as u16).reverse_bits() as u32 >> (16 - len)
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
