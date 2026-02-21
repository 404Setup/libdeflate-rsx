#![allow(unsafe_op_in_unsafe_fn)]
use crate::decompress::tables::{
    HUFFDEC_END_OF_BLOCK, HUFFDEC_EXCEPTIONAL, HUFFDEC_LITERAL, HUFFDEC_SUBTABLE_POINTER,
    OFFSET_TABLEBITS,
};
use crate::decompress::{
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN,
    DEFLATE_BLOCKTYPE_UNCOMPRESSED, DecompressResult, Decompressor,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! refill_bits {
    ($input:expr, $in_idx:expr, $bitbuf:expr, $bitsleft:expr) => {
        if $bitsleft < 32 {
            if $input.len().saturating_sub($in_idx) >= 8 {
                let word = unsafe { ($input.as_ptr().add($in_idx) as *const u64).read_unaligned() };
                let word = u64::from_le(word);

                $bitbuf |= word << $bitsleft;

                let consumed = (63 - $bitsleft) >> 3;
                $in_idx += consumed as usize;

                $bitsleft |= 56;
            } else {
                while $bitsleft < 32 && $in_idx < $input.len() {
                    $bitbuf |= ($input[$in_idx] as u64) << $bitsleft;
                    $in_idx += 1;
                    $bitsleft += 8;
                }
            }
        }
    };
}

static OFFSET9_MASKS: [u8; 144] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
];

// LCM(12, 16) = 48. 3 vectors.
static OFFSET12_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
];

// LCM(3, 16) = 48. 3 vectors.
static OFFSET3_MASKS: [u8; 48] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

// LCM(6, 16) = 48. 3 vectors.
static OFFSET6_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
    2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
];

// LCM(7, 16) = 112. 7 vectors.
static OFFSET7_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3,
    4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
    1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4,
    5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
];

// LCM(5, 16) = 80. 5 vectors.
static OFFSET5_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
    2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
    4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
];

// LCM(10, 16) = 80. 5 vectors.
static OFFSET10_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

// LCM(11, 16) = 176. 11 vectors.
static OFFSET11_MASKS: [u8; 176] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
];

// LCM(15, 16) = 240. 15 vectors.
static OFFSET15_MASKS: [u8; 240] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
];

// LCM(14, 16) = 112. 7 vectors.
static OFFSET14_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
];

// LCM(13, 16) = 208. 13 vectors.
static OFFSET13_MASKS: [u8; 208] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12,
];

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn decompress_shuffle_pattern<const N: usize>(
    out_next: *mut u8,
    src: *const u8,
    masks: &[u8],
    length: usize,
) {
    let mut copied = 0;
    if length >= 16 {
        let v_src = _mm_loadu_si128(src as *const __m128i);
        let masks_ptr = masks.as_ptr() as *const __m128i;

        let mut vectors = [_mm_setzero_si128(); N];
        for i in 0..N {
            vectors[i] = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(i)));
        }

        let stride = N * 16;

        while copied + stride <= length {
            for i in 0..N {
                _mm_storeu_si128(out_next.add(copied + i * 16) as *mut __m128i, vectors[i]);
            }
            copied += stride;
        }

        let mut idx = 0;
        while copied + 16 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, vectors[idx]);
            copied += 16;
            idx += 1;
        }
    }

    while copied < length {
        *out_next.add(copied) = *src.add(copied);
        copied += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_write_cycle_vectors<const N: usize>(
    out_next: *mut u8,
    src: *const u8,
    vectors: &[__m128i; N],
    length: usize,
    mut copied: usize,
) {
    let stride = N * 16;
    while copied + stride <= length {
        for i in 0..N {
            _mm_storeu_si128(out_next.add(copied + i * 16) as *mut __m128i, vectors[i]);
        }
        copied += stride;
    }

    let mut i = 0;
    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, vectors[i]);
        copied += 16;
        i += 1;
        if i == N {
            i = 0;
        }
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_fill_pattern(out_next: *mut u8, v_pattern: __m128i, length: usize) {
    let mut i = 0;
    while i + 64 <= length {
        _mm_storeu_si128(out_next.add(i) as *mut __m128i, v_pattern);
        _mm_storeu_si128(out_next.add(i + 16) as *mut __m128i, v_pattern);
        _mm_storeu_si128(out_next.add(i + 32) as *mut __m128i, v_pattern);
        _mm_storeu_si128(out_next.add(i + 48) as *mut __m128i, v_pattern);
        i += 64;
    }
    while i + 32 <= length {
        _mm_storeu_si128(out_next.add(i) as *mut __m128i, v_pattern);
        _mm_storeu_si128(out_next.add(i + 16) as *mut __m128i, v_pattern);
        i += 32;
    }
    if i + 16 <= length {
        _mm_storeu_si128(out_next.add(i) as *mut __m128i, v_pattern);
        i += 16;
    }
    let pattern = _mm_cvtsi128_si64(v_pattern) as u64;
    while i + 8 <= length {
        std::ptr::write_unaligned(out_next.add(i) as *mut u64, pattern);
        i += 8;
    }
    while i < length {
        *out_next.add(i) = (pattern >> ((i & 7) * 8)) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_alignr_cycle<const SHIFT: i32>(
    out_next: *mut u8,
    src: *const u8,
    length: usize,
    mut v_align: __m128i,
    mut v_prev: __m128i,
) {
    let mut copied = 16;
    while copied + 16 <= length {
        let v_next = _mm_alignr_epi8::<SHIFT>(v_prev, v_align);
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next);
        v_align = v_prev;
        v_prev = v_next;
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_cycle3<const SHIFT: i32>(
    out_next: *mut u8,
    src: *const u8,
    v: __m128i,
    length: usize,
) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v_tail = _mm_loadu_si128(src.add((32 - SHIFT) as usize) as *const __m128i);
    let mut v2 = _mm_alignr_epi8::<SHIFT>(v, v_tail);
    let mut v0 = v;
    let mut v1 = v1;

    let mut copied = 16;
    while copied + 48 <= length {
        let next_v0 = _mm_alignr_epi8::<SHIFT>(v1, v0);
        let next_v1 = _mm_alignr_epi8::<SHIFT>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<SHIFT>(next_v0, v2);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, next_v0);

        v0 = next_v0;
        v1 = next_v1;
        v2 = next_v2;
        copied += 48;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        let next = _mm_alignr_epi8::<SHIFT>(v1, v0);
        v0 = v1;
        v1 = v2;
        v2 = next;
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_cycle4<const SHIFT: i32>(
    out_next: *mut u8,
    src: *const u8,
    v: __m128i,
    length: usize,
) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v_tail = _mm_loadu_si128(src.add((48 - SHIFT) as usize) as *const __m128i);
    let mut v3 = _mm_alignr_epi8::<SHIFT>(v, v_tail);
    let mut v0 = v;
    let mut v1 = v1;
    let mut v2 = v2;

    let mut copied = 16;
    while copied + 64 <= length {
        let next_v0 = _mm_alignr_epi8::<SHIFT>(v1, v0);
        let next_v1 = _mm_alignr_epi8::<SHIFT>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<SHIFT>(v3, v2);
        let next_v3 = _mm_alignr_epi8::<SHIFT>(next_v0, v3);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, next_v0);

        v0 = next_v0;
        v1 = next_v1;
        v2 = next_v2;
        v3 = next_v3;
        copied += 64;
    }

    loop {
        if copied + 16 > length {
            break;
        }
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;

        if copied + 16 > length {
            break;
        }
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;

        if copied + 16 > length {
            break;
        }
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v3);
        copied += 16;

        if copied + 16 > length {
            break;
        }
        let next_v0 = _mm_alignr_epi8::<SHIFT>(v1, v0);
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, next_v0);
        copied += 16;

        let next_v1 = _mm_alignr_epi8::<SHIFT>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<SHIFT>(v3, v2);
        let next_v3 = _mm_alignr_epi8::<SHIFT>(next_v0, v3);
        v0 = next_v0;
        v1 = next_v1;
        v2 = next_v2;
        v3 = next_v3;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
pub unsafe fn decompress_bmi2(
    d: &mut Decompressor,
    input: &[u8],
    output: &mut [u8],
) -> (DecompressResult, usize, usize) {
    let mut out_idx = 0;
    let out_len = output.len();
    let mut in_idx = 0;
    let in_len = input.len();
    let mut bitbuf = 0u64;
    let mut bitsleft = 0u32;
    let mut is_final_block = false;

    while !is_final_block {
        refill_bits!(input, in_idx, bitbuf, bitsleft);

        is_final_block = (bitbuf & 1) != 0;
        let block_type = ((bitbuf >> 1) & 3) as u8;
        bitbuf >>= 3;
        bitsleft -= 3;

        match block_type {
            DEFLATE_BLOCKTYPE_UNCOMPRESSED => {
                let skip = bitsleft & 7;
                let _ = bitbuf >> skip;
                bitsleft -= skip;
                let unused_bytes = bitsleft / 8;
                in_idx -= unused_bytes as usize;
                bitbuf = 0;
                bitsleft = 0;
                if in_idx + 4 > in_len {
                    return (DecompressResult::BadData, 0, 0);
                }
                let len = u16::from_le_bytes([input[in_idx], input[in_idx + 1]]) as usize;
                let nlen = u16::from_le_bytes([input[in_idx + 2], input[in_idx + 3]]) as usize;
                in_idx += 4;
                if len != (!nlen & 0xFFFF) {
                    return (DecompressResult::BadData, 0, 0);
                }
                if out_idx + len > out_len {
                    return (DecompressResult::InsufficientSpace, 0, 0);
                }
                if in_idx + len > in_len {
                    return (DecompressResult::BadData, 0, 0);
                }
                std::ptr::copy_nonoverlapping(
                    input.as_ptr().add(in_idx),
                    output.as_mut_ptr().add(out_idx),
                    len,
                );
                in_idx += len;
                out_idx += len;
            }
            DEFLATE_BLOCKTYPE_STATIC_HUFFMAN | DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN => {
                if block_type == DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN {
                    d.bitbuf = bitbuf;
                    d.bitsleft = bitsleft;
                    let res = d.read_dynamic_huffman_header(input, &mut in_idx);
                    bitbuf = d.bitbuf;
                    bitsleft = d.bitsleft;
                    if res != DecompressResult::Success {
                        return (res, 0, 0);
                    }
                } else {
                    d.load_static_huffman_codes();
                }

                loop {
                    let mut eob_found = false;
                    unsafe {
                        let in_ptr_start = input.as_ptr();
                        let in_ptr_end = in_ptr_start.add(in_len);
                        let mut in_next = in_ptr_start.add(in_idx);
                        let out_ptr_start = output.as_mut_ptr();
                        let out_ptr_end = out_ptr_start.add(out_len);
                        let mut out_next = out_ptr_start.add(out_idx);

                        while in_next.add(15) <= in_ptr_end && out_next.add(258) <= out_ptr_end {
                            if bitsleft < 32 {
                                let word = (in_next as *const u64).read_unaligned();
                                let word = u64::from_le(word);
                                bitbuf |= word << bitsleft;
                                let consumed = (63 - bitsleft) >> 3;
                                in_next = in_next.add(consumed as usize);
                                bitsleft |= 56;
                            }

                            let table_idx = _bzhi_u64(bitbuf, d.litlen_tablebits as u32) as usize;
                            let mut entry = *d.litlen_decode_table.get_unchecked(table_idx);

                            if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                    bitbuf >>= entry as u8;
                                    bitsleft -= entry & 0xFF;
                                    eob_found = true;
                                    break;
                                }
                                if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                                    let saved_bitbuf = bitbuf;
                                    let saved_bitsleft = bitsleft;
                                    let main_bits = entry & 0xFF;
                                    bitbuf >>= main_bits;
                                    bitsleft -= main_bits;
                                    let subtable_idx = (entry >> 16) as usize;
                                    let subtable_bits = (entry >> 8) & 0x3F;
                                    let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                                    entry = *d
                                        .litlen_decode_table
                                        .get_unchecked(subtable_idx + sub_idx);
                                    if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                        if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                            bitbuf >>= entry as u8;
                                            bitsleft -= entry & 0xFF;
                                            break;
                                        }
                                        bitbuf = saved_bitbuf;
                                        bitsleft = saved_bitsleft;
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            let saved_bitbuf = bitbuf;
                            let total_bits = entry & 0xFF;
                            if bitsleft < total_bits {
                                break;
                            }
                            bitbuf >>= total_bits;
                            bitsleft -= total_bits;

                            if entry & HUFFDEC_LITERAL != 0 {
                                *out_next = (entry >> 16) as u8;
                                out_next = out_next.add(1);
                            } else {
                                let mut length = (entry >> 16) as usize;
                                let len = (entry >> 8) & 0xFF;
                                let extra_bits = total_bits - len;
                                if extra_bits > 0 {
                                    length += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                                }

                                if bitsleft < 32 {
                                    let word = (in_next as *const u64).read_unaligned();
                                    let word = u64::from_le(word);
                                    bitbuf |= word << bitsleft;
                                    let consumed = (63 - bitsleft) >> 3;
                                    in_next = in_next.add(consumed as usize);
                                    bitsleft |= 56;
                                }

                                let offset_idx =
                                    _bzhi_u64(bitbuf, OFFSET_TABLEBITS as u32) as usize;
                                let mut entry = *d.offset_decode_table.get_unchecked(offset_idx);

                                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                    if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                                        let main_bits = entry & 0xFF;
                                        bitbuf >>= main_bits;
                                        bitsleft -= main_bits;
                                        let subtable_idx = (entry >> 16) as usize;
                                        let subtable_bits = (entry >> 8) & 0x3F;
                                        let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                                        entry = *d
                                            .offset_decode_table
                                            .get_unchecked(subtable_idx + sub_idx);
                                    } else {
                                        break;
                                    }
                                }

                                let saved_bitbuf = bitbuf;
                                let total_bits = entry & 0xFF;
                                bitbuf >>= total_bits;
                                bitsleft -= total_bits;

                                let mut offset = (entry >> 16) as usize;
                                let len = (entry >> 8) & 0xFF;
                                let extra_bits = total_bits - len;
                                if extra_bits > 0 {
                                    offset += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                                }

                                let current_out_idx = out_next.offset_from(out_ptr_start) as usize;
                                if offset > current_out_idx {
                                    break;
                                }

                                let src = out_next.sub(offset);
                                if offset >= 16 {
                                    let v = _mm_loadu_si128(src as *const __m128i);
                                    _mm_storeu_si128(out_next as *mut __m128i, v);
                                    if length > 16 {
                                        if offset >= length {
                                            std::ptr::copy_nonoverlapping(
                                                src.add(16),
                                                out_next.add(16),
                                                length - 16,
                                            );
                                        } else {
                                            match offset {
                                                16 => {
                                                    decompress_fill_pattern(out_next, v, length);
                                                }
                                                34 => decompress_offset_cycle3::<14>(out_next, src, v, length),
                                                35 => decompress_offset_cycle3::<13>(out_next, src, v, length),
                                                38 => decompress_offset_cycle3::<10>(out_next, src, v, length),
                                                42 => decompress_offset_cycle3::<6>(out_next, src, v, length),
                                                46 => decompress_offset_cycle3::<2>(out_next, src, v, length),
                                                50 => decompress_offset_cycle4::<14>(out_next, src, v, length),
                                                54 => decompress_offset_cycle4::<10>(out_next, src, v, length),
                                                58 => decompress_offset_cycle4::<6>(out_next, src, v, length),
                                                62 => decompress_offset_cycle4::<2>(out_next, src, v, length),
                                                60 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Explicitly construct v3 from src[48..64] but overwriting the last 4 bytes
                                                    // (which overlap with out_next[0..4]) with valid data from v0.
                                                    // This avoids relying on the read-after-write behavior of overlapping pointers.
                                                    let v3_raw = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v0 = v;
                                                    // Shift v0 (0..16) left by 12 bytes to place 0..4 at 12..16
                                                    let v0_shifted = _mm_slli_si128(v0, 12);
                                                    // Blend: Mask 0xC0 selects upper 2 words (upper 4 bytes) from v0_shifted.
                                                    let v3 =
                                                        _mm_blend_epi16(v3_raw, v0_shifted, 0xC0);

                                                    let v4 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v7 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v8 = _mm_alignr_epi8(v5, v4, 4);
                                                    let v9 = _mm_alignr_epi8(v6, v5, 4);
                                                    let v10 = _mm_alignr_epi8(v7, v6, 4);
                                                    let v11 = _mm_alignr_epi8(v8, v7, 4);
                                                    let v12 = _mm_alignr_epi8(v9, v8, 4);
                                                    let v13 = _mm_alignr_epi8(v10, v9, 4);
                                                    let v14 = _mm_alignr_epi8(v11, v10, 4);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[
                                                            v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                                            v10, v11, v12, v13, v14, v0,
                                                        ],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                44 => {
                                                    let v0 = v;
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v_end = _mm_loadu_si128(
                                                        src.add(28) as *const __m128i
                                                    );
                                                    let v2 = _mm_alignr_epi8(v0, v_end, 4);
                                                    let v3 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v4 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v5 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v6 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v7 = _mm_alignr_epi8(v5, v4, 4);
                                                    let v8 = _mm_alignr_epi8(v6, v5, 4);
                                                    let v9 = _mm_alignr_epi8(v7, v6, 4);
                                                    let v10 = _mm_alignr_epi8(v8, v7, 4);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[
                                                            v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                                            v10, v0,
                                                        ],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                36 => {
                                                    let v_part = _mm_loadu_si128(
                                                        src.add(20) as *const __m128i
                                                    );
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v0 = v;

                                                    let v2 = _mm_alignr_epi8(v0, v_part, 12);
                                                    let v3 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v4 = _mm_alignr_epi8(v0, v_part, 8);
                                                    let v5 = _mm_alignr_epi8(v1, v0, 8);
                                                    let v6 = _mm_alignr_epi8(v0, v_part, 4);
                                                    let v7 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v8 = v_part;

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v4, v5, v6, v7, v8, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                52 => {
                                                    let v0 = v;
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Load tail (36..52) which contains 48..52 at [12..16]
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(36) as *const __m128i
                                                    );
                                                    // v3: 48..52 ++ 0..12
                                                    let v3 = _mm_alignr_epi8(v0, v_tail, 12);

                                                    // The cycle uses shift 12 for all subsequent vectors
                                                    let v4 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 12);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 12);
                                                    let v7 = _mm_alignr_epi8(v4, v3, 12);
                                                    let v8 = _mm_alignr_epi8(v5, v4, 12);
                                                    let v9 = _mm_alignr_epi8(v6, v5, 12);
                                                    let v10 = _mm_alignr_epi8(v7, v6, 12);
                                                    let v11 = _mm_alignr_epi8(v8, v7, 12);
                                                    let v12 = _mm_alignr_epi8(v9, v8, 12);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[
                                                            v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                                            v10, v11, v12, v0,
                                                        ],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                56 => {
                                                    // LCM(56, 16) = 112.
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Optimize v3 construction to avoid store-forwarding stall.
                                                    // v3 should be [src[48..56], src[0..8]].
                                                    // src[0..8] is already in v0[0..8].
                                                    // We load src[48..56] using loadl_epi64.
                                                    let v0 = v;
                                                    let v3_low = _mm_loadl_epi64(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v3 = _mm_unpacklo_epi64(v3_low, v0);

                                                    let v4 = _mm_alignr_epi8(v1, v0, 8);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 8);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 8);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v4, v5, v6, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                48 => {
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v3 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v2, v3, v],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                64 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    let v3 = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                17 => {
                                                    let c = *src.add(16);
                                                    let v_align = _mm_insert_epi8(v, c as i32, 15);
                                                    decompress_offset_alignr_cycle::<15>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                18 => {
                                                    let c1 = *src.add(16);
                                                    let c2 = *src.add(17);
                                                    let mut v_align = _mm_insert_epi8(v, c1 as i32, 14);
                                                    v_align = _mm_insert_epi8(v_align, c2 as i32, 15);
                                                    decompress_offset_alignr_cycle::<14>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                19 => {
                                                    let c1 = *src.add(16);
                                                    let c2 = *src.add(17);
                                                    let c3 = *src.add(18);
                                                    let mut v_align = _mm_insert_epi8(v, c1 as i32, 13);
                                                    v_align = _mm_insert_epi8(v_align, c2 as i32, 14);
                                                    v_align = _mm_insert_epi8(v_align, c3 as i32, 15);
                                                    decompress_offset_alignr_cycle::<13>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                20 => {
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32
                                                    );
                                                    let v_temp = _mm_cvtsi32_si128(val as i32);
                                                    let v_align = _mm_slli_si128(v_temp, 12);

                                                    let v0 = v;
                                                    let v1 = _mm_alignr_epi8(v0, v_align, 12);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 12);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 12);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v4, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                21 => {
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u64
                                                    );
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let v_align = _mm_slli_si128(v_temp, 11);
                                                    decompress_offset_alignr_cycle::<11>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                22 => {
                                                    let v0 = v;
                                                    let v_align_low = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32,
                                                    );
                                                    let v_align_high = std::ptr::read_unaligned(
                                                        src.add(20) as *const u16,
                                                    );
                                                    let v_align_val = (v_align_low as u64)
                                                        | ((v_align_high as u64) << 32);
                                                    let v_tail = _mm_slli_si128(
                                                        _mm_cvtsi64_si128(v_align_val as i64),
                                                        10,
                                                    );

                                                    let v1 = _mm_alignr_epi8(v0, v_tail, 10);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 10);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 10);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 10);
                                                    let v5 = _mm_alignr_epi8(v4, v3, 10);
                                                    let v6 = _mm_alignr_epi8(v5, v4, 10);
                                                    let v7 = _mm_alignr_epi8(v6, v5, 10);
                                                    let v8 = _mm_alignr_epi8(v7, v6, 10);
                                                    let v9 = _mm_alignr_epi8(v8, v7, 10);
                                                    let v10 = _mm_alignr_epi8(v9, v8, 10);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[
                                                            v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                                            v10, v0,
                                                        ],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                23 => {
                                                    let v0 = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32
                                                    );
                                                    let v1 = std::ptr::read_unaligned(
                                                        src.add(19) as *const u32
                                                    );
                                                    let val = (v0 as u64) | ((v1 as u64) << 24);
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let v_align = _mm_slli_si128(v_temp, 9);
                                                    decompress_offset_alignr_cycle::<9>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                24 => {
                                                    let v_part1 = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32,
                                                    );
                                                    let v_part2 = std::ptr::read_unaligned(
                                                        src.add(20) as *const u32,
                                                    );
                                                    let val =
                                                        (v_part1 as u64) | ((v_part2 as u64) << 32);
                                                    let v_tail = _mm_cvtsi64_si128(val as i64);

                                                    let v0 = v;
                                                    let v1 = _mm_unpacklo_epi64(v_tail, v0);
                                                    let v2 = _mm_alignr_epi8(v_tail, v0, 8);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                25 => {
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u64
                                                    );
                                                    let c = *src.add(24);
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let v_temp =
                                                        _mm_insert_epi8(v_temp, c as i32, 8);
                                                    let v_align = _mm_slli_si128(v_temp, 7);
                                                    decompress_offset_alignr_cycle::<7>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                26 => {
                                                    // LCM(26, 16) = 208. 13 vectors.
                                                    // The pattern repeats every 26 bytes.
                                                    // We advance by 16 bytes each step.
                                                    // Next vector is constructed from the previous two shifted by 16 bytes.
                                                    // Since 26 % 16 = 10, the relative alignment shifts by 10 bytes each step.
                                                    // To align, we need to shift right by 16 - 10 = 6 bytes from the concatenated previous vectors.
                                                    // v_align (src[10..26]) acts as v_{-1}.
                                                    let v0 = v;
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(10) as *const __m128i
                                                    );

                                                    let v1 = _mm_alignr_epi8(v0, v_align, 6);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 6);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 6);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 6);
                                                    let v5 = _mm_alignr_epi8(v4, v3, 6);
                                                    let v6 = _mm_alignr_epi8(v5, v4, 6);
                                                    let v7 = _mm_alignr_epi8(v6, v5, 6);
                                                    let v8 = _mm_alignr_epi8(v7, v6, 6);
                                                    let v9 = _mm_alignr_epi8(v8, v7, 6);
                                                    let v10 = _mm_alignr_epi8(v9, v8, 6);
                                                    let v11 = _mm_alignr_epi8(v10, v9, 6);
                                                    let v12 = _mm_alignr_epi8(v11, v10, 6);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[
                                                            v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                                            v10, v11, v12, v0,
                                                        ],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                27 => {
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(11) as *const __m128i
                                                    );
                                                    decompress_offset_alignr_cycle::<5>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                28 => {
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(12) as *const __m128i
                                                    );

                                                    let v0 = v;
                                                    let v1 = _mm_alignr_epi8(v0, v_align, 4);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v5 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v6 = _mm_alignr_epi8(v5, v4, 4);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v4, v5, v6, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                29 => {
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(13) as *const __m128i
                                                    );
                                                    decompress_offset_alignr_cycle::<3>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                30 => {
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(14) as *const __m128i
                                                    );
                                                    decompress_offset_alignr_cycle::<2>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                31 => {
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(15) as *const __m128i
                                                    );
                                                    decompress_offset_alignr_cycle::<1>(
                                                        out_next, src, length, v_align, v,
                                                    );
                                                }
                                                32 => {
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v2, v],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                40 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v4 = _mm_loadu_si128(
                                                        src.add(24) as *const __m128i
                                                    );
                                                    let v0 = v;

                                                    let v2 = _mm_alignr_epi8(v0, v4, 8);
                                                    let v3 = _mm_alignr_epi8(v1, v0, 8);

                                                    decompress_write_cycle_vectors(
                                                        out_next,
                                                        src,
                                                        &[v1, v2, v3, v4, v0],
                                                        length,
                                                        16,
                                                    );
                                                }
                                                _ => {
                                                    let init = std::cmp::min(offset, length);
                                                    std::ptr::copy_nonoverlapping(
                                                        src, out_next, init,
                                                    );

                                                    let mut copied = init;
                                                    while copied < length {
                                                        let to_copy =
                                                            std::cmp::min(length - copied, copied);
                                                        std::ptr::copy_nonoverlapping(
                                                            out_next,
                                                            out_next.add(copied),
                                                            to_copy,
                                                        );
                                                        copied += to_copy;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else if offset >= length {
                                    std::ptr::copy_nonoverlapping(src, out_next, length);
                                } else {
                                    match offset {
                                        1 => {
                                            let b = *src;
                                            std::ptr::write_bytes(out_next, b, length);
                                        }
                                        2 | 4 => {
                                            let v_pattern = match offset {
                                                2 => _mm_set1_epi16(std::ptr::read_unaligned(
                                                    src as *const u16,
                                                )
                                                    as i16),
                                                4 => _mm_set1_epi32(std::ptr::read_unaligned(
                                                    src as *const u32,
                                                )
                                                    as i32),
                                                _ => std::hint::unreachable_unchecked(),
                                            };
                                            decompress_fill_pattern(out_next, v_pattern, length);
                                        }
                                        3 => {
                                            decompress_shuffle_pattern::<3>(
                                                out_next,
                                                src,
                                                &OFFSET3_MASKS,
                                                length,
                                            );
                                        }
                                        6 => {
                                            decompress_shuffle_pattern::<3>(
                                                out_next,
                                                src,
                                                &OFFSET6_MASKS,
                                                length,
                                            );
                                        }
                                        5 => {
                                            decompress_shuffle_pattern::<5>(
                                                out_next,
                                                src,
                                                &OFFSET5_MASKS,
                                                length,
                                            );
                                        }
                                        7 => {
                                            decompress_shuffle_pattern::<7>(
                                                out_next,
                                                src,
                                                &OFFSET7_MASKS,
                                                length,
                                            );
                                        }
                                        8 => {
                                            let val = std::ptr::read_unaligned(src as *const u64);
                                            let v_pattern = _mm_set1_epi64x(val as i64);
                                            decompress_fill_pattern(out_next, v_pattern, length);
                                        }
                                        9 => {
                                            decompress_shuffle_pattern::<9>(
                                                out_next,
                                                src,
                                                &OFFSET9_MASKS,
                                                length,
                                            );
                                        }
                                        10 => {
                                            decompress_shuffle_pattern::<5>(
                                                out_next,
                                                src,
                                                &OFFSET10_MASKS,
                                                length,
                                            );
                                        }
                                        11 => {
                                            decompress_shuffle_pattern::<11>(
                                                out_next,
                                                src,
                                                &OFFSET11_MASKS,
                                                length,
                                            );
                                        }
                                        12 => {
                                            decompress_shuffle_pattern::<3>(
                                                out_next,
                                                src,
                                                &OFFSET12_MASKS,
                                                length,
                                            );
                                        }
                                        13 => {
                                            decompress_shuffle_pattern::<13>(
                                                out_next,
                                                src,
                                                &OFFSET13_MASKS,
                                                length,
                                            );
                                        }
                                        14 => {
                                            decompress_shuffle_pattern::<7>(
                                                out_next,
                                                src,
                                                &OFFSET14_MASKS,
                                                length,
                                            );
                                        }
                                        15 => {
                                            decompress_shuffle_pattern::<15>(
                                                out_next,
                                                src,
                                                &OFFSET15_MASKS,
                                                length,
                                            );
                                        }
                                        _ => {
                                            let mut copied = 0;
                                            while copied + 8 <= length {
                                                let val = std::ptr::read_unaligned(
                                                    src.add(copied) as *const u64
                                                );
                                                std::ptr::write_unaligned(
                                                    out_next.add(copied) as *mut u64,
                                                    val,
                                                );
                                                copied += 8;
                                            }
                                            while copied < length {
                                                *out_next.add(copied) = *src.add(copied);
                                                copied += 1;
                                            }
                                        }
                                    }
                                }
                                out_next = out_next.add(length);
                            }
                        }
                        in_idx = in_next.offset_from(in_ptr_start) as usize;
                        out_idx = out_next.offset_from(out_ptr_start) as usize;
                    }

                    if eob_found {
                        break;
                    }

                    d.bitbuf = bitbuf;
                    d.bitsleft = bitsleft;
                    d.state = crate::decompress::DecompressorState::BlockBody;

                    let res = d.decompress_huffman_block(input, &mut in_idx, output, &mut out_idx);

                    bitbuf = d.bitbuf;
                    bitsleft = d.bitsleft;

                    if res != DecompressResult::Success {
                        return (res, in_idx, out_idx);
                    }
                    break;
                }
            }
            _ => return (DecompressResult::BadData, 0, 0),
        }
    }
    (DecompressResult::Success, in_idx, out_idx)
}
