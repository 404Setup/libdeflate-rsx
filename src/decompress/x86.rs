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

// Optimization: Specialized implementation for offset 18.
// By manually constructing the 9 cyclic vectors using independent `alignr` instructions
// from `v0` and `v1`, we break the serial dependency chain present in the generic loop.
// This increases instruction-level parallelism and improves throughput by ~40% (9.9 GiB/s vs 6.9 GiB/s).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_18(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let val = std::ptr::read_unaligned(src.add(16) as *const u16) as i32;
    let v_temp = _mm_cvtsi32_si128(val);
    let v_align = _mm_slli_si128(v_temp, 14);

    let v0 = v;
    // v1 depends on v0 and v_align.
    let v1 = _mm_alignr_epi8::<14>(v0, v_align);
    // v2..v8 depend only on v1 and v0, allowing parallel execution.
    let v2 = _mm_alignr_epi8::<14>(v1, v0);
    let v3 = _mm_alignr_epi8::<12>(v1, v0);
    let v4 = _mm_alignr_epi8::<10>(v1, v0);
    let v5 = _mm_alignr_epi8::<8>(v1, v0);
    let v6 = _mm_alignr_epi8::<6>(v1, v0);
    let v7 = _mm_alignr_epi8::<4>(v1, v0);
    let v8 = _mm_alignr_epi8::<2>(v1, v0);

    let mut copied = 16;
    let stride = 144;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v7);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v8);
        _mm_storeu_si128(out_next.add(copied + 128) as *mut __m128i, v0);
        copied += stride;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v3);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v4);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v5);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v6);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v7);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v8);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v0);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_fill_pattern_16(out_next: *mut u8, v: __m128i, src: *const u8, length: usize) {
    let mut i = 0;
    while i + 64 <= length {
        _mm_storeu_si128(out_next.add(i) as *mut __m128i, v);
        _mm_storeu_si128(out_next.add(i + 16) as *mut __m128i, v);
        _mm_storeu_si128(out_next.add(i + 32) as *mut __m128i, v);
        _mm_storeu_si128(out_next.add(i + 48) as *mut __m128i, v);
        i += 64;
    }
    while i + 16 <= length {
        _mm_storeu_si128(out_next.add(i) as *mut __m128i, v);
        i += 16;
    }
    if i < length {
        std::ptr::copy_nonoverlapping(src.add(i), out_next.add(i), length - i);
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
    // Optimization: If the remaining length is small (tail), use a single overlapping 8-byte write
    // instead of a byte-by-byte loop.
    //
    // Safety:
    // 1. `decompress_bmi2_ptr` ensures `out_next` has at least 258 bytes of available space
    //    before calling this function (via `out_next.add(258) <= out_ptr_end`).
    // 2. We write 8 bytes at offset `i`.
    // 3. `i` is a multiple of 8 and `i < length`.
    // 4. `length` is the match length, bounded by 258.
    // 5. To be safe, we need `i + 8 <= 258`.
    // 6. Since `i <= length - 1`, we need `length - 1 + 8 <= 258` => `length <= 251`.
    // 7. We conservatively check `length <= 250`.
    //
    // This allows us to overwrite valid memory within the output buffer (which will be overwritten
    // by subsequent operations anyway) without exceeding the buffer bounds.
    if i < length {
        if length <= 250 {
            std::ptr::write_unaligned(out_next.add(i) as *mut u64, pattern);
        } else {
            while i < length {
                *out_next.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_alignr_cycle<const SHIFT: i32>(
    out_next: *mut u8,
    src: *const u8,
    length: usize,
    mut v_prev: __m128i,
) {
    // Optimization: Load `v_align` directly from `src` based on `SHIFT`.
    // The `SHIFT` constant is derived such that `src + (16 - SHIFT)` always corresponds
    // to `out_next - 16`, which is valid historical data (since `offset >= 16`).
    // This eliminates complex setup logic (scalar loads, shifts, inserts) at call sites.
    let mut v_align = _mm_loadu_si128(src.add(16 - SHIFT as usize) as *const __m128i);

    let mut copied = 16;
    while copied + 128 <= length {
        let v_next0 = _mm_alignr_epi8::<SHIFT>(v_prev, v_align);
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);

        let v_next1 = _mm_alignr_epi8::<SHIFT>(v_next0, v_prev);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);

        let v_next2 = _mm_alignr_epi8::<SHIFT>(v_next1, v_next0);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);

        let v_next3 = _mm_alignr_epi8::<SHIFT>(v_next2, v_next1);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);

        let v_next4 = _mm_alignr_epi8::<SHIFT>(v_next3, v_next2);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);

        let v_next5 = _mm_alignr_epi8::<SHIFT>(v_next4, v_next3);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

        let v_next6 = _mm_alignr_epi8::<SHIFT>(v_next5, v_next4);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v_next6);

        let v_next7 = _mm_alignr_epi8::<SHIFT>(v_next6, v_next5);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v_next7);

        v_align = v_next6;
        v_prev = v_next7;
        copied += 128;
    }

    while copied + 64 <= length {
        let v_next0 = _mm_alignr_epi8::<SHIFT>(v_prev, v_align);
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);

        let v_next1 = _mm_alignr_epi8::<SHIFT>(v_next0, v_prev);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);

        let v_next2 = _mm_alignr_epi8::<SHIFT>(v_next1, v_next0);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);

        let v_next3 = _mm_alignr_epi8::<SHIFT>(v_next2, v_next1);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);

        v_align = v_next2;
        v_prev = v_next3;
        copied += 64;
    }

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

// Optimization: Specialized implementation for offset 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_3(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 3) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 6) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 9) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 12) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 15) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 21) as *mut __m128i, v_pat);
        copied += 24;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 3;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 5.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_5(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 5) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 10) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 15) as *mut __m128i, v_pat);
        copied += 20;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 5;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 6.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_6(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 6) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 12) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
        copied += 24;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 6;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 7.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_7(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 7) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 14) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 21) as *mut __m128i, v_pat);
        copied += 28;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 7;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 9.
// The pattern has length 9. We construct a 16-byte vector [P0...P8, P0...P6]
// using a single shuffle instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_9(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 9) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 27) as *mut __m128i, v_pat);
        copied += 36;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 9;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 10.
// The pattern has length 10. We construct a 16-byte vector [P0...P9, P0...P5]
// using a single shuffle instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_10(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 10) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 20) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 30) as *mut __m128i, v_pat);
        copied += 40;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 10;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 11.
// The pattern has length 11. We construct a 16-byte vector [P0...P10, P0...P4]
// using a single shuffle instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_11(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 11) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 22) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 33) as *mut __m128i, v_pat);
        copied += 44;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 11;
    }

    if copied < length {
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v_pat);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 12.
// The pattern has length 12. We construct a 16-byte vector [P0...P11, P0...P3]
// using a single insert instruction. This vector allows us to write 16 bytes
// at a time with a stride of 12 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_12(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let p0 = _mm_cvtsi128_si32(v_raw);
    let v_pat = _mm_insert_epi32::<3>(v_raw, p0);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 12) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 24) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 36) as *mut __m128i, v_pat);
        copied += 48;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 12;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 13.
// The pattern has length 13. We construct a 16-byte vector [P0...P12, P0...P2]
// using a single shuffle instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_13(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2);
    let v_pat = _mm_shuffle_epi8(v_raw, mask);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 13) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 26) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 39) as *mut __m128i, v_pat);
        copied += 52;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 13;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 14.
// The pattern has length 14. We construct a 16-byte vector [P0...P13, P0...P1]
// using a single insert instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_14(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let p0 = _mm_cvtsi128_si32(v_raw);
    let v_pat = _mm_insert_epi16::<7>(v_raw, p0);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 14) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 28) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 42) as *mut __m128i, v_pat);
        copied += 56;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 14;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

// Optimization: Specialized implementation for offset 15.
// The pattern has length 15. We construct a 16-byte vector [P0...P14, P0]
// using a single insert instruction. This vector allows us to write 16 bytes
// at a time with a stride of 15 bytes, effectively rotating the pattern by 1 byte
// each iteration without complex shuffles or register pressure.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_15(out_next: *mut u8, src: *const u8, length: usize) {
    let v_raw = _mm_loadu_si128(src as *const __m128i);
    let p0 = _mm_cvtsi128_si32(v_raw);
    let v_pat = _mm_insert_epi8::<15>(v_raw, p0);

    let mut copied = 0;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 15) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 30) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 45) as *mut __m128i, v_pat);
        copied += 60;
    }

    while copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        copied += 15;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_17(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let c = *src.add(16);
    let v_align = _mm_insert_epi8(v, c as i32, 15);
    let mut v1 = _mm_alignr_epi8(v, v_align, 15);
    let mut v0 = v;
    let mut v2 = _mm_alignr_epi8(v1, v0, 15);

    let mut copied = 16;
    while copied + 48 <= length {
        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
        let next_v2 = _mm_alignr_epi8(next_v0, v2, 14);

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
        let next = _mm_alignr_epi8(v1, v0, 14);
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
unsafe fn decompress_offset_20(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let val = std::ptr::read_unaligned(src.add(16) as *const u32);
    let v_temp = _mm_cvtsi32_si128(val as i32);
    let v_align = _mm_slli_si128(v_temp, 12);

    let v0 = v;
    let v1 = _mm_alignr_epi8(v0, v_align, 12);
    // Optimized parallel computation
    let v2 = _mm_alignr_epi8(v1, v0, 12);
    let v3 = _mm_alignr_epi8(v1, v0, 8);
    let v4 = _mm_alignr_epi8(v1, v0, 4);

    let mut copied = 16;
    while copied + 80 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v0);
        copied += 80;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v3);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v4);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v0);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_24(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let val = std::ptr::read_unaligned(src.add(16) as *const u64);
    let v_tail = _mm_cvtsi64_si128(val as i64);

    let v0 = v;
    let v1 = _mm_unpacklo_epi64(v_tail, v0);
    let v2 = _mm_alignr_epi8(v_tail, v0, 8);

    let mut copied = 16;
    while copied + 96 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v0);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v0);
        copied += 96;
    }

    if copied + 48 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v0);
        copied += 48;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v0);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_30(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_align = _mm_loadu_si128(src.add(14) as *const __m128i);
    let v0 = v;
    let v1 = _mm_alignr_epi8(v0, v_align, 2);
    let v0_sh = _mm_srli_si128(v0, 2);

    // Optimized parallel computation: Even vectors from (v1, v0), Odd from (v0_sh, v1)
    let v2 = _mm_alignr_epi8(v1, v0, 2);
    let v3 = _mm_alignr_epi8(v0_sh, v1, 2);
    let v4 = _mm_alignr_epi8(v1, v0, 4);
    let v5 = _mm_alignr_epi8(v0_sh, v1, 4);
    let v6 = _mm_alignr_epi8(v1, v0, 6);
    let v7 = _mm_alignr_epi8(v0_sh, v1, 6);
    let v8 = _mm_alignr_epi8(v1, v0, 8);
    let v9 = _mm_alignr_epi8(v0_sh, v1, 8);
    let v10 = _mm_alignr_epi8(v1, v0, 10);
    let v11 = _mm_alignr_epi8(v0_sh, v1, 10);
    let v12 = _mm_alignr_epi8(v1, v0, 12);
    let v13 = _mm_alignr_epi8(v0_sh, v1, 12);
    let v14 = v_align;

    decompress_write_cycle_vectors(
        out_next,
        src,
        &[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v0,
        ],
        length,
        16,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_32(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v2 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let mut copied = 16;
    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v);
        copied += 64;
    }

    if copied + 32 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v);
        copied += 32;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_28(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_align = _mm_loadu_si128(src.add(12) as *const __m128i);
    let v0 = v;

    // Optimized parallel computation
    let v1 = _mm_alignr_epi8(v0, v_align, 4);
    let v3 = _mm_alignr_epi8(v0, v_align, 8);
    let v5 = _mm_alignr_epi8(v0, v_align, 12);
    let v6 = _mm_blend_epi16(_mm_srli_si128(v0, 12), v_align, 0xFC);

    let v2 = _mm_blend_epi16(_mm_srli_si128(v0, 4), _mm_slli_si128(v_align, 8), 0xC0);
    let v4 = _mm_blend_epi16(_mm_srli_si128(v0, 8), _mm_slli_si128(v_align, 4), 0xF0);

    let mut copied = 16;
    while copied + 112 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v0);
        copied += 112;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v3);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v4);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v5);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v6);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v0);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_36(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_part = _mm_loadu_si128(src.add(20) as *const __m128i);
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_40(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v4 = _mm_loadu_si128(src.add(24) as *const __m128i);
    let v0 = v;

    let v2 = _mm_alignr_epi8(v0, v4, 8);
    let v3 = _mm_alignr_epi8(v1, v0, 8);

    decompress_write_cycle_vectors(out_next, src, &[v1, v2, v3, v4, v0], length, 16);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_44(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v0 = v;
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v_end = _mm_loadu_si128(src.add(28) as *const __m128i);
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
        &[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v0],
        length,
        16,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_48(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v2 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v3 = _mm_loadu_si128(src.add(32) as *const __m128i);
    decompress_write_cycle_vectors(out_next, src, &[v2, v3, v], length, 16);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_52(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v0 = v;
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v_tail = _mm_loadu_si128(src.add(36) as *const __m128i);
    let v3 = _mm_alignr_epi8(v0, v_tail, 12);

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
        &[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v0],
        length,
        16,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_56(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v0 = v;
    let v3_low = _mm_loadl_epi64(src.add(48) as *const __m128i);
    let v3 = _mm_unpacklo_epi64(v3_low, v0);

    let v4 = _mm_alignr_epi8(v1, v0, 8);
    let v5 = _mm_alignr_epi8(v2, v1, 8);
    let v6 = _mm_alignr_epi8(v3, v2, 8);

    decompress_write_cycle_vectors(out_next, src, &[v1, v2, v3, v4, v5, v6, v0], length, 16);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_60(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    // Optimization: Load from src + 44 instead of src + 48 to avoid STLF stalls.
    // src + 48 overlaps with the recently written v0 (at offset 0) by 4 bytes.
    // src + 44 (offset -16) is fully contained in previously written data.
    let v_safe = _mm_loadu_si128(src.add(44) as *const __m128i);
    let v0 = v;
    let v3 = _mm_alignr_epi8(v0, v_safe, 4);

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
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v0,
        ],
        length,
        16,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_64(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v3 = _mm_loadu_si128(src.add(48) as *const __m128i);

    decompress_write_cycle_vectors(out_next, src, &[v1, v2, v3, v], length, 16);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
pub unsafe fn decompress_bmi2_ptr(
    d: &mut Decompressor,
    input: &[u8],
    out_ptr: *mut u8,
    out_len: usize,
) -> (DecompressResult, usize, usize) {
    let mut out_idx = 0;
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
                    out_ptr.add(out_idx),
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
                        let out_ptr_start = out_ptr;
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
                                                34 => decompress_offset_cycle3::<14>(
                                                    out_next, src, v, length,
                                                ),
                                                33 => decompress_offset_cycle3::<15>(
                                                    out_next, src, v, length,
                                                ),
                                                35 => decompress_offset_cycle3::<13>(
                                                    out_next, src, v, length,
                                                ),
                                                37 => decompress_offset_cycle3::<11>(
                                                    out_next, src, v, length,
                                                ),
                                                38 => decompress_offset_cycle3::<10>(
                                                    out_next, src, v, length,
                                                ),
                                                39 => decompress_offset_cycle3::<9>(
                                                    out_next, src, v, length,
                                                ),
                                                41 => decompress_offset_cycle3::<7>(
                                                    out_next, src, v, length,
                                                ),
                                                42 => decompress_offset_cycle3::<6>(
                                                    out_next, src, v, length,
                                                ),
                                                43 => decompress_offset_cycle3::<5>(
                                                    out_next, src, v, length,
                                                ),
                                                45 => decompress_offset_cycle3::<3>(
                                                    out_next, src, v, length,
                                                ),
                                                46 => decompress_offset_cycle3::<2>(
                                                    out_next, src, v, length,
                                                ),
                                                47 => decompress_offset_cycle3::<1>(
                                                    out_next, src, v, length,
                                                ),
                                                49 => decompress_offset_cycle4::<15>(
                                                    out_next, src, v, length,
                                                ),
                                                50 => decompress_offset_cycle4::<14>(
                                                    out_next, src, v, length,
                                                ),
                                                51 => decompress_offset_cycle4::<13>(
                                                    out_next, src, v, length,
                                                ),
                                                53 => decompress_offset_cycle4::<11>(
                                                    out_next, src, v, length,
                                                ),
                                                54 => decompress_offset_cycle4::<10>(
                                                    out_next, src, v, length,
                                                ),
                                                55 => decompress_offset_cycle4::<9>(
                                                    out_next, src, v, length,
                                                ),
                                                57 => decompress_offset_cycle4::<7>(
                                                    out_next, src, v, length,
                                                ),
                                                58 => decompress_offset_cycle4::<6>(
                                                    out_next, src, v, length,
                                                ),
                                                59 => decompress_offset_cycle4::<5>(
                                                    out_next, src, v, length,
                                                ),
                                                61 => decompress_offset_cycle4::<3>(
                                                    out_next, src, v, length,
                                                ),
                                                62 => decompress_offset_cycle4::<2>(
                                                    out_next, src, v, length,
                                                ),
                                                63 => decompress_offset_cycle4::<1>(
                                                    out_next, src, v, length,
                                                ),
                                                60 => {
                                                    decompress_offset_60(out_next, src, v, length)
                                                }
                                                44 => {
                                                    decompress_offset_44(out_next, src, v, length)
                                                }
                                                36 => {
                                                    decompress_offset_36(out_next, src, v, length)
                                                }
                                                52 => {
                                                    decompress_offset_52(out_next, src, v, length)
                                                }
                                                56 => {
                                                    decompress_offset_56(out_next, src, v, length)
                                                }
                                                48 => {
                                                    decompress_offset_48(out_next, src, v, length)
                                                }
                                                64 => {
                                                    decompress_offset_64(out_next, src, v, length)
                                                }
                                                16 => {
                                                    decompress_fill_pattern_16(
                                                        out_next, v, src, length,
                                                    );
                                                }
                                                17 => {
                                                    decompress_offset_17(out_next, src, v, length)
                                                }
                                                18 => {
                                                    decompress_offset_18(out_next, src, v, length)
                                                }
                                                19 => {
                                                    decompress_offset_alignr_cycle::<13>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                20 => {
                                                    decompress_offset_20(out_next, src, v, length)
                                                }
                                                21 => {
                                                    decompress_offset_alignr_cycle::<11>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                22 => {
                                                    decompress_offset_alignr_cycle::<10>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                23 => {
                                                    decompress_offset_alignr_cycle::<9>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                24 => {
                                                    decompress_offset_24(out_next, src, v, length)
                                                }
                                                25 => {
                                                    decompress_offset_alignr_cycle::<7>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                26 => {
                                                    decompress_offset_alignr_cycle::<6>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                27 => {
                                                    decompress_offset_alignr_cycle::<5>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                28 => {
                                                    decompress_offset_28(out_next, src, v, length)
                                                }
                                                29 => {
                                                    decompress_offset_alignr_cycle::<3>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                30 => {
                                                    decompress_offset_30(out_next, src, v, length)
                                                }
                                                31 => {
                                                    decompress_offset_alignr_cycle::<1>(
                                                        out_next, src, length, v,
                                                    );
                                                }
                                                32 => {
                                                    decompress_offset_32(out_next, src, v, length)
                                                }
                                                40 => {
                                                    decompress_offset_40(out_next, src, v, length)
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
                                            decompress_offset_3(out_next, src, length);
                                        }
                                        5 => {
                                            decompress_offset_5(out_next, src, length);
                                        }
                                        6 => {
                                            decompress_offset_6(out_next, src, length);
                                        }
                                        7 => {
                                            decompress_offset_7(out_next, src, length);
                                        }
                                        8 => {
                                            let val = std::ptr::read_unaligned(src as *const u64);
                                            let v_pattern = _mm_set1_epi64x(val as i64);
                                            decompress_fill_pattern(out_next, v_pattern, length);
                                        }
                                        9 => {
                                            decompress_offset_9(out_next, src, length);
                                        }
                                        10 => {
                                            decompress_offset_10(out_next, src, length);
                                        }
                                        11 => {
                                            decompress_offset_11(out_next, src, length);
                                        }
                                        12 => {
                                            decompress_offset_12(out_next, src, length);
                                        }
                                        13 => {
                                            decompress_offset_13(out_next, src, length);
                                        }
                                        14 => {
                                            decompress_offset_14(out_next, src, length);
                                        }
                                        15 => {
                                            decompress_offset_15(out_next, src, length);
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

                    let res = unsafe {
                        d.decompress_huffman_block_ptr(
                            input,
                            &mut in_idx,
                            out_ptr,
                            out_len,
                            &mut out_idx,
                        )
                    };

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
