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

macro_rules! decompress_offset_rotated {
    (
        fn_name: $fn_name:ident,
        offset: $offset:expr,
        min_len: $min_len:expr,
        stride: $stride:expr,
        mask: $mask:expr,
        rotated_vecs: [$(($v_name:ident, $v_mask:expr)),+],
        vars: {
            out_next: $out_next:ident,
            src: $src:ident,
            length: $length:ident,
            v_raw: $v_raw:ident,
            v_pat: $v_pat:ident,
            copied: $copied:ident
        },
        unrolled_stores: $unrolled_stores:block,
        intermediate: $intermediate:block
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "bmi2,ssse3,sse4.1")]
        unsafe fn $fn_name($out_next: *mut u8, $src: *const u8, $length: usize) {
            let $v_raw = _mm_loadu_si128($src as *const __m128i);
            let mask = $mask;
            let $v_pat = _mm_shuffle_epi8($v_raw, mask);

            let mut $copied = 0;

            if $length >= $min_len {
                $(
                    let mask = $v_mask;
                    let $v_name = _mm_shuffle_epi8($v_raw, mask);
                )*

                while $copied + $stride <= $length {
                    $unrolled_stores
                    $copied += $stride;
                }
            }

            $intermediate

            while $copied + 16 <= $length {
                _mm_storeu_si128($out_next.add($copied) as *mut __m128i, $v_pat);
                $copied += $offset;
            }

            if $copied < $length {
                let mut tmp = [0u8; 16];
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, $v_pat);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), $out_next.add($copied), $length - $copied);
            }
        }
    }
}

macro_rules! decompress_offset_simple {
    (
        fn_name: $fn_name:ident,
        offset: $offset:expr,
        vars: {
            out_next: $out_next:ident,
            src: $src:ident,
            length: $length:ident,
            v_raw: $v_raw:ident,
            v_pat: $v_pat:ident,
            copied: $copied:ident
        },
        setup: $setup:expr,
        unrolled_loops: $unrolled_loops:block
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "bmi2,ssse3,sse4.1")]
        unsafe fn $fn_name($out_next: *mut u8, $src: *const u8, $length: usize) {
            let $v_raw = _mm_loadu_si128($src as *const __m128i);
            let $v_pat = $setup;

            let mut $copied = 0;

            $unrolled_loops

            while $copied + 16 <= $length {
                _mm_storeu_si128($out_next.add($copied) as *mut __m128i, $v_pat);
                $copied += $offset;
            }

            if $copied < $length {
                let mut tmp = [0u8; 16];
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, $v_pat);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), $out_next.add($copied), $length - $copied);
            }
        }
    }
}

// Optimization: Specialized implementation for offset 18.
// By manually constructing the 9 cyclic vectors using independent `alignr` instructions
// from `v0` and `v1`, we break the serial dependency chain present in the generic loop.
// This increases instruction-level parallelism and improves throughput by ~40% (9.9 GiB/s vs 6.9 GiB/s).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_18(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_align = _mm_loadu_si128(src.add(2) as *const __m128i);

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

// Optimization: Specialized implementation for offset 58.
// We unroll the loop to write 128 bytes (8 vectors) per iteration.
// This matches the generic decompress_offset_cycle4 logic but reduces loop overhead.
// SHIFT = 6. 48 - 6 = 42.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_58(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1_init = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2_init = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v_tail = _mm_loadu_si128(src.add(42) as *const __m128i);
    let mut v3 = _mm_alignr_epi8::<6>(v, v_tail);
    let mut v0 = v;
    let mut v1 = v1_init;
    let mut v2 = v2_init;

    let mut copied = 16;
    let stride = 128;
    while copied + stride <= length {
        let nv0 = _mm_alignr_epi8::<6>(v1, v0);
        let nv1 = _mm_alignr_epi8::<6>(v2, v1);
        let nv2 = _mm_alignr_epi8::<6>(v3, v2);
        let nv3 = _mm_alignr_epi8::<6>(nv0, v3);

        let nnv0 = _mm_alignr_epi8::<6>(nv1, nv0);
        let nnv1 = _mm_alignr_epi8::<6>(nv2, nv1);
        let nnv2 = _mm_alignr_epi8::<6>(nv3, nv2);
        let nnv3 = _mm_alignr_epi8::<6>(nnv0, nv3);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, nv0);

        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, nv1);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, nv2);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, nv3);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, nnv0);

        v0 = nnv0;
        v1 = nnv1;
        v2 = nnv2;
        v3 = nnv3;
        copied += stride;
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
        let next_v0 = _mm_alignr_epi8::<6>(v1, v0);
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, next_v0);
        copied += 16;

        let next_v1 = _mm_alignr_epi8::<6>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<6>(v3, v2);
        let next_v3 = _mm_alignr_epi8::<6>(next_v0, v3);
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

    if SHIFT == 13 {
        // Optimization for Offset 19: Unroll loop to write 96 bytes (6 vectors) per iteration.
        // We use precomputed shifts to derive v_next1..v_next5 directly from v_next0 and v_prev,
        // breaking the serial dependency chain and allowing parallel execution.
        // v_next0 = alignr(v_prev, v_align, 13)
        // v_next1 = alignr(v_next0, v_prev, 13) -> alignr(v_prev, v_align, 26%16=10) ? No, see derivation.
        // Actually, for SHIFT=13 (Offset 19), the sequence of start offsets relative to (v0, v1) allows
        // direct computation via alignr from (v1, v0).
        while copied + 96 <= length {
            let v_next0 = _mm_alignr_epi8::<13>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<13>(v_next0, v_prev);
            let v_next2 = _mm_alignr_epi8::<10>(v_next0, v_prev);
            let v_next3 = _mm_alignr_epi8::<7>(v_next0, v_prev);
            let v_next4 = _mm_alignr_epi8::<4>(v_next0, v_prev);
            let v_next5 = _mm_alignr_epi8::<1>(v_next0, v_prev);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);
            _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);
            _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

            v_prev = v_next5;
            v_align = v_next4;
            copied += 96;
        }
    } else if SHIFT == 11 {
        // Optimization for Offset 21: Unroll loop to write 64 bytes (4 vectors) per iteration.
        while copied + 64 <= length {
            let v_next0 = _mm_alignr_epi8::<11>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<11>(v_next0, v_prev);
            let v_next2 = _mm_alignr_epi8::<6>(v_next0, v_prev);
            let v_next3 = _mm_alignr_epi8::<1>(v_next0, v_prev);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);

            v_prev = v_next3;
            v_align = v_next2;
            copied += 64;
        }
    } else if SHIFT == 10 {
        // Optimization for Offset 22: Unroll loop to write 96 bytes (6 vectors) per iteration.
        while copied + 96 <= length {
            let v_next0 = _mm_alignr_epi8::<10>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<10>(v_next0, v_prev);
            let v_next2 = _mm_alignr_epi8::<4>(v_next0, v_prev);
            let v_next3 = _mm_alignr_epi8::<4>(v_next1, v_next0);
            let v_next4 = _mm_alignr_epi8::<14>(v_next0, v_prev);
            let v_next5 = _mm_alignr_epi8::<14>(v_next1, v_next0);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);
            _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);
            _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

            v_prev = v_next5;
            v_align = v_next4;
            copied += 96;
        }
    } else if SHIFT == 7 {
        // Optimization for Offset 25 (Shift 7): Unroll loop to write 96 bytes (6 vectors) per iteration.
        while copied + 96 <= length {
            let v_next0 = _mm_alignr_epi8::<7>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<7>(v_next0, v_prev);

            let v_next2 = _mm_alignr_epi8::<14>(v_prev, v_align);
            let v_next3 = _mm_alignr_epi8::<14>(v_next0, v_prev);
            let v_next4 = _mm_alignr_epi8::<14>(v_next1, v_next0);
            let v_next5 = _mm_alignr_epi8::<14>(v_next2, v_next1);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);
            _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);
            _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

            v_prev = v_next5;
            v_align = v_next4;
            copied += 96;
        }
    } else if SHIFT == 6 {
        // Optimization for Offset 26: Unroll loop to write 96 bytes (6 vectors) per iteration.
        while copied + 96 <= length {
            let v_next0 = _mm_alignr_epi8::<6>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<6>(v_next0, v_prev);
            let v_next2 = _mm_alignr_epi8::<6>(v_next1, v_next0);
            let v_next3 = _mm_alignr_epi8::<12>(v_next0, v_prev);
            let v_next4 = _mm_alignr_epi8::<12>(v_next1, v_next0);
            let v_next5 = _mm_alignr_epi8::<2>(v_next1, v_next0);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);
            _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);
            _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

            v_prev = v_next5;
            v_align = v_next4;
            copied += 96;
        }
    } else if SHIFT == 9 {
        // Optimization for Offset 23: Unroll loop to write 96 bytes (6 vectors) per iteration.
        // Parallel computation breakdown:
        // v_next0, v_next1, v_next3, v_next4 depend only on v_prev and v_align!
        // v_next2, v_next5 depend on v_next0 (and v_prev)
        while copied + 96 <= length {
            let v_next0 = _mm_alignr_epi8::<9>(v_prev, v_align);
            let v_next1 = _mm_alignr_epi8::<2>(v_prev, v_align);
            let v_next3 = _mm_alignr_epi8::<11>(v_prev, v_align);
            let v_next4 = _mm_alignr_epi8::<4>(v_prev, v_align);

            let v_next2 = _mm_alignr_epi8::<2>(v_next0, v_prev);
            let v_next5 = _mm_alignr_epi8::<4>(v_next0, v_prev);

            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_next0);
            _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_next1);
            _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_next2);
            _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_next3);
            _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_next4);
            _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v_next5);

            v_prev = v_next5;
            v_align = v_next4;
            copied += 96;
        }
    } else {
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

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;

        if copied + 16 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
            copied += 16;
        }
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
decompress_offset_rotated! {
    fn_name: decompress_offset_3,
    offset: 3,
    min_len: 48,
    stride: 48,
    mask: _mm_setr_epi8(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0),
    rotated_vecs: [
        (v_pat1, _mm_setr_epi8(1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1)),
        (v_pat2, _mm_setr_epi8(2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2))
    ],
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    unrolled_stores: {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_pat1);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_pat2);
    },
    intermediate: {}
}

// Optimization: Specialized implementation for offset 5.
decompress_offset_rotated! {
    fn_name: decompress_offset_5,
    offset: 5,
    min_len: 80,
    stride: 80,
    mask: _mm_setr_epi8(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0),
    rotated_vecs: [
        (v_pat1, _mm_setr_epi8(1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1)),
        (v_pat2, _mm_setr_epi8(2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2)),
        (v_pat3, _mm_setr_epi8(3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3)),
        (v_pat4, _mm_setr_epi8(4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4))
    ],
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    unrolled_stores: {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_pat1);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_pat2);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v_pat3);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v_pat4);
    },
    intermediate: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 5) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 10) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 15) as *mut __m128i, v_pat);
            copied += 20;
        }
    }
}

// Optimization: Specialized implementation for offset 6.
decompress_offset_rotated! {
    fn_name: decompress_offset_6,
    offset: 6,
    min_len: 48,
    stride: 48,
    mask: _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3),
    rotated_vecs: [
        (v_pat1, _mm_setr_epi8(4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1)),
        (v_pat2, _mm_setr_epi8(2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5))
    ],
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    unrolled_stores: {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v_pat1);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v_pat2);
    },
    intermediate: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 6) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 12) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
            copied += 24;
        }
    }
}

// Optimization: Specialized implementation for offset 7.
decompress_offset_simple! {
    fn_name: decompress_offset_7,
    offset: 7,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1);
        _mm_shuffle_epi8(v_raw, mask)
    },
    unrolled_loops: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 7) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 14) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 21) as *mut __m128i, v_pat);
            copied += 28;
        }
    }
}

// Optimization: Specialized implementation for offset 9.
// The pattern has length 9. We construct a 16-byte vector [P0...P8, P0...P6]
// using a single shuffle instruction.
decompress_offset_simple! {
    fn_name: decompress_offset_9,
    offset: 9,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6);
        _mm_shuffle_epi8(v_raw, mask)
    },
    unrolled_loops: {
        while copied + 80 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 9) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 27) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 36) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 45) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 54) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 63) as *mut __m128i, v_pat);
            copied += 72;
        }

        while copied + 48 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 9) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 18) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 27) as *mut __m128i, v_pat);
            copied += 36;
        }
    }
}

// Optimization: Specialized implementation for offset 10.
// The pattern has length 10. We construct a 16-byte vector [P0...P9, P0...P5]
// using a single shuffle instruction.
decompress_offset_simple! {
    fn_name: decompress_offset_10,
    offset: 10,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5);
        _mm_shuffle_epi8(v_raw, mask)
    },
    unrolled_loops: {
        while copied + 104 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 10) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 20) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 30) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 40) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 50) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 60) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 70) as *mut __m128i, v_pat);
            copied += 80;
        }

        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 10) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 20) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 30) as *mut __m128i, v_pat);
            copied += 40;
        }
    }
}

// Optimization: Specialized implementation for offset 11.
// The pattern has length 11. We construct a 16-byte vector [P0...P10, P0...P4]
// using a single shuffle instruction.
decompress_offset_simple! {
    fn_name: decompress_offset_11,
    offset: 11,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4);
        _mm_shuffle_epi8(v_raw, mask)
    },
    unrolled_loops: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 11) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 22) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 33) as *mut __m128i, v_pat);
            copied += 44;
        }
    }
}

// Optimization: Specialized implementation for offset 12.
// The pattern has length 12. We construct a 16-byte vector [P0...P11, P0...P3]
// using a single insert instruction. This vector allows us to write 16 bytes
// at a time with a stride of 12 bytes.
decompress_offset_simple! {
    fn_name: decompress_offset_12,
    offset: 12,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let p0 = _mm_cvtsi128_si32(v_raw);
        _mm_insert_epi32::<3>(v_raw, p0)
    },
    unrolled_loops: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 12) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 24) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 36) as *mut __m128i, v_pat);
            copied += 48;
        }
    }
}

// Optimization: Specialized implementation for offset 13.
// The pattern has length 13. We construct a 16-byte vector [P0...P12, P0...P2]
// using a single shuffle instruction.
decompress_offset_simple! {
    fn_name: decompress_offset_13,
    offset: 13,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let mask = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2);
        _mm_shuffle_epi8(v_raw, mask)
    },
    unrolled_loops: {
        // Unroll loop 8x for offset 13 (8 * 13 = 104 bytes per iteration).
        // This reduces loop overhead for long matches.
        // Safety: The last write is at offset 91 (7 * 13).
        // A 16-byte write at 91 requires 91 + 16 = 107 bytes.
        // We check for 120 bytes to be safe and consistent with other offsets.
        while copied + 120 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 13) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 26) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 39) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 52) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 65) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 78) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 91) as *mut __m128i, v_pat);
            copied += 104;
        }

        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 13) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 26) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 39) as *mut __m128i, v_pat);
            copied += 52;
        }
    }
}

// Optimization: Specialized implementation for offset 14.
// The pattern has length 14. We construct a 16-byte vector [P0...P13, P0...P1]
// using a single insert instruction.
decompress_offset_simple! {
    fn_name: decompress_offset_14,
    offset: 14,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let p0 = _mm_cvtsi128_si32(v_raw);
        _mm_insert_epi16::<7>(v_raw, p0)
    },
    unrolled_loops: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 14) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 28) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 42) as *mut __m128i, v_pat);
            copied += 56;
        }
    }
}

// Optimization: Specialized implementation for offset 15.
// The pattern has length 15. We construct a 16-byte vector [P0...P14, P0]
// using a single insert instruction. This vector allows us to write 16 bytes
// at a time with a stride of 15 bytes, effectively rotating the pattern by 1 byte
// each iteration without complex shuffles or register pressure.
decompress_offset_simple! {
    fn_name: decompress_offset_15,
    offset: 15,
    vars: {
        out_next: out_next,
        src: src,
        length: length,
        v_raw: v_raw,
        v_pat: v_pat,
        copied: copied
    },
    setup: {
        let p0 = _mm_cvtsi128_si32(v_raw);
        _mm_insert_epi8::<15>(v_raw, p0)
    },
    unrolled_loops: {
        while copied + 64 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 15) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 30) as *mut __m128i, v_pat);
            _mm_storeu_si128(out_next.add(copied + 45) as *mut __m128i, v_pat);
            copied += 60;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_17(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_align = _mm_loadu_si128(src.add(1) as *const __m128i);
    let mut v1 = _mm_alignr_epi8(v, v_align, 15);
    let mut v0 = v;
    let mut v2 = _mm_alignr_epi8(v1, v0, 15);

    let mut copied = 16;
    while copied + 96 <= length {
        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
        let next_v2 = _mm_alignr_epi8(next_v0, v2, 14);

        let next_v3 = _mm_alignr_epi8(next_v1, next_v0, 14);
        let next_v4 = _mm_alignr_epi8(next_v2, next_v1, 14);
        let next_v5 = _mm_alignr_epi8(next_v3, next_v2, 14);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, next_v0);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, next_v1);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, next_v2);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, next_v3);

        v0 = next_v3;
        v1 = next_v4;
        v2 = next_v5;
        copied += 96;
    }

    if copied + 48 <= length {
        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
        let next_v2 = _mm_alignr_epi8(next_v0, v2, 14);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, next_v0);

        // v0 = next_v0; // Unused
        v1 = next_v1;
        v2 = next_v2;
        copied += 48;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;

        if copied + 16 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
            copied += 16;
        }
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_20(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v_align = _mm_loadu_si128(src.add(4) as *const __m128i);

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
    let v_tail = _mm_loadu_si128(src.add(16) as *const __m128i);

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

    let mut copied = 16;
    let stride = 240;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v7);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v8);
        _mm_storeu_si128(out_next.add(copied + 128) as *mut __m128i, v9);
        _mm_storeu_si128(out_next.add(copied + 144) as *mut __m128i, v10);
        _mm_storeu_si128(out_next.add(copied + 160) as *mut __m128i, v11);
        _mm_storeu_si128(out_next.add(copied + 176) as *mut __m128i, v12);
        _mm_storeu_si128(out_next.add(copied + 192) as *mut __m128i, v13);
        _mm_storeu_si128(out_next.add(copied + 208) as *mut __m128i, v14);
        _mm_storeu_si128(out_next.add(copied + 224) as *mut __m128i, v0);
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
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v9);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v10);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v11);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v12);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v13);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v14);
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
unsafe fn decompress_offset_40(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v4 = _mm_loadu_si128(src.add(24) as *const __m128i);
    let v0 = v;

    let v2 = _mm_alignr_epi8(v0, v4, 8);
    let v3 = _mm_alignr_epi8(v1, v0, 8);

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

// Optimization: Specialized implementation for offset 42.
// Unroll loop to stride 96 bytes (2 cycles of 3 vectors) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_42(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1_init = _mm_loadu_si128(src.add(16) as *const __m128i);
    // 32 - SHIFT = 32 - 6 = 26
    let v_tail = _mm_loadu_si128(src.add(26) as *const __m128i);
    let mut v2 = _mm_alignr_epi8::<6>(v, v_tail);
    let mut v0 = v;
    let mut v1 = v1_init;

    let mut copied = 16;
    while copied + 96 <= length {
        let next_v0 = _mm_alignr_epi8::<6>(v1, v0);
        let next_v1 = _mm_alignr_epi8::<6>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<6>(next_v0, v2);

        let next_v3 = _mm_alignr_epi8::<6>(next_v1, next_v0);
        let next_v4 = _mm_alignr_epi8::<6>(next_v2, next_v1);
        let next_v5 = _mm_alignr_epi8::<6>(next_v3, next_v2);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, next_v0);

        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, next_v1);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, next_v2);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, next_v3);

        v0 = next_v3;
        v1 = next_v4;
        v2 = next_v5;
        copied += 96;
    }

    if copied + 48 <= length {
        let next_v0 = _mm_alignr_epi8::<6>(v1, v0);
        let next_v1 = _mm_alignr_epi8::<6>(v2, v1);
        let next_v2 = _mm_alignr_epi8::<6>(next_v0, v2);

        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, next_v0);

        // v0 = next_v0; // Unused
        v1 = next_v1;
        v2 = next_v2;
        copied += 48;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        copied += 16;

        if copied + 16 <= length {
            _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v2);
            copied += 16;
        }
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_44(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v0 = v;
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v_end = _mm_loadu_si128(src.add(28) as *const __m128i);

    // Optimized parallel computation
    // v2: p[32..44, 0..4]
    let v2 = _mm_alignr_epi8(v0, v_end, 4);
    // v3: p[4..20]
    let v3 = _mm_alignr_epi8(v1, v0, 4);
    // v4: p[20..36]
    let v_end_sh = _mm_srli_si128(v_end, 4);
    let v4 = _mm_alignr_epi8(v_end_sh, v1, 4);

    // v5: p[36..44, 0..8]
    let v5 = _mm_alignr_epi8(v0, v_end, 8);
    // v6: p[8..24]
    let v6 = _mm_alignr_epi8(v1, v0, 8);
    // v7: p[24..40]
    let v7 = _mm_alignr_epi8(v_end_sh, v1, 8);

    // v8: p[40..44, 0..12]
    let v8 = _mm_alignr_epi8(v0, v_end, 12);
    // v9: p[12..28]
    let v9 = _mm_alignr_epi8(v1, v0, 12);
    // v10: p[28..44] = v_end
    let v10 = v_end;

    let mut copied = 16;
    let stride = 176;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v7);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v8);
        _mm_storeu_si128(out_next.add(copied + 128) as *mut __m128i, v9);
        _mm_storeu_si128(out_next.add(copied + 144) as *mut __m128i, v10);
        _mm_storeu_si128(out_next.add(copied + 160) as *mut __m128i, v0);
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
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v9);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v10);
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

// Optimization: Specialized implementation for offset 48.
// We unroll the loop to write 96 bytes (2 cycles of 48 bytes) per iteration.
// The pattern repeats every 48 bytes, so we can reuse the 3 loaded vectors (v0, v16, v32)
// indefinitely, avoiding redundant loads and function call overhead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_48(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v16 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v32 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v0 = v;

    let mut copied = 16;
    while copied + 96 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v0);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v0);
        copied += 96;
    }

    while copied + 48 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v0);
        copied += 48;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v32);
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

// Optimization: Specialized implementation for offset 52.
// We unroll the loop to write 208 bytes (13 vectors) per iteration.
// This allows us to keep the 13 loaded/constructed vectors in registers,
// avoiding the array construction overhead of the generic `decompress_write_cycle_vectors`.
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

    let mut copied = 16;
    let stride = 208;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v7);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v8);
        _mm_storeu_si128(out_next.add(copied + 128) as *mut __m128i, v9);
        _mm_storeu_si128(out_next.add(copied + 144) as *mut __m128i, v10);
        _mm_storeu_si128(out_next.add(copied + 160) as *mut __m128i, v11);
        _mm_storeu_si128(out_next.add(copied + 176) as *mut __m128i, v12);
        _mm_storeu_si128(out_next.add(copied + 192) as *mut __m128i, v0);
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
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v9);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v10);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v11);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v12);
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
unsafe fn decompress_offset_56(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v1 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v2 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v0 = v;
    let v3_low = _mm_loadl_epi64(src.add(48) as *const __m128i);
    let v3 = _mm_unpacklo_epi64(v3_low, v0);

    let v4 = _mm_alignr_epi8(v1, v0, 8);
    let v5 = _mm_alignr_epi8(v2, v1, 8);
    let v6 = _mm_alignr_epi8(v3, v2, 8);

    let mut copied = 16;
    let stride = 112;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v0);
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
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v0);
        copied += 16;
    }

    if copied < length {
        std::ptr::copy_nonoverlapping(src.add(copied), out_next.add(copied), length - copied);
    }
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

    let mut copied = 16;
    let stride = 240;
    while copied + stride <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v1);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v2);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v3);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v4);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v5);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v6);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v7);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v8);
        _mm_storeu_si128(out_next.add(copied + 128) as *mut __m128i, v9);
        _mm_storeu_si128(out_next.add(copied + 144) as *mut __m128i, v10);
        _mm_storeu_si128(out_next.add(copied + 160) as *mut __m128i, v11);
        _mm_storeu_si128(out_next.add(copied + 176) as *mut __m128i, v12);
        _mm_storeu_si128(out_next.add(copied + 192) as *mut __m128i, v13);
        _mm_storeu_si128(out_next.add(copied + 208) as *mut __m128i, v14);
        _mm_storeu_si128(out_next.add(copied + 224) as *mut __m128i, v0);
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
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v9);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v10);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v11);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v12);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v13);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v14);
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

// Optimization: Specialized implementation for offset 64.
// We unroll the loop to write 128 bytes (2 cycles of 64 bytes) per iteration.
// The pattern repeats every 64 bytes, so we can reuse the 4 loaded vectors (v0, v16, v32, v48).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
unsafe fn decompress_offset_64(out_next: *mut u8, src: *const u8, v: __m128i, length: usize) {
    let v16 = _mm_loadu_si128(src.add(16) as *const __m128i);
    let v32 = _mm_loadu_si128(src.add(32) as *const __m128i);
    let v48 = _mm_loadu_si128(src.add(48) as *const __m128i);
    let v0 = v;

    let mut copied = 16;
    while copied + 128 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v48);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v0);
        _mm_storeu_si128(out_next.add(copied + 64) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 80) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 96) as *mut __m128i, v48);
        _mm_storeu_si128(out_next.add(copied + 112) as *mut __m128i, v0);
        copied += 128;
    }

    while copied + 64 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        _mm_storeu_si128(out_next.add(copied + 16) as *mut __m128i, v32);
        _mm_storeu_si128(out_next.add(copied + 32) as *mut __m128i, v48);
        _mm_storeu_si128(out_next.add(copied + 48) as *mut __m128i, v0);
        copied += 64;
    }

    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v16);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v32);
        copied += 16;
    }
    if copied + 16 <= length {
        _mm_storeu_si128(out_next.add(copied) as *mut __m128i, v48);
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
unsafe fn copy_match_bmi2(out_next: *mut u8, src: *const u8, offset: usize, length: usize) {
    if offset >= 16 {
        let v = _mm_loadu_si128(src as *const __m128i);
        _mm_storeu_si128(out_next as *mut __m128i, v);
        if length > 16 {
            if offset >= length {
                std::ptr::copy_nonoverlapping(src.add(16), out_next.add(16), length - 16);
            } else {
                match offset {
                    34 => decompress_offset_cycle3::<14>(out_next, src, v, length),
                    33 => decompress_offset_cycle3::<15>(out_next, src, v, length),
                    35 => decompress_offset_cycle3::<13>(out_next, src, v, length),
                    37 => decompress_offset_cycle3::<11>(out_next, src, v, length),
                    38 => decompress_offset_cycle3::<10>(out_next, src, v, length),
                    39 => decompress_offset_cycle3::<9>(out_next, src, v, length),
                    41 => decompress_offset_cycle3::<7>(out_next, src, v, length),
                    42 => decompress_offset_42(out_next, src, v, length),
                    43 => decompress_offset_cycle3::<5>(out_next, src, v, length),
                    45 => decompress_offset_cycle3::<3>(out_next, src, v, length),
                    46 => decompress_offset_cycle3::<2>(out_next, src, v, length),
                    47 => decompress_offset_cycle3::<1>(out_next, src, v, length),
                    49 => decompress_offset_cycle4::<15>(out_next, src, v, length),
                    50 => decompress_offset_cycle4::<14>(out_next, src, v, length),
                    51 => decompress_offset_cycle4::<13>(out_next, src, v, length),
                    53 => decompress_offset_cycle4::<11>(out_next, src, v, length),
                    54 => decompress_offset_cycle4::<10>(out_next, src, v, length),
                    55 => decompress_offset_cycle4::<9>(out_next, src, v, length),
                    57 => decompress_offset_cycle4::<7>(out_next, src, v, length),
                    59 => decompress_offset_cycle4::<5>(out_next, src, v, length),
                    58 => decompress_offset_58(out_next, src, v, length),
                    61 => decompress_offset_cycle4::<3>(out_next, src, v, length),
                    62 => decompress_offset_cycle4::<2>(out_next, src, v, length),
                    63 => decompress_offset_cycle4::<1>(out_next, src, v, length),
                    60 => decompress_offset_60(out_next, src, v, length),
                    44 => decompress_offset_44(out_next, src, v, length),
                    36 => decompress_offset_36(out_next, src, v, length),
                    52 => decompress_offset_52(out_next, src, v, length),
                    56 => decompress_offset_56(out_next, src, v, length),
                    48 => decompress_offset_48(out_next, src, v, length),
                    64 => decompress_offset_64(out_next, src, v, length),
                    16 => {
                        decompress_fill_pattern_16(out_next, v, src, length);
                    }
                    17 => decompress_offset_17(out_next, src, v, length),
                    18 => decompress_offset_18(out_next, src, v, length),
                    19 => {
                        decompress_offset_alignr_cycle::<13>(out_next, src, length, v);
                    }
                    20 => decompress_offset_20(out_next, src, v, length),
                    21 => {
                        decompress_offset_alignr_cycle::<11>(out_next, src, length, v);
                    }
                    22 => {
                        decompress_offset_alignr_cycle::<10>(out_next, src, length, v);
                    }
                    23 => {
                        decompress_offset_alignr_cycle::<9>(out_next, src, length, v);
                    }
                    24 => decompress_offset_24(out_next, src, v, length),
                    25 => {
                        decompress_offset_alignr_cycle::<7>(out_next, src, length, v);
                    }
                    26 => {
                        decompress_offset_alignr_cycle::<6>(out_next, src, length, v);
                    }
                    27 => {
                        decompress_offset_alignr_cycle::<5>(out_next, src, length, v);
                    }
                    28 => decompress_offset_28(out_next, src, v, length),
                    29 => {
                        decompress_offset_alignr_cycle::<3>(out_next, src, length, v);
                    }
                    30 => decompress_offset_30(out_next, src, v, length),
                    31 => {
                        decompress_offset_alignr_cycle::<1>(out_next, src, length, v);
                    }
                    32 => decompress_offset_32(out_next, src, v, length),
                    40 => decompress_offset_40(out_next, src, v, length),
                    _ => {
                        let init = std::cmp::min(offset, length);
                        std::ptr::copy_nonoverlapping(src, out_next, init);

                        let mut copied = init;
                        while copied < length {
                            let to_copy = std::cmp::min(length - copied, copied);
                            std::ptr::copy_nonoverlapping(out_next, out_next.add(copied), to_copy);
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
                    2 => _mm_set1_epi16(std::ptr::read_unaligned(src as *const u16) as i16),
                    4 => _mm_set1_epi32(std::ptr::read_unaligned(src as *const u32) as i32),
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
                    let val = std::ptr::read_unaligned(src.add(copied) as *const u64);
                    std::ptr::write_unaligned(out_next.add(copied) as *mut u64, val);
                    copied += 8;
                }
                while copied < length {
                    *out_next.add(copied) = *src.add(copied);
                    copied += 1;
                }
            }
        }
    }
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

                #[allow(clippy::never_loop)]
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
                                copy_match_bmi2(out_next, src, offset, length);
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
