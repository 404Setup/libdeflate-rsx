#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const DIVISOR: u32 = 65521;
const BLOCK_SIZE: usize = 4096;

// Optimization: Manual loop unrolling for Adler32 tail processing.
// This processes 4 bytes per iteration to reduce loop overhead and improve instruction pipelining.
// It is significantly faster than a scalar iterator loop for small tails (e.g., 15-63 bytes).
//
// Safety:
// * `$ptr` must be valid for reads of `$len` bytes.
// * `$s1` and `$s2` must not overflow u32 before modulo (guaranteed by BLOCK_SIZE check in caller).
macro_rules! adler32_tail {
    ($s1:expr, $s2:expr, $ptr:expr, $len:expr) => {
        // We know len < 16 here because larger chunks are handled by SIMD or unrolled loops before calling this macro.
        if $len > 0 {
            if $len >= 8 {
                let v = ($ptr as *const u64).read_unaligned();
                let v_lo = v as u32;
                let v_hi = (v >> 32) as u32;

                let b0 = v_lo & 0xFF;
                let b1 = (v_lo >> 8) & 0xFF;
                let b2 = (v_lo >> 16) & 0xFF;
                let b3 = (v_lo >> 24);

                let b4 = v_hi & 0xFF;
                let b5 = (v_hi >> 8) & 0xFF;
                let b6 = (v_hi >> 16) & 0xFF;
                let b7 = (v_hi >> 24);

                $s2 += ($s1 << 3)
                    + (b0 << 3)
                    + (b1.wrapping_mul(7))
                    + (b2.wrapping_mul(6))
                    + (b3.wrapping_mul(5))
                    + (b4 << 2)
                    + (b5.wrapping_mul(3))
                    + (b6 << 1)
                    + b7;
                $s1 += b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;

                $ptr = $ptr.add(8);
                $len -= 8;
            }
            if $len >= 4 {
                let v = ($ptr as *const u32).read_unaligned();
                let b0 = v & 0xFF;
                let b1 = (v >> 8) & 0xFF;
                let b2 = (v >> 16) & 0xFF;
                let b3 = (v >> 24);

                $s2 += ($s1 << 2) + (b0 << 2) + (b1.wrapping_mul(3)) + (b2 << 1) + b3;
                $s1 += b0 + b1 + b2 + b3;

                $ptr = $ptr.add(4);
                $len -= 4;
            }

            // Remaining 0-3 bytes.
            match $len {
                3 => {
                    let v = ($ptr as *const u16).read_unaligned() as u32;
                    let b2 = *$ptr.add(2) as u32;
                    let b0 = v & 0xFF;
                    let b1 = v >> 8;
                    $s2 += ($s1 << 1) + $s1 + (b0 * 3) + (b1 * 2) + b2;
                    $s1 += b0 + b1 + b2;
                }
                2 => {
                    let v = ($ptr as *const u16).read_unaligned() as u32;
                    let b0 = v & 0xFF;
                    let b1 = v >> 8;
                    $s2 += ($s1 << 1) + (b0 * 2) + b1;
                    $s1 += b0 + b1;
                }
                1 => {
                    let b0 = *$ptr as u32;
                    $s2 += $s1 + b0;
                    $s1 += b0;
                }
                _ => {}
            }
        }
    };
}

#[target_feature(enable = "sse2")]
pub unsafe fn adler32_x86_sse2(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    let mults_a = _mm_set_epi16(25, 26, 27, 28, 29, 30, 31, 32);
    let mults_b = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
    let mults_c = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16);
    let mults_d = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
    let v_zero = _mm_setzero_si128();

    while data.len() >= 32 {
        let mut n = std::cmp::min(data.len(), BLOCK_SIZE);
        n &= !31;

        s2 += s1 * (n as u32);

        let mut v_s1 = _mm_setzero_si128();
        let mut v_s1_sums = _mm_setzero_si128();
        let mut v_byte_sums_a = _mm_setzero_si128();
        let mut v_byte_sums_b = _mm_setzero_si128();
        let mut v_byte_sums_c = _mm_setzero_si128();
        let mut v_byte_sums_d = _mm_setzero_si128();

        let mut chunk_n = n;
        // Optimization: Unroll loop to process 64 bytes per iteration (two 32-byte chunks).
        // This amortizes loop overhead and allows better pipelining of the `sad` and `unpack` operations.
        while chunk_n >= 128 {
            let data_a_1 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let data_b_1 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
            let data_a_2 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
            let data_b_2 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);
            let data_a_3 = _mm_loadu_si128(data.as_ptr().add(64) as *const __m128i);
            let data_b_3 = _mm_loadu_si128(data.as_ptr().add(80) as *const __m128i);
            let data_a_4 = _mm_loadu_si128(data.as_ptr().add(96) as *const __m128i);
            let data_b_4 = _mm_loadu_si128(data.as_ptr().add(112) as *const __m128i);

            // Accumulate byte sums
            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_1, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_1, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_1, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_1, v_zero));

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_2, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_2, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_2, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_2, v_zero));

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_3, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_3, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_3, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_3, v_zero));

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_4, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_4, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_4, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_4, v_zero));

            // SAD calculation
            let sad_1 = _mm_add_epi32(
                _mm_sad_epu8(data_a_1, v_zero),
                _mm_sad_epu8(data_b_1, v_zero),
            );
            let sad_2 = _mm_add_epi32(
                _mm_sad_epu8(data_a_2, v_zero),
                _mm_sad_epu8(data_b_2, v_zero),
            );
            let sad_3 = _mm_add_epi32(
                _mm_sad_epu8(data_a_3, v_zero),
                _mm_sad_epu8(data_b_3, v_zero),
            );
            let sad_4 = _mm_add_epi32(
                _mm_sad_epu8(data_a_4, v_zero),
                _mm_sad_epu8(data_b_4, v_zero),
            );

            // Update v_s1_sums
            // v_s1_sums += 4 * v_s1 (initial) + 3*sad_1 + 2*sad_2 + 1*sad_3
            let s1_x4 = _mm_slli_epi32(v_s1, 2);
            let inc_1 = _mm_add_epi32(
                _mm_add_epi32(sad_1, _mm_slli_epi32(sad_1, 1)), // 3*sad_1
                _mm_add_epi32(_mm_slli_epi32(sad_2, 1), sad_3), // 2*sad_2 + sad_3
            );
            v_s1_sums = _mm_add_epi32(v_s1_sums, _mm_add_epi32(s1_x4, inc_1));

            // Update v_s1
            let total_sad = _mm_add_epi32(_mm_add_epi32(sad_1, sad_2), _mm_add_epi32(sad_3, sad_4));
            v_s1 = _mm_add_epi32(v_s1, total_sad);

            data = &data[128..];
            chunk_n -= 128;
        }

        while chunk_n >= 64 {
            let data_a_1 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let data_b_1 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
            let data_a_2 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
            let data_b_2 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);

            let sad_a_1 = _mm_sad_epu8(data_a_1, v_zero);
            let sad_b_1 = _mm_sad_epu8(data_b_1, v_zero);
            let sad_a_2 = _mm_sad_epu8(data_a_2, v_zero);
            let sad_b_2 = _mm_sad_epu8(data_b_2, v_zero);

            let sad_1 = _mm_add_epi32(sad_a_1, sad_b_1);
            let sad_2 = _mm_add_epi32(sad_a_2, sad_b_2);

            let v_s1_sh = _mm_slli_epi32(v_s1, 1);
            v_s1_sums = _mm_add_epi32(v_s1_sums, _mm_add_epi32(v_s1_sh, sad_1));
            v_s1 = _mm_add_epi32(v_s1, _mm_add_epi32(sad_1, sad_2));

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_1, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_1, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_1, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_1, v_zero));

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a_2, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a_2, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b_2, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b_2, v_zero));

            data = &data[64..];
            chunk_n -= 64;
        }

        while chunk_n >= 32 {
            let data_a = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let data_b = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);

            v_s1_sums = _mm_add_epi32(v_s1_sums, v_s1);

            v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a, v_zero));
            v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a, v_zero));
            v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b, v_zero));
            v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b, v_zero));

            v_s1 = _mm_add_epi32(
                v_s1,
                _mm_add_epi32(_mm_sad_epu8(data_a, v_zero), _mm_sad_epu8(data_b, v_zero)),
            );

            data = &data[32..];
            chunk_n -= 32;
        }

        let mut v_s2 = _mm_add_epi32(
            _mm_madd_epi16(v_byte_sums_a, mults_a),
            _mm_madd_epi16(v_byte_sums_b, mults_b),
        );
        v_s2 = _mm_add_epi32(
            v_s2,
            _mm_add_epi32(
                _mm_madd_epi16(v_byte_sums_c, mults_c),
                _mm_madd_epi16(v_byte_sums_d, mults_d),
            ),
        );

        v_s1_sums = _mm_slli_epi32(v_s1_sums, 5);
        v_s2 = _mm_add_epi32(v_s2, v_s1_sums);

        let mut s1_buf = [0u32; 4];
        let mut s2_buf = [0u32; 4];
        _mm_storeu_si128(s1_buf.as_mut_ptr() as *mut __m128i, v_s1);
        _mm_storeu_si128(s2_buf.as_mut_ptr() as *mut __m128i, v_s2);

        s1 += s1_buf[0] + s1_buf[2];
        s2 += s2_buf[0] + s2_buf[1] + s2_buf[2] + s2_buf[3];

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    if data.len() >= 16 {
        let d = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        let sad = _mm_sad_epu8(d, v_zero);
        let sum_s1 = _mm_cvtsi128_si32(_mm_add_epi32(sad, _mm_srli_si128(sad, 8)));
        s2 += s1 * 16;
        s1 += sum_s1 as u32;

        let d_lo = _mm_unpacklo_epi8(d, v_zero);
        let d_hi = _mm_unpackhi_epi8(d, v_zero);

        let w_lo = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16);
        let w_hi = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);

        let s_lo = _mm_madd_epi16(d_lo, w_lo);
        let s_hi = _mm_madd_epi16(d_hi, w_hi);
        let s = _mm_add_epi32(s_lo, s_hi);

        let s_step = _mm_add_epi32(s, _mm_srli_si128(s, 8));
        let sum_s2 = _mm_cvtsi128_si32(_mm_add_epi32(s_step, _mm_srli_si128(s_step, 4)));
        s2 += sum_s2 as u32;

        data = &data[16..];
    }

    let mut ptr = data.as_ptr();
    let mut len = data.len();
    adler32_tail!(s1, s2, ptr, len);

    (s2 % DIVISOR) << 16 | (s1 % DIVISOR)
}

#[target_feature(enable = "avx2")]
pub unsafe fn adler32_x86_avx2(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;

    let mut ptr = p.as_ptr();
    let mut len = p.len();

    if len > 2048 {
        let align = (ptr as usize) & 31;
        if align != 0 {
            let len_p = std::cmp::min(len, 32 - align);
            let prefix = std::slice::from_raw_parts(ptr, len_p);
            for &b in prefix {
                s1 += b as u32;
                s2 += s1;
            }
            s1 %= DIVISOR;
            s2 %= DIVISOR;
            ptr = ptr.add(len_p);
            len -= len_p;
        }
    }

    // Optimization: Hoist vector constants out of the main loop to avoid redundant loads.
    let weights = _mm256_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    );
    let ones_i16 = _mm256_set1_epi16(1);
    let v_zero = _mm256_setzero_si256();

    while len >= 64 {
        let n = std::cmp::min(len, BLOCK_SIZE);
        let n_rounded = n & !63;

        s2 += s1 * (n_rounded as u32);

        let mut v_s1 = _mm256_setzero_si256();
        let mut v_s1_acc = _mm256_setzero_si256();
        let mut v_inc_acc_a = _mm256_setzero_si256();

        let mut chunk_n = n_rounded;

        let mut v_s2_a = _mm256_setzero_si256();
        let mut v_s2_b = _mm256_setzero_si256();
        let mut v_s2_c = _mm256_setzero_si256();
        let mut v_s2_d = _mm256_setzero_si256();

        if chunk_n >= 256 {
            while chunk_n >= 256 {
                // Parallelize processing of two 128-byte blocks to break dependency chains on v_s1.
                // We compute SADs and other accumulators for both blocks independently, then
                // combine the results for v_s1 and v_s1_acc.
                // Optimization: Reuse v_s2_a..d and v_inc_acc_a to reduce register pressure.
                // This avoids spilling 5 YMM registers to the stack, significantly improving
                // throughput for intermediate buffer sizes (e.g., 512 bytes).

                // Block 1 (0..128)
                let data_a_1 = _mm256_loadu_si256(ptr as *const __m256i);
                let data_b_1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
                let data_a_2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
                let data_b_2 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

                let sad1_a = _mm256_sad_epu8(data_a_1, v_zero);
                let sad2_a = _mm256_sad_epu8(data_b_1, v_zero);
                let sad3_a = _mm256_sad_epu8(data_a_2, v_zero);
                let sad4_a = _mm256_sad_epu8(data_b_2, v_zero);

                let s12_a = _mm256_add_epi32(sad1_a, sad2_a);
                let s34_a = _mm256_add_epi32(sad3_a, sad4_a);
                let sum_sads_a = _mm256_add_epi32(s12_a, s34_a);

                let s12_x2_a = _mm256_slli_epi32(s12_a, 1);
                let inc_part_a = _mm256_add_epi32(_mm256_add_epi32(s12_x2_a, sad1_a), sad3_a);
                v_inc_acc_a = _mm256_add_epi32(v_inc_acc_a, inc_part_a);

                let p1_a = _mm256_maddubs_epi16(data_a_1, weights);
                let s_a_a = _mm256_madd_epi16(p1_a, ones_i16);
                let p2_a = _mm256_maddubs_epi16(data_b_1, weights);
                let s_b_a = _mm256_madd_epi16(p2_a, ones_i16);
                v_s2_a = _mm256_add_epi32(v_s2_a, s_a_a);
                v_s2_b = _mm256_add_epi32(v_s2_b, s_b_a);

                let p3_a = _mm256_maddubs_epi16(data_a_2, weights);
                let s_c_a = _mm256_madd_epi16(p3_a, ones_i16);
                let p4_a = _mm256_maddubs_epi16(data_b_2, weights);
                let s_d_a = _mm256_madd_epi16(p4_a, ones_i16);
                v_s2_c = _mm256_add_epi32(v_s2_c, s_c_a);
                v_s2_d = _mm256_add_epi32(v_s2_d, s_d_a);

                // Block 2 (128..256)
                let data_a_3 = _mm256_loadu_si256(ptr.add(128) as *const __m256i);
                let data_b_3 = _mm256_loadu_si256(ptr.add(160) as *const __m256i);
                let data_a_4 = _mm256_loadu_si256(ptr.add(192) as *const __m256i);
                let data_b_4 = _mm256_loadu_si256(ptr.add(224) as *const __m256i);

                let sad1_b = _mm256_sad_epu8(data_a_3, v_zero);
                let sad2_b = _mm256_sad_epu8(data_b_3, v_zero);
                let sad3_b = _mm256_sad_epu8(data_a_4, v_zero);
                let sad4_b = _mm256_sad_epu8(data_b_4, v_zero);

                let s12_b = _mm256_add_epi32(sad1_b, sad2_b);
                let s34_b = _mm256_add_epi32(sad3_b, sad4_b);
                let sum_sads_b = _mm256_add_epi32(s12_b, s34_b);

                let s12_x2_b = _mm256_slli_epi32(s12_b, 1);
                let inc_part_b = _mm256_add_epi32(_mm256_add_epi32(s12_x2_b, sad1_b), sad3_b);
                v_inc_acc_a = _mm256_add_epi32(v_inc_acc_a, inc_part_b);

                let p1_b = _mm256_maddubs_epi16(data_a_3, weights);
                let s_a_b = _mm256_madd_epi16(p1_b, ones_i16);
                let p2_b = _mm256_maddubs_epi16(data_b_3, weights);
                let s_b_b = _mm256_madd_epi16(p2_b, ones_i16);
                v_s2_a = _mm256_add_epi32(v_s2_a, s_a_b);
                v_s2_b = _mm256_add_epi32(v_s2_b, s_b_b);

                let p3_b = _mm256_maddubs_epi16(data_a_4, weights);
                let s_c_b = _mm256_madd_epi16(p3_b, ones_i16);
                let p4_b = _mm256_maddubs_epi16(data_b_4, weights);
                let s_d_b = _mm256_madd_epi16(p4_b, ones_i16);
                v_s2_c = _mm256_add_epi32(v_s2_c, s_c_b);
                v_s2_d = _mm256_add_epi32(v_s2_d, s_d_b);

                // Update v_s1 and v_s1_acc using accumulated sums
                // v_s1_acc accumulates v_s1 at the start of each 128-byte block.
                // For Block 1: v_s1_acc += v_s1 (current)
                // For Block 2: v_s1_acc += v_s1 + sum_sads_a
                // Combined: v_s1_acc += 2*v_s1 + sum_sads_a
                let v_s1_x2 = _mm256_slli_epi32(v_s1, 1);
                v_s1_acc = _mm256_add_epi32(v_s1_acc, _mm256_add_epi32(v_s1_x2, sum_sads_a));

                // v_s1 accumulates sum_sads from both blocks
                v_s1 = _mm256_add_epi32(v_s1, _mm256_add_epi32(sum_sads_a, sum_sads_b));

                ptr = ptr.add(256);
                chunk_n -= 256;
                len -= 256;
            }
        }

        while chunk_n >= 128 {
            let data_a_1 = _mm256_loadu_si256(ptr as *const __m256i);
            let data_b_1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
            let data_a_2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
            let data_b_2 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

            let sad1 = _mm256_sad_epu8(data_a_1, v_zero);
            let sad2 = _mm256_sad_epu8(data_b_1, v_zero);
            let sad3 = _mm256_sad_epu8(data_a_2, v_zero);
            let sad4 = _mm256_sad_epu8(data_b_2, v_zero);

            let s12 = _mm256_add_epi32(sad1, sad2);
            let s34 = _mm256_add_epi32(sad3, sad4);
            let sum_sads = _mm256_add_epi32(s12, s34);

            let s12_x2 = _mm256_slli_epi32(s12, 1);
            let inc_part = _mm256_add_epi32(_mm256_add_epi32(s12_x2, sad1), sad3);

            v_s1_acc = _mm256_add_epi32(v_s1_acc, v_s1);
            v_inc_acc_a = _mm256_add_epi32(v_inc_acc_a, inc_part);
            v_s1 = _mm256_add_epi32(v_s1, sum_sads);

            let p1 = _mm256_maddubs_epi16(data_a_1, weights);
            let s_a = _mm256_madd_epi16(p1, ones_i16);
            v_s2_a = _mm256_add_epi32(v_s2_a, s_a);

            let p2 = _mm256_maddubs_epi16(data_b_1, weights);
            let s_b = _mm256_madd_epi16(p2, ones_i16);
            v_s2_b = _mm256_add_epi32(v_s2_b, s_b);

            let p3 = _mm256_maddubs_epi16(data_a_2, weights);
            let s_c = _mm256_madd_epi16(p3, ones_i16);
            v_s2_c = _mm256_add_epi32(v_s2_c, s_c);

            let p4 = _mm256_maddubs_epi16(data_b_2, weights);
            let s_d = _mm256_madd_epi16(p4, ones_i16);
            v_s2_d = _mm256_add_epi32(v_s2_d, s_d);

            ptr = ptr.add(128);
            chunk_n -= 128;
            len -= 128;
        }

        let mut v_s2 = _mm256_add_epi32(
            _mm256_add_epi32(v_s2_a, v_s2_b),
            _mm256_add_epi32(v_s2_c, v_s2_d),
        );

        let v_inc_acc = v_inc_acc_a;

        let v_s1_shifted = _mm256_slli_epi32(v_s1_acc, 7);
        let v_inc_shifted = _mm256_slli_epi32(v_inc_acc, 5);
        let mut v_s1_sums = _mm256_add_epi32(v_s1_shifted, v_inc_shifted);

        let mut v_s2_acc_b = _mm256_setzero_si256();

        while chunk_n >= 64 {
            let data_a = _mm256_loadu_si256(ptr as *const __m256i);
            let data_b = _mm256_loadu_si256(ptr.add(32) as *const __m256i);

            // Optimization: Parallelize SAD calculation and s1/s2 updates to reduce dependency chains.
            // By computing sad_a and sad_b in parallel, we can accumulate s1 sums and s1 in larger steps.
            let sad_a = _mm256_sad_epu8(data_a, v_zero);
            let sad_b = _mm256_sad_epu8(data_b, v_zero);

            // Update v_s1_sums:
            // The contribution of the current v_s1 to the sums over the next 64 bytes is:
            // - For the first 32 bytes: v_s1 * 32
            // - For the second 32 bytes: (v_s1 + sad_a) * 32
            // Total: v_s1 * 64 + sad_a * 32
            let v_s1_x64 = _mm256_slli_epi32(v_s1, 6);
            let sad_a_x32 = _mm256_slli_epi32(sad_a, 5);
            v_s1_sums = _mm256_add_epi32(v_s1_sums, _mm256_add_epi32(v_s1_x64, sad_a_x32));

            // Update v_s1: v_s1 += sad_a + sad_b
            v_s1 = _mm256_add_epi32(v_s1, _mm256_add_epi32(sad_a, sad_b));

            // Update v_s2: Calculate partial s2 contributions in parallel
            let p1 = _mm256_maddubs_epi16(data_a, weights);
            let s_a = _mm256_madd_epi16(p1, ones_i16);
            let p2 = _mm256_maddubs_epi16(data_b, weights);
            let s_b = _mm256_madd_epi16(p2, ones_i16);

            v_s2 = _mm256_add_epi32(v_s2, s_a);
            v_s2_acc_b = _mm256_add_epi32(v_s2_acc_b, s_b);

            ptr = ptr.add(64);
            chunk_n -= 64;
            len -= 64;
        }
        v_s2 = _mm256_add_epi32(v_s2, v_s2_acc_b);

        v_s2 = _mm256_add_epi32(v_s2, v_s1_sums);

        let v_s1_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s1, 0),
            _mm256_extracti128_si256(v_s1, 1),
        );
        let v_s2_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s2, 0),
            _mm256_extracti128_si256(v_s2, 1),
        );

        let mut s1_buf = [0u32; 4];
        let mut s2_buf = [0u32; 4];
        _mm_storeu_si128(s1_buf.as_mut_ptr() as *mut __m128i, v_s1_128);
        _mm_storeu_si128(s2_buf.as_mut_ptr() as *mut __m128i, v_s2_128);

        s1 += s1_buf[0] + s1_buf[1] + s1_buf[2] + s1_buf[3];
        s2 += s2_buf[0] + s2_buf[1] + s2_buf[2] + s2_buf[3];

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    if len >= 32 {
        let d = _mm256_loadu_si256(ptr as *const __m256i);

        let sad = _mm256_sad_epu8(d, v_zero);
        let sad_lo = _mm256_castsi256_si128(sad);
        let sad_hi = _mm256_extracti128_si256(sad, 1);
        let sad_sum = _mm_add_epi64(sad_lo, sad_hi);
        let sad_sum = _mm_add_epi64(sad_sum, _mm_unpackhi_epi64(sad_sum, sad_sum));
        let s1_part = _mm_cvtsi128_si32(sad_sum) as u32;

        s2 += s1 * 32;
        s1 += s1_part;

        let w_32 = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let p = _mm256_maddubs_epi16(d, w_32);
        let s = _mm256_madd_epi16(p, ones_i16);

        let s_lo = _mm256_castsi256_si128(s);
        let s_hi = _mm256_extracti128_si256(s, 1);
        let s_sum = _mm_add_epi32(s_lo, s_hi);
        let s_sum = _mm_add_epi32(s_sum, _mm_shuffle_epi32(s_sum, 0x4E));
        let s_sum = _mm_add_epi32(s_sum, _mm_shuffle_epi32(s_sum, 0xB1));

        s2 += _mm_cvtsi128_si32(s_sum) as u32;

        ptr = ptr.add(32);
        len -= 32;
    }

    if len >= 16 {
        let d = _mm_loadu_si128(ptr as *const __m128i);
        let v_zero_xmm = _mm_setzero_si128();

        let sad = _mm_sad_epu8(d, v_zero_xmm);
        let s1_part = _mm_cvtsi128_si32(_mm_add_epi32(sad, _mm_unpackhi_epi64(sad, sad))) as u32;

        s2 += s1 * 16;
        s1 += s1_part;

        let w_16 = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let p = _mm_maddubs_epi16(d, w_16);
        let s = _mm_madd_epi16(p, _mm_set1_epi16(1));

        let s_sum = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        let s_sum = _mm_add_epi32(s_sum, _mm_shuffle_epi32(s_sum, 0xB1));

        s2 += _mm_cvtsi128_si32(s_sum) as u32;

        ptr = ptr.add(16);
        len -= 16;
    }

    adler32_tail!(s1, s2, ptr, len);
    s1 %= DIVISOR;
    s2 %= DIVISOR;

    (s2 << 16) | s1
}

#[target_feature(enable = "avxvnni")]
pub unsafe fn adler32_x86_avx2_vnni(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    if data.len() > 2048 {
        let align = (data.as_ptr() as usize) & 31;
        if align != 0 {
            let len = std::cmp::min(data.len(), 32 - align);
            for &b in &data[..len] {
                s1 += b as u32;
                s2 += s1;
            }
            s1 %= DIVISOR;
            s2 %= DIVISOR;
            data = &data[len..];
        }
    }

    let ones = _mm256_set1_epi8(1);
    let mults = _mm256_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    );

    while data.len() >= 32 {
        let n = std::cmp::min(data.len(), BLOCK_SIZE) & !31;
        s2 += s1 * (n as u32);

        let mut v_s1 = _mm256_setzero_si256();
        let mut v_s2 = _mm256_setzero_si256();
        let mut v_s1_sums = _mm256_setzero_si256();

        let mut chunk_n = n;

        // Optimization: For chunks >= 256 bytes, use an unrolled loop with 8 independent accumulators.
        // This increases instruction-level parallelism to hide the latency of `vpdpbusd` (5 cycles on Golden Cove).
        // We merge the global `v_s2` into `v_s2_a` to save a register, keeping total usage within AVX2 limits (16 YMMs).
        if chunk_n >= 256 {
            let mut ptr = data.as_ptr();
            let mut v_s2_a = v_s2;
            let mut v_s2_b = _mm256_setzero_si256();
            let mut v_s2_c = _mm256_setzero_si256();
            let mut v_s2_d = _mm256_setzero_si256();
            let mut v_s2_e = _mm256_setzero_si256();
            let mut v_s2_f = _mm256_setzero_si256();
            let mut v_s2_g = _mm256_setzero_si256();
            let mut v_s2_h = _mm256_setzero_si256();

            while chunk_n >= 256 {
                // Block 1 (0..128)
                let d1 = _mm256_loadu_si256(ptr as *const __m256i);
                v_s2_a = _mm256_dpbusd_avx_epi32(v_s2_a, d1, mults);
                let u1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d1, ones);

                let d2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
                v_s2_b = _mm256_dpbusd_avx_epi32(v_s2_b, d2, mults);
                let u2 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d2, ones);

                let d3 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
                v_s2_c = _mm256_dpbusd_avx_epi32(v_s2_c, d3, mults);
                let u3 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d3, ones);

                let d4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);
                v_s2_d = _mm256_dpbusd_avx_epi32(v_s2_d, d4, mults);
                let u4 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d4, ones);

                let u12 = _mm256_add_epi32(u1, u2);
                let u34 = _mm256_add_epi32(u3, u4);
                let total_u_a = _mm256_add_epi32(u12, u34);

                let u12_x2 = _mm256_slli_epi32(u12, 1);
                let inc_a = _mm256_add_epi32(_mm256_add_epi32(u12_x2, u1), u3);

                let s1_x4 = _mm256_slli_epi32(v_s1, 2);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, s1_x4);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, inc_a);
                v_s1 = _mm256_add_epi32(v_s1, total_u_a);

                // Block 2 (128..256)
                let d5 = _mm256_loadu_si256(ptr.add(128) as *const __m256i);
                v_s2_e = _mm256_dpbusd_avx_epi32(v_s2_e, d5, mults);
                let u5 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d5, ones);

                let d6 = _mm256_loadu_si256(ptr.add(160) as *const __m256i);
                v_s2_f = _mm256_dpbusd_avx_epi32(v_s2_f, d6, mults);
                let u6 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d6, ones);

                let d7 = _mm256_loadu_si256(ptr.add(192) as *const __m256i);
                v_s2_g = _mm256_dpbusd_avx_epi32(v_s2_g, d7, mults);
                let u7 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d7, ones);

                let d8 = _mm256_loadu_si256(ptr.add(224) as *const __m256i);
                v_s2_h = _mm256_dpbusd_avx_epi32(v_s2_h, d8, mults);
                let u8 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), d8, ones);

                let u56 = _mm256_add_epi32(u5, u6);
                let u78 = _mm256_add_epi32(u7, u8);
                let total_u_b = _mm256_add_epi32(u56, u78);

                let u56_x2 = _mm256_slli_epi32(u56, 1);
                let inc_b = _mm256_add_epi32(_mm256_add_epi32(u56_x2, u5), u7);

                let s1_x8_b = _mm256_slli_epi32(v_s1, 3);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, s1_x8_b);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, inc_b);
                v_s1 = _mm256_add_epi32(v_s1, total_u_b);

                ptr = ptr.add(256);
                chunk_n -= 256;
            }
            v_s2 = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(v_s2_a, v_s2_b),
                    _mm256_add_epi32(v_s2_c, v_s2_d),
                ),
                _mm256_add_epi32(
                    _mm256_add_epi32(v_s2_e, v_s2_f),
                    _mm256_add_epi32(v_s2_g, v_s2_h),
                ),
            );
            let processed = ptr as usize - data.as_ptr() as usize;
            data = &data[processed..];
        }

        while chunk_n >= 128 {
            let mut ptr = data.as_ptr();
            let mut v_s2_a = _mm256_setzero_si256();
            let mut v_s2_b = _mm256_setzero_si256();
            let mut v_s2_c = _mm256_setzero_si256();
            let mut v_s2_d = _mm256_setzero_si256();
            let v_zero = _mm256_setzero_si256();

            while chunk_n >= 128 {
                let d1 = _mm256_loadu_si256(ptr as *const __m256i);
                let d2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
                let d3 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
                let d4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

                let u1 = _mm256_dpbusd_avx_epi32(v_zero, d1, ones);
                let u2 = _mm256_dpbusd_avx_epi32(v_zero, d2, ones);
                let u3 = _mm256_dpbusd_avx_epi32(v_zero, d3, ones);
                let u4 = _mm256_dpbusd_avx_epi32(v_zero, d4, ones);

                v_s2_a = _mm256_dpbusd_avx_epi32(v_s2_a, d1, mults);
                v_s2_b = _mm256_dpbusd_avx_epi32(v_s2_b, d2, mults);
                v_s2_c = _mm256_dpbusd_avx_epi32(v_s2_c, d3, mults);
                v_s2_d = _mm256_dpbusd_avx_epi32(v_s2_d, d4, mults);

                let s1_x4 = _mm256_slli_epi32(v_s1, 2);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, s1_x4);

                let u12 = _mm256_add_epi32(u1, u2);
                let u12_x2 = _mm256_slli_epi32(u12, 1);

                let inc = _mm256_add_epi32(_mm256_add_epi32(u1, u12_x2), u3);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, inc);

                let u34 = _mm256_add_epi32(u3, u4);
                let total_u = _mm256_add_epi32(u12, u34);
                v_s1 = _mm256_add_epi32(v_s1, total_u);

                ptr = ptr.add(128);
                chunk_n -= 128;
            }
            v_s2 = _mm256_add_epi32(
                v_s2,
                _mm256_add_epi32(
                    _mm256_add_epi32(v_s2_a, v_s2_b),
                    _mm256_add_epi32(v_s2_c, v_s2_d),
                ),
            );
            let processed = ptr as usize - data.as_ptr() as usize;
            data = &data[processed..];
        }

        while chunk_n >= 128 {
            let v_zero = _mm256_setzero_si256();
            let d1 = _mm256_loadu_si256(data.as_ptr() as *const _);
            let d2 = _mm256_loadu_si256(data.as_ptr().add(32) as *const _);
            let d3 = _mm256_loadu_si256(data.as_ptr().add(64) as *const _);
            let d4 = _mm256_loadu_si256(data.as_ptr().add(96) as *const _);

            let u1 = _mm256_dpbusd_epi32(v_zero, d1, ones);
            let u2 = _mm256_dpbusd_epi32(v_zero, d2, ones);
            let u3 = _mm256_dpbusd_epi32(v_zero, d3, ones);
            let u4 = _mm256_dpbusd_epi32(v_zero, d4, ones);

            let p1 = _mm256_dpbusd_epi32(v_zero, d1, mults);
            let p2 = _mm256_dpbusd_epi32(v_zero, d2, mults);
            let p3 = _mm256_dpbusd_epi32(v_zero, d3, mults);
            let p4 = _mm256_dpbusd_epi32(v_zero, d4, mults);

            v_s2 = _mm256_add_epi32(
                v_s2,
                _mm256_add_epi32(_mm256_add_epi32(p1, p2), _mm256_add_epi32(p3, p4)),
            );

            let s1_x4 = _mm256_slli_epi32(v_s1, 2);
            let u12 = _mm256_add_epi32(u1, u2);
            let u12_x2 = _mm256_slli_epi32(u12, 1);
            let inc = _mm256_add_epi32(_mm256_add_epi32(u1, u12_x2), u3);
            v_s1_sums = _mm256_add_epi32(v_s1_sums, _mm256_add_epi32(s1_x4, inc));

            let total_u = _mm256_add_epi32(u12, _mm256_add_epi32(u3, u4));
            v_s1 = _mm256_add_epi32(v_s1, total_u);

            data = &data[128..];
            chunk_n -= 128;
        }

        while chunk_n >= 32 {
            let d = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
            v_s1_sums = _mm256_add_epi32(v_s1_sums, v_s1);
            v_s1 = _mm256_dpbusd_avx_epi32(v_s1, d, ones);
            v_s2 = _mm256_dpbusd_avx_epi32(v_s2, d, mults);
            data = &data[32..];
            chunk_n -= 32;
        }

        v_s2 = _mm256_add_epi32(v_s2, _mm256_slli_epi32(v_s1_sums, 5));

        let v_s1_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s1, 0),
            _mm256_extracti128_si256(v_s1, 1),
        );
        let v_s2_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s2, 0),
            _mm256_extracti128_si256(v_s2, 1),
        );

        let v_s1_sum = _mm_add_epi32(v_s1_128, _mm_shuffle_epi32(v_s1_128, 0x31));
        let v_s1_sum = _mm_add_epi32(v_s1_sum, _mm_shuffle_epi32(v_s1_sum, 0x02));

        let v_s2_sum = _mm_add_epi32(v_s2_128, _mm_shuffle_epi32(v_s2_128, 0x31));
        let v_s2_sum = _mm_add_epi32(v_s2_sum, _mm_shuffle_epi32(v_s2_sum, 0x02));

        s1 += _mm_cvtsi128_si32(v_s1_sum) as u32;
        s2 += _mm_cvtsi128_si32(v_s2_sum) as u32;

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    if data.len() >= 16 {
        let d = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        let v_zero_xmm = _mm_setzero_si128();

        let sad = _mm_sad_epu8(d, v_zero_xmm);
        let s1_part = _mm_cvtsi128_si32(_mm_add_epi32(sad, _mm_unpackhi_epi64(sad, sad))) as u32;

        s2 += s1 * 16;
        s1 += s1_part;

        let w_16 = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let p = _mm_maddubs_epi16(d, w_16);
        let s = _mm_madd_epi16(p, _mm_set1_epi16(1));

        let s_sum = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        let s_sum = _mm_add_epi32(s_sum, _mm_shuffle_epi32(s_sum, 0xB1));

        s2 += _mm_cvtsi128_si32(s_sum) as u32;

        let processed = 16;
        data = &data[processed..];
    }

    let mut ptr = data.as_ptr();
    let mut len = data.len();
    adler32_tail!(s1, s2, ptr, len);
    s1 %= DIVISOR;
    s2 %= DIVISOR;

    (s2 << 16) | s1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_avx256(v: __m256i) -> u32 {
    let v128 = _mm_add_epi32(
        _mm256_extracti128_si256(v, 0),
        _mm256_extracti128_si256(v, 1),
    );
    let v64 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, 0x4E));
    let v32 = _mm_add_epi32(v64, _mm_shuffle_epi32(v64, 0xB1));
    _mm_cvtsi128_si32(v32) as u32
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub unsafe fn adler32_x86_avx512_vnni(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    if data.len() > 2048 {
        let align = (data.as_ptr() as usize) & 63;
        if align != 0 {
            let len = std::cmp::min(data.len(), 64 - align);
            for &b in &data[..len] {
                s1 += b as u32;
                s2 += s1;
            }
            s1 %= DIVISOR;
            s2 %= DIVISOR;
            data = &data[len..];
        }
    }

    let ones_u8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let mults = _mm512_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    );

    while data.len() >= 64 {
        let n = std::cmp::min(data.len(), BLOCK_SIZE) & !63;
        s2 += s1 * (n as u32);

        let mut v_s1 = _mm512_setzero_si512();
        let mut v_s2 = _mm512_setzero_si512();
        let mut v_s1_sums = _mm512_setzero_si512();

        let mut chunk_n = n;

        if chunk_n >= 256 {
            let mut ptr = data.as_ptr();
            let mut v_s2_a = _mm512_setzero_si512();
            let mut v_s2_b = _mm512_setzero_si512();
            let mut v_s2_c = _mm512_setzero_si512();
            let mut v_s2_d = _mm512_setzero_si512();
            let mut v_s2_e = _mm512_setzero_si512();
            let mut v_s2_f = _mm512_setzero_si512();
            let mut v_s2_g = _mm512_setzero_si512();
            let mut v_s2_h = _mm512_setzero_si512();

            while chunk_n >= 512 {
                let d1 = _mm512_loadu_si512(ptr as *const _);
                let d2 = _mm512_loadu_si512(ptr.add(64) as *const _);
                let d3 = _mm512_loadu_si512(ptr.add(128) as *const _);
                let d4 = _mm512_loadu_si512(ptr.add(192) as *const _);
                let d5 = _mm512_loadu_si512(ptr.add(256) as *const _);
                let d6 = _mm512_loadu_si512(ptr.add(320) as *const _);
                let d7 = _mm512_loadu_si512(ptr.add(384) as *const _);
                let d8 = _mm512_loadu_si512(ptr.add(448) as *const _);

                v_s2_a = _mm512_add_epi32(
                    v_s2_a,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d1, mults), ones_i16),
                );
                v_s2_b = _mm512_add_epi32(
                    v_s2_b,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d2, mults), ones_i16),
                );
                v_s2_c = _mm512_add_epi32(
                    v_s2_c,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d3, mults), ones_i16),
                );
                v_s2_d = _mm512_add_epi32(
                    v_s2_d,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d4, mults), ones_i16),
                );
                v_s2_e = _mm512_add_epi32(
                    v_s2_e,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d5, mults), ones_i16),
                );
                v_s2_f = _mm512_add_epi32(
                    v_s2_f,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d6, mults), ones_i16),
                );
                v_s2_g = _mm512_add_epi32(
                    v_s2_g,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d7, mults), ones_i16),
                );
                v_s2_h = _mm512_add_epi32(
                    v_s2_h,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d8, mults), ones_i16),
                );

                let u1 = _mm512_madd_epi16(_mm512_maddubs_epi16(d1, ones_u8), ones_i16);
                let u2 = _mm512_madd_epi16(_mm512_maddubs_epi16(d2, ones_u8), ones_i16);
                let u3 = _mm512_madd_epi16(_mm512_maddubs_epi16(d3, ones_u8), ones_i16);
                let u4 = _mm512_madd_epi16(_mm512_maddubs_epi16(d4, ones_u8), ones_i16);
                let u5 = _mm512_madd_epi16(_mm512_maddubs_epi16(d5, ones_u8), ones_i16);
                let u6 = _mm512_madd_epi16(_mm512_maddubs_epi16(d6, ones_u8), ones_i16);
                let u7 = _mm512_madd_epi16(_mm512_maddubs_epi16(d7, ones_u8), ones_i16);
                let u8 = _mm512_madd_epi16(_mm512_maddubs_epi16(d8, ones_u8), ones_i16);

                let u12 = _mm512_add_epi32(u1, u2);
                let u12_x2 = _mm512_slli_epi32(u12, 1);
                let inc_a = _mm512_add_epi32(_mm512_add_epi32(u1, u12_x2), u3);

                let u56 = _mm512_add_epi32(u5, u6);
                let u56_x2 = _mm512_slli_epi32(u56, 1);
                let inc_b = _mm512_add_epi32(_mm512_add_epi32(u5, u56_x2), u7);

                let u34 = _mm512_add_epi32(u3, u4);
                let total_u_a = _mm512_add_epi32(u12, u34);

                let u78 = _mm512_add_epi32(u7, u8);
                let total_u_b = _mm512_add_epi32(u56, u78);

                let u_a_x4 = _mm512_slli_epi32(total_u_a, 2);
                let combined_inc = _mm512_add_epi32(_mm512_add_epi32(inc_a, inc_b), u_a_x4);

                let s1_x8 = _mm512_slli_epi32(v_s1, 3);
                v_s1_sums = _mm512_add_epi32(v_s1_sums, _mm512_add_epi32(s1_x8, combined_inc));

                v_s1 = _mm512_add_epi32(v_s1, _mm512_add_epi32(total_u_a, total_u_b));

                ptr = ptr.add(512);
                chunk_n -= 512;
            }

            while chunk_n >= 256 {
                let d1 = _mm512_loadu_si512(ptr as *const _);
                let d2 = _mm512_loadu_si512(ptr.add(64) as *const _);
                let d3 = _mm512_loadu_si512(ptr.add(128) as *const _);
                let d4 = _mm512_loadu_si512(ptr.add(192) as *const _);

                v_s2_a = _mm512_add_epi32(
                    v_s2_a,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d1, mults), ones_i16),
                );
                v_s2_b = _mm512_add_epi32(
                    v_s2_b,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d2, mults), ones_i16),
                );
                v_s2_c = _mm512_add_epi32(
                    v_s2_c,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d3, mults), ones_i16),
                );
                v_s2_d = _mm512_add_epi32(
                    v_s2_d,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(d4, mults), ones_i16),
                );

                let u1 = _mm512_madd_epi16(_mm512_maddubs_epi16(d1, ones_u8), ones_i16);
                let u2 = _mm512_madd_epi16(_mm512_maddubs_epi16(d2, ones_u8), ones_i16);
                let u3 = _mm512_madd_epi16(_mm512_maddubs_epi16(d3, ones_u8), ones_i16);
                let u4 = _mm512_madd_epi16(_mm512_maddubs_epi16(d4, ones_u8), ones_i16);

                let s1_x4 = _mm512_slli_epi32(v_s1, 2);
                v_s1_sums = _mm512_add_epi32(v_s1_sums, s1_x4);

                let u12 = _mm512_add_epi32(u1, u2);
                let u12_x2 = _mm512_slli_epi32(u12, 1);
                let inc = _mm512_add_epi32(_mm512_add_epi32(u1, u12_x2), u3);
                v_s1_sums = _mm512_add_epi32(v_s1_sums, inc);

                let u34 = _mm512_add_epi32(u3, u4);
                let total_u = _mm512_add_epi32(u12, u34);
                v_s1 = _mm512_add_epi32(v_s1, total_u);

                ptr = ptr.add(256);
                chunk_n -= 256;
            }

            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_a, v_s2_b));
            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_c, v_s2_d));
            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_e, v_s2_f));
            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_g, v_s2_h));

            let processed = ptr as usize - data.as_ptr() as usize;
            data = &data[processed..];
        }

        while chunk_n >= 128 {
            let d1 = _mm512_loadu_si512(data.as_ptr() as *const _);
            let d2 = _mm512_loadu_si512(data.as_ptr().add(64) as *const _);

            let u1 = _mm512_madd_epi16(_mm512_maddubs_epi16(d1, ones_u8), ones_i16);
            let u2 = _mm512_madd_epi16(_mm512_maddubs_epi16(d2, ones_u8), ones_i16);

            let p1 = _mm512_madd_epi16(_mm512_maddubs_epi16(d1, mults), ones_i16);
            let p2 = _mm512_madd_epi16(_mm512_maddubs_epi16(d2, mults), ones_i16);

            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(p1, p2));

            let s1_x2 = _mm512_slli_epi32(v_s1, 1);
            let inc = _mm512_add_epi32(s1_x2, u1);
            v_s1_sums = _mm512_add_epi32(v_s1_sums, inc);

            v_s1 = _mm512_add_epi32(v_s1, _mm512_add_epi32(u1, u2));

            data = &data[128..];
            chunk_n -= 128;
        }

        while chunk_n >= 64 {
            let d = _mm512_loadu_si512(data.as_ptr() as *const _);
            v_s1_sums = _mm512_add_epi32(v_s1_sums, v_s1);
            v_s1 = _mm512_add_epi32(
                v_s1,
                _mm512_madd_epi16(_mm512_maddubs_epi16(d, ones_u8), ones_i16),
            );
            v_s2 = _mm512_add_epi32(
                v_s2,
                _mm512_madd_epi16(_mm512_maddubs_epi16(d, mults), ones_i16),
            );
            data = &data[64..];
            chunk_n -= 64;
        }

        v_s2 = _mm512_add_epi32(v_s2, _mm512_slli_epi32(v_s1_sums, 6));

        let v_s1_256 = _mm256_add_epi32(
            _mm512_extracti64x4_epi64(v_s1, 0),
            _mm512_extracti64x4_epi64(v_s1, 1),
        );
        let v_s2_256 = _mm256_add_epi32(
            _mm512_extracti64x4_epi64(v_s2, 0),
            _mm512_extracti64x4_epi64(v_s2, 1),
        );

        let v_s1_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s1_256, 0),
            _mm256_extracti128_si256(v_s1_256, 1),
        );
        let v_s2_128 = _mm_add_epi32(
            _mm256_extracti128_si256(v_s2_256, 0),
            _mm256_extracti128_si256(v_s2_256, 1),
        );

        let v_s1_sum = _mm_add_epi32(v_s1_128, _mm_shuffle_epi32(v_s1_128, 0x31));
        let v_s1_sum = _mm_add_epi32(v_s1_sum, _mm_shuffle_epi32(v_s1_sum, 0x02));

        let v_s2_sum = _mm_add_epi32(v_s2_128, _mm_shuffle_epi32(v_s2_128, 0x31));
        let v_s2_sum = _mm_add_epi32(v_s2_sum, _mm_shuffle_epi32(v_s2_sum, 0x02));

        s1 += _mm_cvtsi128_si32(v_s1_sum) as u32;
        s2 += _mm_cvtsi128_si32(v_s2_sum) as u32;

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    if data.len() >= 32 {
        let d = _mm256_loadu_si256(data.as_ptr() as *const _);
        let ones_256 = _mm256_set1_epi8(1);
        let mults_256 = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );

        let u = _mm256_dpbusd_epi32(_mm256_setzero_si256(), d, ones_256);
        let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), d, mults_256);

        let sum_u = hsum_epi32_avx256(u);
        let sum_p = hsum_epi32_avx256(p);

        s2 += s1 * 32 + sum_p;
        s1 += sum_u;

        data = &data[32..];
    }

    if data.len() > 0 {
        let len = data.len();
        let mask = (1u32 << len) - 1;
        let d = _mm256_maskz_loadu_epi8(mask, data.as_ptr() as *const i8);

        let s1_part = hsum_epi32_avx256(_mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            d,
            _mm256_set1_epi8(1),
        ));
        let s2_part_raw = hsum_epi32_avx256(_mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            d,
            _mm256_set_epi8(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ),
        ));

        let s2_part = s2_part_raw.wrapping_sub(((32 - len) as u32).wrapping_mul(s1_part));
        s2 = s2.wrapping_add(s1.wrapping_mul(len as u32));
        s2 = s2.wrapping_add(s2_part);
        s1 = s1.wrapping_add(s1_part);

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    (s2 << 16) | s1
}
