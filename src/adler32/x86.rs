#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const DIVISOR: u32 = 65521;

#[target_feature(enable = "sse2")]
pub unsafe fn adler32_x86_sse2(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    while data.len() >= 32 {
        let mut n = std::cmp::min(data.len(), 4096);
        n &= !31;

        s2 += s1 * (n as u32);

        let mut v_s1 = _mm_setzero_si128();
        let mut v_s1_sums = _mm_setzero_si128();
        let mut v_byte_sums_a = _mm_setzero_si128();
        let mut v_byte_sums_b = _mm_setzero_si128();
        let mut v_byte_sums_c = _mm_setzero_si128();
        let mut v_byte_sums_d = _mm_setzero_si128();

        let v_zero = _mm_setzero_si128();

        let mut chunk_n = n;
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

        let mults_a = _mm_set_epi16(25, 26, 27, 28, 29, 30, 31, 32);
        let mults_b = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let mults_c = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16);
        let mults_d = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);

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

    for &b in data {
        s1 += b as u32;
        s2 += s1;
    }

    (s2 % DIVISOR) << 16 | (s1 % DIVISOR)
}

#[target_feature(enable = "avx2")]
pub unsafe fn adler32_x86_avx2(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;

    // Manual pointer tracking to avoid slice overhead in hot loop
    let mut ptr = p.as_ptr();
    let mut len = p.len();

    while len >= 64 {
        let n = std::cmp::min(len, 5552);
        let n_rounded = n & !63;

        s2 += s1 * (n_rounded as u32);

        let mut v_s1 = _mm256_setzero_si256();
        let mut v_s1_acc = _mm256_setzero_si256();
        let mut v_inc_acc_a = _mm256_setzero_si256();
        let mut v_inc_acc_b = _mm256_setzero_si256();

        let mut chunk_n = n_rounded;

        // Use 4 accumulators to increase instruction level parallelism.
        let mut v_s2_a = _mm256_setzero_si256();
        let mut v_s2_b = _mm256_setzero_si256();
        let mut v_s2_c = _mm256_setzero_si256();
        let mut v_s2_d = _mm256_setzero_si256();

        // Only use the heavily unrolled 256-byte loop for larger chunks.
        // For small inputs (e.g. 1KB), the setup and prefetch overhead outweighs the benefits.
        if chunk_n >= 2048 {
            let weights = _mm256_set_epi8(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            );
            let ones_i16 = _mm256_set1_epi16(1);
            let v_zero = _mm256_setzero_si256();

            while chunk_n >= 256 {
                // Manual prefetching removed as it degraded performance on modern CPUs
                // by interfering with hardware prefetchers, especially for L2-resident data.

                {
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
                    let p2 = _mm256_maddubs_epi16(data_b_1, weights);
                    let s_b = _mm256_madd_epi16(p2, ones_i16);
                    v_s2_a = _mm256_add_epi32(v_s2_a, _mm256_add_epi32(s_a, s_b));

                    let p3 = _mm256_maddubs_epi16(data_a_2, weights);
                    let s_c = _mm256_madd_epi16(p3, ones_i16);
                    let p4 = _mm256_maddubs_epi16(data_b_2, weights);
                    let s_d = _mm256_madd_epi16(p4, ones_i16);
                    v_s2_b = _mm256_add_epi32(v_s2_b, _mm256_add_epi32(s_c, s_d));
                }

                {
                    let data_a_1 = _mm256_loadu_si256(ptr.add(128) as *const __m256i);
                    let data_b_1 = _mm256_loadu_si256(ptr.add(160) as *const __m256i);
                    let data_a_2 = _mm256_loadu_si256(ptr.add(192) as *const __m256i);
                    let data_b_2 = _mm256_loadu_si256(ptr.add(224) as *const __m256i);

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
                    v_inc_acc_b = _mm256_add_epi32(v_inc_acc_b, inc_part);
                    v_s1 = _mm256_add_epi32(v_s1, sum_sads);

                    let p1 = _mm256_maddubs_epi16(data_a_1, weights);
                    let s_a = _mm256_madd_epi16(p1, ones_i16);
                    let p2 = _mm256_maddubs_epi16(data_b_1, weights);
                    let s_b = _mm256_madd_epi16(p2, ones_i16);
                    v_s2_c = _mm256_add_epi32(v_s2_c, _mm256_add_epi32(s_a, s_b));

                    let p3 = _mm256_maddubs_epi16(data_a_2, weights);
                    let s_c = _mm256_madd_epi16(p3, ones_i16);
                    let p4 = _mm256_maddubs_epi16(data_b_2, weights);
                    let s_d = _mm256_madd_epi16(p4, ones_i16);
                    v_s2_d = _mm256_add_epi32(v_s2_d, _mm256_add_epi32(s_c, s_d));
                }

                ptr = ptr.add(256);
                chunk_n -= 256;
                len -= 256;
            }
        }

        let weights = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let ones_i16 = _mm256_set1_epi16(1);
        let v_zero = _mm256_setzero_si256();

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

            let p2 = _mm256_maddubs_epi16(data_b_1, weights);
            let s_b = _mm256_madd_epi16(p2, ones_i16);
            v_s2_a = _mm256_add_epi32(v_s2_a, _mm256_add_epi32(s_a, s_b));

            let p3 = _mm256_maddubs_epi16(data_a_2, weights);
            let s_c = _mm256_madd_epi16(p3, ones_i16);

            let p4 = _mm256_maddubs_epi16(data_b_2, weights);
            let s_d = _mm256_madd_epi16(p4, ones_i16);
            v_s2_b = _mm256_add_epi32(v_s2_b, _mm256_add_epi32(s_c, s_d));

            ptr = ptr.add(128);
            chunk_n -= 128;
            len -= 128;
        }

        let mut v_s2 = _mm256_add_epi32(
            _mm256_add_epi32(v_s2_a, v_s2_b),
            _mm256_add_epi32(v_s2_c, v_s2_d),
        );

        let v_inc_acc = _mm256_add_epi32(v_inc_acc_a, v_inc_acc_b);

        let v_s1_shifted = _mm256_slli_epi32(v_s1_acc, 7);
        let v_inc_shifted = _mm256_slli_epi32(v_inc_acc, 5);
        let mut v_s1_sums = _mm256_add_epi32(v_s1_shifted, v_inc_shifted);

        while chunk_n >= 64 {
            let data_a = _mm256_loadu_si256(ptr as *const __m256i);
            let data_b = _mm256_loadu_si256(ptr.add(32) as *const __m256i);

            v_s1_sums = _mm256_add_epi32(v_s1_sums, _mm256_slli_epi32(v_s1, 5));
            let p1 = _mm256_maddubs_epi16(data_a, weights);
            v_s1 = _mm256_add_epi32(v_s1, _mm256_sad_epu8(data_a, v_zero));
            let s_a = _mm256_madd_epi16(p1, ones_i16);
            v_s2 = _mm256_add_epi32(v_s2, s_a);

            v_s1_sums = _mm256_add_epi32(v_s1_sums, _mm256_slli_epi32(v_s1, 5));
            let p2 = _mm256_maddubs_epi16(data_b, weights);
            v_s1 = _mm256_add_epi32(v_s1, _mm256_sad_epu8(data_b, v_zero));
            let s_b = _mm256_madd_epi16(p2, ones_i16);
            v_s2 = _mm256_add_epi32(v_s2, s_b);

            ptr = ptr.add(64);
            chunk_n -= 64;
            len -= 64;
        }

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

    let remaining = core::slice::from_raw_parts(ptr, len);
    if remaining.len() >= 16 {
        let res = adler32_x86_sse2((s2 << 16) | s1, remaining);
        s1 = res & 0xFFFF;
        s2 = res >> 16;
    } else {
        for &b in remaining {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    (s2 << 16) | s1
}

#[target_feature(enable = "avxvnni")]
pub unsafe fn adler32_x86_avx2_vnni(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    let ones = _mm256_set1_epi8(1);
    let mults = _mm256_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    );

    while data.len() >= 32 {
        let n = std::cmp::min(data.len(), 4032) & !31;
        s2 += s1 * (n as u32);

        let mut v_s1 = _mm256_setzero_si256();
        let mut v_s2 = _mm256_setzero_si256();
        let mut v_s1_sums = _mm256_setzero_si256();

        let mut chunk_n = n;

        // Optimization: Unrolled loop for larger chunks (128 bytes per iter).
        // This breaks dependency chains and increases instruction level parallelism.
        if chunk_n >= 2048 {
            let mut ptr = data.as_ptr();
            let mut v_s2_a = _mm256_setzero_si256();
            let mut v_s2_b = _mm256_setzero_si256();
            let v_zero = _mm256_setzero_si256();

            while chunk_n >= 128 {
                let d1 = _mm256_loadu_si256(ptr as *const __m256i);
                let d2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
                let d3 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
                let d4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

                // Calculate intermediate sums for s1 (u1..u4)
                let u1 = _mm256_dpbusd_avx_epi32(v_zero, d1, ones);
                let u2 = _mm256_dpbusd_avx_epi32(v_zero, d2, ones);
                let u3 = _mm256_dpbusd_avx_epi32(v_zero, d3, ones);
                let u4 = _mm256_dpbusd_avx_epi32(v_zero, d4, ones);

                // Update s2 accumulators independently
                v_s2_a = _mm256_dpbusd_avx_epi32(v_s2_a, d1, mults);
                v_s2_b = _mm256_dpbusd_avx_epi32(v_s2_b, d2, mults);
                v_s2_a = _mm256_dpbusd_avx_epi32(v_s2_a, d3, mults);
                v_s2_b = _mm256_dpbusd_avx_epi32(v_s2_b, d4, mults);

                // Update s1_sums
                // formula: S_new = S_old + 4*v_s1 + 3*u1 + 2*u2 + 1*u3
                let s1_x4 = _mm256_slli_epi32(v_s1, 2);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, s1_x4);

                let u12 = _mm256_add_epi32(u1, u2);
                let u12_x2 = _mm256_slli_epi32(u12, 1);
                // inc = u1 + 2(u1+u2) + u3 = 3u1 + 2u2 + u3
                let inc = _mm256_add_epi32(_mm256_add_epi32(u1, u12_x2), u3);
                v_s1_sums = _mm256_add_epi32(v_s1_sums, inc);

                // Update s1
                let u34 = _mm256_add_epi32(u3, u4);
                let total_u = _mm256_add_epi32(u12, u34);
                v_s1 = _mm256_add_epi32(v_s1, total_u);

                ptr = ptr.add(128);
                chunk_n -= 128;
            }
            v_s2 = _mm256_add_epi32(v_s2, _mm256_add_epi32(v_s2_a, v_s2_b));
            let processed = ptr as usize - data.as_ptr() as usize;
            data = &data[processed..];
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
        let res = adler32_x86_sse2((s2 << 16) | s1, data);
        s1 = res & 0xFFFF;
        s2 = res >> 16;
    } else {
        for &b in data {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    (s2 << 16) | s1
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn adler32_x86_avx512_vnni(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    let ones = _mm512_set1_epi8(1);
    let mults = _mm512_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    );

    while data.len() >= 64 {
        let n = std::cmp::min(data.len(), 4032) & !63;
        s2 += s1 * (n as u32);

        let mut v_s1 = _mm512_setzero_si512();
        let mut v_s2 = _mm512_setzero_si512();
        let mut v_s1_sums = _mm512_setzero_si512();

        let mut chunk_n = n;

        // Optimization: Unrolled loop for larger chunks (256 bytes per iter).
        // This breaks dependency chains and increases instruction level parallelism.
        if chunk_n >= 2048 {
            let mut ptr = data.as_ptr();
            let mut v_s2_a = _mm512_setzero_si512();
            let mut v_s2_b = _mm512_setzero_si512();
            let mut v_s2_c = _mm512_setzero_si512();
            let mut v_s2_d = _mm512_setzero_si512();
            let v_zero = _mm512_setzero_si512();

            while chunk_n >= 256 {
                let d1 = _mm512_loadu_si512(ptr as *const _);
                let d2 = _mm512_loadu_si512(ptr.add(64) as *const _);
                let d3 = _mm512_loadu_si512(ptr.add(128) as *const _);
                let d4 = _mm512_loadu_si512(ptr.add(192) as *const _);

                // Update s2 accumulators independently with local weighted sums
                v_s2_a = _mm512_dpbusd_epi32(v_s2_a, d1, mults);
                v_s2_b = _mm512_dpbusd_epi32(v_s2_b, d2, mults);
                v_s2_c = _mm512_dpbusd_epi32(v_s2_c, d3, mults);
                v_s2_d = _mm512_dpbusd_epi32(v_s2_d, d4, mults);

                // Calculate sums of each 64-byte block
                let u1 = _mm512_dpbusd_epi32(v_zero, d1, ones);
                let u2 = _mm512_dpbusd_epi32(v_zero, d2, ones);
                let u3 = _mm512_dpbusd_epi32(v_zero, d3, ones);
                let u4 = _mm512_dpbusd_epi32(v_zero, d4, ones);

                // Update s1_sums
                // Formula: S_new = S_old + 4*v_s1 + 3*u1 + 2*u2 + 1*u3
                let s1_x4 = _mm512_slli_epi32(v_s1, 2); // v_s1 * 4
                v_s1_sums = _mm512_add_epi32(v_s1_sums, s1_x4);

                let u12 = _mm512_add_epi32(u1, u2);
                let u12_x2 = _mm512_slli_epi32(u12, 1); // 2*(u1+u2)
                let inc = _mm512_add_epi32(_mm512_add_epi32(u1, u12_x2), u3); // 3u1 + 2u2 + u3
                v_s1_sums = _mm512_add_epi32(v_s1_sums, inc);

                // Update v_s1
                let u34 = _mm512_add_epi32(u3, u4);
                let total_u = _mm512_add_epi32(u12, u34);
                v_s1 = _mm512_add_epi32(v_s1, total_u);

                ptr = ptr.add(256);
                chunk_n -= 256;
            }

            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_a, v_s2_b));
            v_s2 = _mm512_add_epi32(v_s2, _mm512_add_epi32(v_s2_c, v_s2_d));

            let processed = ptr as usize - data.as_ptr() as usize;
            data = &data[processed..];
        }

        while chunk_n >= 64 {
            let d = _mm512_loadu_si512(data.as_ptr() as *const _);
            v_s1_sums = _mm512_add_epi32(v_s1_sums, v_s1);
            v_s1 = _mm512_dpbusd_epi32(v_s1, d, ones);
            v_s2 = _mm512_dpbusd_epi32(v_s2, d, mults);
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
        let res = adler32_x86_avx2((s2 << 16) | s1, data);
        s1 = res & 0xFFFF;
        s2 = res >> 16;
    } else if data.len() >= 16 {
        let res = adler32_x86_sse2((s2 << 16) | s1, data);
        s1 = res & 0xFFFF;
        s2 = res >> 16;
    } else {
        for &b in data {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    (s2 << 16) | s1
}
