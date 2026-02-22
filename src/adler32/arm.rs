#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

const DIVISOR: u32 = 65521;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn adler32_arm_neon(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    let mults: [u16; 64] = [
        64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
        41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
        18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ];

    let mults_v = [
        vld1q_u16(mults.as_ptr().add(0)),
        vld1q_u16(mults.as_ptr().add(8)),
        vld1q_u16(mults.as_ptr().add(16)),
        vld1q_u16(mults.as_ptr().add(24)),
        vld1q_u16(mults.as_ptr().add(32)),
        vld1q_u16(mults.as_ptr().add(40)),
        vld1q_u16(mults.as_ptr().add(48)),
        vld1q_u16(mults.as_ptr().add(56)),
    ];

    while data.len() > 0 {
        let n = std::cmp::min(data.len(), 5504) & !63;
        if n == 0 {
            break;
        }

        s2 += s1 * (n as u32);

        let mut v_s1 = vdupq_n_u32(0);
        let mut v_s2 = vdupq_n_u32(0);

        let mut v_byte_sums = [vdupq_n_u16(0); 8];

        let mut chunk_n = n;
        while chunk_n >= 64 {
            let data_a = vld1q_u8(data.as_ptr().add(0));
            let data_b = vld1q_u8(data.as_ptr().add(16));
            let data_c = vld1q_u8(data.as_ptr().add(32));
            let data_d = vld1q_u8(data.as_ptr().add(48));

            v_s2 = vaddq_u32(v_s2, v_s1);

            // Optimization: Break dependency chain for v_s1 accumulation.
            // Instead of accumulating pairwise sums serially (4x), we compute them in two parallel
            // branches and then combine. This allows better instruction-level parallelism.
            let tmp_a = vpaddlq_u8(data_a);
            let tmp_c = vpaddlq_u8(data_c);

            let tmp_ab = vpadalq_u8(tmp_a, data_b);
            let tmp_cd = vpadalq_u8(tmp_c, data_d);

            let tmp = vaddq_u16(tmp_ab, tmp_cd);

            v_byte_sums[0] = vaddw_u8(v_byte_sums[0], vget_low_u8(data_a));
            v_byte_sums[1] = vaddw_u8(v_byte_sums[1], vget_high_u8(data_a));
            v_byte_sums[2] = vaddw_u8(v_byte_sums[2], vget_low_u8(data_b));
            v_byte_sums[3] = vaddw_u8(v_byte_sums[3], vget_high_u8(data_b));
            v_byte_sums[4] = vaddw_u8(v_byte_sums[4], vget_low_u8(data_c));
            v_byte_sums[5] = vaddw_u8(v_byte_sums[5], vget_high_u8(data_c));
            v_byte_sums[6] = vaddw_u8(v_byte_sums[6], vget_low_u8(data_d));
            v_byte_sums[7] = vaddw_u8(v_byte_sums[7], vget_high_u8(data_d));

            v_s1 = vpadalq_u16(v_s1, tmp);

            data = &data[64..];
            chunk_n -= 64;
        }

        v_s2 = vshlq_n_u32(v_s2, 6);
        for i in 0..8 {
            v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums[i]), vget_low_u16(mults_v[i]));
            v_s2 = vmlal_high_u16(v_s2, v_byte_sums[i], mults_v[i]);
        }

        s1 += vaddvq_u32(v_s1);
        s2 += vaddvq_u32(v_s2);

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    for &b in data {
        s1 += b as u32;
        s2 += s1;
    }

    (s2 % DIVISOR) << 16 | (s1 % DIVISOR)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn adler32_arm_neon_dotprod(adler: u32, p: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut data = p;

    let mults: [u8; 64] = [
        64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
        41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
        18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ];

    let mults_v = [
        vld1q_u8(mults.as_ptr().add(0)),
        vld1q_u8(mults.as_ptr().add(16)),
        vld1q_u8(mults.as_ptr().add(32)),
        vld1q_u8(mults.as_ptr().add(48)),
    ];
    let ones = vdupq_n_u8(1);

    while data.len() > 0 {
        let n = std::cmp::min(data.len(), 5504) & !63;
        if n == 0 {
            break;
        }

        s2 += s1 * (n as u32);

        let mut v_s1_a = vdupq_n_u32(0);
        let mut v_s1_b = vdupq_n_u32(0);
        let mut v_s1_c = vdupq_n_u32(0);
        let mut v_s1_d = vdupq_n_u32(0);
        let mut v_s2_a = vdupq_n_u32(0);
        let mut v_s2_b = vdupq_n_u32(0);
        let mut v_s2_c = vdupq_n_u32(0);
        let mut v_s2_d = vdupq_n_u32(0);
        let mut v_s1_sums_a = vdupq_n_u32(0);
        let mut v_s1_sums_b = vdupq_n_u32(0);
        let mut v_s1_sums_c = vdupq_n_u32(0);
        let mut v_s1_sums_d = vdupq_n_u32(0);

        let mut chunk_n = n;
        while chunk_n >= 64 {
            let data_a = vld1q_u8(data.as_ptr().add(0));
            let data_b = vld1q_u8(data.as_ptr().add(16));
            let data_c = vld1q_u8(data.as_ptr().add(32));
            let data_d = vld1q_u8(data.as_ptr().add(48));

            v_s1_sums_a = vaddq_u32(v_s1_sums_a, v_s1_a);
            v_s1_a = vdotq_u32(v_s1_a, data_a, ones);
            v_s2_a = vdotq_u32(v_s2_a, data_a, mults_v[0]);

            v_s1_sums_b = vaddq_u32(v_s1_sums_b, v_s1_b);
            v_s1_b = vdotq_u32(v_s1_b, data_b, ones);
            v_s2_b = vdotq_u32(v_s2_b, data_b, mults_v[1]);

            v_s1_sums_c = vaddq_u32(v_s1_sums_c, v_s1_c);
            v_s1_c = vdotq_u32(v_s1_c, data_c, ones);
            v_s2_c = vdotq_u32(v_s2_c, data_c, mults_v[2]);

            v_s1_sums_d = vaddq_u32(v_s1_sums_d, v_s1_d);
            v_s1_d = vdotq_u32(v_s1_d, data_d, ones);
            v_s2_d = vdotq_u32(v_s2_d, data_d, mults_v[3]);

            data = &data[64..];
            chunk_n -= 64;
        }

        let v_s1 = vaddq_u32(vaddq_u32(v_s1_a, v_s1_b), vaddq_u32(v_s1_c, v_s1_d));
        let mut v_s2 = vaddq_u32(vaddq_u32(v_s2_a, v_s2_b), vaddq_u32(v_s2_c, v_s2_d));
        let v_s1_sums = vaddq_u32(
            vaddq_u32(v_s1_sums_a, v_s1_sums_b),
            vaddq_u32(v_s1_sums_c, v_s1_sums_d),
        );

        v_s2 = vaddq_u32(v_s2, vshlq_n_u32(v_s1_sums, 6));

        s1 += vaddvq_u32(v_s1);
        s2 += vaddvq_u32(v_s2);

        s1 %= DIVISOR;
        s2 %= DIVISOR;
    }

    for &b in data {
        s1 += b as u32;
        s2 += s1;
    }

    (s2 % DIVISOR) << 16 | (s1 % DIVISOR)
}
