use std::cmp::min;

const DIVISOR: u32 = 65521;
const MAX_CHUNK_LEN: usize = 5552;

#[inline]
fn adler32_chunk(s1: &mut u32, s2: &mut u32, mut p: &[u8]) {
    let mut n = p.len();

    if n >= 4 {
        let mut s1_sum = 0;
        let mut byte_0_sum = 0;
        let mut byte_1_sum = 0;
        let mut byte_2_sum = 0;
        let mut byte_3_sum = 0;

        while n >= 4 {
            s1_sum += *s1;
            *s1 += p[0] as u32 + p[1] as u32 + p[2] as u32 + p[3] as u32;
            byte_0_sum += p[0] as u32;
            byte_1_sum += p[1] as u32;
            byte_2_sum += p[2] as u32;
            byte_3_sum += p[3] as u32;

            p = &p[4..];
            n -= 4;
        }

        *s2 += (4 * (s1_sum + byte_0_sum)) + (3 * byte_1_sum) + (2 * byte_2_sum) + byte_3_sum;
    }

    for &b in p {
        *s1 += b as u32;
        *s2 += *s1;
    }

    *s1 %= DIVISOR;
    *s2 %= DIVISOR;
}

pub fn adler32_generic(adler: u32, mut buffer: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;
    let mut len = buffer.len();

    while len > 0 {
        let n = min(len, MAX_CHUNK_LEN);
        let (chunk, rest) = buffer.split_at(n);
        buffer = rest;
        len -= n;

        adler32_chunk(&mut s1, &mut s2, chunk);
    }

    (s2 % DIVISOR) << 16 | (s1 % DIVISOR)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;

pub fn adler32(adler: u32, slice: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512vnni") {
            return unsafe { x86::adler32_x86_avx512_vnni(adler, slice) };
        }
        if is_x86_feature_detected!("avxvnni") {
            return unsafe { x86::adler32_x86_avx2_vnni(adler, slice) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::adler32_x86_avx2(adler, slice) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { x86::adler32_x86_sse2(adler, slice) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { arm::adler32_arm_neon_dotprod(adler, slice) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { arm::adler32_arm_neon(adler, slice) };
        }
    }

    adler32_generic(adler, slice)
}