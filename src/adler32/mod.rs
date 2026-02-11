use std::cmp::min;
use std::sync::OnceLock;

const DIVISOR: u32 = 65521;
const MAX_CHUNK_LEN: usize = 5552;

#[inline]
fn adler32_chunk(s1: &mut u32, s2: &mut u32, mut p: &[u8]) {
    let mut s1_local = *s1;
    let mut s2_local = *s2;

    let mut chunks = p.chunks_exact(16);
    for chunk in chunks.by_ref() {
        let b0 = chunk[0] as u32;
        let b1 = chunk[1] as u32;
        let b2 = chunk[2] as u32;
        let b3 = chunk[3] as u32;
        let b4 = chunk[4] as u32;
        let b5 = chunk[5] as u32;
        let b6 = chunk[6] as u32;
        let b7 = chunk[7] as u32;
        let b8 = chunk[8] as u32;
        let b9 = chunk[9] as u32;
        let b10 = chunk[10] as u32;
        let b11 = chunk[11] as u32;
        let b12 = chunk[12] as u32;
        let b13 = chunk[13] as u32;
        let b14 = chunk[14] as u32;
        let b15 = chunk[15] as u32;

        s2_local += (s1_local << 4)
            + (b0 * 16)
            + (b1 * 15)
            + (b2 * 14)
            + (b3 * 13)
            + (b4 * 12)
            + (b5 * 11)
            + (b6 * 10)
            + (b7 * 9)
            + (b8 * 8)
            + (b9 * 7)
            + (b10 * 6)
            + (b11 * 5)
            + (b12 * 4)
            + (b13 * 3)
            + (b14 * 2)
            + (b15 * 1);

        s1_local +=
            b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15;
    }
    p = chunks.remainder();
    let mut n = p.len();

    while n >= 4 {
        let b0 = p[0] as u32;
        let b1 = p[1] as u32;
        let b2 = p[2] as u32;
        let b3 = p[3] as u32;

        s2_local += (s1_local * 4) + (b0 * 4) + (b1 * 3) + (b2 * 2) + (b3 * 1);
        s1_local += b0 + b1 + b2 + b3;

        p = &p[4..];
        n -= 4;
    }

    for &b in p {
        s1_local += b as u32;
        s2_local += s1_local;
    }

    *s1 = s1_local % DIVISOR;
    *s2 = s2_local % DIVISOR;
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

type Adler32Fn = unsafe fn(u32, &[u8]) -> u32;

pub fn adler32(adler: u32, slice: &[u8]) -> u32 {
    // Optimization: Use `std::sync::OnceLock` to cache the best implementation function pointer.
    // This avoids repeated CPU feature detection overhead on every call, which is beneficial for small inputs.
    static IMPL: OnceLock<Adler32Fn> = OnceLock::new();
    let func = IMPL.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512vnni") {
                return x86::adler32_x86_avx512_vnni;
            }
            if is_x86_feature_detected!("avxvnni") {
                return x86::adler32_x86_avx2_vnni;
            }
            if is_x86_feature_detected!("avx2") {
                return x86::adler32_x86_avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return x86::adler32_x86_sse2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return arm::adler32_arm_neon_dotprod;
            }
            if std::arch::is_aarch64_feature_detected!("neon") {
                return arm::adler32_arm_neon;
            }
        }

        adler32_generic
    });

    unsafe { func(adler, slice) }
}
