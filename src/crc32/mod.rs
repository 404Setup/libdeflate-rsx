use crate::crc32_tables::{CRC32_SLICE1_TABLE, CRC32_SLICE8_TABLE};
use std::sync::OnceLock;

pub fn crc32_slice1(mut crc: u32, p: &[u8]) -> u32 {
    for &b in p {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[(crc as u8 ^ b) as usize];
    }
    crc
}

#[inline]
pub fn crc32_slice8(mut crc: u32, p: &[u8]) -> u32 {
    let mut len = p.len();
    let mut ptr = p.as_ptr();

    // Optimization: Unroll loop to process 16 bytes per iteration.
    // This allows lookups for the second 8-byte chunk (which depend only on data)
    // to be interleaved with the dependency chain of the first 8-byte chunk.
    while len >= 16 {
        let va = u64::from_le(unsafe { std::ptr::read_unaligned(ptr as *const u64) });
        let vb = u64::from_le(unsafe { std::ptr::read_unaligned(ptr.add(8) as *const u64) });

        let va1 = va as u32;
        let va2 = (va >> 32) as u32;
        let vb1 = vb as u32;
        let vb2 = (vb >> 32) as u32;

        let idx0 = ((crc ^ va1) as u8) as usize;
        let idx1 = (((crc ^ va1) >> 8) as u8) as usize;
        let idx2 = (((crc ^ va1) >> 16) as u8) as usize;
        let idx3 = (((crc ^ va1) >> 24) as u8) as usize;
        let idx4 = (va2 as u8) as usize;
        let idx5 = ((va2 >> 8) as u8) as usize;
        let idx6 = ((va2 >> 16) as u8) as usize;
        let idx7 = ((va2 >> 24) as u8) as usize;

        // Prefetch/compute indices for the second half that don't depend on CRC
        let idx12 = (vb2 as u8) as usize;
        let idx13 = ((vb2 >> 8) as u8) as usize;
        let idx14 = ((vb2 >> 16) as u8) as usize;
        let idx15 = ((vb2 >> 24) as u8) as usize;

        let t0 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x700 + idx0) };
        let t1 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x600 + idx1) };
        let t2 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x500 + idx2) };
        let t3 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x400 + idx3) };
        let t4 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x300 + idx4) };
        let t5 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x200 + idx5) };
        let t6 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x100 + idx6) };
        let t7 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x000 + idx7) };

        // Start independent lookups for the second chunk early
        let t12 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x300 + idx12) };
        let t13 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x200 + idx13) };
        let t14 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x100 + idx14) };
        let t15 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x000 + idx15) };

        crc = ((t0 ^ t1) ^ (t2 ^ t3)) ^ ((t4 ^ t5) ^ (t6 ^ t7));

        let idx8 = ((crc ^ vb1) as u8) as usize;
        let idx9 = (((crc ^ vb1) >> 8) as u8) as usize;
        let idx10 = (((crc ^ vb1) >> 16) as u8) as usize;
        let idx11 = (((crc ^ vb1) >> 24) as u8) as usize;

        let t8 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x700 + idx8) };
        let t9 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x600 + idx9) };
        let t10 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x500 + idx10) };
        let t11 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x400 + idx11) };

        crc = ((t8 ^ t9) ^ (t10 ^ t11)) ^ ((t12 ^ t13) ^ (t14 ^ t15));

        unsafe {
            ptr = ptr.add(16);
        }
        len -= 16;
    }

    while len >= 8 {
        let v = u64::from_le(unsafe { std::ptr::read_unaligned(ptr as *const u64) });
        let v1 = v as u32;
        let v2 = (v >> 32) as u32;

        let idx0 = ((crc ^ v1) as u8) as usize;
        let idx1 = (((crc ^ v1) >> 8) as u8) as usize;
        let idx2 = (((crc ^ v1) >> 16) as u8) as usize;
        let idx3 = (((crc ^ v1) >> 24) as u8) as usize;
        let idx4 = (v2 as u8) as usize;
        let idx5 = ((v2 >> 8) as u8) as usize;
        let idx6 = ((v2 >> 16) as u8) as usize;
        let idx7 = ((v2 >> 24) as u8) as usize;

        let t0 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x700 + idx0) };
        let t1 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x600 + idx1) };
        let t2 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x500 + idx2) };
        let t3 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x400 + idx3) };
        let t4 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x300 + idx4) };
        let t5 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x200 + idx5) };
        let t6 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x100 + idx6) };
        let t7 = unsafe { *CRC32_SLICE8_TABLE.get_unchecked(0x000 + idx7) };

        // Optimization: Use tree-based XOR reduction to break dependency chains and increase ILP.
        crc = ((t0 ^ t1) ^ (t2 ^ t3)) ^ ((t4 ^ t5) ^ (t6 ^ t7));

        unsafe {
            ptr = ptr.add(8);
        }
        len -= 8;
    }
    if len >= 4 {
        let v = u32::from_le(unsafe { std::ptr::read_unaligned(ptr as *const u32) });
        crc ^= v;
        crc = unsafe {
            *CRC32_SLICE8_TABLE.get_unchecked(0x300 + (crc as u8) as usize)
                ^ *CRC32_SLICE8_TABLE.get_unchecked(0x200 + ((crc >> 8) as u8) as usize)
                ^ *CRC32_SLICE8_TABLE.get_unchecked(0x100 + ((crc >> 16) as u8) as usize)
                ^ *CRC32_SLICE8_TABLE.get_unchecked(0x000 + ((crc >> 24) as u8) as usize)
        };
        unsafe {
            ptr = ptr.add(4);
        }
        len -= 4;
    }
    if len > 0 {
        match len {
            3 => {
                let v = u16::from_le(unsafe { (ptr as *const u16).read_unaligned() }) as u32;
                let b2 = unsafe { *ptr.add(2) } as u32;
                let b0 = v & 0xFF;
                let b1 = v >> 8;

                let idx0 = (crc as u8 as u32) ^ b0;
                let idx1 = ((crc >> 8) as u8 as u32) ^ b1;
                let idx2 = ((crc >> 16) as u8 as u32) ^ b2;

                crc = unsafe {
                    (crc >> 24)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(0x200 + idx0 as usize)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(0x100 + idx1 as usize)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(0x000 + idx2 as usize)
                };
            }
            2 => {
                let v = u16::from_le(unsafe { (ptr as *const u16).read_unaligned() }) as u32;
                let b0 = v & 0xFF;
                let b1 = v >> 8;

                let idx0 = (crc as u8 as u32) ^ b0;
                let idx1 = ((crc >> 8) as u8 as u32) ^ b1;

                crc = unsafe {
                    (crc >> 16)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(0x100 + idx0 as usize)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(0x000 + idx1 as usize)
                };
            }
            1 => {
                let b0 = unsafe { *ptr } as u32;
                crc = unsafe {
                    (crc >> 8)
                        ^ *CRC32_SLICE8_TABLE.get_unchecked(((crc as u8 as u32) ^ b0) as usize)
                };
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
    crc
}

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

type Crc32Fn = unsafe fn(u32, &[u8]) -> u32;

#[inline]
pub fn crc32(crc: u32, slice: &[u8]) -> u32 {
    static IMPL: OnceLock<Crc32Fn> = OnceLock::new();
    let func = IMPL.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("vpclmulqdq")
            {
                return x86::crc32_x86_vpclmulqdq_avx512_vl512;
            }

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq") {
                return x86::crc32_x86_vpclmulqdq_avx2;
            }

            if is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
                return x86::crc32_x86_pclmulqdq;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("crc") {
                return arm::crc32_arm;
            }
        }
        crc32_slice8
    });

    unsafe { !func(!crc, slice) }
}
