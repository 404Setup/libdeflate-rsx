use crate::crc32_tables::{CRC32_SLICE1_TABLE, CRC32_SLICE8_TABLE};
use std::sync::OnceLock;

pub fn crc32_slice1(mut crc: u32, p: &[u8]) -> u32 {
    for &b in p {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[(crc as u8 ^ b) as usize];
    }
    crc
}

pub fn crc32_slice8(mut crc: u32, p: &[u8]) -> u32 {
    let mut len = p.len();
    let mut ptr = p.as_ptr();
    while len >= 8 {
        let v = u64::from_le(unsafe { std::ptr::read_unaligned(ptr as *const u64) });
        let v1 = v as u32;
        let v2 = (v >> 32) as u32;
        crc = CRC32_SLICE8_TABLE[0x700 + ((crc ^ v1) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x600 + (((crc ^ v1) >> 8) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x500 + (((crc ^ v1) >> 16) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x400 + (((crc ^ v1) >> 24) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x300 + (v2 as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x200 + ((v2 >> 8) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x100 + ((v2 >> 16) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x000 + ((v2 >> 24) as u8) as usize];
        unsafe {
            ptr = ptr.add(8);
        }
        len -= 8;
    }
    if len >= 4 {
        let v = u32::from_le(unsafe { std::ptr::read_unaligned(ptr as *const u32) });
        crc ^= v;
        crc = CRC32_SLICE8_TABLE[0x300 + (crc as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x200 + ((crc >> 8) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x100 + ((crc >> 16) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x000 + ((crc >> 24) as u8) as usize];
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

                crc = (crc >> 24)
                    ^ CRC32_SLICE8_TABLE[0x200 + idx0 as usize]
                    ^ CRC32_SLICE8_TABLE[0x100 + idx1 as usize]
                    ^ CRC32_SLICE8_TABLE[0x000 + idx2 as usize];
            }
            2 => {
                let v = u16::from_le(unsafe { (ptr as *const u16).read_unaligned() }) as u32;
                let b0 = v & 0xFF;
                let b1 = v >> 8;

                let idx0 = (crc as u8 as u32) ^ b0;
                let idx1 = ((crc >> 8) as u8 as u32) ^ b1;

                crc = (crc >> 16)
                    ^ CRC32_SLICE8_TABLE[0x100 + idx0 as usize]
                    ^ CRC32_SLICE8_TABLE[0x000 + idx1 as usize];
            }
            1 => {
                let b0 = unsafe { *ptr } as u32;
                crc = (crc >> 8) ^ CRC32_SLICE8_TABLE[((crc as u8 as u32) ^ b0) as usize];
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
