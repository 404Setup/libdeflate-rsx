use crate::crc32_tables::{CRC32_SLICE1_TABLE, CRC32_SLICE8_TABLE};

pub fn crc32_slice1(mut crc: u32, p: &[u8]) -> u32 {
    for &b in p {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[(crc as u8 ^ b) as usize];
    }
    crc
}

pub fn crc32_slice8(mut crc: u32, mut p: &[u8]) -> u32 {
    let mut len = p.len();
    while len > 0 && (p.as_ptr() as usize) & 7 != 0 {
        crc = (crc >> 8) ^ CRC32_SLICE8_TABLE[(crc as u8 ^ p[0]) as usize];
        p = &p[1..];
        len -= 1;
    }
    while len >= 8 {
        let v1 = u32::from_le_bytes(p[0..4].try_into().unwrap());
        let v2 = u32::from_le_bytes(p[4..8].try_into().unwrap());
        crc = CRC32_SLICE8_TABLE[0x700 + ((crc ^ v1) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x600 + (((crc ^ v1) >> 8) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x500 + (((crc ^ v1) >> 16) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x400 + (((crc ^ v1) >> 24) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x300 + (v2 as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x200 + ((v2 >> 8) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x100 + ((v2 >> 16) as u8) as usize]
            ^ CRC32_SLICE8_TABLE[0x000 + ((v2 >> 24) as u8) as usize];
        p = &p[8..];
        len -= 8;
    }
    for &b in p {
        crc = (crc >> 8) ^ CRC32_SLICE8_TABLE[(crc as u8 ^ b) as usize];
    }
    crc
}

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

pub fn crc32(crc: u32, slice: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("vpclmulqdq")
        {
            return unsafe { !x86::crc32_x86_vpclmulqdq_avx512_vl512(!crc, slice) };
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq") {
            return unsafe { !x86::crc32_x86_vpclmulqdq_avx2(!crc, slice) };
        }

        if is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
            return unsafe { !x86::crc32_x86_pclmulqdq(!crc, slice) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            return unsafe { !arm::crc32_arm(!crc, slice) };
        }
    }
    !crc32_slice8(!crc, slice)
}