#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
pub unsafe fn crc32_arm(mut crc: u32, p: &[u8]) -> u32 {
    let mut data = p;
    let mut len = data.len();

    if len > 0 {
        let align = (data.as_ptr() as usize) & 7;
        if align != 0 {
            let n = std::cmp::min(len, 8 - align);
            for _ in 0..n {
                crc = __crc32b(crc, data[0]);
                data = &data[1..];
                len -= 1;
            }
        }
    }

    while len >= 64 {
        let ptr = data.as_ptr() as *const u64;
        crc = __crc32d(crc, *ptr);
        crc = __crc32d(crc, *ptr.add(1));
        crc = __crc32d(crc, *ptr.add(2));
        crc = __crc32d(crc, *ptr.add(3));
        crc = __crc32d(crc, *ptr.add(4));
        crc = __crc32d(crc, *ptr.add(5));
        crc = __crc32d(crc, *ptr.add(6));
        crc = __crc32d(crc, *ptr.add(7));
        data = &data[64..];
        len -= 64;
    }

    while len >= 8 {
        crc = __crc32d(crc, *(data.as_ptr() as *const u64));
        data = &data[8..];
        len -= 8;
    }

    for &b in data {
        crc = __crc32b(crc, b);
    }

    crc
}
