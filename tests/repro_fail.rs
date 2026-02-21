#[cfg(test)]
mod tests {
    use libdeflate::{crc32, adler32};

    #[test]
    fn test_crc32_len_20() {
        let size = 20;
        let data: Vec<u8> = (0..size).map(|i| (i % 255) as u8).collect();
        let res = crc32(0, &data);
        // Expected value from libdeflater (or known good source)
        // From failure log: right: 1004404644
        assert_eq!(res, 1004404644, "CRC32 mismatch for size 20");
    }

    #[test]
    fn test_crc32_len_4() {
        let size = 4;
        let data: Vec<u8> = (0..size).map(|i| (i % 255) as u8).collect();
        let res = crc32(0, &data);
        // Calculate expected for 0, 1, 2, 3
        // libdeflater calc or manual
        // Let's assume the previous pass implied len 4 is interesting.
        // I'll print it if I can't match it.
        // For now, let's just see if len 20 matches.
    }

    #[test]
    fn test_adler32_huge_repro() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 255) as u8).collect();
        let res = adler32(1, &data);
        // From failure log: right: 1336954002
        assert_eq!(res, 1336954002, "Adler32 mismatch for size 10000");
    }
}
