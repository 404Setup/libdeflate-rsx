use libdeflate::{Compressor, Decompressor};

#[test]
fn test_offset_3_pattern() {
    let mut compressor = Compressor::new(1).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABC' repeated. Offset 3.
    let pattern: Vec<u8> = b"ABC".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_38_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 38. Offset 38.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_56_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 56. Offset 56.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrst"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_52_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 52. Offset 52.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnop"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_32_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 32. Offset 32.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_26_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 26. Offset 26.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_27_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 27. Offset 27.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_28_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 28. Offset 28.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ01"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_25_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 25. Offset 25.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXY"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_20_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 20. Offset 20.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRST"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_21_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 21. Offset 21.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTU"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_22_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 22. Offset 22.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUV"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_18_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 18. Offset 18.
    let pattern: Vec<u8> = b"123456789012345678"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_14_large() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // 100 KB of repeated pattern. Offset 14.
    let pattern_len = 100 * 1024;
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMN"
        .iter()
        .cloned()
        .cycle()
        .take(pattern_len)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_12_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDEFGHIJKL' repeated. Offset 12.
    let pattern: Vec<u8> = b"ABCDEFGHIJKL".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_5_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDE' repeated. Offset 5.
    let pattern: Vec<u8> = b"ABCDE".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_6_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDEF' repeated. Offset 6.
    let pattern: Vec<u8> = b"ABCDEF".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_7_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDEFG' repeated. Offset 7.
    let pattern: Vec<u8> = b"ABCDEFG".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_8_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDEFGH' repeated. Offset 8.
    let pattern: Vec<u8> = b"ABCDEFGH".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_9_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCDEFGHI' repeated. Offset 9.
    let pattern: Vec<u8> = b"ABCDEFGHI".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_15_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 15. Offset 15.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNO"
        .iter()
        .cloned()
        .cycle()
        .take(1000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_14_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 14. Offset 14.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMN"
        .iter()
        .cloned()
        .cycle()
        .take(1000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_13_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 13. Offset 13.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLM"
        .iter()
        .cloned()
        .cycle()
        .take(1000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_17_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 17. Offset 17.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMN123"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_16_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 16. Offset 16.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOP"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_19_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 19. Offset 19.
    let pattern: Vec<u8> = b"1234567890123456789"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_23_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 23. Offset 23.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVW"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_24_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 24. Offset 24.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWX"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_29_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 29. Offset 29.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_30_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 30. Offset 30.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_31_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 31. Offset 31.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ01234"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_40_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 40. Offset 40.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_64_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 64. Offset 64.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz01"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_48_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 48. Offset 48.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_36_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 36. Offset 36.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_44_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 44. Offset 44.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_60_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 60. Offset 60.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwx"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_58_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 58. Offset 58.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuv"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_42_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 42. Offset 42.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_50_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern length 50. Offset 50.
    // Use unique bytes to ensure no internal matches.
    let pattern: Vec<u8> = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmn"
        .iter()
        .cloned()
        .cycle()
        .take(10000)
        .collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor
        .decompress_deflate(&compressed, pattern.len())
        .unwrap();
    assert_eq!(decompressed, pattern);
}
