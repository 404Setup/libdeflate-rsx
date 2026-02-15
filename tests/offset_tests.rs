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
