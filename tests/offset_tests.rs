use libdeflate::{Compressor, Decompressor};

#[test]
fn test_offset_2_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'AB' repeated
    let pattern = b"ABABABABABABABABABABABABABABABAB";
    let compressed = compressor.compress_deflate(pattern).unwrap();

    let decompressed = decompressor.decompress_deflate(&compressed, pattern.len()).unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_4_pattern() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    // Pattern 'ABCD' repeated
    let pattern = b"ABCDABCDABCDABCDABCDABCDABCDABCD";
    let compressed = compressor.compress_deflate(pattern).unwrap();

    let decompressed = decompressor.decompress_deflate(&compressed, pattern.len()).unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_2_long() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    let pattern: Vec<u8> = b"AB".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor.decompress_deflate(&compressed, pattern.len()).unwrap();
    assert_eq!(decompressed, pattern);
}

#[test]
fn test_offset_4_long() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    let pattern: Vec<u8> = b"ABCD".iter().cloned().cycle().take(1000).collect();
    let compressed = compressor.compress_deflate(&pattern).unwrap();

    let decompressed = decompressor.decompress_deflate(&compressed, pattern.len()).unwrap();
    assert_eq!(decompressed, pattern);
}
