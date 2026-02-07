use libdeflate::{Compressor, Decompressor};

#[test]
fn test_parallel_compression_roundtrip() {
    let size = 1024 * 1024;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    
    let compressed = compressor.compress_deflate(&data).unwrap();
    assert!(compressed.len() > 0);

    let decompressed = decompressor.decompress_deflate(&compressed, size).unwrap();
    
    assert_eq!(decompressed, data);
}