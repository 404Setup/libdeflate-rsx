use libdeflate::{Compressor, Decompressor};

#[test]
fn test_parallel_deflate_1mb() {
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

#[test]
fn test_parallel_deflate_boundary() {
    // 256KB + 1 byte
    let size = 256 * 1024 + 1;
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

#[test]
fn test_parallel_deflate_large() {
    // 10MB
    let size = 10 * 1024 * 1024;
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

#[test]
fn test_parallel_zlib_large() {
    let size = 5 * 1024 * 1024;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i.wrapping_mul(3) % 251) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    let compressed = compressor.compress_zlib(&data).unwrap();
    assert!(compressed.len() > 0);

    let decompressed = decompressor.decompress_zlib(&compressed, size).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_parallel_gzip_large() {
    let size = 5 * 1024 * 1024;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i.wrapping_mul(7) % 251) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();

    let compressed = compressor.compress_gzip(&data).unwrap();
    assert!(compressed.len() > 0);

    let decompressed = decompressor.decompress_gzip(&compressed, size).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_parallel_insufficient_space() {
    let size = 1024 * 1024; // 1MB, triggers parallel
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();

    // Provide a buffer that is too small
    let mut output = vec![0u8; 100]; // Definitely too small

    let result = compressor.compress_deflate_into(&data, &mut output);
    assert!(result.is_err());
    // We expect "Insufficient space" or similar error
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Insufficient space") || err.to_string().contains("Compression failed"));
}
