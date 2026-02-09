use libdeflate::{Compressor, Decompressor, adler32, crc32};

#[test]
fn test_adler32_empty() {
    let buf = [];
    assert_eq!(adler32(1, &buf), 1);
}

#[test]
fn test_adler32_small() {
    let data = b"A";
    let res = adler32(1, data);
    assert_eq!(res, 4325442);

    let data = b"AB";
    let res = adler32(1, data);
    assert_eq!(res, 12976260);
}

#[test]
fn test_adler32_simple() {
    let data = b"adler32";
    let res = adler32(1, data);
    assert_eq!(res, 178520686);

    let data = b"Hello, World!";
    let res = adler32(1, data);
    assert_eq!(res, 530449514);
}

#[test]
fn test_adler32_large() {
    let data = vec![0u8; 1000];
    let expected = 65536001;
    let res = adler32(1, &data);
    assert_eq!(res, expected);
}

#[test]
fn test_crc32_empty() {
    let buf = [];
    assert_eq!(crc32(0, &buf), 0);
}

#[test]
fn test_crc32_simple() {
    let data = b"Hello, World!";
    let res = crc32(0, data);
    assert_eq!(res, 0xEC4AC3D0);
}

#[test]
fn test_crc32_large() {
    let mut data = Vec::new();
    for i in 0..100 {
        data.push(i as u8);
    }
    let res = crc32(0, &data);
    assert_eq!(res, 1489580789);
}

#[test]
fn test_compress_decompress_deflate() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for deflate compression.";
    
    let compressed = compressor.compress_deflate(data).unwrap();
    let decompressed = decompressor.decompress_deflate(&compressed, data.len()).unwrap();
    
    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_decompress_zlib() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for zlib compression.";
    
    let compressed = compressor.compress_zlib(data).unwrap();
    let decompressed = decompressor.decompress_zlib(&compressed, data.len()).unwrap();
    
    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_decompress_gzip() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for gzip compression.";
    
    let compressed = compressor.compress_gzip(data).unwrap();
    let decompressed = decompressor.decompress_gzip(&compressed, data.len()).unwrap();
    
    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_levels() {
    let data = vec![b'a'; 10000];
    let mut c0 = Compressor::new(0).unwrap();
    let comp0 = c0.compress_deflate(&data).unwrap();
    assert!(comp0.len() > 10000);

    let mut c1 = Compressor::new(1).unwrap();
    let comp1 = c1.compress_deflate(&data).unwrap();
    assert!(comp1.len() < comp0.len());

    let mut c12 = Compressor::new(12).unwrap();
    let comp12 = c12.compress_deflate(&data).unwrap();
    assert!(comp12.len() <= comp1.len());
}

#[test]
fn test_decompress_errors() {
    let mut d = Decompressor::new();
    let bad_data = [0u8, 1, 2, 3];
    assert!(d.decompress_deflate(&bad_data, 100).is_err());
    assert!(d.decompress_zlib(&bad_data, 100).is_err());
    assert!(d.decompress_gzip(&bad_data, 100).is_err());
}

#[test]
fn test_buffer_reuse() {
    let mut c = Compressor::new(6).unwrap();
    let mut d = Decompressor::new();
    
    let data1 = b"Data set 1";
    let comp1 = c.compress_deflate(data1).unwrap();
    let decomp1 = d.decompress_deflate(&comp1, data1.len()).unwrap();
    assert_eq!(data1.to_vec(), decomp1);

    let data2 = b"Data set 2 - different content";
    let comp2 = c.compress_deflate(data2).unwrap();
    let decomp2 = d.decompress_deflate(&comp2, data2.len()).unwrap();
    assert_eq!(data2.to_vec(), decomp2);
}
#[test]
fn test_compress_bound_overflow_check() {
    let mut compressor = Compressor::new(1).unwrap();
    let size = usize::MAX - 100;

    let bound = compressor.deflate_compress_bound(size);
    assert!(bound >= size);

    let bound = compressor.zlib_compress_bound(size);
    assert!(bound >= size);

    let bound = compressor.gzip_compress_bound(size);
    assert!(bound >= size);
}

#[test]
fn test_compress_deflate_insufficient_space() {
    let mut compressor = Compressor::new(6).unwrap();
    let data = b"Hello world! This is a test string for deflate compression.";

    // Create an output buffer that is too small for the compressed data
    let mut output = vec![0u8; 5];

    let result = compressor.compress_deflate_into(data, &mut output);

    match result {
        Ok(_) => panic!("Expected compression to fail due to insufficient space"),
        Err(e) => {
            assert_eq!(e.kind(), std::io::ErrorKind::Other);
            assert_eq!(e.to_string(), "Insufficient space");
        }
    }
}
