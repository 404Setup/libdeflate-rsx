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
fn test_adler32_huge() {
    let data: Vec<u8> = (0..10000).map(|i| (i % 255) as u8).collect();
    let res = adler32(1, &data);
    assert_eq!(res, 1336954002);
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
    let decompressed = decompressor
        .decompress_deflate(&compressed, data.len())
        .unwrap();

    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_decompress_zlib() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for zlib compression.";

    let compressed = compressor.compress_zlib(data).unwrap();
    let decompressed = decompressor
        .decompress_zlib(&compressed, data.len())
        .unwrap();

    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_decompress_gzip() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for gzip compression.";

    let compressed = compressor.compress_gzip(data).unwrap();
    let decompressed = decompressor
        .decompress_gzip(&compressed, data.len())
        .unwrap();

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
fn test_compress_gzip_into_success() {
    let mut compressor = Compressor::new(6).unwrap();
    let mut decompressor = Decompressor::new();
    let data = b"Hello world! This is a test string for gzip compression into buffer.";

    let bound = compressor.gzip_compress_bound(data.len());
    let mut output = vec![0u8; bound];

    let size = compressor.compress_gzip_into(data, &mut output).unwrap();
    assert!(size > 0);
    assert!(size <= bound);

    let decompressed = decompressor
        .decompress_gzip(&output[..size], data.len())
        .unwrap();
    assert_eq!(data.to_vec(), decompressed);
}

#[test]
fn test_compress_gzip_into_insufficient_space() {
    let mut compressor = Compressor::new(6).unwrap();
    let data = b"Hello world! This is a test string for gzip compression.";

    let mut output = vec![0u8; 10];
    let result = compressor.compress_gzip_into(data, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_new_compressor_invalid_level() {
    let res = Compressor::new(-1);
    assert!(res.is_err());
    let err = res.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        err.to_string(),
        "Compression level must be between 0 and 12"
    );

    let res = Compressor::new(13);
    assert!(res.is_err());
    let err = res.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        err.to_string(),
        "Compression level must be between 0 and 12"
    );
}

#[test]
fn test_compress_insufficient_space_panic_prevention() {
    let mut compressor = Compressor::new(1).unwrap();
    let data = vec![0u8; 10000];
    let mut output = vec![0u8; 1];

    let res = compressor.compress_deflate_into(&data, &mut output);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::Other);
}

struct BitWriter {
    data: Vec<u8>,
    bit_buffer: u32,
    bits_in_buffer: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    fn write_bits(&mut self, value: u32, count: u32) {
        self.bit_buffer |= value << self.bits_in_buffer;
        self.bits_in_buffer += count;
        while self.bits_in_buffer >= 8 {
            self.data.push(self.bit_buffer as u8);
            self.bit_buffer >>= 8;
            self.bits_in_buffer -= 8;
        }
    }

    fn write_huffman(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            let bit = (code >> i) & 1;
            self.write_bits(bit, 1);
        }
    }

    fn flush(&mut self) -> Vec<u8> {
        if self.bits_in_buffer > 0 {
            self.data.push(self.bit_buffer as u8);
        }
        self.data.clone()
    }
}

#[test]
fn test_offset_3_bug() {
    let mut writer = BitWriter::new();

    writer.write_bits(3, 3);

    writer.write_huffman(0b01110001, 8);

    writer.write_huffman(0b01110010, 8);

    writer.write_huffman(0b01110011, 8);

    writer.write_huffman(0b0001000, 7);

    writer.write_huffman(0b00010, 5);

    writer.write_huffman(0b0000000, 7);

    let input = writer.flush();

    let mut decompressor = Decompressor::new();
    let mut expected = b"ABC".to_vec();
    expected.extend_from_slice(b"ABCABCABCA");

    let result = decompressor.decompress_deflate(&input, 1024);

    match result {
        Ok(output) => {
            if output != expected {
                println!("Output: {:?}", String::from_utf8_lossy(&output));
                println!("Expect: {:?}", String::from_utf8_lossy(&expected));
                panic!("Decompression mismatch");
            }
        }
        Err(e) => panic!("Decompression failed: {}", e),
    }
}

#[test]
fn test_offset_3_large_match() {
    let mut writer = BitWriter::new();

    writer.write_bits(3, 3);

    writer.write_huffman(0b01110001, 8);
    writer.write_huffman(0b01110010, 8);
    writer.write_huffman(0b01110011, 8);

    writer.write_huffman(0b0001111, 7);
    writer.write_bits(3, 2);

    writer.write_huffman(0b00010, 5);

    writer.write_huffman(0b0000000, 7);

    let input = writer.flush();

    let mut decompressor = Decompressor::new();
    let mut expected = b"ABC".to_vec();
    for _ in 0..10 {
        expected.extend_from_slice(b"ABC");
    }

    let result = decompressor.decompress_deflate(&input, 1024);

    match result {
        Ok(output) => {
            if output != expected {
                for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
                    if a != b {
                        println!(
                            "Mismatch at index {}: got {}, expected {}",
                            i, *a as char, *b as char
                        );
                        break;
                    }
                }
                panic!("Decompression mismatch");
            }
        }
        Err(e) => panic!("Decompression failed: {}", e),
    }
}

#[test]
fn test_crc32_tails_vs_reference() {
    // Verify CRC32 against libdeflater (reference implementation)
    // specifically checking sizes that trigger different tail handling paths.
    let sizes = [
        0, 1, 7, 8, 15, 16,      // Small
        20, 28, 31, 32,          // Medium with tails
        100, 108, 128,           // Larger with tails
        1024, 1036               // Block + tails
    ];

    for &size in &sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 255) as u8).collect();
        let my_res = crc32(0, &data);
        let ref_res = libdeflater::crc32(&data);
        assert_eq!(my_res, ref_res, "CRC32 mismatch for size {}", size);
    }
}
