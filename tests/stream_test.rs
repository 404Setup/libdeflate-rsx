use libdeflate::stream::{DeflateDecoder, DeflateEncoder};
use std::io::{Cursor, Read, Write};

#[test]
fn test_stream_round_trip() {
    let mut data = Vec::with_capacity(10000);
    for i in 0..10000 {
        data.push((i % 256) as u8);
    }

    let mut encoder = DeflateEncoder::new(Vec::new(), 6);
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder = DeflateDecoder::new(Cursor::new(compressed));
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_stream_small_chunks() {
    let mut data = Vec::with_capacity(10000);
    for i in 0..10000 {
        data.push((i % 256) as u8);
    }

    let mut encoder = DeflateEncoder::new(Vec::new(), 6);
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder = DeflateDecoder::new(Cursor::new(compressed));
    let mut decompressed = Vec::new();
    let mut buf = [0u8; 10];
    loop {
        let n = decoder.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buf[..n]);
    }

    assert_eq!(data, decompressed);
}
