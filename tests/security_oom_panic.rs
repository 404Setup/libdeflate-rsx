use libdeflate::stream::DeflateEncoder;
use std::io::Write;

#[test]
fn test_deflate_encoder_functionality() {
    let data = vec![0u8; 1024 * 1024];
    let mut encoder = DeflateEncoder::new(Vec::new(), 6);
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();
    assert!(compressed.len() > 0);
}
