use libdeflate::Compressor;
use libdeflate::decompress::{Decompressor, DecompressResult};

#[test]
fn test_decompress_reuse_mixed() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("ssse3") {
        println!("BMI2 and SSSE3 detected, using optimized path");
    } else {
        println!("Optimized path NOT used");
    }

    // Create enough data to force dynamic huffman blocks
    let mut data = Vec::new();
    for i in 0..1000 {
        data.extend_from_slice(b"This is a repeating string to force dynamic huffman encoding. ");
        data.push((i % 256) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();
    let compressed = compressor.compress_deflate(&data).unwrap();

    let mut decompressor = Decompressor::new();

    // First decompression (one-shot)
    let mut out1 = vec![0u8; data.len()];
    let (res, _, size) = decompressor.decompress(&compressed, &mut out1);
    assert_eq!(res, DecompressResult::Success);
    assert_eq!(size, data.len());
    assert_eq!(out1, data);

    println!("First decompression successful. Reusing decompressor...");

    // Second decompression (streaming)
    let mut out2 = vec![0u8; data.len()];
    let mut out_idx = 0;
    let (res, _in_read, out_written) = decompressor.decompress_streaming(&compressed, &mut out2, &mut out_idx);

    assert_eq!(res, DecompressResult::Success, "Streaming decompression failed: {:?}", res);
    assert_eq!(out_written, data.len());
    assert_eq!(&out2[..out_written], data);
}
