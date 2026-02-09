use libdeflate::batch::{BatchCompressor, BatchDecompressor};

#[test]
fn test_batch_compress_decompress_roundtrip() {
    let inputs: Vec<&[u8]> = vec![
        b"Hello world! This is a test string for deflate compression.",
        b"Another test string.",
        b"Repeating pattern repeating pattern repeating pattern repeating pattern.",
        b"Short",
        &[0u8; 1000], // Highly compressible
    ];

    // Create compressor with level 6 (default-ish)
    let compressor = BatchCompressor::new(6);
    let compressed_batch = compressor.compress_batch(&inputs);

    assert_eq!(compressed_batch.len(), inputs.len());

    // Prepare for decompression
    let max_out_sizes: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
    let compressed_refs: Vec<&[u8]> = compressed_batch.iter().map(|v| v.as_slice()).collect();

    // Create decompressor
    let decompressor = BatchDecompressor::new();
    let decompressed_batch = decompressor.decompress_batch(&compressed_refs, &max_out_sizes);

    assert_eq!(decompressed_batch.len(), inputs.len());

    for (i, result) in decompressed_batch.iter().enumerate() {
        match result {
            Some(decompressed) => {
                assert_eq!(decompressed.as_slice(), inputs[i], "Mismatch at index {}", i);
            },
            None => panic!("Decompression failed for input index {}", i),
        }
    }
}

#[test]
fn test_batch_empty() {
    let compressor = BatchCompressor::new(6);
    let compressed = compressor.compress_batch(&[]);
    assert!(compressed.is_empty());

    let decompressor = BatchDecompressor::new();
    let decompressed = decompressor.decompress_batch(&[], &[]);
    assert!(decompressed.is_empty());
}

#[test]
fn test_batch_empty_input() {
    let inputs: Vec<&[u8]> = vec![b"", b"Not empty"];
    let compressor = BatchCompressor::new(6);
    let compressed = compressor.compress_batch(&inputs);

    assert_eq!(compressed.len(), 2);
    // Empty input should produce a valid DEFLATE stream (non-empty)
    assert!(!compressed[0].is_empty());

    let max_out_sizes = vec![0, 9];
    let compressed_refs: Vec<&[u8]> = compressed.iter().map(|v| v.as_slice()).collect();

    let decompressor = BatchDecompressor::new();
    let decompressed = decompressor.decompress_batch(&compressed_refs, &max_out_sizes);

    assert_eq!(decompressed.len(), 2);
    assert_eq!(decompressed[0], Some(Vec::new()));
    assert_eq!(decompressed[1], Some(b"Not empty".to_vec()));
}

#[test]
fn test_batch_decompress_error() {
    let invalid_data = vec![0u8, 1, 2, 3, 4, 5]; // Not a valid deflate stream
    let inputs: Vec<&[u8]> = vec![&invalid_data];
    let max_out_sizes = vec![100];

    let decompressor = BatchDecompressor::new();
    let decompressed = decompressor.decompress_batch(&inputs, &max_out_sizes);

    assert_eq!(decompressed.len(), 1);
    assert_eq!(decompressed[0], None);
}

#[test]
fn test_batch_decompress_insufficient_buffer() {
    let input = b"Hello world!";
    let compressor = BatchCompressor::new(6);
    let compressed = compressor.compress_batch(&[input]);

    let compressed_refs: Vec<&[u8]> = compressed.iter().map(|v| v.as_slice()).collect();

    // Buffer too small
    let max_out_sizes = vec![input.len() - 1];

    let decompressor = BatchDecompressor::new();
    let decompressed = decompressor.decompress_batch(&compressed_refs, &max_out_sizes);

    assert_eq!(decompressed.len(), 1);
    assert_eq!(decompressed[0], None);
}
