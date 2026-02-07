use libdeflate::batch::{BatchCompressor, BatchDecompressor};
use std::time::Instant;

fn main() {
    let data1 = b"Hello world. This is the first string.".repeat(100);
    let data2 = b"Another string for batch processing.".repeat(100);
    let data3 = b"Short string.".to_vec();
    let data4 = vec![b'A'; 10000];

    let inputs = vec![
        data1.as_slice(),
        data2.as_slice(),
        data3.as_slice(),
        data4.as_slice(),
    ];

    println!("Batch compressing {} items...", inputs.len());
    let start = Instant::now();

    let compressor = BatchCompressor::new(6);
    let compressed_data = compressor.compress_batch(&inputs);

    let duration = start.elapsed();
    println!("Compression took: {:?}", duration);

    for (i, data) in compressed_data.iter().enumerate() {
        println!(
            "Item {}: Original size: {}, Compressed size: {}",
            i,
            inputs[i].len(),
            data.len()
        );
    }

    println!("Batch decompressing...");
    let decompressor = BatchDecompressor::new();

    let max_sizes: Vec<usize> = inputs.iter().map(|i| i.len()).collect();

    let compressed_refs: Vec<&[u8]> = compressed_data.iter().map(|v| v.as_slice()).collect();
    let decompressed_results = decompressor.decompress_batch(&compressed_refs, &max_sizes);

    for (i, result) in decompressed_results.iter().enumerate() {
        match result {
            Some(data) => {
                assert_eq!(data.as_slice(), inputs[i]);
                println!("Item {}: Decompression successful.", i);
            }
            None => println!("Item {}: Decompression failed!", i),
        }
    }
}
