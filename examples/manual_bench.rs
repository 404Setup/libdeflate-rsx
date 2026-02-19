use libdeflate::Compressor;
use std::time::Instant;

fn main() {
    let size = 256 * 1024;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }

    let mut compressor = Compressor::new(6).unwrap();
    let bound = compressor.deflate_compress_bound(size);
    let mut out_buf = vec![0u8; bound];

    // Warmup
    for _ in 0..10 {
        compressor
            .compress_deflate_into(&data, &mut out_buf)
            .unwrap();
    }

    let start = Instant::now();
    let iterations = 2000;
    let mut total_bytes = 0;
    for _ in 0..iterations {
        let size = compressor
            .compress_deflate_into(&data, &mut out_buf)
            .unwrap();
        total_bytes += size;
    }
    let duration = start.elapsed();

    println!("Compressed {} iterations of {} bytes", iterations, size);
    println!("Total time: {:?}", duration);
    println!(
        "Throughput: {:.2} GiB/s",
        (iterations as f64 * size as f64) / duration.as_secs_f64() / 1024.0 / 1024.0 / 1024.0
    );
}
