use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use libdeflate::Compressor;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn read_file(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect(&format!("Failed to open file {}", path));
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");
    data
}

fn bench_parallel_compress(c: &mut Criterion) {
    let files = [
        ("L", "bench_data/data_L.bin"),
    ];
    let levels = [1, 6]; // Test fast and default levels

    let mut group = c.benchmark_group("ParallelCompress");

    for (name, path) in &files {
        if !Path::new(path).exists() {
            println!("Skipping {} because file not found", name);
            continue;
        }
        let data = read_file(path);
        let size = data.len();

        // Ensure buffer is large enough
        let mut out_buf = vec![0u8; size + size / 2 + 1024];

        for &level in &levels {
            group.throughput(Throughput::Bytes(size as u64));
            group.sample_size(10); // Large files take time, fewer samples

            group.bench_with_input(
                BenchmarkId::new(format!("Level {}", level), name),
                &size,
                |b, &_size| {
                    let mut compressor = Compressor::new(level).unwrap();
                    b.iter(|| {
                        compressor.compress_deflate_into(&data, &mut out_buf).unwrap_or(0)
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_compress,
);
criterion_main!(benches);
