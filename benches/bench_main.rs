use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use libdeflate::{Compressor, Decompressor, adler32, crc32};
use libdeflate::crc32::crc32_slice8;
use libdeflate::batch;
use libdeflate::stream;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn read_file(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect(&format!("Failed to open file {}", path));
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");
    data
}

fn bench_crc32_slice8(c: &mut Criterion) {
    let path = "bench_data/data_L.bin";
    if !Path::new(path).exists() {
        return;
    }
    let data = read_file(path);
    let size = data.len();

    let mut group = c.benchmark_group("CRC32 Slice8");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs slice8", &size, |b, &_size| {
        b.iter(|| crc32_slice8(0, &data));
    });

    group.finish();
}

fn bench_checksums(c: &mut Criterion) {
    let files = [
        ("XXS", "bench_data/data_XXS.bin"),
        ("S", "bench_data/data_S.bin"),
        ("L", "bench_data/data_L.bin"),
    ];

    let mut group = c.benchmark_group("Checksums");

    for (name, path) in &files {
        if !Path::new(path).exists() {
            continue;
        }
        let data = read_file(path);
        let size = data.len();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new(format!("Adler32/libdeflate-rs {}", name), size), &size, |b, &_size| {
            b.iter(|| adler32(1, &data));
        });

        group.bench_with_input(BenchmarkId::new(format!("Adler32/libdeflater {}", name), size), &size, |b, &_size| {
            b.iter(|| libdeflater::adler32(&data));
        });

        if *name == "L" {
             group.bench_with_input("CRC32/libdeflate-rs", &size, |b, &_size| {
                b.iter(|| crc32(0, &data));
            });

            group.bench_with_input("CRC32/libdeflater", &size, |b, &_size| {
                b.iter(|| libdeflater::crc32(&data));
            });
        }
    }

    group.finish();
}

fn bench_compress(c: &mut Criterion) {
    let files = [
        ("XXS", "bench_data/data_XXS.bin"),
        ("XS", "bench_data/data_XS.bin"),
        ("S", "bench_data/data_S.bin"),
        ("M", "bench_data/data_M.bin"),
        ("L", "bench_data/data_L.bin"),
        ("XL", "bench_data/data_XL.bin"),
    ];
    let levels = [1, 6, 9];

    let mut group = c.benchmark_group("Compress");

    for (name, path) in &files {
        if !Path::new(path).exists() {
            continue;
        }
        let data = read_file(path);
        let size = data.len();

        let mut out_buf = vec![0u8; size + size / 2 + 1024];

        for &level in &levels {
            group.throughput(Throughput::Bytes(size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("libdeflate-rs {} Level {}", name, level), size),
                &size,
                |b, &_size| {
                    let mut compressor = Compressor::new(level).unwrap();
                    b.iter(|| {
                        compressor.compress_deflate_into(&data, &mut out_buf).unwrap_or(0)
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("libdeflater {} Level {}", name, level), size),
                &size,
                |b, &_size| {
                    let mut compressor = libdeflater::Compressor::new(
                        libdeflater::CompressionLvl::new(level).unwrap(),
                    );
                    b.iter(|| compressor.deflate_compress(&data, &mut out_buf).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let files = [
        ("XXS", "bench_data/data_XXS.bin"),
        ("XS", "bench_data/data_XS.bin"),
        ("S", "bench_data/data_S.bin"),
        ("M", "bench_data/data_M.bin"),
        ("L", "bench_data/data_L.bin"),
        ("XL", "bench_data/data_XL.bin"),
    ];
    let levels = [1, 6, 9];

    let mut group = c.benchmark_group("Decompress");

    for (name, path) in &files {
        if !Path::new(path).exists() {
            continue;
        }
        let original_data = read_file(path);
        let size = original_data.len();

        for &level in &levels {
            let mut compressor = Compressor::new(level).unwrap();
            let mut compressed_data = vec![0u8; size + size / 2 + 1024];
            let compressed_size = compressor.compress_deflate_into(
                &original_data,
                &mut compressed_data,
            ).unwrap();

            let mut out_buf = vec![0u8; size];

            group.throughput(Throughput::Bytes(size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("libdeflate-rs {} Level {}", name, level), size),
                &size,
                |b, &_size| {
                    let mut decompressor = Decompressor::new();
                    b.iter(|| {
                        decompressor.decompress_deflate_into(
                            &compressed_data[..compressed_size],
                            &mut out_buf,
                        ).unwrap_or(0)
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("libdeflater {} Level {}", name, level), size),
                &size,
                |b, &_size| {
                    let mut decompressor = libdeflater::Decompressor::new();
                    b.iter(|| {
                        let res = decompressor
                            .deflate_decompress(&compressed_data[..compressed_size], &mut out_buf);
                        if let Err(_) = res {
                            // Ignore errors for bench continuity
                        }
                        res.unwrap_or(0)
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_stream(c: &mut Criterion) {
    let path = "bench_data/data_M.bin";
    if !Path::new(path).exists() {
        return;
    }
    let data = read_file(path);
    let size = data.len();

    let mut group = c.benchmark_group("Stream Processing");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("DeflateEncoder", &size, |b, &_size| {
        b.iter(|| {
            let sink = std::io::sink();
            let mut encoder = stream::DeflateEncoder::new(sink, 6).with_buffer_size(64 * 1024);
            encoder.write_all(&data).unwrap();
            encoder.finish().unwrap();
        });
    });

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor.compress_deflate_into(&data, &mut compressed_data).unwrap();
    let compressed_slice = &compressed_data[..compressed_size];

    group.bench_with_input("DeflateDecoder", &size, |b, &_size| {
        b.iter(|| {
            let mut decoder = stream::DeflateDecoder::new(compressed_slice);
            let mut buf = vec![0u8; 64 * 1024];
            while let Ok(n) = decoder.read(&mut buf) {
                if n == 0 { break; }
            }
        });
    });

    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    let path = "bench_data/data_M.bin";
    if !Path::new(path).exists() {
        return;
    }
    let data = read_file(path);

    let chunk_size = 32 * 1024;
    let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();
    let total_size: usize = chunks.iter().map(|c| c.len()).sum();

    let mut group = c.benchmark_group("Batch Processing");
    group.throughput(Throughput::Bytes(total_size as u64));

    group.bench_with_input("BatchCompressor", &total_size, |b, &_size| {
        let compressor = batch::BatchCompressor::new(6);
        b.iter(|| compressor.compress_batch(&chunks));
    });

    let compressor = batch::BatchCompressor::new(6);
    let compressed_chunks = compressor.compress_batch(&chunks);
    let compressed_refs: Vec<&[u8]> = compressed_chunks.iter().map(|v| v.as_slice()).collect();
    let max_sizes: Vec<usize> = chunks.iter().map(|c| c.len()).collect();

    group.bench_with_input("BatchDecompressor", &total_size, |b, &_size| {
        let decompressor = batch::BatchDecompressor::new();
        b.iter(|| decompressor.decompress_batch(&compressed_refs, &max_sizes));
    });

    group.finish();
}

fn bench_parallel_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Alloc");
    let level = 1;

    // Case 1: Compressible data (data_L.bin)
    let path = "bench_data/data_L.bin";
    if Path::new(path).exists() {
        let data = read_file(path);
        let size = data.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input("Compress Parallel Compressible", &size, |b, &_size| {
            let mut compressor = Compressor::new(level).unwrap();
            let bound = compressor.deflate_compress_bound(size);
            let mut out_buf = vec![0u8; bound];
            b.iter(|| {
                 compressor.compress_deflate_into(&data, &mut out_buf).unwrap_or(0)
            });
        });
    }

    // Case 2: Incompressible data (data_random_L.bin)
    let path_random = "bench_data/data_random_L.bin";
    if Path::new(path_random).exists() {
        let data = read_file(path_random);
        let size = data.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input("Compress Parallel Incompressible", &size, |b, &_size| {
            let mut compressor = Compressor::new(level).unwrap();
            let bound = compressor.deflate_compress_bound(size);
            let mut out_buf = vec![0u8; bound];
            b.iter(|| {
                 compressor.compress_deflate_into(&data, &mut out_buf).unwrap_or(0)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_crc32_slice8,
    bench_checksums,
    bench_compress,
    bench_decompress,
    bench_stream,
    bench_batch,
    bench_parallel_alloc,
);
criterion_main!(benches);
