use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use libdeflate::batch;
use libdeflate::compress::bitstream::Bitstream;
use libdeflate::crc32::{crc32_slice1, crc32_slice8};
use libdeflate::stream;
use libdeflate::{Compressor, Decompressor, adler32, crc32};
use std::fs::File;
use std::io::{Read, Write};
use std::mem::MaybeUninit;
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
        b.iter(|| crc32_slice8(0, std::hint::black_box(&data)));
    });

    group.finish();
}

fn bench_decompress_offset8_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    // 8 bytes pattern
    let pattern = b"12345678";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset8 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset8 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_adler32_nano(c: &mut Criterion) {
    let sizes = [16, 31, 32, 48, 63];
    let mut group = c.benchmark_group("Adler32 Nano");

    for size in sizes {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("libdeflate-rs", size),
            &size,
            |b, &_size| {
                b.iter(|| adler32(1, &data));
            },
        );

        group.bench_with_input(BenchmarkId::new("libdeflater", size), &size, |b, &_size| {
            b.iter(|| libdeflater::adler32(&data));
        });
    }
    group.finish();
}

fn bench_decompress_offset28(c: &mut Criterion) {
    let path = "bench_data/data_offset28.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset28");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset28", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset29(c: &mut Criterion) {
    let path = "bench_data/data_offset29.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset29");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset29", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset30(c: &mut Criterion) {
    let path = "bench_data/data_offset30.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset30");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset30", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset31(c: &mut Criterion) {
    let path = "bench_data/data_offset31.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset31");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset31", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset25(c: &mut Criterion) {
    let path = "bench_data/data_offset25.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset25");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset25", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset22(c: &mut Criterion) {
    let path = "bench_data/data_offset22.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset22");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset22", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset15(c: &mut Criterion) {
    let path = "bench_data/data_offset15.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset15");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset15", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset1(c: &mut Criterion) {
    let path = "bench_data/data_offset1.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset1");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset1", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset2(c: &mut Criterion) {
    let path = "bench_data/data_offset2.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset2");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset2", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset4(c: &mut Criterion) {
    let path = "bench_data/data_offset4.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset4");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset4", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_crc32_micro(c: &mut Criterion) {
    // Sizes chosen to test tail handling (assuming 16-byte SIMD blocks):
    // 20: 16*1 + 4  (tail 4, should use slice1)
    // 28: 16*1 + 12 (tail 12, should use slice8 + slice1)
    // 100: 16*6 + 4 (tail 4, should use slice1)
    // 108: 16*6 + 12 (tail 12, should use slice8 + slice1)
    // 1024: 16*64 (tail 0)
    let sizes = [20, 28, 48, 50, 60, 64, 80, 100, 108, 127, 128, 1024];
    let mut group = c.benchmark_group("CRC32 Micro");

    for size in sizes {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("libdeflate-rs", size),
            &size,
            |b, &_size| {
                b.iter(|| crc32(0, &data));
            },
        );

        group.bench_with_input(BenchmarkId::new("libdeflater", size), &size, |b, &_size| {
            b.iter(|| libdeflater::crc32(&data));
        });
    }
    group.finish();
}

fn bench_adler32_micro(c: &mut Criterion) {
    let sizes = [128, 256, 384, 512, 1024, 2048, 4096, 8192];
    let mut group = c.benchmark_group("Adler32 Micro");

    for size in sizes {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("libdeflate-rs", size),
            &size,
            |b, &_size| {
                b.iter(|| adler32(1, &data));
            },
        );

        group.bench_with_input(BenchmarkId::new("libdeflater", size), &size, |b, &_size| {
            b.iter(|| libdeflater::adler32(&data));
        });
    }
    group.finish();
}

fn bench_adler32_tail(c: &mut Criterion) {
    // Sizes to test tail optimization: 1, 2, 3 (small), 7 (4+3), 15 (8+4+3), 31 (16+8+4+3)
    let sizes = [1, 2, 3, 7, 15, 31];
    let mut group = c.benchmark_group("Adler32 Tail");

    for size in sizes {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("libdeflate-rs", size),
            &size,
            |b, &_size| {
                b.iter(|| adler32(1, &data));
            },
        );

        group.bench_with_input(BenchmarkId::new("libdeflater", size), &size, |b, &_size| {
            b.iter(|| libdeflater::adler32(&data));
        });
    }
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

        group.bench_with_input(
            BenchmarkId::new(format!("Adler32/libdeflate-rs {}", name), size),
            &size,
            |b, &_size| {
                b.iter(|| adler32(1, &data));
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Adler32/libdeflater {}", name), size),
            &size,
            |b, &_size| {
                b.iter(|| libdeflater::adler32(&data));
            },
        );

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
    let levels = [1, 6, 9, 10];

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
                        compressor
                            .compress_deflate_into(&data, &mut out_buf)
                            .unwrap_or(0)
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
            let compressed_size = compressor
                .compress_deflate_into(&original_data, &mut compressed_data)
                .unwrap();

            let mut out_buf = vec![0u8; size];

            group.throughput(Throughput::Bytes(size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("libdeflate-rs {} Level {}", name, level), size),
                &size,
                |b, &_size| {
                    let mut decompressor = Decompressor::new();
                    b.iter(|| {
                        decompressor
                            .decompress_deflate_into(
                                &compressed_data[..compressed_size],
                                &mut out_buf,
                            )
                            .unwrap_or(0)
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
    let compressed_size = compressor
        .compress_deflate_into(&data, &mut compressed_data)
        .unwrap();
    let compressed_slice = &compressed_data[..compressed_size];

    group.bench_with_input("DeflateDecoder", &size, |b, &_size| {
        b.iter(|| {
            let mut decoder = stream::DeflateDecoder::new(compressed_slice);
            let mut buf = vec![0u8; 64 * 1024];
            while let Ok(n) = decoder.read(&mut buf) {
                if n == 0 {
                    break;
                }
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
                compressor
                    .compress_deflate_into(&data, &mut out_buf)
                    .unwrap_or(0)
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
                compressor
                    .compress_deflate_into(&data, &mut out_buf)
                    .unwrap_or(0)
            });
        });
    }

    group.finish();
}

fn bench_decompress_offset8(c: &mut Criterion) {
    let path = "bench_data/data_offset8.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset8");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset8", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset7(c: &mut Criterion) {
    let path = "bench_data/data_offset7.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset7");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset7", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset3(c: &mut Criterion) {
    let path = "bench_data/data_offset3.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset3");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset3", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset3_small(c: &mut Criterion) {
    let path = "bench_data/data_offset3_small.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset3 small");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset3 small", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset9_small(c: &mut Criterion) {
    let path = "bench_data/data_offset9_small.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset9 small");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset9 small", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset5(c: &mut Criterion) {
    let path = "bench_data/data_offset5.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset5");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset5", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset9(c: &mut Criterion) {
    let path = "bench_data/data_offset9.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset9");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset9", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset12(c: &mut Criterion) {
    let path = "bench_data/data_offset12.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset12");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset12", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset10(c: &mut Criterion) {
    let path = "bench_data/data_offset10.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset10");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset10", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset11(c: &mut Criterion) {
    let path = "bench_data/data_offset11.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset11");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset11", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset14(c: &mut Criterion) {
    let path = "bench_data/data_offset14.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset14");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset14", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset13(c: &mut Criterion) {
    let path = "bench_data/data_offset13.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset13");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset13", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset16(c: &mut Criterion) {
    let path = "bench_data/data_offset16.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset16");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset16", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset17(c: &mut Criterion) {
    let path = "bench_data/data_offset17.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset17");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset17", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset19(c: &mut Criterion) {
    let path = "bench_data/data_offset19.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset19");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset19", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset20(c: &mut Criterion) {
    let path = "bench_data/data_offset20.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset20");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset20", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset18(c: &mut Criterion) {
    let path = "bench_data/data_offset18.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset18");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset18", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset21(c: &mut Criterion) {
    let path = "bench_data/data_offset21.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset21");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset21", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset10_micro(c: &mut Criterion) {
    let size = 200;
    let pattern = b"1234567890";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset10 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset10 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset17_micro(c: &mut Criterion) {
    let size = 200;
    let pattern = b"12345678901234567";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset17 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset17 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset18_micro(c: &mut Criterion) {
    let size = 200;
    let pattern = b"123456789012345678";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset18 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset18 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_compress_micro(c: &mut Criterion) {
    let size = 256 * 1024;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }

    let mut group = c.benchmark_group("Compress Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs level 6", &size, |b, &_size| {
        let mut compressor = Compressor::new(6).unwrap();
        let bound = compressor.deflate_compress_bound(size);
        let mut out_buf = vec![0u8; bound];
        b.iter(|| {
            compressor
                .compress_deflate_into(&data, &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_bitstream_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bitstream Micro");
    let size = 1024 * 1024; // 1MB buffer

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("write_bits_upto_32 (28 bits)", |b| {
        let mut buffer: Vec<MaybeUninit<u8>> = Vec::with_capacity(size + 1024);
        unsafe {
            buffer.set_len(size + 1024);
        }
        b.iter(|| {
            let mut bs = Bitstream::new(&mut buffer);
            // Write 1MB of data using 28-bit writes
            let iterations = (size * 8) / 28;
            for _ in 0..iterations {
                unsafe {
                    bs.write_bits_upto_32(0x1234567, 28);
                }
            }
            bs.flush();
        });
    });

    group.bench_function("write_bits (13 bits)", |b| {
        let mut buffer: Vec<MaybeUninit<u8>> = Vec::with_capacity(size + 1024);
        unsafe {
            buffer.set_len(size + 1024);
        }
        b.iter(|| {
            let mut bs = Bitstream::new(&mut buffer);
            let iterations = (size * 8) / 13;
            for _ in 0..iterations {
                bs.write_bits(0x1234, 13);
            }
            bs.flush();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bitstream_micro,
    bench_compress_micro,
    bench_decompress_offset10_micro,
    bench_decompress_offset40_micro,
    bench_decompress_offset17_micro,
    bench_decompress_offset18_micro,
    bench_crc32_slice8,
    bench_checksums,
    bench_compress,
    bench_decompress,
    bench_stream,
    bench_batch,
    bench_parallel_alloc,
    bench_adler32_nano,
    bench_adler32_micro,
    bench_adler32_tail,
    bench_crc32_micro,
    bench_decompress_offset8,
    bench_decompress_offset3,
    bench_decompress_offset3_small,
    bench_decompress_offset9_small,
    bench_decompress_offset5,
    bench_decompress_offset7,
    bench_decompress_offset6_micro,
    bench_decompress_offset1,
    bench_decompress_offset2,
    bench_decompress_offset4,
    bench_decompress_offset9,
    bench_decompress_offset10,
    bench_decompress_offset11,
    bench_decompress_offset12,
    bench_decompress_offset13,
    bench_decompress_offset14,
    bench_decompress_offset15,
    bench_decompress_offset16,
    bench_decompress_offset17,
    bench_decompress_offset18,
    bench_decompress_offset19,
    bench_decompress_offset20,
    bench_decompress_offset21,
    bench_decompress_offset22,
    bench_decompress_offset23,
    bench_decompress_offset24,
    bench_decompress_offset25,
    bench_decompress_offset26,
    bench_decompress_offset27,
    bench_decompress_offset28,
    bench_decompress_offset29,
    bench_decompress_offset30,
    bench_decompress_offset31,
    bench_decompress_offset32,
    bench_crc32_slice8_tail,
    bench_crc32_small,
    bench_decompress_offset64_micro,
    bench_decompress_offset56_micro,
    bench_decompress_offset48_micro,
    bench_decompress_offset8_micro,
);
criterion_main!(benches);

fn bench_decompress_offset56_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    // 56 bytes pattern
    let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrst";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset56 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset56 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset48_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    // 48 bytes pattern
    let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset48 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset48 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset32(c: &mut Criterion) {
    let path = "bench_data/data_offset32.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset32");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset32", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset23(c: &mut Criterion) {
    let path = "bench_data/data_offset23.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset23");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset23", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset27(c: &mut Criterion) {
    let path = "bench_data/data_offset27.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset27");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset27", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset26(c: &mut Criterion) {
    let path = "bench_data/data_offset26.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset26");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset26", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset24(c: &mut Criterion) {
    let path = "bench_data/data_offset24.bin";
    if !Path::new(path).exists() {
        return;
    }
    let original_data = read_file(path);
    let size = original_data.len();

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset24");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset24", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset40_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"; // 40 bytes
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset40 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset40 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_crc32_slice8_tail(c: &mut Criterion) {
    let mut group = c.benchmark_group("CRC32 Slice8 Tail");

    let size = 15; // 8 + 4 + 3
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_with_input("15 bytes", &size, |b, &_size| {
        b.iter(|| crc32_slice8(0, std::hint::black_box(&data)));
    });

    let size = 3;
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_with_input("3 bytes", &size, |b, &_size| {
        b.iter(|| crc32_slice8(0, std::hint::black_box(&data)));
    });

    group.finish();
}

fn bench_crc32_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("CRC32 Small");
    let sizes = [1, 2, 3, 4];

    for size in sizes {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("slice1", size), &size, |b, &_size| {
            b.iter(|| crc32_slice1(0, std::hint::black_box(&data)));
        });

        group.bench_with_input(BenchmarkId::new("slice8", size), &size, |b, &_size| {
            b.iter(|| crc32_slice8(0, std::hint::black_box(&data)));
        });
    }
    group.finish();
}

fn bench_decompress_offset6_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    let pattern = b"123456"; // 6 bytes
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset6 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset6 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}

fn bench_decompress_offset64_micro(c: &mut Criterion) {
    let size = 1024 * 1024; // 1MB
    // 64 bytes pattern
    let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz01";
    let mut original_data = Vec::with_capacity(size);
    while original_data.len() < size {
        original_data.extend_from_slice(pattern);
    }
    original_data.truncate(size);

    let mut compressor = Compressor::new(6).unwrap();
    let mut compressed_data = vec![0u8; size + size / 2 + 1024];
    let compressed_size = compressor
        .compress_deflate_into(&original_data, &mut compressed_data)
        .unwrap();

    let mut out_buf = vec![0u8; size];

    let mut group = c.benchmark_group("Decompress offset64 Micro");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input("libdeflate-rs offset64 micro", &size, |b, &_size| {
        let mut decompressor = Decompressor::new();
        b.iter(|| {
            decompressor
                .decompress_deflate_into(&compressed_data[..compressed_size], &mut out_buf)
                .unwrap_or(0)
        });
    });

    group.finish();
}
