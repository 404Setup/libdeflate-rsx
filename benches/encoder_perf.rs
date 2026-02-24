use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use libdeflate::stream::DeflateEncoder;
use std::io::Write;

fn bench_encoder_parallel(c: &mut Criterion) {
    let size = 10 * 1024 * 1024; // 10MB
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }

    let mut group = c.benchmark_group("DeflateEncoder Parallel");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("write_all 10MB", |b| {
        b.iter(|| {
            let sink = std::io::sink();
            let mut encoder = DeflateEncoder::new(sink, 6); // Default 1MB buffer
            encoder.write_all(&data).unwrap();
            encoder.finish().unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_encoder_parallel);
criterion_main!(benches);
