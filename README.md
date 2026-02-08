# libdeflate-rs

A complete Rust port of libdeflate, without C code.

## Feature

- Includes streaming processing API
- Includes batch processing API
- A highly optimized implementation, faster than C binding
- Support full concurrency, greatly improving processing speed

## Bug

- Adler32 poor performance
- Compress/S-level performance is poor
- Decompress/S-level poor performance at the beginning of the level

## Usage

**I'm still fixing a bug that caused the performance degradation, so I can't import it for now.**

```toml
[dependencies]
libdeflate = "0.1.0"
```

## Examples

See [examples](examples)

## SIMD

Currently, the instruction set support is the same as libdeflate.

For architectures/instruction sets that are not supported, a fallback to the slower
implementation will be implemented.

## Environment

- Rust 1.93

## Run Benchmark

**"Full concurrency" is the basic capability of libdeflate-rs, and it is not only effective in Benchmark.**

```bash
py3 gen_bench_data.py
cargo bench
```

## License

2026 404Setup. All rights reserved. Source code is licensed under a BSD-3-Clause License.
