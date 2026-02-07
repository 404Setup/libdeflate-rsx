# libdeflate rust x

A complete Rust port of libdeflate, without C code.

**libdeflate-rsx is currently available, but is still undergoing internal integrity testing.**

## Feature
- Includes streaming processing API
- Includes batch processing API
- Performance is almost identical to libdeflate.

## SIMD
Currently, the instruction set support is the same as libdeflate.

For architectures/instruction sets that are not supported, a fallback to the slower 
implementation will be implemented.

## Environment
- Rust 1.93

## Run Benchmark
```bash
py3 gen_bench_data.py
cargo bench
```

## License
2026 404Setup. All rights reserved. Source code is licensed under a BSD-3-Clause License.
