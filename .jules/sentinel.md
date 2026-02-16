## 2025-02-23 - Prevention of OOM Panics in API
**Vulnerability:** The public API `compress_deflate` and `decompress_deflate` used `vec![0u8; size]`, which panics on allocation failure. This allows DoS attacks via memory exhaustion.
**Learning:** Rust's default allocation panics. For libraries exposing allocation based on user input, always use `Vec::try_reserve` or similar fallible allocation methods.
**Prevention:** Audit all `vec![]` macros or `Vec::with_capacity` calls where the size is user-controlled.
