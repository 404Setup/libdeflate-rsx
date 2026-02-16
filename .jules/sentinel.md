## 2025-02-23 - Prevention of OOM Panics in API
**Vulnerability:** The public API `compress_deflate` and `decompress_deflate` used `vec![0u8; size]`, which panics on allocation failure. This allows DoS attacks via memory exhaustion.
**Learning:** Rust's default allocation panics. For libraries exposing allocation based on user input, always use `Vec::try_reserve` or similar fallible allocation methods.
**Prevention:** Audit all `vec![]` macros or `Vec::with_capacity` calls where the size is user-controlled.

## 2025-02-19 - State Corruption in Mixed Decompression Modes
**Vulnerability:** A `Decompressor` reused for both streaming (`decompress_streaming`) and one-shot (`decompress`) operations could enter a corrupted state. The one-shot x86 BMI2 optimization overwrote internal Huffman tables without resetting the state machine. If streaming was subsequently resumed on the same instance, it would use the clobbered tables on the pending data stream, leading to data corruption or crashes.
**Learning:** "Stateless" optimization paths in stateful objects must explicitly reset or invalidate the object's state to prevent side effects on future stateful operations. Performance optimizations that bypass the main state machine are risky if they share mutable state (like tables).
**Prevention:** When implementing optimized fast-paths that bypass the standard state machine, verify they leave the object in a consistent known state (e.g., `Start` or `Invalid`). Add regression tests that mix different API modes (streaming vs one-shot) on the same object instance.
