## 2025-02-23 - Prevention of OOM Panics in API
**Vulnerability:** The public API `compress_deflate` and `decompress_deflate` used `vec![0u8; size]`, which panics on allocation failure. This allows DoS attacks via memory exhaustion.
**Learning:** Rust's default allocation panics. For libraries exposing allocation based on user input, always use `Vec::try_reserve` or similar fallible allocation methods.
**Prevention:** Audit all `vec![]` macros or `Vec::with_capacity` calls where the size is user-controlled.

## 2025-02-19 - State Corruption in Mixed Decompression Modes
**Vulnerability:** A `Decompressor` reused for both streaming (`decompress_streaming`) and one-shot (`decompress`) operations could enter a corrupted state. The one-shot x86 BMI2 optimization overwrote internal Huffman tables without resetting the state machine. If streaming was subsequently resumed on the same instance, it would use the clobbered tables on the pending data stream, leading to data corruption or crashes.
**Learning:** "Stateless" optimization paths in stateful objects must explicitly reset or invalidate the object's state to prevent side effects on future stateful operations. Performance optimizations that bypass the main state machine are risky if they share mutable state (like tables).
**Prevention:** When implementing optimized fast-paths that bypass the standard state machine, verify they leave the object in a consistent known state (e.g., `Start` or `Invalid`). Add regression tests that mix different API modes (streaming vs one-shot) on the same object instance.

## 2025-02-23 - Heap Buffer Over-read in Decompression Optimization
**Vulnerability:** The internal `prepare_pattern` function used an unsafe optimization for offsets 5 and 6, reading 8 bytes (`u64`) from the output buffer to extract a 5 or 6-byte pattern. This caused a heap buffer over-read of up to 3 bytes when the pattern source was near the end of the allocated buffer. While often benign due to allocator padding, it is Undefined Behavior and could cause crashes or information leaks.
**Learning:** Performance optimizations using `unsafe` pointer arithmetic must strictly respect buffer boundaries. Loading a full word (`u64`) to mask out bytes is only safe if the buffer extends beyond the needed bytes.
**Prevention:** When optimizing small reads with wider loads, always verify the buffer size or use a composition of smaller safe loads (e.g., `u32` + `u8` instead of `u64`). Add boundary tests for `unsafe` logic.
