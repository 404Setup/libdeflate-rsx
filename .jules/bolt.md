## 2024-05-24 - [Fallback Loop Optimization]
**Learning:** Porting SIMD optimizations from a primary fast loop to a secondary fallback loop yielded a 5x throughput improvement (1.2 GB/s -> 5.9 GB/s) for offset 9. However, for offset 8, the existing scalar unrolled loop (~11 GB/s) was slightly faster than the ported SIMD version (~8.9 GB/s), likely due to specific instruction scheduling or overheads.
**Action:** Always benchmark fallback paths when optimizing. Don't assume SIMD is automatically faster than well-unrolled scalar code for small constant offsets; verify each case.
## 2024-05-23 - EOB Panic in Subtables
**Learning:** `libdeflate-rs` had a critical bug where End-Of-Block (EOB) symbols located inside Huffman subtables were not checked for their "Exceptional" status in the Fast Loop (and Safe Loop). This caused the decoder to treat EOB as a Match with corrupted length (because EOB entry generation mixes flags into the length field), leading to integer underflow panics (`total_bits - len`) or garbage memory access.
**Action:** Always verify that Exceptional/Special entries are handled after *every* table lookup, including nested/subtable lookups. Added checks for `HUFFDEC_EXCEPTIONAL` after subtable loads in `src/decompress/x86.rs`.
## 2024-05-23 - AVX2 Register Pressure in Manual Unrolling
**Learning:** When manually unrolling AVX2 loops (e.g., match finding), using `_mm256_cmpeq_epi8(v1, v2)` inside error paths forces the compiler to keep `v1` and `v2` alive, increasing register pressure.
**Action:** Reuse the already computed `xor` vector (used for the fast check) by comparing it against zero: `_mm256_cmpeq_epi8(xor, zero)`. This frees up registers for the unrolled loop.

## 2024-05-24 - CRC32 Tail Optimization
**Learning:** For SIMD algorithms (like CRC32 PCLMULQDQ), the cost of handling small tails (e.g., < 16 bytes) with a scalar byte-loop can dominate performance for small inputs. Using an intermediate optimized scalar path (e.g., slice-by-4 using existing slice-by-8 tables) can significantly improve throughput (33% gain for 28 bytes) by avoiding the slow byte-by-byte loop.
**Action:** When implementing SIMD fallbacks, ensure the scalar fallback is also optimized for chunks smaller than the SIMD width but larger than a single byte.

## 2026-06-03 - [Adler32 AVX2 Register Spilling]
**Learning:** An aggressive 256-byte loop unrolling in `adler32_x86_avx2` using 8 separate vector accumulators caused a 36% throughput regression for 512-byte inputs compared to 384 bytes. The excessive use of YMM registers (accumulators + constants + temporaries) forced the compiler to spill to the stack.
**Action:** When unrolling loops for ILP, carefully balance the number of independent accumulators against the register file size. Reusing accumulators (4 instead of 8) eliminated spills while maintaining instruction-level parallelism for the heavy arithmetic instructions.

## 2026-06-03 - [Tail Loop Overhead]
**Learning:** When implementing tail handling for small inputs (e.g., < 64 bytes) using SIMD, wrapping the logic in a `while` loop caused significant regression (e.g., 31 bytes took 19.9ns vs 13.4ns baseline), likely due to branch prediction overhead or loop mechanics. Unrolling the loop into a sequence of `if` checks (straight-line code) recovered the performance and unlocked gains (13.0ns).
**Action:** For hot paths handling small fixed-size chunks (tails), prefer unrolled `if` sequences over loops to minimize control flow overhead.

## 2026-06-04 - [Adler32 AVX2 VNNI Optimization]
**Learning:** Optimizing `adler32_x86_avx2_vnni` by unrolling to 256 bytes (8 accumulators) yielded a 44% throughput improvement for 256-byte inputs. However, holding intermediate `u` vectors for batch reduction caused register spilling (17+ registers needed).
**Action:** To fit 8 accumulators within 16 AVX2 registers, interleave the reduction of temporary vectors (`u`) with the accumulation steps (`v_s2`), allowing registers to be freed earlier. Merging the global accumulator into a local one and generating `v_zero` on-the-fly also saved registers.

## 2026-06-04 - [Vector Precomputation vs Alignr Chain]
**Learning:** For overlapping patterns where offset is a multiple of 8 (e.g., offset 24), breaking the `alignr` dependency chain by precomputing all vectors in the cycle (LCM of offset and vector size) allowed for effective loop unrolling. This yielded a 32% throughput improvement (7.7 GiB/s -> 10.2 GiB/s) by increasing ILP compared to the serial dependency of iterative `alignr`.
**Action:** When optimizing decompression loops for specific offsets, determine if the pattern cycle is short enough to precompute fully. If so, prefer storing precomputed vectors in an unrolled loop over calculating the next vector from the previous one.
