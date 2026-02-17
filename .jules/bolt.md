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
