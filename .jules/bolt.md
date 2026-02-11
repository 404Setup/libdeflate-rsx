## 2024-05-23 - [Deflate Accessor Optimization]
**Learning:** Replacing dynamic if-else chains with static lookup tables for `get_length_slot` improved compression throughput by ~22%.
**Action:** When implementing Huffman coding or similar algorithms, precompute static tables for frequently accessed mappings instead of computing them on the fly, especially in hot loops.

## 2024-05-24 - [SIMD Dependency Breaking]
**Learning:** Splitting SIMD accumulators in hot loops (Adler32) to break dependency chains improved throughput by ~38% on large inputs, despite increased register pressure and recombination logic.
**Action:** In compute-bound loops, identify long dependency chains (e.g. `sum += f(sum)`) and try to split them into parallel accumulations (e.g. `sum1 += f1, sum2 += f2`) even if it adds overhead.
