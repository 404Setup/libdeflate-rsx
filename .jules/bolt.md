## 2024-05-23 - [Deflate Accessor Optimization]
**Learning:** Replacing dynamic if-else chains with static lookup tables for `get_length_slot` improved compression throughput by ~22%.
**Action:** When implementing Huffman coding or similar algorithms, precompute static tables for frequently accessed mappings instead of computing them on the fly, especially in hot loops.

## 2024-05-24 - [SIMD Dependency Breaking]
**Learning:** Splitting SIMD accumulators in hot loops (Adler32) to break dependency chains improved throughput by ~38% on large inputs, despite increased register pressure and recombination logic.
**Action:** In compute-bound loops, identify long dependency chains (e.g. `sum += f(sum)`) and try to split them into parallel accumulations (e.g. `sum1 += f1, sum2 += f2`) even if it adds overhead.

## 2024-05-25 - [Adler32 AVX2 Loop Optimization]
**Learning:** Slice overhead (bounds checks and slice creation) in hot inner loops can be significant, even for simple operations. In the Adler32 AVX2 implementation, replacing `&data[..]` slicing with manual raw pointer arithmetic improved throughput on large inputs (16MB) by ~27% (8.5 GiB/s -> 10.8 GiB/s), surpassing the C-based `libdeflater` implementation.
**Action:** When optimizing extremely hot loops where every cycle counts, consider using `unsafe` pointer arithmetic to eliminate slice overhead, provided that bounds are carefully checked beforehand.
