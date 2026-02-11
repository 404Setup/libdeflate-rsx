## 2024-05-23 - [Deflate Accessor Optimization]
**Learning:** Replacing dynamic if-else chains with static lookup tables for `get_length_slot` improved compression throughput by ~22%.
**Action:** When implementing Huffman coding or similar algorithms, precompute static tables for frequently accessed mappings instead of computing them on the fly, especially in hot loops.
