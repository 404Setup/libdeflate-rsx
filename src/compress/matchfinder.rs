use crate::common::*;
use std::cmp::min;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const MATCHFINDER_HASH_ORDER: usize = 15;
pub const MATCHFINDER_HASH_SIZE: usize = 1 << MATCHFINDER_HASH_ORDER;
pub const MATCHFINDER_WINDOW_SIZE: usize = 32768;

// Function pointer for the optimized match length calculation.
// This avoids repeated feature detection inside the hot loop.
type MatchLenFn = unsafe fn(*const u8, *const u8, usize) -> usize;

pub trait MatchFinderTrait {
    fn reset(&mut self);
    fn prepare(&mut self, len: usize);
    fn advance(&mut self, len: usize);
    fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize);
    fn skip_match(&mut self, data: &[u8], pos: usize, max_depth: usize);
    fn skip_positions(&mut self, data: &[u8], pos: usize, count: usize, max_depth: usize) {
        for i in 0..count {
            self.skip_match(data, pos + i, max_depth);
        }
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize);
}

impl MatchFinderTrait for MatchFinder {
    fn reset(&mut self) {
        self.reset();
    }
    fn prepare(&mut self, len: usize) {
        self.prepare(len);
    }
    fn advance(&mut self, len: usize) {
        self.advance(len);
    }
    fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        self.find_match(data, pos, max_depth)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, _max_depth: usize) {
        self.skip_match(data, pos);
    }
    fn skip_positions(&mut self, data: &[u8], pos: usize, count: usize, _max_depth: usize) {
        self.skip_positions(data, pos, count);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        self.find_matches(data, pos, max_depth, matches)
    }
}

impl MatchFinderTrait for HtMatchFinder {
    fn reset(&mut self) {
        self.reset();
    }
    fn prepare(&mut self, len: usize) {
        self.prepare(len);
    }
    fn advance(&mut self, len: usize) {
        self.advance(len);
    }
    fn find_match(&mut self, data: &[u8], pos: usize, _max_depth: usize) -> (usize, usize) {
        self.find_match(data, pos)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, _max_depth: usize) {
        self.skip_match(data, pos);
    }
    fn skip_positions(&mut self, data: &[u8], pos: usize, count: usize, _max_depth: usize) {
        self.skip_positions(data, pos, count);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        _max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        let (len, offset) = self.find_match(data, pos);
        matches.clear();
        if len >= 3 {
            matches.push((len as u16, offset as u16));
            (len, offset)
        } else {
            (0, 0)
        }
    }
}

impl MatchFinderTrait for BtMatchFinder {
    fn reset(&mut self) {
        self.reset();
    }
    fn prepare(&mut self, len: usize) {
        self.prepare(len);
    }
    fn advance(&mut self, len: usize) {
        self.advance(len);
    }
    fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        self.find_match(data, pos, max_depth)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, max_depth: usize) {
        let mut matches = Vec::new();
        self.advance_one_byte(data, pos, 0, 0, max_depth, &mut matches, false);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        matches.clear();
        self.advance_one_byte(
            data,
            pos,
            DEFLATE_MAX_MATCH_LEN,
            DEFLATE_MAX_MATCH_LEN,
            max_depth,
            matches,
            true,
        );
        if let Some(&(len, offset)) = matches.last() {
            (len as usize, offset as usize)
        } else {
            (0, 0)
        }
    }
}

#[inline(always)]
unsafe fn match_len_sw(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;
    while len + 8 <= max_len {
        let a_val = (a.add(len) as *const u64).read_unaligned();
        let b_val = (b.add(len) as *const u64).read_unaligned();
        if a_val != b_val {
            #[cfg(target_endian = "little")]
            return len + ((a_val ^ b_val).trailing_zeros() / 8) as usize;
            #[cfg(target_endian = "big")]
            return len + ((a_val ^ b_val).leading_zeros() / 8) as usize;
        }
        len += 8;
    }

    if len + 4 <= max_len {
        let a_val = (a.add(len) as *const u32).read_unaligned();
        let b_val = (b.add(len) as *const u32).read_unaligned();
        if a_val != b_val {
            #[cfg(target_endian = "little")]
            return len + ((a_val ^ b_val).trailing_zeros() / 8) as usize;
            #[cfg(target_endian = "big")]
            return len + ((a_val ^ b_val).leading_zeros() / 8) as usize;
        }
        len += 4;
    }

    while len < max_len && *a.add(len) == *b.add(len) {
        len += 1;
    }
    len
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn match_len_sse2(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    // Unroll loop to process 64 bytes per iteration
    while len + 64 <= max_len {
        let v1 = _mm_loadu_si128(a.add(len) as *const __m128i);
        let v2 = _mm_loadu_si128(b.add(len) as *const __m128i);
        let cmp1 = _mm_cmpeq_epi8(v1, v2);
        let mask1 = _mm_movemask_epi8(cmp1) as u32;

        let v3 = _mm_loadu_si128(a.add(len + 16) as *const __m128i);
        let v4 = _mm_loadu_si128(b.add(len + 16) as *const __m128i);
        let cmp2 = _mm_cmpeq_epi8(v3, v4);
        let mask2 = _mm_movemask_epi8(cmp2) as u32;

        let v5 = _mm_loadu_si128(a.add(len + 32) as *const __m128i);
        let v6 = _mm_loadu_si128(b.add(len + 32) as *const __m128i);
        let cmp3 = _mm_cmpeq_epi8(v5, v6);
        let mask3 = _mm_movemask_epi8(cmp3) as u32;

        let v7 = _mm_loadu_si128(a.add(len + 48) as *const __m128i);
        let v8 = _mm_loadu_si128(b.add(len + 48) as *const __m128i);
        let cmp4 = _mm_cmpeq_epi8(v7, v8);
        let mask4 = _mm_movemask_epi8(cmp4) as u32;

        // Combine masks into a single 64-bit check
        let combined = (mask1 as u64)
            | ((mask2 as u64) << 16)
            | ((mask3 as u64) << 32)
            | ((mask4 as u64) << 48);

        if combined != 0xFFFFFFFFFFFFFFFF {
            if mask1 != 0xFFFF {
                return len + (!mask1).trailing_zeros() as usize;
            }
            if mask2 != 0xFFFF {
                return len + 16 + (!mask2).trailing_zeros() as usize;
            }
            if mask3 != 0xFFFF {
                return len + 32 + (!mask3).trailing_zeros() as usize;
            }
            return len + 48 + (!mask4).trailing_zeros() as usize;
        }

        len += 64;
    }

    while len + 16 <= max_len {
        let v1 = _mm_loadu_si128(a.add(len) as *const __m128i);
        let v2 = _mm_loadu_si128(b.add(len) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(v1, v2);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF {
            return len + (!mask).trailing_zeros() as usize;
        }
        len += 16;
    }
    len + match_len_sw(a.add(len), b.add(len), max_len - len)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn match_len_avx2(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    // Fail fast for short matches (common case in greedy/lazy parsing)
    if len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a as *const __m256i);
        let v2 = _mm256_loadu_si256(b as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(v1, v2);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFFFFFF {
            return (!mask).trailing_zeros() as usize;
        }
        len += 32;
    }

    while len + 128 <= max_len {
        let v1 = _mm256_loadu_si256(a.add(len) as *const __m256i);
        let v2 = _mm256_loadu_si256(b.add(len) as *const __m256i);
        let xor1 = _mm256_xor_si256(v1, v2);

        let v3 = _mm256_loadu_si256(a.add(len + 32) as *const __m256i);
        let v4 = _mm256_loadu_si256(b.add(len + 32) as *const __m256i);
        let xor2 = _mm256_xor_si256(v3, v4);

        let v5 = _mm256_loadu_si256(a.add(len + 64) as *const __m256i);
        let v6 = _mm256_loadu_si256(b.add(len + 64) as *const __m256i);
        let xor3 = _mm256_xor_si256(v5, v6);

        let v7 = _mm256_loadu_si256(a.add(len + 96) as *const __m256i);
        let v8 = _mm256_loadu_si256(b.add(len + 96) as *const __m256i);
        let xor4 = _mm256_xor_si256(v7, v8);

        let or1 = _mm256_or_si256(xor1, xor2);
        let or2 = _mm256_or_si256(xor3, xor4);
        let or_all = _mm256_or_si256(or1, or2);

        if _mm256_testz_si256(or_all, or_all) == 1 {
            len += 128;
            continue;
        }

        if _mm256_testz_si256(xor1, xor1) == 0 {
            let cmp = _mm256_cmpeq_epi8(v1, v2);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            return len + (!mask).trailing_zeros() as usize;
        }
        if _mm256_testz_si256(xor2, xor2) == 0 {
            let cmp = _mm256_cmpeq_epi8(v3, v4);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            return len + 32 + (!mask).trailing_zeros() as usize;
        }
        if _mm256_testz_si256(xor3, xor3) == 0 {
            let cmp = _mm256_cmpeq_epi8(v5, v6);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            return len + 64 + (!mask).trailing_zeros() as usize;
        }

        let cmp = _mm256_cmpeq_epi8(v7, v8);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        return len + 96 + (!mask).trailing_zeros() as usize;
    }

    while len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a.add(len) as *const __m256i);
        let v2 = _mm256_loadu_si256(b.add(len) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(v1, v2);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFFFFFF {
            return len + (!mask).trailing_zeros() as usize;
        }
        len += 32;
    }
    len + match_len_sse2(a.add(len), b.add(len), max_len - len)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn match_len_avx512(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    // Fail fast for short matches (common case in greedy/lazy parsing).
    // This avoids loading the second 64-byte chunk if the first one mismatches.
    if len + 64 <= max_len {
        let v1 = _mm512_loadu_si512(a as *const _);
        let v2 = _mm512_loadu_si512(b as *const _);
        let mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if mask != u64::MAX {
            return (!mask).trailing_zeros() as usize;
        }
        len += 64;
    }

    // Optimize: Unroll loop to process 128 bytes per iteration.
    // This reduces loop overhead and allows pipelining of loads.
    while len + 128 <= max_len {
        let v1 = _mm512_loadu_si512(a.add(len) as *const _);
        let v2 = _mm512_loadu_si512(b.add(len) as *const _);
        let mask1 = _mm512_cmpeq_epi8_mask(v1, v2);

        let v3 = _mm512_loadu_si512(a.add(len + 64) as *const _);
        let v4 = _mm512_loadu_si512(b.add(len + 64) as *const _);
        let mask2 = _mm512_cmpeq_epi8_mask(v3, v4);

        if (mask1 & mask2) == u64::MAX {
            len += 128;
            continue;
        }

        if mask1 != u64::MAX {
            return len + (!mask1).trailing_zeros() as usize;
        }
        return len + 64 + (!mask2).trailing_zeros() as usize;
    }

    while len + 64 <= max_len {
        let v1 = _mm512_loadu_si512(a.add(len) as *const _);
        let v2 = _mm512_loadu_si512(b.add(len) as *const _);
        let mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if mask != u64::MAX {
            return len + (!mask).trailing_zeros() as usize;
        }
        len += 64;
    }
    len + match_len_avx2(a.add(len), b.add(len), max_len - len)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl,avx512bw")]
unsafe fn match_len_avx10(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    // Fail fast for short matches.
    // Checks the first 32 bytes to avoid loading 96 extra bytes for short matches.
    if len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a as *const _);
        let v2 = _mm256_loadu_si256(b as *const _);
        let mask = _mm256_cmpeq_epi8_mask(v1, v2);
        if mask != 0xFFFFFFFF {
            return (!mask).trailing_zeros() as usize;
        }
        len += 32;
    }

    // Optimize: Unroll loop to process 128 bytes per iteration using 256-bit vectors.
    // This maintains throughput while avoiding potential frequency throttling associated
    // with 512-bit vectors on some architectures, and aligns with AVX10 philosophy.
    while len + 128 <= max_len {
        let v1 = _mm256_loadu_si256(a.add(len) as *const _);
        let v2 = _mm256_loadu_si256(b.add(len) as *const _);
        let mask1 = _mm256_cmpeq_epi8_mask(v1, v2);

        let v3 = _mm256_loadu_si256(a.add(len + 32) as *const _);
        let v4 = _mm256_loadu_si256(b.add(len + 32) as *const _);
        let mask2 = _mm256_cmpeq_epi8_mask(v3, v4);

        let v5 = _mm256_loadu_si256(a.add(len + 64) as *const _);
        let v6 = _mm256_loadu_si256(b.add(len + 64) as *const _);
        let mask3 = _mm256_cmpeq_epi8_mask(v5, v6);

        let v7 = _mm256_loadu_si256(a.add(len + 96) as *const _);
        let v8 = _mm256_loadu_si256(b.add(len + 96) as *const _);
        let mask4 = _mm256_cmpeq_epi8_mask(v7, v8);

        if (mask1 & mask2 & mask3 & mask4) == 0xFFFFFFFF {
            len += 128;
            continue;
        }

        if mask1 != 0xFFFFFFFF {
            return len + (!mask1).trailing_zeros() as usize;
        }
        if mask2 != 0xFFFFFFFF {
            return len + 32 + (!mask2).trailing_zeros() as usize;
        }
        if mask3 != 0xFFFFFFFF {
            return len + 64 + (!mask3).trailing_zeros() as usize;
        }
        return len + 96 + (!mask4).trailing_zeros() as usize;
    }

    while len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a.add(len) as *const _);
        let v2 = _mm256_loadu_si256(b.add(len) as *const _);
        let mask = _mm256_cmpeq_epi8_mask(v1, v2);
        if mask != 0xFFFFFFFF {
            return len + (!mask).trailing_zeros() as usize;
        }
        len += 32;
    }

    let rem = max_len - len;
    if rem > 0 {
        let k = (1u32 << rem).wrapping_sub(1);
        let v1 = _mm256_maskz_loadu_epi8(k, a.add(len) as *const _);
        let v2 = _mm256_maskz_loadu_epi8(k, b.add(len) as *const _);
        let mask = _mm256_cmpeq_epi8_mask(v1, v2);
        len += min(rem, (!mask).trailing_zeros() as usize);
    }
    len
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn match_len_neon(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;
    while len + 16 <= max_len {
        let v1 = vld1q_u8(a.add(len));
        let v2 = vld1q_u8(b.add(len));
        let xor = veorq_u8(v1, v2);
        if vmaxvq_u8(xor) == 0 {
            len += 16;
        } else {
            let mut bytes = [0u8; 16];
            vst1q_u8(bytes.as_mut_ptr(), xor);
            for i in 0..16 {
                if bytes[i] != 0 {
                    return len + i;
                }
            }
            return len;
        }
    }
    len + match_len_sw(a.add(len), b.add(len), max_len - len)
}

// Selects the best available match length implementation at runtime.
// This is called once during initialization to avoid checking CPU features
// inside the compression loop.
fn get_match_len_func() -> MatchLenFn {
    #[cfg(target_arch = "x86_64")]
    {
        // Prioritize AVX10/256-bit implementation if AVX512VL is available.
        // This avoids frequency throttling on hybrid/consumer CPUs while still using AVX512 features.
        if is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx512bw") {
            return match_len_avx10;
        }
        if is_x86_feature_detected!("avx512bw") {
            return match_len_avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return match_len_avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return match_len_sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return match_len_neon;
        }
    }
    match_len_sw
}

pub struct MatchFinder {
    pub hash_tab: Vec<i32>,
    pub prev_tab: Vec<u16>,
    pub base_offset: usize,
    match_len: MatchLenFn,
}

impl MatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            prev_tab: vec![0; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
            match_len: get_match_len_func(),
        }
    }

    pub fn reset(&mut self) {
        self.hash_tab.fill(-1);
        self.prev_tab.fill(0);
        self.base_offset = 0;
    }

    pub fn prepare(&mut self, len: usize) {
        if self.base_offset + len > i32::MAX as usize {
            self.reset();
        }
    }

    pub fn advance(&mut self, len: usize) {
        self.base_offset += len;
    }

    #[inline(always)]
    unsafe fn find_match_impl<F>(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        mut on_match: F,
    ) -> (usize, usize)
    where
        F: FnMut(usize, usize),
    {
        if pos.checked_add(3).map_or(true, |end| end > data.len()) {
            return (0, 0);
        }

        let src = data.as_ptr().add(pos);
        let src_val;

        if pos + 4 <= data.len() {
            src_val = (src as *const u32).read_unaligned() & 0xFFFFFF;
        } else {
            src_val = ((src.read() as u32) << 0)
                | ((src.add(1).read() as u32) << 8)
                | ((src.add(2).read() as u32) << 16);
        }

        let h = (src_val.wrapping_mul(0x1E35A7BD)) >> (32 - MATCHFINDER_HASH_ORDER);

        let abs_pos = self.base_offset + pos;
        let h_idx = h as usize;
        let cur_pos = *self.hash_tab.get_unchecked(h_idx);
        *self.hash_tab.get_unchecked_mut(h_idx) = abs_pos as i32;

        if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
            *self
                .prev_tab
                .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
            return (0, 0);
        }

        let prev_offset = abs_pos - (cur_pos as usize);
        *self
            .prev_tab
            .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = if prev_offset > 0xFFFF {
            0
        } else {
            prev_offset as u16
        };

        let mut best_len = 0;
        let mut best_offset = 0;
        let mut depth = 0;
        let mut cur_pos_i32 = cur_pos;

        while cur_pos_i32 != -1 && depth < max_depth {
            let p_abs = cur_pos_i32 as usize;
            if p_abs < self.base_offset {
                break;
            }
            let offset = abs_pos - p_abs;
            if offset > DEFLATE_MAX_MATCH_OFFSET {
                break;
            }

            let p_rel = p_abs - self.base_offset;
            let match_ptr = data.as_ptr().add(p_rel);

            let match_val;
            if p_rel + 4 <= data.len() {
                match_val = (match_ptr as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                match_val = ((match_ptr.read() as u32) << 0)
                    | ((match_ptr.add(1).read() as u32) << 8)
                    | ((match_ptr.add(2).read() as u32) << 16);
            }

            if pos + best_len < data.len() && match_val == src_val {
                if best_len < 3 || *match_ptr.add(best_len) == *src.add(best_len) {
                    let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                    let len = (self.match_len)(match_ptr, src, max_len);

                    if len > best_len {
                        best_len = len;
                        best_offset = offset;
                        on_match(len, offset);
                        if len == DEFLATE_MAX_MATCH_LEN {
                            break;
                        }
                    }
                }
            }

            let prev_offset_val = *self
                .prev_tab
                .get_unchecked(p_abs & (MATCHFINDER_WINDOW_SIZE - 1));
            if prev_offset_val == 0 {
                break;
            }
            cur_pos_i32 -= prev_offset_val as i32;
            depth += 1;
        }

        (best_len, best_offset)
    }

    pub fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        matches.clear();
        unsafe {
            self.find_match_impl(data, pos, max_depth, |len, offset| {
                if len >= 3 {
                    matches.push((len as u16, offset as u16));
                }
            })
        }
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        unsafe { self.find_match_impl(data, pos, max_depth, |_, _| {}) }
    }
    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos.checked_add(3).map_or(true, |end| end > data.len()) {
            return;
        }
        unsafe {
            let src = data.as_ptr().add(pos);
            let src_val;
            if pos + 4 <= data.len() {
                src_val = (src as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                src_val = ((src.read() as u32) << 0)
                    | ((src.add(1).read() as u32) << 8)
                    | ((src.add(2).read() as u32) << 16);
            }

            let h = (src_val.wrapping_mul(0x1E35A7BD)) >> (32 - MATCHFINDER_HASH_ORDER);

            let abs_pos = self.base_offset + pos;
            let cur_pos = *self.hash_tab.get_unchecked(h as usize);
            *self.hash_tab.get_unchecked_mut(h as usize) = abs_pos as i32;

            if cur_pos != -1 && (cur_pos as usize) >= self.base_offset {
                let prev_offset = abs_pos - (cur_pos as usize);
                *self
                    .prev_tab
                    .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) =
                    if prev_offset > 0xFFFF {
                        0
                    } else {
                        prev_offset as u16
                    }
            } else {
                *self
                    .prev_tab
                    .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
            }
        }
    }

    pub fn skip_positions(&mut self, data: &[u8], pos: usize, count: usize) {
        if count == 0 {
            return;
        }
        if pos
            .checked_add(count + 3)
            .map_or(true, |end| end > data.len())
        {
            for i in 0..count {
                self.skip_match(data, pos + i);
            }
            return;
        }

        unsafe {
            let mut ptr = data.as_ptr().add(pos);
            let end_ptr = ptr.add(count);
            let mut abs_pos = self.base_offset + pos;

            while ptr < end_ptr {
                let src_val = (ptr as *const u32).read_unaligned() & 0xFFFFFF;
                let h = (src_val.wrapping_mul(0x1E35A7BD)) >> (32 - MATCHFINDER_HASH_ORDER);
                let h_idx = h as usize;

                let cur_pos = *self.hash_tab.get_unchecked(h_idx);
                *self.hash_tab.get_unchecked_mut(h_idx) = abs_pos as i32;

                if cur_pos != -1 && (cur_pos as usize) >= self.base_offset {
                    let prev_offset = abs_pos - (cur_pos as usize);
                    let val = if prev_offset > 0xFFFF {
                        0
                    } else {
                        prev_offset as u16
                    };
                    *self
                        .prev_tab
                        .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = val;
                } else {
                    *self
                        .prev_tab
                        .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
                }

                ptr = ptr.add(1);
                abs_pos += 1;
            }
        }
    }

}

pub struct HtMatchFinder {
    pub hash_tab: Vec<i32>,
    pub base_offset: usize,
    match_len: MatchLenFn,
}

impl HtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            base_offset: 0,
            match_len: get_match_len_func(),
        }
    }

    pub fn reset(&mut self) {
        self.hash_tab.fill(-1);
        self.base_offset = 0;
    }

    pub fn prepare(&mut self, len: usize) {
        if self.base_offset + len > i32::MAX as usize {
            self.reset();
        }
    }

    pub fn advance(&mut self, len: usize) {
        self.base_offset += len;
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize) -> (usize, usize) {
        if pos.checked_add(3).map_or(true, |end| end > data.len()) {
            return (0, 0);
        }

        unsafe {
            let src = data.as_ptr().add(pos);
            let src_val;

            if pos + 4 <= data.len() {
                src_val = (src as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                src_val = ((src.read() as u32) << 0)
                    | ((src.add(1).read() as u32) << 8)
                    | ((src.add(2).read() as u32) << 16);
            }

            let h = (src_val.wrapping_mul(0x1E35A7BD)) >> (32 - MATCHFINDER_HASH_ORDER);

            let abs_pos = self.base_offset + pos;
            let h_idx = h as usize;
            let cur_pos = *self.hash_tab.get_unchecked(h_idx);
            *self.hash_tab.get_unchecked_mut(h_idx) = abs_pos as i32;

            if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
                return (0, 0);
            }

            let p_abs = cur_pos as usize;
            let offset = abs_pos - p_abs;
            if offset > DEFLATE_MAX_MATCH_OFFSET {
                return (0, 0);
            }

            let p_rel = p_abs - self.base_offset;
            let match_ptr = data.as_ptr().add(p_rel);

            let match_val;
            if p_rel + 4 <= data.len() {
                match_val = (match_ptr as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                match_val = ((match_ptr.read() as u32) << 0)
                    | ((match_ptr.add(1).read() as u32) << 8)
                    | ((match_ptr.add(2).read() as u32) << 16);
            }

            if src_val == match_val {
                let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                let len = (self.match_len)(match_ptr, src, max_len);
                if len >= 3 {
                    return (len, offset);
                }
            }
        }
        (0, 0)
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos.checked_add(3).map_or(true, |end| end > data.len()) {
            return;
        }
        unsafe {
            let src = data.as_ptr().add(pos);
            let src_val;
            if pos + 4 <= data.len() {
                src_val = (src as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                src_val = ((src.read() as u32) << 0)
                    | ((src.add(1).read() as u32) << 8)
                    | ((src.add(2).read() as u32) << 16);
            }
            let h = (src_val.wrapping_mul(0x1E35A7BD)) >> (32 - MATCHFINDER_HASH_ORDER);

            let abs_pos = self.base_offset + pos;
            *self.hash_tab.get_unchecked_mut(h as usize) = abs_pos as i32;
        }
    }

    pub fn skip_positions(&mut self, _data: &[u8], _pos: usize, _count: usize) {
        // For HtMatchFinder (Level 1), skipping hash updates inside matches provides
        // a massive speed boost with minimal compression loss.
    }
}

pub struct BtMatchFinder {
    pub hash3_tab: Vec<[i32; 2]>,
    pub hash4_tab: Vec<i32>,
    pub child_tab: Vec<[i32; 2]>,
    pub base_offset: usize,
    match_len: MatchLenFn,
}

impl BtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash3_tab: vec![[-1; 2]; 1 << 16],
            hash4_tab: vec![-1; 1 << 16],
            child_tab: vec![[0; 2]; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
            match_len: get_match_len_func(),
        }
    }

    pub fn reset(&mut self) {
        self.hash3_tab.fill([-1; 2]);
        self.hash4_tab.fill(-1);
        self.child_tab.fill([0; 2]);
        self.base_offset = 0;
    }

    pub fn prepare(&mut self, len: usize) {
        if self.base_offset + len > i32::MAX as usize {
            self.reset();
        }
    }

    pub fn advance(&mut self, len: usize) {
        self.base_offset += len;
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        if pos.checked_add(4).map_or(true, |end| end > data.len()) {
            return (0, 0);
        }

        unsafe {
            let src = data.as_ptr().add(pos);
            let h3 = (((src.read() as u32) << 16)
                | ((src.add(1).read() as u32) << 8)
                | (src.add(2).read() as u32))
                .wrapping_mul(0x1E35A7BD);
            let h3 = (h3 >> 16) as usize;

            let h4 = src.cast::<u32>().read_unaligned().wrapping_mul(0x1E35A7BD);
            let h4 = (h4 >> 16) as usize;

            let abs_pos = self.base_offset + pos;

            let cur_node_3 = (*self.hash3_tab.get_unchecked(h3))[0];
            (*self.hash3_tab.get_unchecked_mut(h3))[0] = abs_pos as i32;
            let cur_node_3_2 = (*self.hash3_tab.get_unchecked(h3))[1];
            (*self.hash3_tab.get_unchecked_mut(h3))[1] = cur_node_3;

            let cutoff = (abs_pos as i32).wrapping_sub(MATCHFINDER_WINDOW_SIZE as i32);

            let mut best_len = 0;
            let mut best_offset = 0;

            if cur_node_3 != -1 && cur_node_3 > cutoff && (cur_node_3 as usize) >= self.base_offset
            {
                let p_abs = cur_node_3 as usize;
                let p_rel = p_abs - self.base_offset;
                let match_ptr = data.as_ptr().add(p_rel);
                if *match_ptr == *src
                    && *match_ptr.add(1) == *src.add(1)
                    && *match_ptr.add(2) == *src.add(2)
                {
                    best_len = 3;
                    best_offset = abs_pos - p_abs;
                } else if cur_node_3_2 != -1
                    && cur_node_3_2 > cutoff
                    && (cur_node_3_2 as usize) >= self.base_offset
                {
                    let p2_abs = cur_node_3_2 as usize;
                    let p2_rel = p2_abs - self.base_offset;
                    let match_ptr_2 = data.as_ptr().add(p2_rel);
                    if *match_ptr_2 == *src
                        && *match_ptr_2.add(1) == *src.add(1)
                        && *match_ptr_2.add(2) == *src.add(2)
                    {
                        best_len = 3;
                        best_offset = abs_pos - p2_abs;
                    }
                }
            }

            let mut cur_node = *self.hash4_tab.get_unchecked(h4);
            *self.hash4_tab.get_unchecked_mut(h4) = abs_pos as i32;

            let child_idx = abs_pos & (MATCHFINDER_WINDOW_SIZE - 1);

            if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                *self.child_tab.get_unchecked_mut(child_idx) = [-1, -1];
                return (best_len, best_offset);
            }

            let mut depth_remaining = max_depth;
            let max_len_clamped = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);

            let mut pending_lt_node = child_idx;
            let mut pending_lt_child = 0;
            let mut pending_gt_node = child_idx;
            let mut pending_gt_child = 1;

            loop {
                let p_abs = cur_node as usize;
                let p_child_idx = p_abs & (MATCHFINDER_WINDOW_SIZE - 1);
                let p_rel = p_abs - self.base_offset;
                let match_ptr = data.as_ptr().add(p_rel);

                let len = (self.match_len)(match_ptr, src, max_len_clamped);

                if len > best_len {
                    best_len = len;
                    best_offset = abs_pos - p_abs;
                    if len == max_len_clamped {
                        let children = *self.child_tab.get_unchecked(p_child_idx);
                        (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                            children[0];
                        (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                            children[1];
                        return (best_len, best_offset);
                    }
                }

                if len < max_len_clamped && *match_ptr.add(len) < *src.add(len) {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                        cur_node;
                    pending_lt_node = p_child_idx;
                    pending_lt_child = 1;
                    cur_node = (*self.child_tab.get_unchecked(p_child_idx))[1];
                } else {
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                        cur_node;
                    pending_gt_node = p_child_idx;
                    pending_gt_child = 0;
                    cur_node = (*self.child_tab.get_unchecked(p_child_idx))[0];
                }

                if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                    return (best_len, best_offset);
                }

                depth_remaining -= 1;
                if depth_remaining == 0 {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                    return (best_len, best_offset);
                }
            }
        }
    }

    pub fn advance_one_byte(
        &mut self,
        data: &[u8],
        pos: usize,
        max_len: usize,
        nice_len: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
        record_matches: bool,
    ) {
        if pos.checked_add(4).map_or(true, |end| end > data.len()) {
            return;
        }

        unsafe {
            let src = data.as_ptr().add(pos);
            let h3 = (((src.read() as u32) << 16)
                | ((src.add(1).read() as u32) << 8)
                | (src.add(2).read() as u32))
                .wrapping_mul(0x1E35A7BD);
            let h3 = (h3 >> 16) as usize;

            let h4 = src.cast::<u32>().read_unaligned().wrapping_mul(0x1E35A7BD);
            let h4 = (h4 >> 16) as usize;

            let abs_pos = self.base_offset + pos;

            let cur_node_3 = (*self.hash3_tab.get_unchecked(h3))[0];
            (*self.hash3_tab.get_unchecked_mut(h3))[0] = abs_pos as i32;
            let cur_node_3_2 = (*self.hash3_tab.get_unchecked(h3))[1];
            (*self.hash3_tab.get_unchecked_mut(h3))[1] = cur_node_3;

            let cutoff = (abs_pos as i32).wrapping_sub(MATCHFINDER_WINDOW_SIZE as i32);

            if record_matches {
                if cur_node_3 != -1
                    && cur_node_3 > cutoff
                    && (cur_node_3 as usize) >= self.base_offset
                {
                    let p_abs = cur_node_3 as usize;
                    let p_rel = p_abs - self.base_offset;
                    let match_ptr = data.as_ptr().add(p_rel);
                    if *match_ptr == *src
                        && *match_ptr.add(1) == *src.add(1)
                        && *match_ptr.add(2) == *src.add(2)
                    {
                        matches.push((3, (abs_pos - p_abs) as u16));
                    } else if cur_node_3_2 != -1
                        && cur_node_3_2 > cutoff
                        && (cur_node_3_2 as usize) >= self.base_offset
                    {
                        let p2_abs = cur_node_3_2 as usize;
                        let p2_rel = p2_abs - self.base_offset;
                        let match_ptr_2 = data.as_ptr().add(p2_rel);
                        if *match_ptr_2 == *src
                            && *match_ptr_2.add(1) == *src.add(1)
                            && *match_ptr_2.add(2) == *src.add(2)
                        {
                            matches.push((3, (abs_pos - p2_abs) as u16));
                        }
                    }
                }
            }

            let mut cur_node = *self.hash4_tab.get_unchecked(h4);
            *self.hash4_tab.get_unchecked_mut(h4) = abs_pos as i32;

            let child_idx = abs_pos & (MATCHFINDER_WINDOW_SIZE - 1);

            if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                *self.child_tab.get_unchecked_mut(child_idx) = [-1, -1];
                return;
            }

            let mut depth_remaining = max_depth;
            let mut best_len = 3;

            let mut pending_lt_node = child_idx;
            let mut pending_lt_child = 0;

            let mut pending_gt_node = child_idx;
            let mut pending_gt_child = 1;

            let max_len_clamped = min(max_len, data.len() - pos);

            loop {
                let p_abs = cur_node as usize;
                let p_child_idx = p_abs & (MATCHFINDER_WINDOW_SIZE - 1);
                let p_rel = p_abs - self.base_offset;
                let match_ptr = data.as_ptr().add(p_rel);

                let len = (self.match_len)(match_ptr, src, max_len_clamped);

                if record_matches && len > best_len {
                    best_len = len;
                    matches.push((len as u16, (abs_pos - p_abs) as u16));
                    if len >= nice_len {
                        let children = *self.child_tab.get_unchecked(p_child_idx);
                        (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                            children[0];
                        (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                            children[1];
                        return;
                    }
                }

                if len < max_len_clamped && *match_ptr.add(len) < *src.add(len) {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                        cur_node;
                    pending_lt_node = p_child_idx;
                    pending_lt_child = 1;
                    cur_node = (*self.child_tab.get_unchecked(p_child_idx))[1];
                } else {
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                        cur_node;
                    pending_gt_node = p_child_idx;
                    pending_gt_child = 0;
                    cur_node = (*self.child_tab.get_unchecked(p_child_idx))[0];
                }

                if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                    return;
                }

                depth_remaining -= 1;
                if depth_remaining == 0 {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                    return;
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_finder_consistency() {
        let mut mf1 = MatchFinder::new();
        let mut mf2 = MatchFinder::new();
        let data = b"abcdeabcdeabcde";
        mf1.prepare(data.len());
        mf2.prepare(data.len());

        let max_depth = 10;
        let mut matches = Vec::new();

        for i in 0..5 {
            mf1.find_match(data, i, max_depth);

            mf2.find_match(data, i, max_depth);
        }

        let (len1, offset1) = mf1.find_match(data, 5, max_depth);

        let (len2, offset2) = mf2.find_matches(data, 5, max_depth, &mut matches);

        assert_eq!(len1, len2, "Lengths should match");
        assert_eq!(offset1, offset2, "Offsets should match");
        assert_eq!(len1, 10, "Should find match of length 10");
        assert_eq!(offset1, 5, "Should find match at offset 5");

        assert!(!matches.is_empty(), "Matches vector should not be empty");
        assert_eq!(
            matches.last(),
            Some(&(10, 5)),
            "Last match should be best match"
        );
    }

    #[test]
    fn test_skip_match_overflow() {
        let mut mf = MatchFinder::new();
        let data = b"some data";
        mf.skip_match(data, usize::MAX);
    }

    #[test]
    fn test_ht_match_overflow() {
        let mut mf = HtMatchFinder::new();
        let data = b"some data";
        mf.skip_match(data, usize::MAX);
    }

    #[test]
    fn test_bt_match_overflow() {
        let mut mf = BtMatchFinder::new();
        let data = b"some data";
        mf.skip_match(data, usize::MAX, 10);
    }

    #[test]
    fn test_match_len_selection() {
        let mf = MatchFinder::new();
        let a = b"abcdef";
        let b = b"abcxyz";
        unsafe {
            let len = (mf.match_len)(a.as_ptr(), b.as_ptr(), 6);
            assert_eq!(len, 3);
        }
    }

    #[test]
    fn test_match_len_avx2_explicit() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            // Create a buffer large enough to test 128-byte unrolling
            let mut a = vec![0u8; 300];
            let mut b = vec![0u8; 300];
            for i in 0..300 {
                a[i] = (i % 256) as u8;
                b[i] = (i % 256) as u8;
            }

            unsafe {
                // Test > 128 bytes match
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 200);

                // Test exact 128 bytes match
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 128);
                assert_eq!(len, 128);

                // Test < 128 bytes match
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 100);
                assert_eq!(len, 100);

                // Test mismatch in first 32 bytes
                b[10] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 10);
                b[10] = a[10]; // Reset

                // Test mismatch in second 32 bytes (offset 40)
                b[40] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 40);
                b[40] = a[40]; // Reset

                // Test mismatch in third 32 bytes (offset 70)
                b[70] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 70);
                b[70] = a[70]; // Reset

                // Test mismatch in fourth 32 bytes (offset 100)
                b[100] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 100);
                b[100] = a[100]; // Reset

                // Test mismatch after 128 bytes (offset 130)
                b[130] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 130);
                b[130] = a[130]; // Reset
            }
        }
    }

    #[test]
    fn test_match_len_avx10_explicit() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx512bw") {
            let a = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let b = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

            unsafe {
                // Test full match
                let len = match_len_avx10(a.as_ptr(), b.as_ptr(), a.len());
                assert_eq!(len, a.len());

                // Test partial match
                let len = match_len_avx10(a.as_ptr(), b.as_ptr(), 10);
                assert_eq!(len, 10);

                // Test mismatch
                let c = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXY!";
                let len = match_len_avx10(a.as_ptr(), c.as_ptr(), a.len());
                assert_eq!(len, a.len() - 1);

                // Test small lengths (tail handling)
                for i in 0..35 {
                    let len = match_len_avx10(a.as_ptr(), b.as_ptr(), i);
                    assert_eq!(len, i);
                }

                // Test mismatch in tail
                let d = b"abcdefghijklmnopqrstuvwxyz012345!"; // 33 chars, mismatch at 32
                // a has '6' at 32.
                let len = match_len_avx10(a.as_ptr(), d.as_ptr(), 33);
                assert_eq!(len, 32);
            }
        }
    }
}
