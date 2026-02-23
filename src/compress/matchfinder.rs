use crate::common::*;
use std::cmp::min;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const MATCHFINDER_HASH_ORDER: usize = 15;
pub const MATCHFINDER_HASH_SIZE: usize = 1 << MATCHFINDER_HASH_ORDER;
pub const MATCHFINDER_WINDOW_SIZE: usize = 32768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchLenStrategy {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx10,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

trait MatchLen {
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize;
}

struct ScalarStrategy;
impl MatchLen for ScalarStrategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_sw(a, b, max_len)
    }
}

#[cfg(target_arch = "x86_64")]
struct Sse2Strategy;
#[cfg(target_arch = "x86_64")]
impl MatchLen for Sse2Strategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_sse2(a, b, max_len)
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2Strategy;
#[cfg(target_arch = "x86_64")]
impl MatchLen for Avx2Strategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_avx2(a, b, max_len)
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx512Strategy;
#[cfg(target_arch = "x86_64")]
impl MatchLen for Avx512Strategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_avx512(a, b, max_len)
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx10Strategy;
#[cfg(target_arch = "x86_64")]
impl MatchLen for Avx10Strategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_avx10(a, b, max_len)
    }
}

#[cfg(target_arch = "aarch64")]
struct NeonStrategy;
#[cfg(target_arch = "aarch64")]
impl MatchLen for NeonStrategy {
    #[inline(always)]
    unsafe fn calc(a: *const u8, b: *const u8, max_len: usize) -> usize {
        match_len_neon(a, b, max_len)
    }
}

pub trait MatchFinderTrait {
    fn reset(&mut self);
    fn prepare(&mut self, len: usize);
    fn advance(&mut self, len: usize);
    fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
    ) -> (usize, usize);
    fn skip_match(&mut self, data: &[u8], pos: usize, max_depth: usize, nice_len: usize);
    fn skip_positions(
        &mut self,
        data: &[u8],
        pos: usize,
        count: usize,
        max_depth: usize,
        nice_len: usize,
    ) {
        for i in 0..count {
            self.skip_match(data, pos + i, max_depth, nice_len);
        }
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
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
    fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
    ) -> (usize, usize) {
        self.find_match(data, pos, max_depth, nice_len)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, _max_depth: usize, _nice_len: usize) {
        self.skip_match(data, pos);
    }
    fn skip_positions(
        &mut self,
        data: &[u8],
        pos: usize,
        count: usize,
        _max_depth: usize,
        _nice_len: usize,
    ) {
        self.skip_positions(data, pos, count);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        self.find_matches(data, pos, max_depth, nice_len, matches)
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
    fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        _max_depth: usize,
        _nice_len: usize,
    ) -> (usize, usize) {
        self.find_match(data, pos)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, _max_depth: usize, _nice_len: usize) {
        self.skip_match(data, pos);
    }
    fn skip_positions(
        &mut self,
        data: &[u8],
        pos: usize,
        count: usize,
        _max_depth: usize,
        _nice_len: usize,
    ) {
        self.skip_positions(data, pos, count);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        _max_depth: usize,
        _nice_len: usize,
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
    fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
    ) -> (usize, usize) {
        self.find_match(data, pos, max_depth, nice_len)
    }
    fn skip_match(&mut self, data: &[u8], pos: usize, max_depth: usize, nice_len: usize) {
        self.skip_match(data, pos, max_depth, nice_len);
    }
    fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        self.find_matches(data, pos, max_depth, nice_len, matches)
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

    if len + 16 <= max_len {
        let v1 = _mm_loadu_si128(a.add(len) as *const __m128i);
        let v2 = _mm_loadu_si128(b.add(len) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(v1, v2);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF {
            return len + (!mask).trailing_zeros() as usize;
        }
        len += 16;
    }

    while len + 64 <= max_len {
        let v1 = _mm_loadu_si128(a.add(len) as *const __m128i);
        let v2 = _mm_loadu_si128(b.add(len) as *const __m128i);
        let cmp1 = _mm_cmpeq_epi8(v1, v2);
        let mask1 = _mm_movemask_epi8(cmp1) as u32;
        if mask1 != 0xFFFF {
            return len + (!mask1).trailing_zeros() as usize;
        }

        let v3 = _mm_loadu_si128(a.add(len + 16) as *const __m128i);
        let v4 = _mm_loadu_si128(b.add(len + 16) as *const __m128i);
        let cmp2 = _mm_cmpeq_epi8(v3, v4);
        let mask2 = _mm_movemask_epi8(cmp2) as u32;
        if mask2 != 0xFFFF {
            return len + 16 + (!mask2).trailing_zeros() as usize;
        }

        let v5 = _mm_loadu_si128(a.add(len + 32) as *const __m128i);
        let v6 = _mm_loadu_si128(b.add(len + 32) as *const __m128i);
        let cmp3 = _mm_cmpeq_epi8(v5, v6);
        let mask3 = _mm_movemask_epi8(cmp3) as u32;
        if mask3 != 0xFFFF {
            return len + 32 + (!mask3).trailing_zeros() as usize;
        }

        let v7 = _mm_loadu_si128(a.add(len + 48) as *const __m128i);
        let v8 = _mm_loadu_si128(b.add(len + 48) as *const __m128i);
        let cmp4 = _mm_cmpeq_epi8(v7, v8);
        let mask4 = _mm_movemask_epi8(cmp4) as u32;
        if mask4 != 0xFFFF {
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

    if len < max_len {
        if max_len >= 16 {
            let v1 = _mm_loadu_si128(a.add(max_len - 16) as *const __m128i);
            let v2 = _mm_loadu_si128(b.add(max_len - 16) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(v1, v2);
            let mask = _mm_movemask_epi8(cmp) as u32;
            if mask != 0xFFFF {
                return max_len - 16 + (!mask).trailing_zeros() as usize;
            }
            return max_len;
        }
        return len + match_len_sw(a.add(len), b.add(len), max_len - len);
    }
    len
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn match_len_avx2(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    if len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a as *const __m256i);
        let v2 = _mm256_loadu_si256(b as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(v1, v2);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFFFFFF {
            return (!mask).trailing_zeros() as usize;
        }
        len += 32;
    } else {
        return match_len_sse2(a, b, max_len);
    }

    let v_zero = _mm256_setzero_si256();

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

        if _mm256_testz_si256(or1, or1) == 0 {
            if _mm256_testz_si256(xor1, xor1) == 0 {
                let cmp = _mm256_cmpeq_epi8(xor1, v_zero);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                return len + (!mask).trailing_zeros() as usize;
            } else {
                let cmp = _mm256_cmpeq_epi8(xor2, v_zero);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                return len + 32 + (!mask).trailing_zeros() as usize;
            }
        } else {
            if _mm256_testz_si256(xor3, xor3) == 0 {
                let cmp = _mm256_cmpeq_epi8(xor3, v_zero);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                return len + 64 + (!mask).trailing_zeros() as usize;
            } else {
                let cmp = _mm256_cmpeq_epi8(xor4, v_zero);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                return len + 96 + (!mask).trailing_zeros() as usize;
            }
        }
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

    if len < max_len {
        let v1 = _mm256_loadu_si256(a.add(max_len - 32) as *const __m256i);
        let v2 = _mm256_loadu_si256(b.add(max_len - 32) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(v1, v2);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFFFFFF {
            return max_len - 32 + (!mask).trailing_zeros() as usize;
        }
        return max_len;
    }
    len
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn match_len_avx512(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    if len + 64 <= max_len {
        let v1 = _mm512_loadu_si512(a as *const _);
        let v2 = _mm512_loadu_si512(b as *const _);
        let mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if mask != u64::MAX {
            return (!mask).trailing_zeros() as usize;
        }
        len += 64;
    }

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

    if len + 32 <= max_len {
        let v1 = _mm256_loadu_si256(a as *const _);
        let v2 = _mm256_loadu_si256(b as *const _);
        let mask = _mm256_cmpeq_epi8_mask(v1, v2);
        if mask != 0xFFFFFFFF {
            return (!mask).trailing_zeros() as usize;
        }
        len += 32;
    }

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

        if (mask1 & mask2) != 0xFFFFFFFF {
            if mask1 != 0xFFFFFFFF {
                return len + (!mask1).trailing_zeros() as usize;
            }
            return len + 32 + (!mask2).trailing_zeros() as usize;
        } else {
            if mask3 != 0xFFFFFFFF {
                return len + 64 + (!mask3).trailing_zeros() as usize;
            }
            return len + 96 + (!mask4).trailing_zeros() as usize;
        }
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

fn get_match_len_strategy() -> MatchLenStrategy {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx512bw") {
            return MatchLenStrategy::Avx10;
        }
        if is_x86_feature_detected!("avx512bw") {
            return MatchLenStrategy::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return MatchLenStrategy::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return MatchLenStrategy::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return MatchLenStrategy::Neon;
        }
    }
    MatchLenStrategy::Scalar
}

pub struct MatchFinder {
    pub hash_tab: Vec<i32>,
    pub prev_tab: Vec<u16>,
    pub base_offset: usize,
    match_len: MatchLenStrategy,
}

impl MatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            prev_tab: vec![0; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
            match_len: get_match_len_strategy(),
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

    #[inline(always)]
    unsafe fn find_match_impl<F, M: MatchLen>(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
        mut on_match: F,
    ) -> (usize, usize)
    where
        F: FnMut(usize, usize),
    {
        if pos.checked_add(3).map_or(true, |end| end > data.len()) {
            return (0, 0);
        }

        let safe_to_read_u32 = pos + 4 <= data.len();

        let src = data.as_ptr().add(pos);
        let src_val;
        let mut src_val_4 = 0;

        if safe_to_read_u32 {
            src_val_4 = (src as *const u32).read_unaligned();
            src_val = src_val_4 & 0xFFFFFF;
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

            if pos + best_len >= data.len() {
                break;
            }

            let mut match_ok = true;
            if best_len >= 3 {
                if *match_ptr.add(best_len) != *src.add(best_len) {
                    match_ok = false;
                }
            }

            if match_ok {
                if safe_to_read_u32 {
                    let match_val_4 = (match_ptr as *const u32).read_unaligned();
                    if match_val_4 == src_val_4 {
                        let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                        let len = M::calc(match_ptr, src, max_len);

                        if len > best_len {
                            best_len = len;
                            best_offset = offset;
                            on_match(len, offset);
                            if len >= nice_len || len == DEFLATE_MAX_MATCH_LEN {
                                break;
                            }
                        }
                    } else if best_len < 3 && (match_val_4 & 0xFFFFFF) == src_val {
                        best_len = 3;
                        best_offset = offset;
                        on_match(3, offset);
                    }
                } else {
                    let match_val;
                    if p_rel + 4 <= data.len() {
                        match_val = (match_ptr as *const u32).read_unaligned() & 0xFFFFFF;
                    } else {
                        match_val = ((match_ptr.read() as u32) << 0)
                            | ((match_ptr.add(1).read() as u32) << 8)
                            | ((match_ptr.add(2).read() as u32) << 16);
                    }

                    if match_val == src_val {
                        let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                        let len = M::calc(match_ptr, src, max_len);

                        if len > best_len {
                            best_len = len;
                            best_offset = offset;
                            on_match(len, offset);
                            if len >= nice_len || len == DEFLATE_MAX_MATCH_LEN {
                                break;
                            }
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
        nice_len: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        matches.clear();
        let mut on_match = |len: usize, offset: usize| {
            if len >= 3 {
                matches.push((len as u16, offset as u16));
            }
        };
        unsafe {
            match self.match_len {
                MatchLenStrategy::Scalar => self.find_match_impl::<_, ScalarStrategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => self.find_match_impl::<_, Sse2Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => self.find_match_impl::<_, Avx2Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => self.find_match_impl::<_, Avx512Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => self.find_match_impl::<_, Avx10Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => self.find_match_impl::<_, NeonStrategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
            }
        }
    }

    pub fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
    ) -> (usize, usize) {
        let mut on_match = |_: usize, _: usize| {};
        unsafe {
            match self.match_len {
                MatchLenStrategy::Scalar => self.find_match_impl::<_, ScalarStrategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => self.find_match_impl::<_, Sse2Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => self.find_match_impl::<_, Avx2Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => self.find_match_impl::<_, Avx512Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => self.find_match_impl::<_, Avx10Strategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => self.find_match_impl::<_, NeonStrategy>(
                    data,
                    pos,
                    max_depth,
                    nice_len,
                    &mut on_match,
                ),
            }
        }
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
    match_len: MatchLenStrategy,
}

impl HtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            base_offset: 0,
            match_len: get_match_len_strategy(),
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

        let safe_to_read_u32 = pos + 4 <= data.len();

        unsafe {
            let src = data.as_ptr().add(pos);
            let src_val;

            if safe_to_read_u32 {
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
            if safe_to_read_u32 {
                match_val = (match_ptr as *const u32).read_unaligned() & 0xFFFFFF;
            } else if p_rel + 4 <= data.len() {
                match_val = (match_ptr as *const u32).read_unaligned() & 0xFFFFFF;
            } else {
                match_val = ((match_ptr.read() as u32) << 0)
                    | ((match_ptr.add(1).read() as u32) << 8)
                    | ((match_ptr.add(2).read() as u32) << 16);
            }

            if src_val == match_val {
                let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                let len = match self.match_len {
                    MatchLenStrategy::Scalar => match_len_sw(match_ptr, src, max_len),
                    #[cfg(target_arch = "x86_64")]
                    MatchLenStrategy::Sse2 => match_len_sse2(match_ptr, src, max_len),
                    #[cfg(target_arch = "x86_64")]
                    MatchLenStrategy::Avx2 => match_len_avx2(match_ptr, src, max_len),
                    #[cfg(target_arch = "x86_64")]
                    MatchLenStrategy::Avx512 => match_len_avx512(match_ptr, src, max_len),
                    #[cfg(target_arch = "x86_64")]
                    MatchLenStrategy::Avx10 => match_len_avx10(match_ptr, src, max_len),
                    #[cfg(target_arch = "aarch64")]
                    MatchLenStrategy::Neon => match_len_neon(match_ptr, src, max_len),
                };
                return (len, offset);
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
    }
}

trait MatchVisitor {
    fn on_hash3_match(&mut self, len: usize, offset: usize);
    fn on_match(&mut self, len: usize, offset: usize);
}

struct BestMatchVisitor {
    best_len: usize,
    best_offset: usize,
}

impl BestMatchVisitor {
    fn new() -> Self {
        Self {
            best_len: 0,
            best_offset: 0,
        }
    }
}

impl MatchVisitor for BestMatchVisitor {
    #[inline(always)]
    fn on_hash3_match(&mut self, len: usize, offset: usize) {
        self.best_len = len;
        self.best_offset = offset;
    }

    #[inline(always)]
    fn on_match(&mut self, len: usize, offset: usize) {
        if len > self.best_len {
            self.best_len = len;
            self.best_offset = offset;
        }
    }
}

struct AllMatchesVisitor<'a> {
    matches: &'a mut Vec<(u16, u16)>,
    best_len: usize,
}

impl<'a> AllMatchesVisitor<'a> {
    fn new(matches: &'a mut Vec<(u16, u16)>) -> Self {
        Self {
            matches,
            best_len: 3,
        }
    }
}

impl<'a> MatchVisitor for AllMatchesVisitor<'a> {
    #[inline(always)]
    fn on_hash3_match(&mut self, len: usize, offset: usize) {
        self.matches.push((len as u16, offset as u16));
    }

    #[inline(always)]
    fn on_match(&mut self, len: usize, offset: usize) {
        if len > self.best_len {
            self.best_len = len;
            self.matches.push((len as u16, offset as u16));
        }
    }
}

struct NoOpVisitor;

impl MatchVisitor for NoOpVisitor {
    #[inline(always)]
    fn on_hash3_match(&mut self, _len: usize, _offset: usize) {}

    #[inline(always)]
    fn on_match(&mut self, _len: usize, _offset: usize) {}
}

pub struct BtMatchFinder {
    pub hash3_tab: Vec<[i32; 2]>,
    pub hash4_tab: Vec<i32>,
    pub child_tab: Vec<[i32; 2]>,
    pub base_offset: usize,
    match_len: MatchLenStrategy,
}

impl BtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash3_tab: vec![[-1; 2]; 1 << 16],
            hash4_tab: vec![-1; 1 << 16],
            child_tab: vec![[0; 2]; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
            match_len: get_match_len_strategy(),
        }
    }

    pub fn reset(&mut self) {
        self.hash3_tab.fill([-1; 2]);
        self.hash4_tab.fill(-1);
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
    unsafe fn advance_one_byte_generic<M: MatchLen, V: MatchVisitor>(
        &mut self,
        data: &[u8],
        pos: usize,
        max_len: usize,
        nice_len: usize,
        max_depth: usize,
        mut visitor: V,
    ) -> V {
        if pos.checked_add(4).map_or(true, |end| end > data.len()) {
            return visitor;
        }

        let src = data.as_ptr().add(pos);
        let val = src.cast::<u32>().read_unaligned();
        let h3 = (val.to_le() & 0xFFFFFF).wrapping_mul(0x1E35A7BD);
        let h3 = (h3 >> 16) as usize;

        let h4 = val.wrapping_mul(0x1E35A7BD);
        let h4 = (h4 >> 16) as usize;

        let abs_pos = self.base_offset + pos;

        let cur_node_3 = (*self.hash3_tab.get_unchecked(h3))[0];
        (*self.hash3_tab.get_unchecked_mut(h3))[0] = abs_pos as i32;
        let cur_node_3_2 = (*self.hash3_tab.get_unchecked(h3))[1];
        (*self.hash3_tab.get_unchecked_mut(h3))[1] = cur_node_3;

        let cutoff = (abs_pos as i32).wrapping_sub(MATCHFINDER_WINDOW_SIZE as i32);

        if cur_node_3 != -1 && cur_node_3 > cutoff && (cur_node_3 as usize) >= self.base_offset {
            let p_abs = cur_node_3 as usize;
            let p_rel = p_abs - self.base_offset;
            let match_ptr = data.as_ptr().add(p_rel);
            if *match_ptr == *src
                && *match_ptr.add(1) == *src.add(1)
                && *match_ptr.add(2) == *src.add(2)
            {
                visitor.on_hash3_match(3, abs_pos - p_abs);
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
                    visitor.on_hash3_match(3, abs_pos - p2_abs);
                }
            }
        }

        let mut cur_node = *self.hash4_tab.get_unchecked(h4);
        *self.hash4_tab.get_unchecked_mut(h4) = abs_pos as i32;

        let child_idx = abs_pos & (MATCHFINDER_WINDOW_SIZE - 1);

        if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
            *self.child_tab.get_unchecked_mut(child_idx) = [-1, -1];
            return visitor;
        }

        let mut depth_remaining = max_depth;

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

            let len = M::calc(match_ptr, src, max_len_clamped);

            visitor.on_match(len, abs_pos - p_abs);

            if len >= nice_len || len == max_len_clamped {
                let children = *self.child_tab.get_unchecked(p_child_idx);
                (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                    children[0];
                (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                    children[1];
                return visitor;
            }

            if len < max_len_clamped && *match_ptr.add(len) < *src.add(len) {
                (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = cur_node;
                pending_lt_node = p_child_idx;
                pending_lt_child = 1;
                cur_node = (*self.child_tab.get_unchecked(p_child_idx))[1];
            } else {
                (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = cur_node;
                pending_gt_node = p_child_idx;
                pending_gt_child = 0;
                cur_node = (*self.child_tab.get_unchecked(p_child_idx))[0];
            }

            if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                return visitor;
            }

            depth_remaining -= 1;
            if depth_remaining == 0 {
                (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                return visitor;
            }
        }
    }

    pub fn find_match(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
    ) -> (usize, usize) {
        unsafe {
            let visitor = BestMatchVisitor::new();
            let visitor = match self.match_len {
                MatchLenStrategy::Scalar => self.advance_one_byte_generic::<ScalarStrategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => self.advance_one_byte_generic::<Sse2Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => self.advance_one_byte_generic::<Avx2Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => self.advance_one_byte_generic::<Avx512Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => self.advance_one_byte_generic::<Avx10Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => self.advance_one_byte_generic::<NeonStrategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
            };
            (visitor.best_len, visitor.best_offset)
        }
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize, max_depth: usize, nice_len: usize) {
        unsafe {
            let visitor = NoOpVisitor;
            match self.match_len {
                MatchLenStrategy::Scalar => {
                    self.advance_one_byte_generic::<ScalarStrategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => {
                    self.advance_one_byte_generic::<Sse2Strategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => {
                    self.advance_one_byte_generic::<Avx2Strategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => {
                    self.advance_one_byte_generic::<Avx512Strategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => {
                    self.advance_one_byte_generic::<Avx10Strategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => {
                    self.advance_one_byte_generic::<NeonStrategy, _>(
                        data,
                        pos,
                        DEFLATE_MAX_MATCH_LEN,
                        nice_len,
                        max_depth,
                        visitor,
                    );
                }
            }
        }
    }

    pub fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        nice_len: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        matches.clear();
        unsafe {
            let visitor = AllMatchesVisitor::new(matches);
            let visitor = match self.match_len {
                MatchLenStrategy::Scalar => self.advance_one_byte_generic::<ScalarStrategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => self.advance_one_byte_generic::<Sse2Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => self.advance_one_byte_generic::<Avx2Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => self.advance_one_byte_generic::<Avx512Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => self.advance_one_byte_generic::<Avx10Strategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => self.advance_one_byte_generic::<NeonStrategy, _>(
                    data,
                    pos,
                    DEFLATE_MAX_MATCH_LEN,
                    nice_len,
                    max_depth,
                    visitor,
                ),
            };
            if let Some(&(len, offset)) = visitor.matches.last() {
                (len as usize, offset as usize)
            } else {
                (0, 0)
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
            mf1.find_match(data, i, max_depth, 258);

            mf2.find_match(data, i, max_depth, 258);
        }

        let (len1, offset1) = mf1.find_match(data, 5, max_depth, 258);

        let (len2, offset2) = mf2.find_matches(data, 5, max_depth, 258, &mut matches);

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
        mf.skip_match(data, usize::MAX, 10, 258);
    }

    #[test]
    fn test_match_len_selection() {
        let mf = MatchFinder::new();
        let a = b"abcdef";
        let b = b"abcxyz";
        unsafe {
            let len = match mf.match_len {
                MatchLenStrategy::Scalar => match_len_sw(a.as_ptr(), b.as_ptr(), 6),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Sse2 => match_len_sse2(a.as_ptr(), b.as_ptr(), 6),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx2 => match_len_avx2(a.as_ptr(), b.as_ptr(), 6),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx512 => match_len_avx512(a.as_ptr(), b.as_ptr(), 6),
                #[cfg(target_arch = "x86_64")]
                MatchLenStrategy::Avx10 => match_len_avx10(a.as_ptr(), b.as_ptr(), 6),
                #[cfg(target_arch = "aarch64")]
                MatchLenStrategy::Neon => match_len_neon(a.as_ptr(), b.as_ptr(), 6),
            };
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

    #[test]
    fn test_match_len_sse2_explicit() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("sse2") {
            let mut a = vec![0u8; 300];
            let mut b = vec![0u8; 300];
            for i in 0..300 {
                a[i] = (i % 256) as u8;
                b[i] = (i % 256) as u8;
            }

            unsafe {
                // Test > 64 bytes match
                let len = match_len_sse2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 200);

                // Test mismatch in first 16 bytes
                b[10] = 0xFF;
                let len = match_len_sse2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 10);
                b[10] = a[10];

                // Test mismatch in second 16 bytes (offset 20)
                b[20] = 0xFF;
                let len = match_len_sse2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 20);
                b[20] = a[20];

                // Test mismatch in unrolled loop (offset 70)
                b[70] = 0xFF;
                let len = match_len_sse2(a.as_ptr(), b.as_ptr(), 200);
                assert_eq!(len, 70);
                b[70] = a[70];

                // Test match length preventing unrolled loop (max_len=70)
                // With optimization: consumes 16 bytes. len=16. 16+64 > 70. Skips unrolled.
                // Tail loop handles rest.
                let len = match_len_sse2(a.as_ptr(), b.as_ptr(), 70);
                assert_eq!(len, 70);
            }
        }
    }

    #[test]
    fn test_match_len_avx2_tail_overlap() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let mut a = vec![0u8; 64];
            let mut b = vec![0u8; 64];
            for i in 0..64 {
                a[i] = i as u8;
                b[i] = i as u8;
            }

            unsafe {
                // Case 1: Max len 35, mismatch at 34
                b[34] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 35);
                assert_eq!(len, 34);
                b[34] = a[34];

                // Case 2: Max len 35, mismatch at 32
                b[32] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 35);
                assert_eq!(len, 32);
                b[32] = a[32];

                // Case 3: Max len 35, full match
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 35);
                assert_eq!(len, 35);

                // Case 4: Max len 32 (boundary)
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 32);
                assert_eq!(len, 32);

                // Case 5: Max len 32, mismatch at 31
                b[31] = 0xFF;
                let len = match_len_avx2(a.as_ptr(), b.as_ptr(), 32);
                assert_eq!(len, 31);
                b[31] = a[31];
            }
        }
    }
}
