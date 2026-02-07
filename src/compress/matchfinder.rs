use crate::common::*;
use std::cmp::min;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const MATCHFINDER_HASH_ORDER: usize = 15;
pub const MATCHFINDER_HASH_SIZE: usize = 1 << MATCHFINDER_HASH_ORDER;
pub const MATCHFINDER_WINDOW_SIZE: usize = 32768;

#[inline(always)]
fn match_len(a: &[u8], b: &[u8], max_len: usize) -> usize {
    let mut len = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if max_len >= 64 && is_x86_feature_detected!("avx512bw") {
            unsafe {
                while len + 64 <= max_len {
                    let v1 = _mm512_loadu_si512(a.as_ptr().add(len) as *const _);
                    let v2 = _mm512_loadu_si512(b.as_ptr().add(len) as *const _);
                    let mask = _mm512_cmpeq_epi8_mask(v1, v2);
                    if mask != u64::MAX {
                        return len + (!mask).trailing_zeros() as usize;
                    }
                    len += 64;
                }
            }
        }

        if max_len >= 32 && is_x86_feature_detected!("avx2") {
            unsafe {
                while len + 32 <= max_len {
                    let v1 = _mm256_loadu_si256(a.as_ptr().add(len) as *const __m256i);
                    let v2 = _mm256_loadu_si256(b.as_ptr().add(len) as *const __m256i);
                    let cmp = _mm256_cmpeq_epi8(v1, v2);
                    let mask = _mm256_movemask_epi8(cmp) as u32;
                    if mask != 0xFFFFFFFF {
                        return len + (!mask).trailing_zeros() as usize;
                    }
                    len += 32;
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        if max_len >= 16 && is_x86_feature_detected!("sse2") {
            unsafe {
                while len + 16 <= max_len {
                    let v1 = _mm_loadu_si128(a.as_ptr().add(len) as *const __m128i);
                    let v2 = _mm_loadu_si128(b.as_ptr().add(len) as *const __m128i);
                    let cmp = _mm_cmpeq_epi8(v1, v2);
                    let mask = _mm_movemask_epi8(cmp) as u32;
                    if mask != 0xFFFF {
                        return len + (!mask).trailing_zeros() as usize;
                    }
                    len += 16;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if max_len >= 16 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                while len + 16 <= max_len {
                    let v1 = vld1q_u8(a.as_ptr().add(len));
                    let v2 = vld1q_u8(b.as_ptr().add(len));
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
            }
        }
    }

    while len + 8 <= max_len {
        unsafe {
            let a_val = (a.as_ptr().add(len) as *const u64).read_unaligned();
            let b_val = (b.as_ptr().add(len) as *const u64).read_unaligned();
            if a_val != b_val {
                #[cfg(target_endian = "little")]
                return len + ((a_val ^ b_val).trailing_zeros() / 8) as usize;
                #[cfg(target_endian = "big")]
                return len + ((a_val ^ b_val).leading_zeros() / 8) as usize;
            }
        }
        len += 8;
    }

    if len + 4 <= max_len {
        unsafe {
            let a_val = (a.as_ptr().add(len) as *const u32).read_unaligned();
            let b_val = (b.as_ptr().add(len) as *const u32).read_unaligned();
            if a_val != b_val {
                #[cfg(target_endian = "little")]
                return len + ((a_val ^ b_val).trailing_zeros() / 8) as usize;
                #[cfg(target_endian = "big")]
                return len + ((a_val ^ b_val).leading_zeros() / 8) as usize;
            }
        }
        len += 4;
    }

    while len < max_len && a[len] == b[len] {
        len += 1;
    }
    len
}

pub struct MatchFinder {
    pub hash_tab: Vec<i32>,
    pub prev_tab: Vec<u16>,
    pub base_offset: usize,
}

impl MatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            prev_tab: vec![0; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
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
    pub fn hash(bytes: &[u8]) -> usize {
        unsafe {
            let h = ((*bytes.get_unchecked(0) as u32) << 16)
                | ((*bytes.get_unchecked(1) as u32) << 8)
                | (*bytes.get_unchecked(2) as u32);
            let h = h.wrapping_mul(0x1E35A7BD);
            (h >> (32 - MATCHFINDER_HASH_ORDER)) as usize
        }
    }

    pub fn find_matches(
        &mut self,
        data: &[u8],
        pos: usize,
        max_depth: usize,
        matches: &mut Vec<(u16, u16)>,
    ) -> (usize, usize) {
        matches.clear();
        if pos + 3 > data.len() {
            return (0, 0);
        }

        let abs_pos = self.base_offset + pos;
        let h = Self::hash(&data[pos..]);

        let cur_pos = unsafe { *self.hash_tab.get_unchecked(h) };
        unsafe { *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32 };

        if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
             unsafe {
                 *self.prev_tab.get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
             }
             return (0, 0);
        }

        let prev_offset = abs_pos - (cur_pos as usize);
        unsafe {
            *self
                .prev_tab
                .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = if prev_offset > 0xFFFF {
                0
            } else {
                prev_offset as u16
            }
        };

        /*if prev_offset > DEFLATE_MAX_MATCH_OFFSET {
        }*/

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

            if pos + best_len < data.len() && data[p_rel + best_len] == data[pos + best_len] {
                let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                let len = match_len(&data[p_rel..], &data[pos..], max_len);

                if len > best_len {
                    best_len = len;
                    best_offset = offset;
                    if len >= 3 {
                        matches.push((len as u16, offset as u16));
                    }
                    if len == DEFLATE_MAX_MATCH_LEN {
                        break;
                    }
                }
            }

            let prev_offset_val = unsafe {
                *self
                    .prev_tab
                    .get_unchecked(p_abs & (MATCHFINDER_WINDOW_SIZE - 1))
            };
            if prev_offset_val == 0 {
                break;
            }
            cur_pos_i32 -= prev_offset_val as i32;
            depth += 1;
        }

        (best_len, best_offset)
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        if pos + 3 > data.len() {
            return (0, 0);
        }

        let abs_pos = self.base_offset + pos;
        let h = Self::hash(&data[pos..]);
        let cur_pos = unsafe { *self.hash_tab.get_unchecked(h) };
        unsafe { *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32 };

        if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
             unsafe {
                 *self.prev_tab.get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
             }
             return (0, 0);
        }

        let prev_offset = abs_pos - (cur_pos as usize);
        unsafe {
            *self
                .prev_tab
                .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = if prev_offset > 0xFFFF {
                0
            } else {
                prev_offset as u16
            }
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

            if pos + best_len < data.len() && data[p_rel + best_len] == data[pos + best_len] {
                let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                let len = match_len(&data[p_rel..], &data[pos..], max_len);

                if len > best_len {
                    best_len = len;
                    best_offset = offset;
                    if len == DEFLATE_MAX_MATCH_LEN {
                        break;
                    }
                }
            }

            let prev_offset_val = unsafe {
                *self
                    .prev_tab
                    .get_unchecked(p_abs & (MATCHFINDER_WINDOW_SIZE - 1))
            };
            if prev_offset_val == 0 {
                break;
            }
            cur_pos_i32 -= prev_offset_val as i32;
            depth += 1;
        }

        (best_len, best_offset)
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
            return;
        }
        let abs_pos = self.base_offset + pos;
        let h = Self::hash(&data[pos..]);
        let cur_pos = unsafe { *self.hash_tab.get_unchecked(h) };
        unsafe { *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32 };

        if cur_pos != -1 && (cur_pos as usize) >= self.base_offset {
             let prev_offset = abs_pos - (cur_pos as usize);
             unsafe {
                *self
                    .prev_tab
                    .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = if prev_offset > 0xFFFF {
                    0
                } else {
                    prev_offset as u16
                }
            };
        } else {
             unsafe {
                 *self.prev_tab.get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
             }
        }
    }
}

pub struct HtMatchFinder {
    pub hash_tab: Vec<i32>,
    pub base_offset: usize,
}

impl HtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash_tab: vec![-1; MATCHFINDER_HASH_SIZE],
            base_offset: 0,
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
        if pos + 3 > data.len() {
            return (0, 0);
        }

        let abs_pos = self.base_offset + pos;
        let h = MatchFinder::hash(&data[pos..]);
        let cur_pos = unsafe { *self.hash_tab.get_unchecked(h) };
        unsafe { *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32 };

        if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
            return (0, 0);
        }

        let p_abs = cur_pos as usize;
        let offset = abs_pos - p_abs;
        if offset > DEFLATE_MAX_MATCH_OFFSET {
            return (0, 0);
        }
        
        let p_rel = p_abs - self.base_offset;

        if data[p_rel] == data[pos] && data[p_rel + 1] == data[pos + 1] && data[p_rel + 2] == data[pos + 2] {
            let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
            let len = match_len(&data[p_rel..], &data[pos..], max_len);
            if len >= 3 {
                return (len, offset);
            }
        }

        (0, 0)
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
            return;
        }
        let abs_pos = self.base_offset + pos;
        let h = MatchFinder::hash(&data[pos..]);
        unsafe { *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32 };
    }
}

pub struct BtMatchFinder {
    pub hash3_tab: Vec<[i32; 2]>,
    pub hash4_tab: Vec<i32>,
    pub child_tab: Vec<[i32; 2]>,
    pub base_offset: usize,
}

impl BtMatchFinder {
    pub fn new() -> Self {
        Self {
            hash3_tab: vec![[-1; 2]; 1 << 16],
            hash4_tab: vec![-1; 1 << 16],
            child_tab: vec![[0; 2]; MATCHFINDER_WINDOW_SIZE],
            base_offset: 0,
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

    #[inline(always)]
    fn hash3(bytes: &[u8]) -> usize {
        unsafe {
            let h = ((*bytes.get_unchecked(0) as u32) << 16)
                | ((*bytes.get_unchecked(1) as u32) << 8)
                | (*bytes.get_unchecked(2) as u32);
            let h = h.wrapping_mul(0x1E35A7BD);
            (h >> (32 - 16)) as usize
        }
    }

    #[inline(always)]
    fn hash4(bytes: &[u8]) -> usize {
        unsafe {
            let slice = bytes.get_unchecked(0..4);
            let h = u32::from_le_bytes(slice.try_into().unwrap_unchecked());
            let h = h.wrapping_mul(0x1E35A7BD);
            (h >> (32 - 16)) as usize
        }
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        if pos + 4 > data.len() {
            return (0, 0);
        }

        let abs_pos = self.base_offset + pos;
        let h3 = Self::hash3(&data[pos..]);
        let h4 = Self::hash4(&data[pos..]);

        let cur_node_3 = unsafe { (*self.hash3_tab.get_unchecked(h3))[0] };
        unsafe { (*self.hash3_tab.get_unchecked_mut(h3))[0] = abs_pos as i32 };
        let cur_node_3_2 = unsafe { (*self.hash3_tab.get_unchecked(h3))[1] };
        unsafe { (*self.hash3_tab.get_unchecked_mut(h3))[1] = cur_node_3 };

        let cutoff = (abs_pos as i32).wrapping_sub(MATCHFINDER_WINDOW_SIZE as i32);

        let mut best_len = 0;
        let mut best_offset = 0;

        if cur_node_3 != -1 && cur_node_3 > cutoff && (cur_node_3 as usize) >= self.base_offset {
            let p_abs = cur_node_3 as usize;
            let p_rel = p_abs - self.base_offset;
            if data[p_rel] == data[pos] && data[p_rel + 1] == data[pos + 1] && data[p_rel + 2] == data[pos + 2]
            {
                best_len = 3;
                best_offset = abs_pos - p_abs;
            } else if cur_node_3_2 != -1 && cur_node_3_2 > cutoff && (cur_node_3_2 as usize) >= self.base_offset {
                let p2_abs = cur_node_3_2 as usize;
                let p2_rel = p2_abs - self.base_offset;
                if data[p2_rel] == data[pos]
                    && data[p2_rel + 1] == data[pos + 1]
                    && data[p2_rel + 2] == data[pos + 2]
                {
                    best_len = 3;
                    best_offset = abs_pos - p2_abs;
                }
            }
        }

        let mut cur_node = unsafe { *self.hash4_tab.get_unchecked(h4) };
        unsafe { *self.hash4_tab.get_unchecked_mut(h4) = abs_pos as i32 };

        let child_idx = abs_pos & (MATCHFINDER_WINDOW_SIZE - 1);

        if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
            unsafe { *self.child_tab.get_unchecked_mut(child_idx) = [-1, -1] };
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

            let len = match_len(&data[p_rel..], &data[pos..], max_len_clamped);

            if len > best_len {
                best_len = len;
                best_offset = abs_pos - p_abs;
                if len == max_len_clamped {
                    let children = unsafe { *self.child_tab.get_unchecked(p_child_idx) };
                    unsafe {
                        (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                            children[0];
                        (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                            children[1];
                    }
                    return (best_len, best_offset);
                }
            }

            if len < max_len_clamped && data[p_rel + len] < data[pos + len] {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                        cur_node
                };
                pending_lt_node = p_child_idx;
                pending_lt_child = 1;
                cur_node = unsafe { (*self.child_tab.get_unchecked(p_child_idx))[1] };
            } else {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                        cur_node
                };
                pending_gt_node = p_child_idx;
                pending_gt_child = 0;
                cur_node = unsafe { (*self.child_tab.get_unchecked(p_child_idx))[0] };
            }

            if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                }
                return (best_len, best_offset);
            }

            depth_remaining -= 1;
            if depth_remaining == 0 {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                }
                return (best_len, best_offset);
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
        if pos + 4 > data.len() {
            return;
        }

        let abs_pos = self.base_offset + pos;
        let h3 = Self::hash3(&data[pos..]);
        let h4 = Self::hash4(&data[pos..]);

        let cur_node_3 = unsafe { (*self.hash3_tab.get_unchecked(h3))[0] };
        unsafe { (*self.hash3_tab.get_unchecked_mut(h3))[0] = abs_pos as i32 };
        let cur_node_3_2 = unsafe { (*self.hash3_tab.get_unchecked(h3))[1] };
        unsafe { (*self.hash3_tab.get_unchecked_mut(h3))[1] = cur_node_3 };

        let cutoff = (abs_pos as i32).wrapping_sub(MATCHFINDER_WINDOW_SIZE as i32);

        if record_matches {
            if cur_node_3 != -1 && cur_node_3 > cutoff && (cur_node_3 as usize) >= self.base_offset {
                let p_abs = cur_node_3 as usize;
                let p_rel = p_abs - self.base_offset;
                if data[p_rel] == data[pos]
                    && data[p_rel + 1] == data[pos + 1]
                    && data[p_rel + 2] == data[pos + 2]
                {
                    matches.push((3, (abs_pos - p_abs) as u16));
                } else if cur_node_3_2 != -1 && cur_node_3_2 > cutoff && (cur_node_3_2 as usize) >= self.base_offset {
                    let p2_abs = cur_node_3_2 as usize;
                    let p2_rel = p2_abs - self.base_offset;
                    if data[p2_rel] == data[pos]
                        && data[p2_rel + 1] == data[pos + 1]
                        && data[p2_rel + 2] == data[pos + 2]
                    {
                        matches.push((3, (abs_pos - p2_abs) as u16));
                    }
                }
            }
        }

        let mut cur_node = unsafe { *self.hash4_tab.get_unchecked(h4) };
        unsafe { *self.hash4_tab.get_unchecked_mut(h4) = abs_pos as i32 };

        let child_idx = abs_pos & (MATCHFINDER_WINDOW_SIZE - 1);

        if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
            unsafe { *self.child_tab.get_unchecked_mut(child_idx) = [-1, -1] };
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

            let len = match_len(&data[p_rel..], &data[pos..], max_len_clamped);

            if record_matches && len > best_len {
                best_len = len;
                matches.push((len as u16, (abs_pos - p_abs) as u16));
                if len >= nice_len {
                    let children = unsafe { *self.child_tab.get_unchecked(p_child_idx) };
                    unsafe {
                        (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                            children[0];
                        (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                            children[1];
                    }
                    return;
                }
            }

            if len < max_len_clamped && data[p_rel + len] < data[pos + len] {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] =
                        cur_node
                };
                pending_lt_node = p_child_idx;
                pending_lt_child = 1;
                cur_node = unsafe { (*self.child_tab.get_unchecked(p_child_idx))[1] };
            } else {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] =
                        cur_node
                };
                pending_gt_node = p_child_idx;
                pending_gt_child = 0;
                cur_node = unsafe { (*self.child_tab.get_unchecked(p_child_idx))[0] };
            }

            if cur_node == -1 || cur_node <= cutoff || (cur_node as usize) < self.base_offset {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                }
                return;
            }

            depth_remaining -= 1;
            if depth_remaining == 0 {
                unsafe {
                    (*self.child_tab.get_unchecked_mut(pending_lt_node))[pending_lt_child] = -1;
                    (*self.child_tab.get_unchecked_mut(pending_gt_node))[pending_gt_child] = -1;
                }
                return;
            }
        }
    }
}