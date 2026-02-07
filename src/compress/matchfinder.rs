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
unsafe fn match_len_ptr(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let mut len = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if max_len >= 64 && is_x86_feature_detected!("avx512bw") {
            while len + 64 <= max_len {
                let v1 = _mm512_loadu_si512(a.add(len) as *const _);
                let v2 = _mm512_loadu_si512(b.add(len) as *const _);
                let mask = _mm512_cmpeq_epi8_mask(v1, v2);
                if mask != u64::MAX {
                    return len + (!mask).trailing_zeros() as usize;
                }
                len += 64;
            }
        }

        if max_len >= 32 && is_x86_feature_detected!("avx2") {
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
        }

        #[cfg(target_arch = "x86_64")]
        if max_len >= 16 && is_x86_feature_detected!("sse2") {
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
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if max_len >= 16 && std::arch::is_aarch64_feature_detected!("neon") {
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
        }
    }

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

        unsafe {
            let src = data.as_ptr().add(pos);
            let h = (((src.read() as u32) << 16)
                | ((src.add(1).read() as u32) << 8)
                | (src.add(2).read() as u32))
                .wrapping_mul(0x1E35A7BD);
            let h = (h >> (32 - MATCHFINDER_HASH_ORDER)) as usize;

            let abs_pos = self.base_offset + pos;
            let cur_pos = *self.hash_tab.get_unchecked(h);
            *self.hash_tab.get_unchecked_mut(h) = abs_pos as i32;

            if cur_pos == -1 || (cur_pos as usize) < self.base_offset {
                *self
                    .prev_tab
                    .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
                return (0, 0);
            }

            let prev_offset = abs_pos - (cur_pos as usize);
            *self
                .prev_tab
                .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) =
                if prev_offset > 0xFFFF {
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

                if pos + best_len < data.len() && *match_ptr.add(best_len) == *src.add(best_len) {
                    let max_len = min(DEFLATE_MAX_MATCH_LEN, data.len() - pos);
                    let len = match_len_ptr(match_ptr, src, max_len);

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
    }

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        if pos + 3 > data.len() {
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
                *self
                    .prev_tab
                    .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) = 0;
                return (0, 0);
            }

            let prev_offset = abs_pos - (cur_pos as usize);
            *self
                .prev_tab
                .get_unchecked_mut(abs_pos & (MATCHFINDER_WINDOW_SIZE - 1)) =
                if prev_offset > 0xFFFF {
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
                        let len = match_len_ptr(match_ptr, src, max_len);

                        if len > best_len {
                            best_len = len;
                            best_offset = offset;
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
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
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
                let len = match_len_ptr(match_ptr, src, max_len);
                if len >= 3 {
                    return (len, offset);
                }
            }
        }
        (0, 0)
    }

    pub fn skip_match(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
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

    pub fn find_match(&mut self, data: &[u8], pos: usize, max_depth: usize) -> (usize, usize) {
        if pos + 4 > data.len() {
            return (0, 0);
        }

        unsafe {
            let src = data.as_ptr().add(pos);
            let h3 = (((src.read() as u32) << 16)
                | ((src.add(1).read() as u32) << 8)
                | (src.add(2).read() as u32))
                .wrapping_mul(0x1E35A7BD);
            let h3 = (h3 >> 16) as usize;

            let h4 = (src.cast::<u32>().read_unaligned()).wrapping_mul(0x1E35A7BD);
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

                let len = match_len_ptr(match_ptr, src, max_len_clamped);

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
        if pos + 4 > data.len() {
            return;
        }

        unsafe {
            let src = data.as_ptr().add(pos);
            let h3 = (((src.read() as u32) << 16)
                | ((src.add(1).read() as u32) << 8)
                | (src.add(2).read() as u32))
                .wrapping_mul(0x1E35A7BD);
            let h3 = (h3 >> 16) as usize;

            let h4 = (src.cast::<u32>().read_unaligned()).wrapping_mul(0x1E35A7BD);
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

                let len = match_len_ptr(match_ptr, src, max_len_clamped);

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
