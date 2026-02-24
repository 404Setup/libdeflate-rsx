pub mod bitstream;

mod huffman_comp;
mod matchfinder;

use self::bitstream::Bitstream;
use self::huffman_comp::make_huffman_code;
use self::matchfinder::{BtMatchFinder, HtMatchFinder, MatchFinder, MatchFinderTrait};
use crate::common::*;
use rayon::prelude::*;
use std::cmp::min;
use std::io;
use std::mem::MaybeUninit;
use std::sync::OnceLock;

const LENGTH_WRITE_TABLE: [u32; 260] = [
    3, 3, 3, 3, 16777220, 33554437, 50331654, 67108871, 83886088, 100663305, 117440522, 134283275,
    134283275, 151060493, 151060493, 167837711, 167837711, 184614929, 184614929, 201457683,
    201457683, 201457683, 201457683, 218234903, 218234903, 218234903, 218234903, 235012123,
    235012123, 235012123, 235012123, 251789343, 251789343, 251789343, 251789343, 268632099,
    268632099, 268632099, 268632099, 268632099, 268632099, 268632099, 268632099, 285409323,
    285409323, 285409323, 285409323, 285409323, 285409323, 285409323, 285409323, 302186547,
    302186547, 302186547, 302186547, 302186547, 302186547, 302186547, 302186547, 318963771,
    318963771, 318963771, 318963771, 318963771, 318963771, 318963771, 318963771, 335806531,
    335806531, 335806531, 335806531, 335806531, 335806531, 335806531, 335806531, 335806531,
    335806531, 335806531, 335806531, 335806531, 335806531, 335806531, 335806531, 352583763,
    352583763, 352583763, 352583763, 352583763, 352583763, 352583763, 352583763, 352583763,
    352583763, 352583763, 352583763, 352583763, 352583763, 352583763, 352583763, 369360995,
    369360995, 369360995, 369360995, 369360995, 369360995, 369360995, 369360995, 369360995,
    369360995, 369360995, 369360995, 369360995, 369360995, 369360995, 369360995, 386138227,
    386138227, 386138227, 386138227, 386138227, 386138227, 386138227, 386138227, 386138227,
    386138227, 386138227, 386138227, 386138227, 386138227, 386138227, 386138227, 402980995,
    402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995,
    402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995,
    402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995,
    402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 402980995, 419758243,
    419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243,
    419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243,
    419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243,
    419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 419758243, 436535491,
    436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491,
    436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491,
    436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491,
    436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 436535491, 453312739,
    453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739,
    453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739,
    453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 453312739,
    453312739, 453312739, 453312739, 453312739, 453312739, 453312739, 469762306, 3,
];

const LENGTH_EXTRA_BITS_TABLE: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const OFFSET_BASE_TABLE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

const OFFSET_EXTRA_BITS_TABLE: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

// Precomputed table for get_offset_slot.
// For offsets <= 256, table[offset] gives the slot.
// For offsets > 256, table[256 + ((offset - 1) >> 7)] gives the slot.
// This reduces the table size from 32KB to 512 bytes, improving cache locality.
const OFFSET_SLOT_TABLE_512: [u8; 512] = {
    let mut table = [0; 512];

    // Fill 0..=256 (used for direct lookup)
    let mut offset: usize = 1;
    while offset <= 256 {
        let slot = if offset <= 2 {
            offset - 1
        } else {
            let off = (offset - 1) as u32;
            let l = 31 - off.leading_zeros();
            ((2 * l) + ((off >> (l - 1)) & 1)) as usize
        };
        table[offset] = slot as u8;
        offset += 1;
    }

    // Fill 257..511 (used for (offset - 1) >> 7)
    // Index i corresponds to k = i - 256.
    // k = (offset - 1) >> 7.
    let mut k: u32 = 0;
    while k < 256 {
        // We only access this part for offset > 256, which implies k >= 2.
        if k >= 2 {
            // Pick a representative offset value. k << 7 works because the slot
            // depends only on the MSB and the bit below it, which are preserved
            // in k for k >= 2.
            let off = k << 7;
            let l = 31 - off.leading_zeros();
            let slot = ((2 * l) + ((off >> (l - 1)) & 1)) as usize;
            table[(256 + k) as usize] = slot as u8;
        }
        k += 1;
    }

    table
};

const OFF_IDX_TABLE: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, // 0-7: offset < 256
    1, 1, 1, 1, // 8-11: offset < 4096
    2, 2, 2, // 12-14: offset < 32768
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // 15-31: offset >= 32768
];

// Mapping from offset slot to observation index.
// Slots 0-15 (offsets 1-256) -> 0
// Slots 16-23 (offsets 257-4096) -> 1
// Slots 24-29 (offsets 4097-32768) -> 2
// Note: Offset 32768 (Slot 29) is effectively mapped to Type 2,
// whereas OFF_IDX_TABLE maps it to Type 3. This minor deviation is acceptable.
const SLOT_TO_OBS_IDX: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-15
    1, 1, 1, 1, 1, 1, 1, 1, // 16-23
    2, 2, 2, 2, 2, 2, // 24-29
    0, 0, // 30-31
];

pub const MAX_LITLEN_CODEWORD_LEN: usize = 14;
pub const MAX_OFFSET_CODEWORD_LEN: usize = 15;
pub const MAX_PRE_CODEWORD_LEN: usize = 7;

fn gen_codewords_from_lens(lens: &[u8], codewords: &mut [u32], max_len: usize) {
    let mut len_counts = [0u32; 16];
    for &l in lens {
        if l > 0 {
            len_counts[l as usize] += 1;
        }
    }
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for len in 1..=max_len {
        code = (code + len_counts[len - 1]) << 1;
        next_code[len] = code;
    }
    for i in 0..lens.len() {
        if lens[i] > 0 {
            let c = next_code[lens[i] as usize];
            next_code[lens[i] as usize] += 1;
            codewords[i] = (c as u16).reverse_bits() as u32 >> (16 - lens[i]);
        }
    }
}

struct StaticTables {
    litlen_lens: [u8; DEFLATE_NUM_LITLEN_SYMS],
    offset_lens: [u8; DEFLATE_NUM_OFFSET_SYMS],
    litlen_table: [u64; DEFLATE_NUM_LITLEN_SYMS],
    offset_table: [u64; DEFLATE_NUM_OFFSET_SYMS],
    match_len_table: [u64; DEFLATE_MAX_MATCH_LEN + 1],
}

fn compute_static_tables() -> StaticTables {
    let mut litlen_lens = [0u8; DEFLATE_NUM_LITLEN_SYMS];
    let mut offset_lens = [0u8; DEFLATE_NUM_OFFSET_SYMS];
    let mut litlen_codewords = [0u32; DEFLATE_NUM_LITLEN_SYMS];
    let mut offset_codewords = [0u32; DEFLATE_NUM_OFFSET_SYMS];
    let mut litlen_table = [0u64; DEFLATE_NUM_LITLEN_SYMS];
    let mut offset_table = [0u64; DEFLATE_NUM_OFFSET_SYMS];
    let mut match_len_table = [0u64; DEFLATE_MAX_MATCH_LEN + 1];

    let mut i = 0;
    while i < 144 {
        litlen_lens[i] = 8;
        i += 1;
    }
    while i < 256 {
        litlen_lens[i] = 9;
        i += 1;
    }
    while i < 280 {
        litlen_lens[i] = 7;
        i += 1;
    }
    while i < 288 {
        litlen_lens[i] = 8;
        i += 1;
    }
    for i in 0..32 {
        offset_lens[i] = 5;
    }

    gen_codewords_from_lens(&litlen_lens, &mut litlen_codewords, 9);
    gen_codewords_from_lens(&offset_lens, &mut offset_codewords, 5);

    for i in 0..DEFLATE_NUM_LITLEN_SYMS {
        litlen_table[i] = (litlen_codewords[i] as u64) | ((litlen_lens[i] as u64) << 32);
    }
    for i in 0..DEFLATE_NUM_OFFSET_SYMS {
        let mut entry = (offset_codewords[i] as u64) | ((offset_lens[i] as u64) << 32);
        if i < 30 {
            // SAFETY: Arrays are static consts of size 30.
            entry |= (unsafe { *OFFSET_EXTRA_BITS_TABLE.get_unchecked(i) } as u64) << 40;
            entry |= (unsafe { *OFFSET_BASE_TABLE.get_unchecked(i) } as u64) << 48;
        }
        offset_table[i] = entry;
    }

    for len in 3..=DEFLATE_MAX_MATCH_LEN {
        let len_info = unsafe { *LENGTH_WRITE_TABLE.get_unchecked(len) };
        let slot = (len_info >> 24) as usize;
        let extra = (len_info >> 16) as u8;
        let base = len_info as u16;

        let huff_entry = unsafe { *litlen_table.get_unchecked(257 + slot) };
        let code = huff_entry as u16;
        let huff_len = (huff_entry >> 32) as u8;

        match_len_table[len] = (code as u64)
            | ((huff_len as u64) << 16)
            | ((extra as u64) << 24)
            | ((base as u64) << 32);
    }

    StaticTables {
        litlen_lens,
        offset_lens,
        litlen_table,
        offset_table,
        match_len_table,
    }
}

#[derive(Debug, PartialEq, Eq)]
#[must_use = "Compression result must be checked to ensure data integrity"]
pub enum CompressResult {
    Success,
    InsufficientSpace,
}

#[derive(Clone, Copy)]
struct Sequence {
    litrunlen: u32,
    length: u16,
    offset: u16,
}

impl Sequence {
    #[inline(always)]
    fn new(litrunlen: u32, len: u16, offset: u16, off_slot: u8) -> Self {
        Self {
            litrunlen,
            length: len | ((off_slot as u16) << 9),
            offset,
        }
    }

    #[inline(always)]
    fn len(&self) -> u16 {
        self.length & 0x1FF
    }

    #[inline(always)]
    fn off_slot(&self) -> usize {
        (self.length >> 9) as usize
    }
}

#[derive(Clone, Copy)]
struct DPNode {
    cost: u32,
    length: u16,
    offset: u16,
}

const NUM_LITERAL_OBSERVATION_TYPES: usize = 8;
const NUM_MATCH_OBSERVATION_TYPES: usize = 2;
const NUM_OFFSET_OBSERVATION_TYPES: usize = 4;
const NUM_OBSERVATION_TYPES: usize =
    NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES + NUM_OFFSET_OBSERVATION_TYPES;
const NUM_OBSERVATIONS_PER_BLOCK_CHECK: u32 = 2048;

struct BlockSplitStats {
    new_observations: [u32; NUM_OBSERVATION_TYPES],
    observations: [u32; NUM_OBSERVATION_TYPES],
    num_new_observations: u32,
    num_observations: u32,
}

impl BlockSplitStats {
    fn new() -> Self {
        Self {
            new_observations: [0; NUM_OBSERVATION_TYPES],
            observations: [0; NUM_OBSERVATION_TYPES],
            num_new_observations: 0,
            num_observations: 0,
        }
    }

    fn reset(&mut self) {
        self.new_observations.fill(0);
        self.observations.fill(0);
        self.num_new_observations = 0;
        self.num_observations = 0;
    }

    #[inline(always)]
    fn observe_literal(&mut self, lit: u8) {
        unsafe {
            *self.new_observations.get_unchecked_mut((lit >> 5) as usize) += 1;
        }
        self.num_new_observations += 1;
    }

    #[inline(always)]
    fn observe_match(&mut self, length: usize, offset: usize) {
        let len_idx = NUM_LITERAL_OBSERVATION_TYPES + if length >= 8 { 1 } else { 0 };
        unsafe {
            *self.new_observations.get_unchecked_mut(len_idx) += 1;
        }

        // Optimization: Use table lookup to avoid branch mispredictions.
        // offset is always >= 1, so bsr32 is safe.
        debug_assert!(offset >= 1);
        let off_idx_base = unsafe { *OFF_IDX_TABLE.get_unchecked(bsr32(offset as u32) as usize) };

        let off_idx =
            NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES + off_idx_base as usize;
        unsafe {
            *self.new_observations.get_unchecked_mut(off_idx) += 1;
        }

        self.num_new_observations += 2;
    }

    #[inline(always)]
    fn observe_match_with_slot(&mut self, length: usize, off_slot: usize) {
        let len_idx = NUM_LITERAL_OBSERVATION_TYPES + if length >= 8 { 1 } else { 0 };
        unsafe {
            *self.new_observations.get_unchecked_mut(len_idx) += 1;
        }

        let off_idx_base = unsafe { *SLOT_TO_OBS_IDX.get_unchecked(off_slot) };
        let off_idx =
            NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES + off_idx_base as usize;
        unsafe {
            *self.new_observations.get_unchecked_mut(off_idx) += 1;
        }

        self.num_new_observations += 2;
    }

    fn merge_new_observations(&mut self) {
        for i in 0..NUM_OBSERVATION_TYPES {
            unsafe {
                *self.observations.get_unchecked_mut(i) += *self.new_observations.get_unchecked(i);
            }
        }
        self.num_observations += self.num_new_observations;
        self.new_observations.fill(0);
        self.num_new_observations = 0;
    }

    fn do_end_block_check(&self, block_length: usize) -> bool {
        if self.num_observations == 0 {
            return false;
        }
        let mut old_bits = 0;
        let mut new_bits = 0;

        let log2_num_obs = bsr32(self.num_observations);
        let log2_num_new_obs = bsr32(self.num_new_observations);

        for i in 0..NUM_OBSERVATION_TYPES {
            unsafe {
                let new_obs = *self.new_observations.get_unchecked(i);
                if new_obs > 0 {
                    let log2_obs = bsr32(*self.observations.get_unchecked(i) + 1);
                    let cost_old = log2_num_obs.saturating_sub(log2_obs);
                    old_bits += new_obs * cost_old;

                    let log2_new_obs = bsr32(new_obs + 1);
                    let cost_new = log2_num_new_obs.saturating_sub(log2_new_obs);
                    new_bits += new_obs * cost_new;
                }
            }
        }
        (old_bits as i32 - new_bits as i32) > (block_length as i32 / 16)
    }

    #[inline(always)]
    fn should_end_block(&mut self, block_length: usize, input_remaining: usize) -> bool {
        // Optimization: Fast path for the common case where we are far from any block limit.
        // This avoids checking `input_remaining` (which requires a subtraction) and other
        // conditions in the hottest path (executed for every literal/match).
        if self.num_new_observations < NUM_OBSERVATIONS_PER_BLOCK_CHECK
            && block_length < SOFT_MAX_BLOCK_LENGTH
        {
            return false;
        }

        if input_remaining <= MIN_BLOCK_LENGTH {
            return false;
        }
        if block_length >= SOFT_MAX_BLOCK_LENGTH {
            return true;
        }

        // If we reach here, we know `block_length < SOFT_MAX_BLOCK_LENGTH`.
        // Combined with the failure of the fast path check above, this implies that
        // `self.num_new_observations >= NUM_OBSERVATIONS_PER_BLOCK_CHECK`.
        // So we can proceed directly to the block split check without re-verifying the count.
        if block_length >= MIN_BLOCK_LENGTH {
            if self.do_end_block_check(block_length) {
                return true;
            }
            self.merge_new_observations();
        }
        false
    }
}

enum MatchFinderEnum {
    Chain(MatchFinder),
    Table(HtMatchFinder),
    Bt(BtMatchFinder),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlushMode {
    None,
    Sync,
    Finish,
}

pub struct Compressor {
    pub compression_level: usize,
    pub max_search_depth: usize,
    pub nice_match_length: usize,
    pub litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    pub offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
    pub litlen_codewords: [u32; DEFLATE_NUM_LITLEN_SYMS],
    pub litlen_lens: [u8; DEFLATE_NUM_LITLEN_SYMS],
    pub offset_codewords: [u32; DEFLATE_NUM_OFFSET_SYMS],
    pub offset_lens: [u8; DEFLATE_NUM_OFFSET_SYMS],

    pub litlen_table: [u64; DEFLATE_NUM_LITLEN_SYMS],
    pub offset_table: [u64; DEFLATE_NUM_OFFSET_SYMS],
    pub match_len_table: [u64; DEFLATE_MAX_MATCH_LEN + 1],

    pub literal_costs: [u32; 256],
    pub length_costs: [u32; DEFLATE_MAX_MATCH_LEN + 1],
    pub offset_slot_costs: [u32; 32],

    mf: Option<MatchFinderEnum>,
    sequences: Vec<Sequence>,
    dp_costs: Vec<u32>,
    dp_path: Vec<u32>,
    split_stats: BlockSplitStats,
    matches: Vec<(u16, u16)>,
}

impl Compressor {
    pub fn new(level: usize) -> Self {
        let mut c = Self {
            compression_level: level,
            max_search_depth: 0,
            nice_match_length: 0,
            litlen_freqs: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_freqs: [0; DEFLATE_NUM_OFFSET_SYMS],
            litlen_codewords: [0; DEFLATE_NUM_LITLEN_SYMS],
            litlen_lens: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_codewords: [0; DEFLATE_NUM_OFFSET_SYMS],
            offset_lens: [0; DEFLATE_NUM_OFFSET_SYMS],
            litlen_table: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_table: [0; DEFLATE_NUM_OFFSET_SYMS],
            match_len_table: [0; DEFLATE_MAX_MATCH_LEN + 1],
            literal_costs: [0; 256],
            length_costs: [0; DEFLATE_MAX_MATCH_LEN + 1],
            offset_slot_costs: [0; 32],
            mf: Some(if level == 1 {
                MatchFinderEnum::Table(HtMatchFinder::new())
            } else if level >= 10 {
                MatchFinderEnum::Bt(BtMatchFinder::new())
            } else {
                MatchFinderEnum::Chain(MatchFinder::new())
            }),
            sequences: if level == 0 {
                Vec::new()
            } else {
                Vec::with_capacity(50000)
            },
            dp_costs: if level >= 10 {
                Vec::with_capacity(300000)
            } else {
                Vec::new()
            },
            dp_path: if level >= 10 {
                Vec::with_capacity(300000)
            } else {
                Vec::new()
            },
            split_stats: BlockSplitStats::new(),
            matches: if level >= 10 {
                Vec::with_capacity(32)
            } else {
                Vec::new()
            },
        };
        c.init_params();
        c
    }

    fn update_huffman_tables(&mut self) {
        for i in 0..DEFLATE_NUM_LITLEN_SYMS {
            self.litlen_table[i] =
                (self.litlen_codewords[i] as u64) | ((self.litlen_lens[i] as u64) << 32);
        }
        for i in 0..DEFLATE_NUM_OFFSET_SYMS {
            let mut entry =
                (self.offset_codewords[i] as u64) | ((self.offset_lens[i] as u64) << 32);
            if i < 30 {
                // SAFETY: Arrays are static consts of size 30.
                entry |= (unsafe { *OFFSET_EXTRA_BITS_TABLE.get_unchecked(i) } as u64) << 40;
                entry |= (unsafe { *OFFSET_BASE_TABLE.get_unchecked(i) } as u64) << 48;
            }
            self.offset_table[i] = entry;
        }

        for len in 3..=DEFLATE_MAX_MATCH_LEN {
            let len_info = unsafe { *LENGTH_WRITE_TABLE.get_unchecked(len) };
            let slot = (len_info >> 24) as usize;
            let extra = (len_info >> 16) as u8;
            let base = len_info as u16;

            let huff_entry = unsafe { *self.litlen_table.get_unchecked(257 + slot) };
            let code = huff_entry as u16;
            let huff_len = (huff_entry >> 32) as u8;

            self.match_len_table[len] = (code as u64)
                | ((huff_len as u64) << 16)
                | ((extra as u64) << 24)
                | ((base as u64) << 32);
        }
    }

    fn init_params(&mut self) {
        match self.compression_level {
            0 => {
                self.max_search_depth = 0;
                self.nice_match_length = 0;
            }
            1 => {
                self.max_search_depth = 2;
                self.nice_match_length = 32;
            }
            2 => {
                self.max_search_depth = 6;
                self.nice_match_length = 10;
            }
            3 => {
                self.max_search_depth = 12;
                self.nice_match_length = 14;
            }
            4 => {
                self.max_search_depth = 16;
                self.nice_match_length = 30;
            }
            5 => {
                self.max_search_depth = 16;
                self.nice_match_length = 30;
            }
            6 => {
                self.max_search_depth = 35;
                self.nice_match_length = 65;
            }
            7 => {
                self.max_search_depth = 100;
                self.nice_match_length = 130;
            }
            8 => {
                self.max_search_depth = 300;
                self.nice_match_length = 258;
            }
            9 => {
                self.max_search_depth = 600;
                self.nice_match_length = 258;
            }
            10 => {
                self.max_search_depth = 35;
                self.nice_match_length = 75;
            }
            11 => {
                self.max_search_depth = 100;
                self.nice_match_length = 150;
            }
            12 => {
                self.max_search_depth = 300;
                self.nice_match_length = 258;
            }
            _ => {
                self.max_search_depth = 300;
                self.nice_match_length = 258;
            }
        }
    }

    fn compress_loop<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        bs: &mut Bitstream,
        flush_mode: FlushMode,
    ) -> (CompressResult, usize, u32) {
        let mut in_idx = 0;
        mf.prepare(input.len());

        while in_idx < input.len() {
            let processed = if self.compression_level >= 10 {
                self.compress_near_optimal_block(
                    mf,
                    input,
                    in_idx,
                    bs,
                    flush_mode == FlushMode::Finish,
                )
            } else {
                let lazy_depth = if self.compression_level >= 8 {
                    2
                } else if self.compression_level >= 5 {
                    1
                } else {
                    0
                };
                self.compress_greedy_block(
                    mf,
                    input,
                    in_idx,
                    bs,
                    lazy_depth,
                    flush_mode == FlushMode::Finish,
                )
            };

            if processed == 0 {
                mf.advance(input.len());
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            in_idx += processed;
        }

        if in_idx == 0 && flush_mode == FlushMode::Finish {
            let start_out = bs.out_idx;
            if self.compression_level >= 10 {
                self.compress_near_optimal_block(mf, input, 0, bs, true);
            } else {
                self.compress_greedy_block(mf, input, 0, bs, 0, true);
            }
            if bs.out_idx == start_out {
                mf.advance(input.len());
                return (CompressResult::InsufficientSpace, 0, 0);
            }
        }

        if flush_mode == FlushMode::Sync {
            if !bs.write_bits(0, 3) {
                mf.advance(input.len());
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            let (res, _) = bs.flush();
            if !res {
                mf.advance(input.len());
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            if bs.out_idx + 4 > bs.output.len() {
                mf.advance(input.len());
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            bs.output[bs.out_idx].write(0);
            bs.output[bs.out_idx + 1].write(0);
            bs.output[bs.out_idx + 2].write(0xFF);
            bs.output[bs.out_idx + 3].write(0xFF);
            bs.out_idx += 4;
        }

        let (res, valid_bits) = bs.flush();
        if !res {
            mf.advance(input.len());
            return (CompressResult::InsufficientSpace, 0, 0);
        }

        mf.advance(input.len());
        (CompressResult::Success, bs.out_idx, valid_bits)
    }

    pub fn compress(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush_mode: FlushMode,
    ) -> (CompressResult, usize, u32) {
        if input.len() > 256 * 1024 {
            let chunk_size = 256 * 1024;
            let chunks: Vec<&[u8]> = input.chunks(chunk_size).collect();

            let compressed_chunks_res: Vec<io::Result<Vec<u8>>> = chunks
                .par_iter()
                .enumerate()
                .map_init(
                    || {
                        (
                            Compressor::new(self.compression_level),
                            Vec::with_capacity(chunk_size + chunk_size / 2),
                        )
                    },
                    |(compressor, buf), (i, chunk)| {
                        let is_last = i == chunks.len() - 1;
                        let mode = if is_last { flush_mode } else { FlushMode::Sync };

                        let bound = Self::deflate_compress_bound(chunk.len());
                        if buf.capacity() < bound {
                            buf.reserve(bound - buf.len());
                        }
                        unsafe {
                            buf.set_len(bound);
                        }

                        let buf_uninit = unsafe {
                            std::slice::from_raw_parts_mut(
                                buf.as_mut_ptr() as *mut MaybeUninit<u8>,
                                buf.len(),
                            )
                        };

                        let (res, size, _) = compressor.compress(chunk, buf_uninit, mode);
                        if res == CompressResult::Success {
                            unsafe {
                                buf.set_len(size);
                            }
                            if size < buf.capacity() / 2 {
                                Ok(buf.to_vec())
                            } else {
                                Ok(std::mem::replace(
                                    buf,
                                    Vec::with_capacity(chunk_size + chunk_size / 2),
                                ))
                            }
                        } else {
                            Err(io::Error::other("Compression failed"))
                        }
                    },
                )
                .collect();

            let mut out_idx = 0;
            for res in compressed_chunks_res {
                match res {
                    Ok(data) => {
                        if out_idx + data.len() > output.len() {
                            return (CompressResult::InsufficientSpace, 0, 0);
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                output.as_mut_ptr().add(out_idx) as *mut u8,
                                data.len(),
                            );
                        }
                        out_idx += data.len();
                    }
                    Err(_) => return (CompressResult::InsufficientSpace, 0, 0),
                }
            }
            return (CompressResult::Success, out_idx, 0);
        }

        if self.compression_level == 0 {
            return self.compress_uncompressed(input, output, flush_mode);
        }

        let mut bs = Bitstream::new(output);

        let mut mf_enum = self.mf.take().unwrap();

        let res = match &mut mf_enum {
            MatchFinderEnum::Chain(mf) => self.compress_loop(mf, input, &mut bs, flush_mode),
            MatchFinderEnum::Table(mf) => self.compress_loop(mf, input, &mut bs, flush_mode),
            MatchFinderEnum::Bt(mf) => self.compress_loop(mf, input, &mut bs, flush_mode),
        };

        self.mf = Some(mf_enum);
        res
    }

    fn compress_to_size_loop<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        _final_block: bool,
    ) -> usize {
        let mut in_idx = 0;
        let mut total_bits = 0;
        mf.prepare(input.len());

        while in_idx < input.len() {
            let (processed, bits) = if self.compression_level < 2 {
                self.calculate_block_size_fast(mf, input, in_idx)
            } else if self.compression_level >= 10 {
                self.calculate_block_size_near_optimal(mf, input, in_idx)
            } else {
                self.calculate_block_size_greedy_lazy(mf, input, in_idx)
            };

            in_idx += processed;
            total_bits += bits;
        }

        mf.advance(input.len());
        total_bits.div_ceil(8)
    }

    fn calculate_block_size_fast<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        in_idx: usize,
    ) -> (usize, usize) {
        let processed = self.accumulate_greedy_frequencies(mf, input, in_idx, 0);
        self.load_static_huffman_codes();
        let bits = 3 + self.calculate_block_data_size();
        (processed, bits)
    }

    fn calculate_block_size_greedy_lazy<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        in_idx: usize,
    ) -> (usize, usize) {
        let lazy_depth = if self.compression_level >= 8 {
            2
        } else if self.compression_level >= 5 {
            1
        } else {
            0
        };
        let processed = self.decide_greedy_sequences(mf, input, in_idx, lazy_depth);

        make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &self.litlen_freqs,
            &mut self.litlen_lens,
            &mut self.litlen_codewords,
        );
        make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &self.offset_freqs,
            &mut self.offset_lens,
            &mut self.offset_codewords,
        );

        let bits = 3 + self.calculate_dynamic_header_size() + self.calculate_block_data_size();
        (processed, bits)
    }

    fn calculate_block_size_near_optimal<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        in_idx: usize,
    ) -> (usize, usize) {
        self.split_stats.reset();
        let mut p = in_idx;
        while p < input.len() {
            let block_len = p - in_idx;
            if self
                .split_stats
                .should_end_block(block_len, input.len() - p)
            {
                break;
            }
            let (len, offset) =
                mf.find_match(input, p, self.max_search_depth, self.nice_match_length);
            if len >= 3 {
                self.split_stats.observe_match(len, offset);
                p += len;
                for i in 1..len {
                    mf.skip_match(
                        input,
                        p - len + i,
                        self.max_search_depth,
                        self.nice_match_length,
                    );
                }
            } else {
                self.split_stats.observe_literal(input[p]);
                p += 1;
            }
        }
        let processed = p - in_idx;
        let block_input = &input[in_idx..in_idx + processed];

        self.sequences.clear();
        let mut cur_in_idx = 0;
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        mf.reset();

        while cur_in_idx < block_input.len() {
            let (len, offset) = mf.find_match(
                block_input,
                cur_in_idx,
                self.max_search_depth,
                self.nice_match_length,
            );
            if len >= 3 {
                self.sequences.push(Sequence {
                    litrunlen: 0,
                    length: len as u16,
                    offset: offset as u16,
                });
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[self.get_offset_slot(offset)] += 1;
                mf.skip_positions(
                    block_input,
                    cur_in_idx + 1,
                    len - 1,
                    self.max_search_depth,
                    self.nice_match_length,
                );
                cur_in_idx += len;
            } else {
                self.litlen_freqs[block_input[cur_in_idx] as usize] += 1;
                cur_in_idx += 1;
            }
        }
        self.litlen_freqs[256] += 1;

        make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &self.litlen_freqs,
            &mut self.litlen_lens,
            &mut self.litlen_codewords,
        );
        make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &self.offset_freqs,
            &mut self.offset_lens,
            &mut self.offset_codewords,
        );

        self.update_costs();

        self.dp_costs.clear();
        self.dp_costs.resize(processed + 1, 0x3FFFFFFF);
        self.dp_costs[0] = 0;

        self.dp_path.clear();
        if self.dp_path.capacity() < processed + 1 {
            self.dp_path.reserve(processed + 1 - self.dp_path.len());
        }
        unsafe {
            self.dp_path.set_len(processed + 1);
        }

        mf.reset();
        let mut pos = 0;
        while pos < processed {
            let cur_cost = self.dp_costs[pos];
            if cur_cost >= 0x3FFFFFFF {
                pos += 1;
                continue;
            }

            let lit_cost = self.litlen_lens[block_input[pos] as usize] as u32;
            if cur_cost + lit_cost < self.dp_costs[pos + 1] {
                self.dp_costs[pos + 1] = cur_cost + lit_cost;
                self.dp_path[pos + 1] = 1_u32;
            }

            mf.find_matches(
                block_input,
                pos,
                self.max_search_depth,
                self.nice_match_length,
                &mut self.matches,
            );
            let mut best_len = 0;
            for &(len, offset) in &self.matches {
                let len = len as usize;
                if pos + len > processed {
                    continue;
                }
                if len > best_len {
                    best_len = len;
                }
                let cost = self.get_match_cost(len, offset as usize);
                if cur_cost + cost < self.dp_costs[pos + len] {
                    self.dp_costs[pos + len] = cur_cost + cost;
                    self.dp_path[pos + len] = (len as u32) | ((offset as u32) << 16);
                }
            }

            if best_len >= self.nice_match_length {
                let skip = best_len;
                mf.skip_positions(
                    block_input,
                    pos + 1,
                    skip - 1,
                    self.max_search_depth,
                    self.nice_match_length,
                );
                pos += skip;
            } else {
                pos += 1;
            }
        }

        self.sequences.clear();
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        self.litlen_freqs[256] = 1;

        let mut pos = processed;
        let mut path_nodes = Vec::with_capacity(processed / 3);
        while pos > 0 {
            let packed = self.dp_path[pos];
            let length = (packed & 0xFFFF) as u16;
            let offset = (packed >> 16) as u16;
            path_nodes.push((length, offset));
            pos -= length as usize;
        }

        let mut litrunlen = 0;
        let mut cur_pos = 0;
        for &(length, offset) in path_nodes.iter().rev() {
            if length == 1 {
                self.litlen_freqs[block_input[cur_pos] as usize] += 1;
                litrunlen += 1;
                cur_pos += 1;
            } else {
                let off_slot = self.get_offset_slot(offset as usize);
                self.sequences
                    .push(Sequence::new(litrunlen, length, offset, off_slot as u8));
                self.litlen_freqs[257 + self.get_length_slot(length as usize)] += 1;
                self.offset_freqs[off_slot] += 1;
                litrunlen = 0;
                cur_pos += length as usize;
            }
        }
        self.sequences.push(Sequence::new(litrunlen, 0, 0, 0));

        make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &self.litlen_freqs,
            &mut self.litlen_lens,
            &mut self.litlen_codewords,
        );
        make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &self.offset_freqs,
            &mut self.offset_lens,
            &mut self.offset_codewords,
        );

        let bits = 3 + self.calculate_dynamic_header_size() + self.calculate_block_data_size();
        (processed, bits)
    }

    pub fn compress_to_size(&mut self, input: &[u8], final_block: bool) -> usize {
        if self.compression_level == 0 {
            let num_blocks = input.len() / 65535
                + if !input.len().is_multiple_of(65535) || (input.is_empty() && final_block) {
                    1
                } else {
                    0
                };
            return input.len() + num_blocks * 5;
        }

        let mut mf_enum = self.mf.take().unwrap();

        let res = match &mut mf_enum {
            MatchFinderEnum::Chain(mf) => self.compress_to_size_loop(mf, input, final_block),
            MatchFinderEnum::Table(mf) => self.compress_to_size_loop(mf, input, final_block),
            MatchFinderEnum::Bt(mf) => self.compress_to_size_loop(mf, input, final_block),
        };

        self.mf = Some(mf_enum);
        res
    }

    fn accumulate_greedy_frequencies<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        start_pos: usize,
        _lazy_depth: u32,
    ) -> usize {
        self.split_stats.reset();
        let mut in_idx = start_pos;
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);

        while in_idx < input.len() {
            let block_len = in_idx - start_pos;
            if self
                .split_stats
                .should_end_block(block_len, input.len() - in_idx)
            {
                break;
            }

            let (len, offset) =
                mf.find_match(input, in_idx, self.max_search_depth, self.nice_match_length);

            if len >= 3 {
                let off_slot = self.get_offset_slot(offset);
                self.split_stats.observe_match_with_slot(len, off_slot);
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[off_slot] += 1;
                mf.skip_positions(
                    input,
                    in_idx + 1,
                    len - 1,
                    self.max_search_depth,
                    self.nice_match_length,
                );
                in_idx += len;
            } else {
                self.split_stats.observe_literal(input[in_idx]);
                self.litlen_freqs[input[in_idx] as usize] += 1;
                in_idx += 1;
            }
        }
        self.litlen_freqs[256] += 1;
        in_idx - start_pos
    }

    fn calculate_block_data_size(&self) -> usize {
        let mut bits = 0;

        bits += self
            .litlen_freqs
            .iter()
            .zip(self.litlen_lens.iter())
            .map(|(&freq, &len)| (freq as usize) * (len as usize))
            .sum::<usize>();

        bits += self
            .offset_freqs
            .iter()
            .zip(self.offset_lens.iter())
            .map(|(&freq, &len)| (freq as usize) * (len as usize))
            .sum::<usize>();

        bits += self.litlen_freqs[257..286]
            .iter()
            .zip(LENGTH_EXTRA_BITS_TABLE.iter())
            .map(|(&freq, &extra)| (freq as usize) * (extra as usize))
            .sum::<usize>();

        bits += self.offset_freqs[0..30]
            .iter()
            .zip(OFFSET_EXTRA_BITS_TABLE.iter())
            .map(|(&freq, &extra)| (freq as usize) * (extra as usize))
            .sum::<usize>();

        bits
    }

    fn calculate_dynamic_header_size(&self) -> usize {
        let mut bits = 5 + 5 + 4;

        let mut num_litlen_syms = DEFLATE_NUM_LITLEN_SYMS;
        while num_litlen_syms > 257 && self.litlen_lens[num_litlen_syms - 1] == 0 {
            num_litlen_syms -= 1;
        }
        let mut num_offset_syms = DEFLATE_NUM_OFFSET_SYMS;
        while num_offset_syms > 1 && self.offset_lens[num_offset_syms - 1] == 0 {
            num_offset_syms -= 1;
        }

        let mut lens = Vec::with_capacity(num_litlen_syms + num_offset_syms);
        lens.extend_from_slice(&self.litlen_lens[..num_litlen_syms]);
        lens.extend_from_slice(&self.offset_lens[..num_offset_syms]);

        let mut precode_freqs = [0u32; 19];
        let mut i = 0;
        while i < lens.len() {
            let len = lens[i];
            let mut run = 1;
            while i + run < lens.len() && lens[i + run] == len {
                run += 1;
            }
            if len == 0 {
                while run >= 11 {
                    precode_freqs[18] += 1;
                    run -= min(run, 138);
                }
                if run >= 3 {
                    precode_freqs[17] += 1;
                    run -= min(run, 10);
                }
            } else if run >= 4 {
                precode_freqs[len as usize] += 1;
                run -= 1;
                while run >= 3 {
                    precode_freqs[16] += 1;
                    run -= min(run, 6);
                }
            }
            while run > 0 {
                precode_freqs[len as usize] += 1;
                run -= 1;
            }
            i += lens[i..].iter().take_while(|&&l| l == len).count();
        }

        let mut precode_lens = [0u8; 19];
        let mut precode_codewords = [0u32; 19];
        make_huffman_code(
            19,
            7,
            &precode_freqs,
            &mut precode_lens,
            &mut precode_codewords,
        );

        let permutation = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut num_precode_syms = 19;
        while num_precode_syms > 4 && precode_lens[permutation[num_precode_syms - 1]] == 0 {
            num_precode_syms -= 1;
        }

        bits += num_precode_syms * 3;

        for (sym, &freq) in precode_freqs.iter().enumerate() {
            if freq > 0 {
                let extra = if sym == 16 {
                    2
                } else if sym == 17 {
                    3
                } else if sym == 18 {
                    7
                } else {
                    0
                };
                bits += (freq as usize) * ((precode_lens[sym] as usize) + extra);
            }
        }

        bits
    }

    fn decide_greedy_sequences<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        start_pos: usize,
        lazy_depth: u32,
    ) -> usize {
        self.sequences.clear();
        self.split_stats.reset();
        let mut litrunlen = 0;
        let mut in_idx = start_pos;
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);

        while in_idx < input.len() {
            let block_len = in_idx - start_pos;
            if self
                .split_stats
                .should_end_block(block_len, input.len() - in_idx)
            {
                break;
            }

            let (mut len, mut offset) =
                mf.find_match(input, in_idx, self.max_search_depth, self.nice_match_length);

            if len >= 3 {
                let mut skipped = 0;
                // Optimization: Skip lazy check if the current match is "nice" enough.
                if lazy_depth >= 1 && in_idx + 1 < input.len() && len < self.nice_match_length {
                    let (next_len, next_offset) = mf.find_match(
                        input,
                        in_idx + 1,
                        self.max_search_depth,
                        self.nice_match_length,
                    );
                    if next_len > len {
                        if lazy_depth >= 2 && in_idx + 2 < input.len() {
                            let (next2_len, next2_offset) = mf.find_match(
                                input,
                                in_idx + 2,
                                self.max_search_depth,
                                self.nice_match_length,
                            );
                            if next2_len > next_len {
                                self.split_stats.observe_literal(input[in_idx]);
                                self.litlen_freqs[input[in_idx] as usize] += 1;
                                litrunlen += 1;
                                in_idx += 1;

                                self.split_stats.observe_literal(input[in_idx]);
                                self.litlen_freqs[input[in_idx] as usize] += 1;
                                litrunlen += 1;
                                in_idx += 1;

                                len = next2_len;
                                offset = next2_offset;
                            } else {
                                self.split_stats.observe_literal(input[in_idx]);
                                self.litlen_freqs[input[in_idx] as usize] += 1;
                                litrunlen += 1;
                                in_idx += 1;

                                len = next_len;
                                offset = next_offset;
                                skipped = 1;
                            }
                        } else {
                            self.split_stats.observe_literal(input[in_idx]);
                            self.litlen_freqs[input[in_idx] as usize] += 1;
                            litrunlen += 1;
                            in_idx += 1;

                            len = next_len;
                            offset = next_offset;
                        }
                    } else {
                        skipped = 1;
                    }
                }

                let off_slot = self.get_offset_slot(offset);
                self.sequences.push(Sequence::new(
                    litrunlen,
                    len as u16,
                    offset as u16,
                    off_slot as u8,
                ));
                self.split_stats.observe_match_with_slot(len, off_slot);
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[off_slot] += 1;
                litrunlen = 0;
                if len - 1 > skipped {
                    mf.skip_positions(
                        input,
                        in_idx + 1 + skipped,
                        len - 1 - skipped,
                        self.max_search_depth,
                        self.nice_match_length,
                    );
                }
                in_idx += len;
            } else {
                self.split_stats.observe_literal(input[in_idx]);
                self.litlen_freqs[input[in_idx] as usize] += 1;
                litrunlen += 1;
                in_idx += 1;
            }
        }
        self.sequences.push(Sequence::new(litrunlen, 0, 0, 0));
        self.litlen_freqs[256] += 1;
        in_idx - start_pos
    }

    fn write_dynamic_block_with_sequences(
        &self,
        input: &[u8],
        start_pos: usize,
        bs: &mut Bitstream,
        is_final: bool,
    ) -> bool {
        if !bs.write_bits(if is_final { 1 } else { 0 }, 1) {
            return false;
        }
        if !bs.write_bits(2, 2) {
            return false;
        }
        if !self.write_dynamic_huffman_header_impl(bs) {
            return false;
        }
        let mut in_pos = start_pos;
        for seq in &self.sequences {
            if seq.litrunlen > 0 {
                // Check if we have enough space to write all literals in this run without bounds checks.
                // Heuristic: Each literal expands to at most 15 bits. Writing flushes every 32 bits (4 bytes).
                // Max expansion is roughly 15/8 ~= 1.875 bytes per literal. 2 bytes is a safe upper bound.
                // We add 16 bytes margin for the last flush (8 bytes) and safety.
                if bs.out_idx + 16 + (seq.litrunlen as usize * 2) < bs.output.len() {
                    let mut lit_remain = seq.litrunlen as usize;
                    while lit_remain >= 4 {
                        // SAFETY: We verified sufficient buffer space above.
                        // `write_literals_2` writes at most 30 bits and may flush 4 bytes.
                        // We do this twice, so max 60 bits + flush overhead.
                        // The loop precondition checks for space.
                        unsafe {
                            self.write_literals_2(bs, input[in_pos], input[in_pos + 1]);
                            self.write_literals_2(bs, input[in_pos + 2], input[in_pos + 3]);
                        }
                        in_pos += 4;
                        lit_remain -= 4;
                    }
                    while lit_remain >= 2 {
                        // SAFETY: We verified sufficient buffer space above.
                        // `write_literals_2` writes at most 30 bits and may flush 4 bytes.
                        unsafe { self.write_literals_2(bs, input[in_pos], input[in_pos + 1]) };
                        in_pos += 2;
                        lit_remain -= 2;
                    }
                    if lit_remain > 0 {
                        // SAFETY: We verified sufficient buffer space above.
                        // `write_literal_fast` writes at most 15 bits and may flush 4 bytes.
                        // The buffer margin guarantees `out_idx + 8 <= output.len()` for the `write_bits_unchecked_fast` call.
                        unsafe { self.write_literal_fast(bs, input[in_pos]) };
                        in_pos += 1;
                    }
                } else {
                    for _ in 0..seq.litrunlen {
                        if !self.write_literal(bs, input[in_pos]) {
                            return false;
                        }
                        in_pos += 1;
                    }
                }
            }
            let len = seq.len();
            if len >= 3 {
                let offset = seq.offset as usize;
                let off_slot = seq.off_slot();
                if bs.out_idx + 16 < bs.output.len() {
                    unsafe {
                        self.write_match_fast(bs, len as usize, offset, off_slot);
                    }
                } else if !self.write_match(bs, len as usize, offset, off_slot) {
                    return false;
                }
                in_pos += len as usize;
            }
        }
        if !self.write_sym(bs, 256) {
            return false;
        }
        true
    }

    fn compress_uncompressed(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        flush_mode: FlushMode,
    ) -> (CompressResult, usize, u32) {
        let mut bs = Bitstream::new(output);
        let mut in_idx = 0;
        while in_idx < input.len() {
            let bfinal = if in_idx + 65535 >= input.len() && flush_mode == FlushMode::Finish {
                1
            } else {
                0
            };
            let block_len = min(65535, input.len() - in_idx);
            if !bs.write_bits(bfinal, 1) || !bs.write_bits(0, 2) {
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            let (res, _) = bs.flush();
            if !res {
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            if bs.out_idx + 4 + block_len > bs.output.len() {
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            let len = block_len as u16;
            let nlen = !len;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    len.to_le_bytes().as_ptr(),
                    bs.output.as_mut_ptr().add(bs.out_idx) as *mut u8,
                    2,
                );
                std::ptr::copy_nonoverlapping(
                    nlen.to_le_bytes().as_ptr(),
                    bs.output.as_mut_ptr().add(bs.out_idx + 2) as *mut u8,
                    2,
                );
            }
            bs.out_idx += 4;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr().add(in_idx),
                    bs.output.as_mut_ptr().add(bs.out_idx) as *mut u8,
                    block_len,
                );
            }
            bs.out_idx += block_len;
            in_idx += block_len;
        }
        if flush_mode == FlushMode::Sync {
            if bs.out_idx + 5 > bs.output.len() {
                return (CompressResult::InsufficientSpace, 0, 0);
            }
            bs.output[bs.out_idx].write(0);
            bs.out_idx += 1;
            bs.output[bs.out_idx].write(0);
            bs.output[bs.out_idx + 1].write(0);
            bs.output[bs.out_idx + 2].write(0xFF);
            bs.output[bs.out_idx + 3].write(0xFF);
            bs.out_idx += 4;
        }

        (CompressResult::Success, bs.out_idx, 0)
    }

    fn compress_greedy_block<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        start_pos: usize,
        bs: &mut Bitstream,
        lazy_depth: u32,
        final_block: bool,
    ) -> usize {
        if self.compression_level >= 2 {
            let processed = self.decide_greedy_sequences(mf, input, start_pos, lazy_depth);
            let is_final = (start_pos + processed >= input.len()) && final_block;
            make_huffman_code(
                DEFLATE_NUM_LITLEN_SYMS,
                MAX_LITLEN_CODEWORD_LEN,
                &self.litlen_freqs,
                &mut self.litlen_lens,
                &mut self.litlen_codewords,
            );
            make_huffman_code(
                DEFLATE_NUM_OFFSET_SYMS,
                MAX_OFFSET_CODEWORD_LEN,
                &self.offset_freqs,
                &mut self.offset_lens,
                &mut self.offset_codewords,
            );
            self.update_huffman_tables();
            if !self.write_dynamic_block_with_sequences(input, start_pos, bs, is_final) {
                return 0;
            }
            return processed;
        }

        self.load_static_huffman_codes();
        self.sequences.clear();
        let mut litrunlen = 0;
        let mut in_idx = start_pos;
        self.split_stats.reset();

        if input.len() <= 65536 {
            while in_idx < input.len() {
                let (len, offset) =
                    mf.find_match(input, in_idx, self.max_search_depth, self.nice_match_length);
                if len >= 3 {
                    self.sequences.push(Sequence::new(
                        litrunlen,
                        len as u16,
                        offset as u16,
                        self.get_offset_slot(offset) as u8,
                    ));
                    litrunlen = 0;
                    mf.skip_positions(
                        input,
                        in_idx + 1,
                        len - 1,
                        self.max_search_depth,
                        self.nice_match_length,
                    );
                    in_idx += len;
                } else {
                    litrunlen += 1;
                    in_idx += 1;
                }
            }
        } else {
            while in_idx < input.len() {
                let block_len = in_idx - start_pos;
                if self
                    .split_stats
                    .should_end_block(block_len, input.len() - in_idx)
                {
                    break;
                }
                let (len, offset) =
                    mf.find_match(input, in_idx, self.max_search_depth, self.nice_match_length);
                if len >= 3 {
                    self.split_stats.observe_match(len, offset);
                    self.sequences.push(Sequence::new(
                        litrunlen,
                        len as u16,
                        offset as u16,
                        self.get_offset_slot(offset) as u8,
                    ));
                    litrunlen = 0;
                    mf.skip_positions(
                        input,
                        in_idx + 1,
                        len - 1,
                        self.max_search_depth,
                        self.nice_match_length,
                    );
                    in_idx += len;
                } else {
                    self.split_stats.observe_literal(input[in_idx]);
                    litrunlen += 1;
                    in_idx += 1;
                }
            }
        }
        self.sequences.push(Sequence::new(litrunlen, 0, 0, 0));

        let processed = in_idx - start_pos;
        let is_final = (start_pos + processed >= input.len()) && final_block;
        if !bs.write_bits(if is_final { 1 } else { 0 }, 1) {
            return 0;
        }
        if !bs.write_bits(1, 2) {
            return 0;
        }

        let mut in_pos = start_pos;
        for seq in &self.sequences {
            if seq.litrunlen > 0 {
                if bs.out_idx + 16 + (seq.litrunlen as usize * 2) < bs.output.len() {
                    let mut lit_remain = seq.litrunlen as usize;
                    while lit_remain >= 4 {
                        unsafe {
                            self.write_literals_2(bs, input[in_pos], input[in_pos + 1]);
                            self.write_literals_2(bs, input[in_pos + 2], input[in_pos + 3]);
                        }
                        in_pos += 4;
                        lit_remain -= 4;
                    }
                    while lit_remain >= 2 {
                        unsafe { self.write_literals_2(bs, input[in_pos], input[in_pos + 1]) };
                        in_pos += 2;
                        lit_remain -= 2;
                    }
                    if lit_remain > 0 {
                        unsafe { self.write_literal_fast(bs, input[in_pos]) };
                        in_pos += 1;
                    }
                } else {
                    for _ in 0..seq.litrunlen {
                        if !self.write_literal(bs, input[in_pos]) {
                            return 0;
                        }
                        in_pos += 1;
                    }
                }
            }
            let len = seq.len();
            if len >= 3 {
                let offset = seq.offset as usize;
                let off_slot = seq.off_slot();
                if bs.out_idx + 16 < bs.output.len() {
                    unsafe {
                        self.write_match_fast(bs, len as usize, offset, off_slot);
                    }
                } else if !self.write_match(bs, len as usize, offset, off_slot) {
                    return 0;
                }
                in_pos += len as usize;
            }
        }
        if !self.write_sym(bs, 256) {
            return 0;
        }
        processed
    }

    fn compress_near_optimal_block<T: MatchFinderTrait>(
        &mut self,
        mf: &mut T,
        input: &[u8],
        start_pos: usize,
        bs: &mut Bitstream,
        final_block: bool,
    ) -> usize {
        self.split_stats.reset();
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        let mut in_idx = start_pos;

        while in_idx < input.len() {
            let block_len = in_idx - start_pos;
            if self
                .split_stats
                .should_end_block(block_len, input.len() - in_idx)
            {
                break;
            }
            let (len, offset) =
                mf.find_match(input, in_idx, self.max_search_depth, self.nice_match_length);
            if len >= 3 {
                let off_slot = self.get_offset_slot(offset);
                self.split_stats.observe_match_with_slot(len, off_slot);
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[off_slot] += 1;
                mf.skip_positions(
                    input,
                    in_idx + 1,
                    len - 1,
                    self.max_search_depth,
                    self.nice_match_length,
                );
                in_idx += len;
            } else {
                self.split_stats.observe_literal(input[in_idx]);
                self.litlen_freqs[input[in_idx] as usize] += 1;
                in_idx += 1;
            }
        }

        let processed = in_idx - start_pos;
        let block_input = &input[start_pos..start_pos + processed];
        let is_final = (start_pos + processed >= input.len()) && final_block;

        self.sequences.clear();
        self.litlen_freqs[256] += 1;

        make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &self.litlen_freqs,
            &mut self.litlen_lens,
            &mut self.litlen_codewords,
        );
        make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &self.offset_freqs,
            &mut self.offset_lens,
            &mut self.offset_codewords,
        );

        self.update_costs();

        self.dp_costs.clear();
        self.dp_costs.resize(processed + 1, 0x3FFFFFFF);
        self.dp_costs[0] = 0;

        self.dp_path.clear();
        if self.dp_path.capacity() < processed + 1 {
            self.dp_path.reserve(processed + 1 - self.dp_path.len());
        }
        unsafe {
            self.dp_path.set_len(processed + 1);
        }

        mf.reset();
        let mut pos = 0;
        while pos < processed {
            let cur_cost = self.dp_costs[pos];
            if cur_cost >= 0x3FFFFFFF {
                pos += 1;
                continue;
            }

            let lit_cost = self.litlen_lens[block_input[pos] as usize] as u32;
            if cur_cost + lit_cost < self.dp_costs[pos + 1] {
                self.dp_costs[pos + 1] = cur_cost + lit_cost;
                self.dp_path[pos + 1] = 1_u32;
            }

            mf.find_matches(
                block_input,
                pos,
                self.max_search_depth,
                self.nice_match_length,
                &mut self.matches,
            );
            let mut best_len = 0;
            for &(len, offset) in &self.matches {
                let len = len as usize;
                if pos + len > processed {
                    continue;
                }
                if len > best_len {
                    best_len = len;
                }
                let cost = self.get_match_cost(len, offset as usize);
                if cur_cost + cost < self.dp_costs[pos + len] {
                    self.dp_costs[pos + len] = cur_cost + cost;
                    self.dp_path[pos + len] = (len as u32) | ((offset as u32) << 16);
                }
            }

            if best_len >= self.nice_match_length {
                let skip = best_len;
                mf.skip_positions(
                    block_input,
                    pos + 1,
                    skip - 1,
                    self.max_search_depth,
                    self.nice_match_length,
                );
                pos += skip;
            } else {
                pos += 1;
            }
        }

        self.sequences.clear();
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        self.litlen_freqs[256] = 1;

        let mut pos = processed;
        let mut path_nodes = Vec::with_capacity(processed / 3);
        while pos > 0 {
            let packed = self.dp_path[pos];
            let length = (packed & 0xFFFF) as u16;
            let offset = (packed >> 16) as u16;
            path_nodes.push((length, offset));
            pos -= length as usize;
        }

        let mut litrunlen = 0;
        let mut cur_pos = 0;
        for &(length, offset) in path_nodes.iter().rev() {
            if length == 1 {
                self.litlen_freqs[block_input[cur_pos] as usize] += 1;
                litrunlen += 1;
                cur_pos += 1;
            } else {
                let off_slot = self.get_offset_slot(offset as usize);
                self.sequences
                    .push(Sequence::new(litrunlen, length, offset, off_slot as u8));
                self.litlen_freqs[257 + self.get_length_slot(length as usize)] += 1;
                self.offset_freqs[off_slot] += 1;
                litrunlen = 0;
                cur_pos += length as usize;
            }
        }
        self.sequences.push(Sequence::new(litrunlen, 0, 0, 0));

        make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &self.litlen_freqs,
            &mut self.litlen_lens,
            &mut self.litlen_codewords,
        );
        make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &self.offset_freqs,
            &mut self.offset_lens,
            &mut self.offset_codewords,
        );
        self.update_huffman_tables();

        if !self.write_dynamic_block_with_sequences(input, start_pos, bs, is_final) {
            return 0;
        }
        processed
    }

    fn write_dynamic_huffman_header_impl(&self, bs: &mut Bitstream) -> bool {
        let mut num_litlen_syms = DEFLATE_NUM_LITLEN_SYMS;
        while num_litlen_syms > 257 && self.litlen_lens[num_litlen_syms - 1] == 0 {
            num_litlen_syms -= 1;
        }
        let mut num_offset_syms = DEFLATE_NUM_OFFSET_SYMS;
        while num_offset_syms > 1 && self.offset_lens[num_offset_syms - 1] == 0 {
            num_offset_syms -= 1;
        }
        if !bs.write_bits((num_litlen_syms - 257) as u32, 5) {
            return false;
        }
        if !bs.write_bits((num_offset_syms - 1) as u32, 5) {
            return false;
        }
        let mut lens = [0u8; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS];
        let lens_len = num_litlen_syms + num_offset_syms;
        lens[..num_litlen_syms].copy_from_slice(&self.litlen_lens[..num_litlen_syms]);
        lens[num_litlen_syms..lens_len].copy_from_slice(&self.offset_lens[..num_offset_syms]);

        let mut precode_freqs = [0u32; 19];
        let mut precode_items = [0u16; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS];
        let mut num_precode_items = 0;
        let mut i = 0;
        while i < lens_len {
            let len = lens[i];
            let total_run = lens[i..lens_len].iter().take_while(|&&l| l == len).count();
            let mut run = total_run;

            if len == 0 {
                while run >= 11 {
                    let c = min(run, 138);
                    precode_items[num_precode_items] = (18 << 8) | ((c - 11) as u16);
                    num_precode_items += 1;
                    precode_freqs[18] += 1;
                    run -= c;
                }
                if run >= 3 {
                    let c = min(run, 10);
                    precode_items[num_precode_items] = (17 << 8) | ((c - 3) as u16);
                    num_precode_items += 1;
                    precode_freqs[17] += 1;
                    run -= c;
                }
            } else if run >= 4 {
                precode_items[num_precode_items] = (len as u16) << 8;
                num_precode_items += 1;
                precode_freqs[len as usize] += 1;
                run -= 1;
                while run >= 3 {
                    let c = min(run, 6);
                    precode_items[num_precode_items] = (16 << 8) | ((c - 3) as u16);
                    num_precode_items += 1;
                    precode_freqs[16] += 1;
                    run -= c;
                }
            }
            while run > 0 {
                precode_items[num_precode_items] = (len as u16) << 8;
                num_precode_items += 1;
                precode_freqs[len as usize] += 1;
                run -= 1;
            }
            i += total_run;
        }
        let mut precode_lens = [0u8; 19];
        let mut precode_codewords = [0u32; 19];
        make_huffman_code(
            19,
            7,
            &precode_freqs,
            &mut precode_lens,
            &mut precode_codewords,
        );
        let mut num_precode_syms = 19;
        let permutation = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        while num_precode_syms > 4 && precode_lens[permutation[num_precode_syms - 1]] == 0 {
            num_precode_syms -= 1;
        }
        if !bs.write_bits((num_precode_syms - 4) as u32, 4) {
            return false;
        }
        for j in 0..num_precode_syms {
            if !bs.write_bits(precode_lens[permutation[j]] as u32, 3) {
                return false;
            }
        }
        for &item in precode_items[..num_precode_items].iter() {
            let sym = (item >> 8) as usize;
            let extra = (item & 0xFF) as u32;
            if !bs.write_bits(precode_codewords[sym], precode_lens[sym] as u32) {
                return false;
            }
            if sym == 16 {
                if !bs.write_bits(extra, 2) {
                    return false;
                }
            } else if sym == 17 {
                if !bs.write_bits(extra, 3) {
                    return false;
                }
            } else if sym == 18
                && !bs.write_bits(extra, 7) {
                    return false;
                }
        }
        true
    }

    fn load_static_huffman_codes(&mut self) {
        static STATIC_TABLES: OnceLock<StaticTables> = OnceLock::new();
        let tables = STATIC_TABLES.get_or_init(compute_static_tables);

        self.litlen_lens.copy_from_slice(&tables.litlen_lens);
        self.offset_lens.copy_from_slice(&tables.offset_lens);
        self.litlen_table.copy_from_slice(&tables.litlen_table);
        self.offset_table.copy_from_slice(&tables.offset_table);
        self.match_len_table
            .copy_from_slice(&tables.match_len_table);
    }

    #[inline(always)]
    unsafe fn write_literal_fast(&self, bs: &mut Bitstream, lit: u8) {
        let sym = lit as usize;
        let entry = *self.litlen_table.get_unchecked(sym);
        bs.write_bits_unchecked_fast(entry as u32, (entry >> 32) as u32)
    }

    #[inline(always)]
    unsafe fn write_literals_2(&self, bs: &mut Bitstream, lit1: u8, lit2: u8) {
        let entry1 = *self.litlen_table.get_unchecked(lit1 as usize);
        let entry2 = *self.litlen_table.get_unchecked(lit2 as usize);

        let code1 = entry1 as u32;
        let len1 = (entry1 >> 32) as u32;

        let code2 = entry2 as u32;
        let len2 = (entry2 >> 32) as u32;

        bs.write_bits_unchecked_fast(code1 | (code2 << len1), len1 + len2);
    }

    fn write_literal(&self, bs: &mut Bitstream, lit: u8) -> bool {
        let sym = lit as usize;
        let entry = unsafe { *self.litlen_table.get_unchecked(sym) };
        unsafe { bs.write_bits_unchecked(entry as u32, (entry >> 32) as u32) }
    }
    fn write_sym(&self, bs: &mut Bitstream, sym: usize) -> bool {
        let entry = unsafe { *self.litlen_table.get_unchecked(sym) };
        unsafe { bs.write_bits_unchecked(entry as u32, (entry >> 32) as u32) }
    }

    #[inline(always)]
    unsafe fn write_match_fast(
        &self,
        bs: &mut Bitstream,
        len: usize,
        offset: usize,
        off_slot: usize,
    ) {
        let entry = *self.match_len_table.get_unchecked(len);
        let code = entry as u16 as u32;
        let huff_len = (entry >> 16) as u8 as u32;
        let extra_bits = (entry >> 24) as u8 as u32;
        let base = (entry >> 32) as u16 as u32;

        let len_val = code | ((len as u32).wrapping_sub(base) << huff_len);
        let len_len = huff_len + extra_bits;

        let entry = *self.offset_table.get_unchecked(off_slot);
        let off_code = entry as u32;
        let off_len = (entry >> 32) as u8 as u32;
        let extra_bits = (entry >> 40) as u8 as u32;
        let base = (entry >> 48) as u16 as u32;

        let off_val = off_code | ((offset as u32).wrapping_sub(base) << off_len);
        let off_len_total = off_len + extra_bits;

        if len_len + off_len_total <= 32 {
            bs.write_bits_unchecked_fast(len_val | (off_val << len_len), len_len + off_len_total);
        } else {
            bs.write_bits_unchecked_fast(len_val, len_len);
            bs.write_bits_unchecked_fast(off_val, off_len_total);
        }
    }

    fn write_match(&self, bs: &mut Bitstream, len: usize, offset: usize, off_slot: usize) -> bool {
        let entry = unsafe { *self.match_len_table.get_unchecked(len) };
        let code = entry as u16 as u32;
        let huff_len = (entry >> 16) as u8 as u32;
        let extra_bits = (entry >> 24) as u8 as u32;
        let base = (entry >> 32) as u16 as u32;

        let len_val = code | ((len as u32).wrapping_sub(base) << huff_len);
        let len_len = huff_len + extra_bits;

        if !unsafe { bs.write_bits_upto_32(len_val, len_len) } {
            return false;
        }

        let entry = unsafe { *self.offset_table.get_unchecked(off_slot) };
        let off_code = entry as u32;
        let off_len = (entry >> 32) as u8 as u32;
        let extra_bits = (entry >> 40) as u8 as u32;
        let base = (entry >> 48) as u16 as u32;

        let off_val = off_code | ((offset as u32).wrapping_sub(base) << off_len);
        let off_len_total = off_len + extra_bits;

        unsafe { bs.write_bits_upto_32(off_val, off_len_total) }
    }

    #[inline(always)]
    fn get_length_slot(&self, len: usize) -> usize {
        debug_assert!(len < LENGTH_WRITE_TABLE.len());
        // SAFETY: The match finder guarantees len <= DEFLATE_MAX_MATCH_LEN (258),
        // which is within the table bounds (260).
        unsafe { (*LENGTH_WRITE_TABLE.get_unchecked(len) >> 24) as usize }
    }

    #[inline(always)]
    fn get_offset_slot(&self, offset: usize) -> usize {
        debug_assert!(offset <= 32768);
        if offset <= 256 {
            // SAFETY: table has size 512, offset <= 256 is within bounds.
            unsafe { *OFFSET_SLOT_TABLE_512.get_unchecked(offset) as usize }
        } else {
            // SAFETY: offset <= 32768 implies (offset - 1) >> 7 <= 255.
            // Index <= 256 + 255 = 511. Within bounds.
            unsafe { *OFFSET_SLOT_TABLE_512.get_unchecked(256 + ((offset - 1) >> 7)) as usize }
        }
    }

    fn update_costs(&mut self) {
        for len in 3..=DEFLATE_MAX_MATCH_LEN {
            let len_info = unsafe { *LENGTH_WRITE_TABLE.get_unchecked(len) };
            let len_slot = (len_info >> 24) as usize;
            let len_extra_bits = (len_info >> 16) & 0xFF ;

            let len_cost =
                unsafe { *self.litlen_lens.get_unchecked(257 + len_slot) } as u32 + len_extra_bits;
            unsafe { *self.length_costs.get_unchecked_mut(len) = len_cost };
        }
        for slot in 0..30 {
            let extra_bits = unsafe { *OFFSET_EXTRA_BITS_TABLE.get_unchecked(slot) } as u32;
            let off_cost = unsafe { *self.offset_lens.get_unchecked(slot) } as u32 + extra_bits;
            unsafe { *self.offset_slot_costs.get_unchecked_mut(slot) = off_cost };
        }
    }

    #[inline(always)]
    fn get_match_cost(&self, len: usize, offset: usize) -> u32 {
        unsafe {
            let len_cost = *self.length_costs.get_unchecked(len);
            let off_slot = self.get_offset_slot(offset);
            let off_cost = *self.offset_slot_costs.get_unchecked(off_slot);
            len_cost + off_cost
        }
    }

    pub fn deflate_compress_bound(size: usize) -> usize {
        size.saturating_add((size / 65535 + 1) * 5 + 10)
    }

    pub fn zlib_compress_bound(size: usize) -> usize {
        ZLIB_MIN_OVERHEAD.saturating_add(Self::deflate_compress_bound(size))
    }

    pub fn gzip_compress_bound(size: usize) -> usize {
        GZIP_MIN_OVERHEAD.saturating_add(Self::deflate_compress_bound(size))
    }

    pub fn compress_zlib(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> (CompressResult, usize) {
        if output.len() < ZLIB_MIN_OVERHEAD {
            return (CompressResult::InsufficientSpace, 0);
        }
        let mut out_idx = 0;
        let mut hdr = (ZLIB_CM_DEFLATE as u16) << 8;
        hdr |= (ZLIB_CINFO_32K_WINDOW as u16) << 12;
        let level_hint = if self.compression_level < 2 {
            ZLIB_FASTEST_COMPRESSION
        } else if self.compression_level < 6 {
            ZLIB_FAST_COMPRESSION
        } else if self.compression_level < 8 {
            ZLIB_DEFAULT_COMPRESSION
        } else {
            ZLIB_SLOWEST_COMPRESSION
        };
        hdr |= (level_hint as u16) << 6;
        hdr |= 31 - (hdr % 31);
        unsafe {
            std::ptr::copy_nonoverlapping(
                hdr.to_be_bytes().as_ptr(),
                output.as_mut_ptr().add(0) as *mut u8,
                2,
            );
        }
        out_idx += 2;
        let out_len = output.len();
        let (res, deflate_size, _) = self.compress(
            input,
            &mut output[out_idx..out_len - ZLIB_FOOTER_SIZE],
            FlushMode::Finish,
        );
        if res != CompressResult::Success {
            return (res, 0);
        }
        out_idx += deflate_size;
        let adler = crate::adler32::adler32(1, input);
        unsafe {
            std::ptr::copy_nonoverlapping(
                adler.to_be_bytes().as_ptr(),
                output.as_mut_ptr().add(out_idx) as *mut u8,
                4,
            );
        }
        out_idx += 4;
        (CompressResult::Success, out_idx)
    }

    pub fn compress_gzip(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> (CompressResult, usize) {
        if output.len() < GZIP_MIN_OVERHEAD {
            return (CompressResult::InsufficientSpace, 0);
        }
        let mut out_idx = 0;
        output[0].write(GZIP_ID1);
        output[1].write(GZIP_ID2);
        output[2].write(GZIP_CM_DEFLATE);
        output[3].write(0);
        unsafe {
            std::ptr::copy_nonoverlapping(
                GZIP_MTIME_UNAVAILABLE.to_le_bytes().as_ptr(),
                output.as_mut_ptr().add(4) as *mut u8,
                4,
            );
        }
        let mut xfl = 0u8;
        if self.compression_level < 2 {
            xfl |= GZIP_XFL_FASTEST_COMPRESSION;
        } else if self.compression_level >= 8 {
            xfl |= GZIP_XFL_SLOWEST_COMPRESSION;
        }
        output[8].write(xfl);
        output[9].write(GZIP_OS_UNKNOWN);
        out_idx += 10;
        let out_len = output.len();
        let (res, deflate_size, _) = self.compress(
            input,
            &mut output[out_idx..out_len - GZIP_FOOTER_SIZE],
            FlushMode::Finish,
        );
        if res != CompressResult::Success {
            return (res, 0);
        }
        out_idx += deflate_size;
        let crc = crate::crc32::crc32(0, input);
        unsafe {
            std::ptr::copy_nonoverlapping(
                crc.to_le_bytes().as_ptr(),
                output.as_mut_ptr().add(out_idx) as *mut u8,
                4,
            );
        }
        out_idx += 4;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (input.len() as u32).to_le_bytes().as_ptr(),
                output.as_mut_ptr().add(out_idx) as *mut u8,
                4,
            );
        }
        out_idx += 4;
        (CompressResult::Success, out_idx)
    }
}
