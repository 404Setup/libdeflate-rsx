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

const OFFSET_SLOT_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
];

pub const MAX_LITLEN_CODEWORD_LEN: usize = 14;
pub const MAX_OFFSET_CODEWORD_LEN: usize = 15;
pub const MAX_PRE_CODEWORD_LEN: usize = 7;

#[derive(Debug, PartialEq, Eq)]
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

        let off_idx = NUM_LITERAL_OBSERVATION_TYPES
            + NUM_MATCH_OBSERVATION_TYPES
            + if offset < 256 {
                0
            } else if offset < 4096 {
                1
            } else if offset < 32768 {
                2
            } else {
                3
            };
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
        if input_remaining <= MIN_BLOCK_LENGTH {
            return false;
        }
        if block_length >= SOFT_MAX_BLOCK_LENGTH {
            return true;
        }
        if self.num_new_observations >= NUM_OBSERVATIONS_PER_BLOCK_CHECK
            && block_length >= MIN_BLOCK_LENGTH
        {
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
    dp_nodes: Vec<DPNode>,
    split_stats: BlockSplitStats,
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
            dp_nodes: if level >= 10 {
                Vec::with_capacity(300000)
            } else {
                Vec::new()
            },
            split_stats: BlockSplitStats::new(),
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
                            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
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
        (total_bits + 7) / 8
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
            let (len, offset) = mf.find_match(input, p, self.max_search_depth);
            if len >= 3 {
                self.split_stats.observe_match(len, offset);
                p += len;
                for i in 1..len {
                    mf.skip_match(input, p - len + i, self.max_search_depth);
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
            let (len, offset) = mf.find_match(block_input, cur_in_idx, self.max_search_depth);
            if len >= 3 {
                self.sequences.push(Sequence {
                    litrunlen: 0,
                    length: len as u16,
                    offset: offset as u16,
                });
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[self.get_offset_slot(offset)] += 1;
                mf.skip_positions(block_input, cur_in_idx + 1, len - 1, self.max_search_depth);
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

        self.dp_nodes.clear();
        self.dp_nodes.resize(
            processed + 1,
            DPNode {
                cost: 0x3FFFFFFF,
                length: 0,
                offset: 0,
            },
        );
        self.dp_nodes[0].cost = 0;

        mf.reset();
        let mut matches = Vec::new();
        for pos in 0..processed {
            let cur_cost = self.dp_nodes[pos].cost;
            if cur_cost >= 0x3FFFFFFF {
                continue;
            }

            let lit_cost = self.litlen_lens[block_input[pos] as usize] as u32;
            if cur_cost + lit_cost < self.dp_nodes[pos + 1].cost {
                self.dp_nodes[pos + 1] = DPNode {
                    cost: cur_cost + lit_cost,
                    length: 1,
                    offset: 0,
                };
            }

            mf.find_matches(block_input, pos, self.max_search_depth, &mut matches);
            for &(len, offset) in &matches {
                let len = len as usize;
                if pos + len > processed {
                    continue;
                }
                let cost = self.get_match_cost(len, offset as usize);
                if cur_cost + cost < self.dp_nodes[pos + len].cost {
                    self.dp_nodes[pos + len] = DPNode {
                        cost: cur_cost + cost,
                        length: len as u16,
                        offset,
                    };
                }
            }
        }

        self.sequences.clear();
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        self.litlen_freqs[256] = 1;

        let mut pos = processed;
        while pos > 0 {
            let node = self.dp_nodes[pos];
            if node.length == 1 {
                self.litlen_freqs[block_input[pos - 1] as usize] += 1;
            } else {
                self.litlen_freqs[257 + self.get_length_slot(node.length as usize)] += 1;
                self.offset_freqs[self.get_offset_slot(node.offset as usize)] += 1;
            }
            pos -= node.length as usize;
        }

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
                + if input.len() % 65535 != 0 || (input.len() == 0 && final_block) {
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
        lazy_depth: u32,
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

            let (len, offset) = mf.find_match(input, in_idx, self.max_search_depth);

            if len >= 3 {
                if lazy_depth >= 1 && in_idx + 1 < input.len() {
                    let (_next_len, _next_offset) =
                        mf.find_match(input, in_idx + 1, self.max_search_depth);
                }

                self.split_stats.observe_match(len, offset);
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[self.get_offset_slot(offset)] += 1;
                mf.skip_positions(input, in_idx + 1, len - 1, self.max_search_depth);
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
        for i in 0..288 {
            if self.litlen_freqs[i] > 0 {
                bits += (self.litlen_freqs[i] as usize) * (self.litlen_lens[i] as usize);
            }
        }
        for i in 0..32 {
            if self.offset_freqs[i] > 0 {
                bits += (self.offset_freqs[i] as usize) * (self.offset_lens[i] as usize);
            }
        }
        for i in 257..286 {
            if self.litlen_freqs[i] > 0 {
                bits += (self.litlen_freqs[i] as usize) * (self.get_length_extra_bits(i - 257));
            }
        }
        for i in 0..30 {
            if self.offset_freqs[i] > 0 {
                bits += (self.offset_freqs[i] as usize) * (self.get_offset_extra_bits(i));
            }
        }
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
            } else {
                if run >= 4 {
                    precode_freqs[len as usize] += 1;
                    run -= 1;
                    while run >= 3 {
                        precode_freqs[16] += 1;
                        run -= min(run, 6);
                    }
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

            let (mut len, mut offset) = mf.find_match(input, in_idx, self.max_search_depth);

            if len >= 3 {
                if lazy_depth >= 1 && in_idx + 1 < input.len() {
                    let (next_len, next_offset) =
                        mf.find_match(input, in_idx + 1, self.max_search_depth);
                    if next_len > len {
                        if lazy_depth >= 2 && in_idx + 2 < input.len() {
                            let (next2_len, next2_offset) =
                                mf.find_match(input, in_idx + 2, self.max_search_depth);
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
                            }
                        } else {
                            self.split_stats.observe_literal(input[in_idx]);
                            self.litlen_freqs[input[in_idx] as usize] += 1;
                            litrunlen += 1;
                            in_idx += 1;

                            len = next_len;
                            offset = next_offset;
                        }
                    }
                }

                self.sequences.push(Sequence {
                    litrunlen,
                    length: len as u16,
                    offset: offset as u16,
                });
                self.split_stats.observe_match(len, offset);
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[self.get_offset_slot(offset)] += 1;
                litrunlen = 0;
                mf.skip_positions(input, in_idx + 1, len - 1, self.max_search_depth);
                in_idx += len;
            } else {
                self.split_stats.observe_literal(input[in_idx]);
                self.litlen_freqs[input[in_idx] as usize] += 1;
                litrunlen += 1;
                in_idx += 1;
            }
        }
        self.sequences.push(Sequence {
            litrunlen,
            length: 0,
            offset: 0,
        });
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
            for _ in 0..seq.litrunlen {
                if !self.write_literal(bs, input[in_pos]) {
                    return false;
                }
                in_pos += 1;
            }
            if seq.length >= 3 {
                if !self.write_match(bs, seq.length as usize, seq.offset as usize) {
                    return false;
                }
                in_pos += seq.length as usize;
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
                let (len, offset) = mf.find_match(input, in_idx, self.max_search_depth);
                if len >= 3 {
                    self.sequences.push(Sequence {
                        litrunlen,
                        length: len as u16,
                        offset: offset as u16,
                    });
                    litrunlen = 0;
                    mf.skip_positions(input, in_idx + 1, len - 1, self.max_search_depth);
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
                let (len, offset) = mf.find_match(input, in_idx, self.max_search_depth);
                if len >= 3 {
                    self.split_stats.observe_match(len, offset);
                    self.sequences.push(Sequence {
                        litrunlen,
                        length: len as u16,
                        offset: offset as u16,
                    });
                    litrunlen = 0;
                    mf.skip_positions(input, in_idx + 1, len - 1, self.max_search_depth);
                    in_idx += len;
                } else {
                    self.split_stats.observe_literal(input[in_idx]);
                    litrunlen += 1;
                    in_idx += 1;
                }
            }
        }
        self.sequences.push(Sequence {
            litrunlen,
            length: 0,
            offset: 0,
        });

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
            for _ in 0..seq.litrunlen {
                if !self.write_literal(bs, input[in_pos]) {
                    return 0;
                }
                in_pos += 1;
            }
            if seq.length >= 3 {
                if !self.write_match(bs, seq.length as usize, seq.offset as usize) {
                    return 0;
                }
                in_pos += seq.length as usize;
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
        let mut in_idx = start_pos;

        while in_idx < input.len() {
            let block_len = in_idx - start_pos;
            if self
                .split_stats
                .should_end_block(block_len, input.len() - in_idx)
            {
                break;
            }
            let (len, offset) = mf.find_match(input, in_idx, self.max_search_depth);
            if len >= 3 {
                self.split_stats.observe_match(len, offset);
                mf.skip_positions(input, in_idx + 1, len - 1, self.max_search_depth);
                in_idx += len;
            } else {
                self.split_stats.observe_literal(input[in_idx]);
                in_idx += 1;
            }
        }

        let processed = in_idx - start_pos;
        let block_input = &input[start_pos..start_pos + processed];
        let is_final = (start_pos + processed >= input.len()) && final_block;

        self.sequences.clear();
        let mut litrunlen = 0;
        let mut cur_in_idx = 0;
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);

        mf.reset();

        while cur_in_idx < block_input.len() {
            let (len, offset) = mf.find_match(block_input, cur_in_idx, self.max_search_depth);
            if len >= 3 {
                self.sequences.push(Sequence {
                    litrunlen,
                    length: len as u16,
                    offset: offset as u16,
                });
                self.litlen_freqs[257 + self.get_length_slot(len)] += 1;
                self.offset_freqs[self.get_offset_slot(offset)] += 1;
                litrunlen = 0;
                cur_in_idx += len;
                for i in 1..len {
                    mf.skip_match(block_input, cur_in_idx - len + i, self.max_search_depth);
                }
            } else {
                self.litlen_freqs[block_input[cur_in_idx] as usize] += 1;
                litrunlen += 1;
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

        self.dp_nodes.clear();
        self.dp_nodes.resize(
            processed + 1,
            DPNode {
                cost: 0x3FFFFFFF,
                length: 0,
                offset: 0,
            },
        );
        self.dp_nodes[0].cost = 0;

        mf.reset();
        let mut matches = Vec::new();
        for pos in 0..processed {
            let cur_cost = self.dp_nodes[pos].cost;
            if cur_cost >= 0x3FFFFFFF {
                continue;
            }

            let lit_cost = self.litlen_lens[block_input[pos] as usize] as u32;
            if cur_cost + lit_cost < self.dp_nodes[pos + 1].cost {
                self.dp_nodes[pos + 1] = DPNode {
                    cost: cur_cost + lit_cost,
                    length: 1,
                    offset: 0,
                };
            }

            mf.find_matches(block_input, pos, self.max_search_depth, &mut matches);
            for &(len, offset) in &matches {
                let len = len as usize;
                if pos + len > processed {
                    continue;
                }
                let cost = self.get_match_cost(len, offset as usize);
                if cur_cost + cost < self.dp_nodes[pos + len].cost {
                    self.dp_nodes[pos + len] = DPNode {
                        cost: cur_cost + cost,
                        length: len as u16,
                        offset,
                    };
                }
            }
        }

        self.sequences.clear();
        self.litlen_freqs.fill(0);
        self.offset_freqs.fill(0);
        self.litlen_freqs[256] = 1;

        let mut pos = processed;
        let mut path = Vec::with_capacity(processed);
        while pos > 0 {
            let node = self.dp_nodes[pos];
            path.push(node);
            pos -= node.length as usize;
        }
        path.reverse();

        let mut litrunlen = 0;
        let mut cur_pos = 0;
        for node in path {
            if node.length == 1 {
                self.litlen_freqs[block_input[cur_pos] as usize] += 1;
                litrunlen += 1;
                cur_pos += 1;
            } else {
                self.sequences.push(Sequence {
                    litrunlen,
                    length: node.length,
                    offset: node.offset,
                });
                self.litlen_freqs[257 + self.get_length_slot(node.length as usize)] += 1;
                self.offset_freqs[self.get_offset_slot(node.offset as usize)] += 1;
                litrunlen = 0;
                cur_pos += node.length as usize;
            }
        }
        self.sequences.push(Sequence {
            litrunlen,
            length: 0,
            offset: 0,
        });

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
            } else {
                if run >= 4 {
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
            } else if sym == 18 {
                if !bs.write_bits(extra, 7) {
                    return false;
                }
            }
        }
        true
    }

    fn load_static_huffman_codes(&mut self) {
        let mut i = 0;
        while i < 144 {
            self.litlen_lens[i] = 8;
            i += 1;
        }
        while i < 256 {
            self.litlen_lens[i] = 9;
            i += 1;
        }
        while i < 280 {
            self.litlen_lens[i] = 7;
            i += 1;
        }
        while i < 288 {
            self.litlen_lens[i] = 8;
            i += 1;
        }
        for i in 0..32 {
            self.offset_lens[i] = 5;
        }
        fn gen_codewords_from_lens(lens: &[u8], codewords: &mut [u32], max_len: usize) {
            let mut len_counts = vec![0u32; max_len + 1];
            for &l in lens {
                if l > 0 {
                    len_counts[l as usize] += 1;
                }
            }
            let mut next_code = vec![0u32; max_len + 1];
            let mut code = 0u32;
            for len in 1..=max_len {
                code = (code + len_counts[len - 1]) << 1;
                next_code[len] = code;
            }
            for i in 0..lens.len() {
                if lens[i] > 0 {
                    let mut c = next_code[lens[i] as usize];
                    next_code[lens[i] as usize] += 1;
                    let mut rev = 0u32;
                    for _ in 0..lens[i] {
                        rev = (rev << 1) | (c & 1);
                        c >>= 1;
                    }
                    codewords[i] = rev;
                }
            }
        }
        gen_codewords_from_lens(&self.litlen_lens, &mut self.litlen_codewords, 9);
        gen_codewords_from_lens(&self.offset_lens, &mut self.offset_codewords, 5);
        self.update_huffman_tables();
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
    fn write_match(&self, bs: &mut Bitstream, len: usize, offset: usize) -> bool {
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

        let off_slot = self.get_offset_slot(offset);
        let entry = unsafe { *self.offset_table.get_unchecked(off_slot) };
        let off_code = entry as u32;
        let off_len = (entry >> 32) as u8 as u32;
        let extra_bits = (entry >> 40) as u8 as u32;
        let base = (entry >> 48) as u16 as u32;

        let off_val = off_code | ((offset as u32).wrapping_sub(base) << off_len);
        let off_len_total = off_len + extra_bits;

        unsafe { bs.write_bits_upto_32(off_val, off_len_total) }
    }

    fn get_length_slot(&self, len: usize) -> usize {
        (LENGTH_WRITE_TABLE[len] >> 24) as usize
    }

    fn get_length_extra_bits(&self, slot: usize) -> usize {
        LENGTH_EXTRA_BITS_TABLE[slot] as usize
    }
    fn get_offset_slot(&self, offset: usize) -> usize {
        let off = (offset - 1) as u32;
        if off < 256 {
            return unsafe { *OFFSET_SLOT_TABLE.get_unchecked(off as usize) as usize };
        }
        let l = 31 - off.leading_zeros();
        let slot = (2 * l) as usize;
        slot + ((off >> (l - 1)) as usize & 1)
    }
    fn get_offset_extra_bits(&self, slot: usize) -> usize {
        OFFSET_EXTRA_BITS_TABLE[slot] as usize
    }
    fn get_match_cost(&self, len: usize, offset: usize) -> u32 {
        let len_info = unsafe { *LENGTH_WRITE_TABLE.get_unchecked(len) };
        let len_slot = (len_info >> 24) as usize;
        let len_extra_bits = ((len_info >> 16) & 0xFF) as u32;

        let len_cost = self.litlen_lens[257 + len_slot] as u32 + len_extra_bits;
        let off_slot = self.get_offset_slot(offset);
        let off_cost =
            self.offset_lens[off_slot] as u32 + self.get_offset_extra_bits(off_slot) as u32;
        len_cost + off_cost
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
