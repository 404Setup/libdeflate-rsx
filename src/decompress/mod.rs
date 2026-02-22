#![deny(unsafe_op_in_unsafe_fn)]
mod tables;

use self::tables::*;
use crate::common::*;
use std::cmp::min;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

pub use self::tables::{
    LITLEN_ENOUGH, LITLEN_TABLEBITS, OFFSET_ENOUGH, OFFSET_TABLEBITS, PRECODE_ENOUGH,
    PRECODE_TABLEBITS,
};

macro_rules! refill_bits {
    ($input:expr, $in_idx:expr, $bitbuf:expr, $bitsleft:expr) => {
        if $bitsleft < 32 {
            if $input.len().saturating_sub($in_idx) >= 8 {
                let word = unsafe { ($input.as_ptr().add($in_idx) as *const u64).read_unaligned() };
                let word = u64::from_le(word);
                $bitbuf |= word << $bitsleft;
                let consumed = (63 - $bitsleft) >> 3;
                $in_idx += consumed as usize;
                $bitsleft |= 56;
            } else {
                while $bitsleft < 32 && $in_idx < $input.len() {
                    $bitbuf |= ($input[$in_idx] as u64) << $bitsleft;
                    $in_idx += 1;
                    $bitsleft += 8;
                }
            }
        }
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompressorState {
    Start,
    BlockHeader,
    DynamicHeader,
    BlockBody,
    BlockBodyOffset { length: usize, extra_bits: u32 },
    UncompressedHeader,
    UncompressedBody { len: usize },
    Done,
}

#[derive(Clone)]
pub struct Decompressor {
    pub(crate) precode_decode_table: [u32; PRECODE_ENOUGH],
    pub(crate) litlen_decode_table: [u32; LITLEN_ENOUGH],
    pub(crate) offset_decode_table: [u32; OFFSET_ENOUGH],

    pub(crate) precode_lens: [u8; DEFLATE_NUM_PRECODE_SYMS],
    pub(crate) lens: [u8; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS + DEFLATE_MAX_LENS_OVERRUN],
    pub(crate) sorted_syms: [u16; DEFLATE_MAX_NUM_SYMS],

    pub(crate) litlen_tablebits: usize,
    pub(crate) static_codes_loaded: bool,

    pub(crate) bitbuf: u64,
    pub(crate) bitsleft: u32,
    pub(crate) state: DecompressorState,
    pub(crate) is_final_block: bool,
}

struct StaticHuffmanData {
    offset_decode_table: [u32; OFFSET_ENOUGH],
    litlen_decode_table: [u32; LITLEN_ENOUGH],
    lens: [u8; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS + DEFLATE_MAX_LENS_OVERRUN],
    litlen_tablebits: usize,
}

static STATIC_HUFFMAN_DATA: std::sync::OnceLock<StaticHuffmanData> = std::sync::OnceLock::new();

#[derive(Debug, PartialEq, Eq)]
#[must_use = "Decompression result must be checked for errors"]
pub enum DecompressResult {
    Success,
    BadData,
    ShortOutput,
    InsufficientSpace,
    ShortInput,
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            precode_decode_table: [0; PRECODE_ENOUGH],
            litlen_decode_table: [0; LITLEN_ENOUGH],
            offset_decode_table: [0; OFFSET_ENOUGH],
            precode_lens: [0; DEFLATE_NUM_PRECODE_SYMS],
            lens: [0; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS + DEFLATE_MAX_LENS_OVERRUN],
            sorted_syms: [0; DEFLATE_MAX_NUM_SYMS],
            litlen_tablebits: LITLEN_TABLEBITS,
            static_codes_loaded: false,
            bitbuf: 0,
            bitsleft: 0,
            state: DecompressorState::Start,
            is_final_block: false,
        }
    }

    fn build_precode_decode_table(&mut self) -> bool {
        build_decode_table(
            &mut self.precode_decode_table,
            &self.precode_lens,
            DEFLATE_NUM_PRECODE_SYMS,
            &PRECODE_DECODE_RESULTS,
            PRECODE_TABLEBITS,
            DEFLATE_MAX_PRE_CODEWORD_LEN,
            &mut self.sorted_syms,
            None,
        )
    }

    fn build_litlen_decode_table(&mut self, num_litlen_syms: usize) -> bool {
        build_decode_table(
            &mut self.litlen_decode_table,
            &self.lens[..num_litlen_syms],
            num_litlen_syms,
            &LITLEN_DECODE_RESULTS,
            LITLEN_TABLEBITS,
            DEFLATE_MAX_LITLEN_CODEWORD_LEN,
            &mut self.sorted_syms,
            Some(&mut self.litlen_tablebits),
        )
    }

    fn build_offset_decode_table(
        &mut self,
        num_litlen_syms: usize,
        num_offset_syms: usize,
    ) -> bool {
        build_decode_table(
            &mut self.offset_decode_table,
            &self.lens[num_litlen_syms..num_litlen_syms + num_offset_syms],
            num_offset_syms,
            &OFFSET_DECODE_RESULTS,
            OFFSET_TABLEBITS,
            DEFLATE_MAX_OFFSET_CODEWORD_LEN,
            &mut self.sorted_syms,
            None,
        )
    }

    pub fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> (DecompressResult, usize, usize) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("ssse3")
                && is_x86_feature_detected!("sse4.1")
            {
                let res = unsafe { x86::decompress_bmi2(self, input, output) };
                // Security: Reset state because x86 implementation clobbers internal tables.
                // This prevents state corruption if the Decompressor is reused for streaming.
                self.state = DecompressorState::Start;
                self.is_final_block = false;
                self.bitbuf = 0;
                self.bitsleft = 0;
                return res;
            }
        }

        self.bitbuf = 0;
        self.bitsleft = 0;
        self.state = DecompressorState::Start;
        self.is_final_block = false;

        let mut out_idx = 0;
        self.decompress_streaming(input, output, &mut out_idx)
    }

    pub fn decompress_streaming(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        out_idx: &mut usize,
    ) -> (DecompressResult, usize, usize) {
        let mut in_idx = 0;
        let start_out_idx = *out_idx;

        loop {
            match self.state {
                DecompressorState::Start => {
                    refill_bits!(input, in_idx, self.bitbuf, self.bitsleft);
                    if self.bitsleft < 3 {
                        return (
                            DecompressResult::ShortInput,
                            in_idx,
                            *out_idx - start_out_idx,
                        );
                    }
                    self.is_final_block = (self.bitbuf & 1) != 0;
                    let block_type = ((self.bitbuf >> 1) & 3) as u8;
                    self.bitbuf >>= 3;
                    self.bitsleft -= 3;

                    match block_type {
                        DEFLATE_BLOCKTYPE_UNCOMPRESSED => {
                            self.state = DecompressorState::UncompressedHeader
                        }
                        DEFLATE_BLOCKTYPE_STATIC_HUFFMAN => {
                            self.load_static_huffman_codes();
                            self.state = DecompressorState::BlockBody;
                        }
                        DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN => {
                            self.state = DecompressorState::DynamicHeader
                        }
                        _ => return (DecompressResult::BadData, in_idx, *out_idx - start_out_idx),
                    }
                }
                DecompressorState::DynamicHeader => {
                    let res = self.read_dynamic_huffman_header(input, &mut in_idx);
                    if res == DecompressResult::Success {
                        self.state = DecompressorState::BlockBody;
                    } else {
                        return (res, in_idx, *out_idx - start_out_idx);
                    }
                }
                DecompressorState::BlockBody | DecompressorState::BlockBodyOffset { .. } => {
                    let res = self.decompress_huffman_block(input, &mut in_idx, output, out_idx);
                    if res == DecompressResult::Success {
                        if self.is_final_block {
                            self.state = DecompressorState::Done;
                            return (DecompressResult::Success, in_idx, *out_idx - start_out_idx);
                        } else {
                            self.state = DecompressorState::Start;
                        }
                    } else {
                        return (res, in_idx, *out_idx - start_out_idx);
                    }
                }
                DecompressorState::UncompressedHeader => {
                    let skip = self.bitsleft & 7;
                    self.bitbuf >>= skip;
                    self.bitsleft -= skip;
                    let unused_bytes = self.bitsleft / 8;
                    in_idx = in_idx.saturating_sub(unused_bytes as usize);
                    self.bitbuf = 0;
                    self.bitsleft = 0;
                    if in_idx + 4 > input.len() {
                        return (
                            DecompressResult::ShortInput,
                            in_idx,
                            *out_idx - start_out_idx,
                        );
                    }
                    let len = u16::from_le_bytes([input[in_idx], input[in_idx + 1]]) as usize;
                    let nlen = u16::from_le_bytes([input[in_idx + 2], input[in_idx + 3]]) as usize;
                    in_idx += 4;
                    if len != (!nlen & 0xFFFF) {
                        return (DecompressResult::BadData, in_idx, *out_idx - start_out_idx);
                    }
                    self.state = DecompressorState::UncompressedBody { len };
                }
                DecompressorState::UncompressedBody { len } => {
                    let remaining = len;
                    let available_in = input.len() - in_idx;
                    let available_out = output.len() - *out_idx;
                    let copy_len = min(remaining, min(available_in, available_out));

                    output[*out_idx..*out_idx + copy_len]
                        .copy_from_slice(&input[in_idx..in_idx + copy_len]);
                    in_idx += copy_len;
                    *out_idx += copy_len;
                    let new_len = remaining - copy_len;

                    if new_len == 0 {
                        if self.is_final_block {
                            self.state = DecompressorState::Done;
                            return (DecompressResult::Success, in_idx, *out_idx - start_out_idx);
                        } else {
                            self.state = DecompressorState::Start;
                        }
                    } else {
                        self.state = DecompressorState::UncompressedBody { len: new_len };
                        if available_out == 0 {
                            return (
                                DecompressResult::InsufficientSpace,
                                in_idx,
                                *out_idx - start_out_idx,
                            );
                        }
                        if available_in == 0 {
                            return (
                                DecompressResult::ShortInput,
                                in_idx,
                                *out_idx - start_out_idx,
                            );
                        }
                    }
                }
                DecompressorState::Done => {
                    return (DecompressResult::Success, in_idx, *out_idx - start_out_idx);
                }
                _ => return (DecompressResult::BadData, in_idx, *out_idx - start_out_idx),
            }
        }
    }

    fn load_static_huffman_codes(&mut self) {
        if self.static_codes_loaded {
            return;
        }

        let data = STATIC_HUFFMAN_DATA.get_or_init(|| {
            let mut d = Decompressor::new();
            let mut i = 0;
            while i < 144 {
                d.lens[i] = 8;
                i += 1;
            }
            while i < 256 {
                d.lens[i] = 9;
                i += 1;
            }
            while i < 280 {
                d.lens[i] = 7;
                i += 1;
            }
            while i < 288 {
                d.lens[i] = 8;
                i += 1;
            }
            while i < 288 + 32 {
                d.lens[i] = 5;
                i += 1;
            }
            d.build_offset_decode_table(288, 32);
            d.build_litlen_decode_table(288);

            StaticHuffmanData {
                offset_decode_table: d.offset_decode_table,
                litlen_decode_table: d.litlen_decode_table,
                lens: d.lens,
                litlen_tablebits: d.litlen_tablebits,
            }
        });

        self.offset_decode_table
            .copy_from_slice(&data.offset_decode_table);
        self.litlen_decode_table
            .copy_from_slice(&data.litlen_decode_table);
        self.lens.copy_from_slice(&data.lens);
        self.litlen_tablebits = data.litlen_tablebits;
        self.static_codes_loaded = true;
    }

    fn read_dynamic_huffman_header(
        &mut self,
        input: &[u8],
        in_idx: &mut usize,
    ) -> DecompressResult {
        refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
        if self.bitsleft < 14 {
            return DecompressResult::ShortInput;
        }
        let num_litlen_syms = 257 + ((self.bitbuf & 0x1F) as usize);
        let num_offset_syms = 1 + (((self.bitbuf >> 5) & 0x1F) as usize);
        let num_precode_syms = 4 + (((self.bitbuf >> 10) & 0xF) as usize);
        self.bitbuf >>= 14;
        self.bitsleft -= 14;
        let permutation = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        for i in 0..num_precode_syms {
            refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
            if self.bitsleft < 3 {
                return DecompressResult::ShortInput;
            }
            self.precode_lens[permutation[i]] = (self.bitbuf & 7) as u8;
            self.bitbuf >>= 3;
            self.bitsleft -= 3;
        }
        for i in num_precode_syms..19 {
            self.precode_lens[permutation[i]] = 0;
        }
        if !self.build_precode_decode_table() {
            return DecompressResult::BadData;
        }
        let mut i = 0;
        let total_syms = num_litlen_syms + num_offset_syms;
        while i < total_syms {
            refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
            let entry =
                self.precode_decode_table[(self.bitbuf & ((1 << PRECODE_TABLEBITS) - 1)) as usize];
            let total_bits = entry & 0xFF;
            if self.bitsleft < total_bits {
                return DecompressResult::ShortInput;
            }
            self.bitbuf >>= total_bits;
            self.bitsleft -= total_bits;
            let presym = entry >> 16;
            if presym < 16 {
                self.lens[i] = presym as u8;
                i += 1;
            } else if presym == 16 {
                if i == 0 {
                    return DecompressResult::BadData;
                }
                let rep_val = self.lens[i - 1];
                if self.bitsleft < 2 {
                    return DecompressResult::ShortInput;
                }
                let rep_count = 3 + ((self.bitbuf & 3) as usize);
                self.bitbuf >>= 2;
                self.bitsleft -= 2;
                for _ in 0..rep_count {
                    if i < total_syms {
                        self.lens[i] = rep_val;
                        i += 1;
                    }
                }
            } else if presym == 17 {
                if self.bitsleft < 3 {
                    return DecompressResult::ShortInput;
                }
                let rep_count = 3 + ((self.bitbuf & 7) as usize);
                self.bitbuf >>= 3;
                self.bitsleft -= 3;
                for _ in 0..rep_count {
                    if i < total_syms {
                        self.lens[i] = 0;
                        i += 1;
                    }
                }
            } else {
                if self.bitsleft < 7 {
                    return DecompressResult::ShortInput;
                }
                let rep_count = 11 + ((self.bitbuf & 0x7F) as usize);
                self.bitbuf >>= 7;
                self.bitsleft -= 7;
                for _ in 0..rep_count {
                    if i < total_syms {
                        self.lens[i] = 0;
                        i += 1;
                    }
                }
            }
        }
        if i != total_syms {
            return DecompressResult::BadData;
        }
        if !self.build_offset_decode_table(num_litlen_syms, num_offset_syms) {
            return DecompressResult::BadData;
        }
        if !self.build_litlen_decode_table(num_litlen_syms) {
            return DecompressResult::BadData;
        }
        self.static_codes_loaded = false;
        DecompressResult::Success
    }

    pub(crate) fn decompress_huffman_block(
        &mut self,
        input: &[u8],
        in_idx: &mut usize,
        output: &mut [u8],
        out_idx: &mut usize,
    ) -> DecompressResult {
        let litlen_tablemask = (1 << self.litlen_tablebits) - 1;

        if let DecompressorState::BlockBodyOffset {
            length,
            extra_bits: _,
        } = self.state
        {
            refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
            let mut entry =
                self.offset_decode_table[(self.bitbuf as usize) & ((1 << OFFSET_TABLEBITS) - 1)];
            if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                let main_bits = entry & 0xFF;
                if self.bitsleft < main_bits {
                    return DecompressResult::ShortInput;
                }
                self.bitbuf >>= main_bits;
                self.bitsleft -= main_bits;
                let subtable_idx = (entry >> 16) as usize;
                let subtable_bits = (entry >> 8) & 0x3F;
                entry = self.offset_decode_table
                    [subtable_idx + ((self.bitbuf as usize) & ((1 << subtable_bits) - 1))];
            }
            let total_bits = entry & 0xFF;
            if self.bitsleft < total_bits {
                return DecompressResult::ShortInput;
            }
            self.bitbuf >>= total_bits;
            self.bitsleft -= total_bits;
            let mut offset = (entry >> 16) as usize;
            let len = (entry >> 8) & 0xFF;
            let extra_bits = total_bits - len;
            if extra_bits > 0 {
                offset += (self.bitbuf as usize) & ((1 << extra_bits) - 1);
                self.bitbuf >>= extra_bits;
                self.bitsleft -= extra_bits;
            }

            if offset > *out_idx {
                return DecompressResult::BadData;
            }
            let dest = *out_idx;
            let src = dest - offset;
            if dest + length > output.len() {
                return DecompressResult::InsufficientSpace;
            }
            unsafe {
                let out_ptr = output.as_mut_ptr();
                if offset >= length {
                    std::ptr::copy_nonoverlapping(out_ptr.add(src), out_ptr.add(dest), length);
                } else if offset == 1 {
                    let b = *out_ptr.add(src);
                    std::ptr::write_bytes(out_ptr.add(dest), b, length);
                } else if offset < 8 {
                    let src_ptr = out_ptr.add(src);
                    let dest_ptr = out_ptr.add(dest);
                    if offset == 1 || offset == 2 || offset == 4 {
                        let pattern = prepare_pattern(offset, src_ptr);
                        let mut i = 0;
                        while i + 32 <= length {
                            std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                            std::ptr::write_unaligned(dest_ptr.add(i + 8) as *mut u64, pattern);
                            std::ptr::write_unaligned(dest_ptr.add(i + 16) as *mut u64, pattern);
                            std::ptr::write_unaligned(dest_ptr.add(i + 24) as *mut u64, pattern);
                            i += 32;
                        }
                        while i + 8 <= length {
                            std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                            i += 8;
                        }
                        while i < length {
                            *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                            i += 1;
                        }
                    } else {
                        let pattern = prepare_pattern(offset, src_ptr);
                        let mut i = 0;
                        while i + 8 <= length {
                            std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                            i += offset;
                        }
                        while i < length {
                            *dest_ptr.add(i) = *src_ptr.add(i);
                            i += 1;
                        }
                    }
                } else {
                    let mut copied = 0;
                    while copied < length {
                        let copy_len = std::cmp::min(offset, length - copied);
                        std::ptr::copy_nonoverlapping(
                            out_ptr.add(src + copied),
                            out_ptr.add(dest + copied),
                            copy_len,
                        );
                        copied += copy_len;
                    }
                }
            }
            *out_idx += length;
            self.state = DecompressorState::BlockBody;
        }

        let mut bitbuf = self.bitbuf;
        let mut bitsleft = self.bitsleft;
        let in_ptr_start = input.as_ptr();
        let in_ptr_end = unsafe { in_ptr_start.add(input.len()) };
        let mut in_next = unsafe { in_ptr_start.add(*in_idx) };
        let out_ptr_start = output.as_mut_ptr();
        let out_ptr_end = unsafe { out_ptr_start.add(output.len()) };
        let mut out_next = unsafe { out_ptr_start.add(*out_idx) };

        unsafe {
            while in_next.add(15) <= in_ptr_end && out_next.add(258) <= out_ptr_end {
                if bitsleft < 32 {
                    let word = (in_next as *const u64).read_unaligned();
                    let word = u64::from_le(word);
                    bitbuf |= word << bitsleft;
                    let consumed = (63 - bitsleft) >> 3;
                    in_next = in_next.add(consumed as usize);
                    bitsleft |= 56;
                }

                let entry = *self
                    .litlen_decode_table
                    .get_unchecked((bitbuf as usize) & litlen_tablemask);

                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                    break;
                }

                if entry & HUFFDEC_LITERAL != 0 {
                    let total_bits = entry & 0xFF;
                    bitbuf >>= total_bits;
                    bitsleft -= total_bits;
                    *out_next = (entry >> 16) as u8;
                    out_next = out_next.add(1);
                } else {
                    let len = (entry >> 8) & 0xFF;
                    bitbuf >>= len;
                    bitsleft -= len;

                    let mut length = (entry >> 16) as usize;
                    let total_bits = entry & 0xFF;
                    let extra_bits = total_bits - len;

                    if extra_bits > 0 {
                        length += (bitbuf as usize) & ((1 << extra_bits) - 1);
                        bitbuf >>= extra_bits;
                        bitsleft -= extra_bits;
                    }

                    if bitsleft < 32 {
                        let word = (in_next as *const u64).read_unaligned();
                        let word = u64::from_le(word);
                        bitbuf |= word << bitsleft;
                        let consumed = (63 - bitsleft) >> 3;
                        in_next = in_next.add(consumed as usize);
                        bitsleft |= 56;
                    }

                    let mut off_entry = *self
                        .offset_decode_table
                        .get_unchecked((bitbuf as usize) & ((1 << OFFSET_TABLEBITS) - 1));

                    if off_entry & HUFFDEC_EXCEPTIONAL != 0 {
                        if off_entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                            let main_bits = off_entry & 0xFF;
                            bitbuf >>= main_bits;
                            bitsleft -= main_bits;
                            let subtable_idx = (off_entry >> 16) as usize;
                            let subtable_bits = (off_entry >> 8) & 0x3F;
                            off_entry = *self.offset_decode_table.get_unchecked(
                                subtable_idx + ((bitbuf as usize) & ((1 << subtable_bits) - 1)),
                            );
                        } else {
                            break;
                        }
                    }

                    let len_off = (off_entry >> 8) & 0xFF;
                    bitbuf >>= len_off;
                    bitsleft -= len_off;

                    let mut offset = (off_entry >> 16) as usize;
                    let total_bits_off = off_entry & 0xFF;
                    let extra_bits_off = total_bits_off - len_off;

                    if extra_bits_off > 0 {
                        offset += (bitbuf as usize) & ((1 << extra_bits_off) - 1);
                        bitbuf >>= extra_bits_off;
                        bitsleft -= extra_bits_off;
                    }

                    let current_out_idx = out_next.offset_from(out_ptr_start) as usize;
                    if offset > current_out_idx {
                        break;
                    }

                    let src = out_next.sub(offset);
                    if offset < 8 {
                        if offset == 1 || offset == 2 || offset == 4 {
                            let pattern = prepare_pattern(offset, src);
                            let mut i = 0;
                            while i + 32 <= length {
                                (out_next.add(i) as *mut u64).write_unaligned(pattern);
                                (out_next.add(i + 8) as *mut u64).write_unaligned(pattern);
                                (out_next.add(i + 16) as *mut u64).write_unaligned(pattern);
                                (out_next.add(i + 24) as *mut u64).write_unaligned(pattern);
                                i += 32;
                            }
                            while i + 8 <= length {
                                (out_next.add(i) as *mut u64).write_unaligned(pattern);
                                i += 8;
                            }
                            while i < length {
                                *out_next.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                i += 1;
                            }
                        } else {
                            let pattern = prepare_pattern(offset, src);
                            let mut i = 0;
                            while i + 8 <= length {
                                (out_next.add(i) as *mut u64).write_unaligned(pattern);
                                i += offset;
                            }
                            while i < length {
                                *out_next.add(i) = *src.add(i);
                                i += 1;
                            }
                        }
                    } else {
                        if offset >= length {
                            std::ptr::copy_nonoverlapping(src, out_next, length);
                        } else {
                            // Optimization: Use u64 copy loop for overlapping case with offset >= 8.
                            // This avoids function call overhead of copy_nonoverlapping for small chunks.
                            // Since offset >= 8, we can read 8 bytes and write 8 bytes safely
                            // (the read source is at least 8 bytes behind the write destination).
                            let src_ptr = src;
                            let dest_ptr = out_next;
                            let mut i = 0;
                            while i + 8 <= length {
                                let val = (src_ptr.add(i) as *const u64).read_unaligned();
                                (dest_ptr.add(i) as *mut u64).write_unaligned(val);
                                i += 8;
                            }
                            while i < length {
                                *dest_ptr.add(i) = *src_ptr.add(i);
                                i += 1;
                            }
                        }
                    }
                    out_next = out_next.add(length);
                }
            }
        }

        self.bitbuf = bitbuf;
        self.bitsleft = bitsleft;
        *in_idx = unsafe { in_next.offset_from(in_ptr_start) as usize };
        *out_idx = unsafe { out_next.offset_from(out_ptr_start) as usize };

        loop {
            while *in_idx + 15 < input.len() && *out_idx + 258 < output.len() {
                if self.bitsleft < 32 {
                    let word =
                        unsafe { (input.as_ptr().add(*in_idx) as *const u64).read_unaligned() };
                    let word = u64::from_le(word);
                    self.bitbuf |= word << self.bitsleft;
                    let consumed = (63 - self.bitsleft) >> 3;
                    *in_idx += consumed as usize;
                    self.bitsleft |= 56;
                }

                let entry = unsafe {
                    *self
                        .litlen_decode_table
                        .get_unchecked((self.bitbuf as usize) & litlen_tablemask)
                };

                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                    break;
                }

                let saved_bitbuf = self.bitbuf;
                let total_bits = entry & 0xFF;
                self.bitbuf >>= total_bits;
                self.bitsleft -= total_bits;

                if entry & HUFFDEC_LITERAL != 0 {
                    unsafe {
                        *output.get_unchecked_mut(*out_idx) = (entry >> 16) as u8;
                    }
                    *out_idx += 1;
                } else {
                    let mut length = (entry >> 16) as usize;
                    let len = (entry >> 8) & 0xFF;
                    let extra_bits = total_bits - len;
                    if extra_bits > 0 {
                        length += ((saved_bitbuf >> len) as usize) & ((1 << extra_bits) - 1);
                    }

                    if self.bitsleft < 32 {
                        let word =
                            unsafe { (input.as_ptr().add(*in_idx) as *const u64).read_unaligned() };
                        let word = u64::from_le(word);
                        self.bitbuf |= word << self.bitsleft;
                        let consumed = (63 - self.bitsleft) >> 3;
                        *in_idx += consumed as usize;
                        self.bitsleft |= 56;
                    }

                    let mut off_entry = unsafe {
                        *self
                            .offset_decode_table
                            .get_unchecked((self.bitbuf as usize) & ((1 << OFFSET_TABLEBITS) - 1))
                    };

                    if off_entry & HUFFDEC_EXCEPTIONAL != 0 {
                        if off_entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                            let main_bits = off_entry & 0xFF;
                            self.bitbuf >>= main_bits;
                            self.bitsleft -= main_bits;
                            let subtable_idx = (off_entry >> 16) as usize;
                            let subtable_bits = (off_entry >> 8) & 0x3F;
                            off_entry = unsafe {
                                *self.offset_decode_table.get_unchecked(
                                    subtable_idx
                                        + ((self.bitbuf as usize) & ((1 << subtable_bits) - 1)),
                                )
                            };
                        } else {
                            break;
                        }
                    }

                    let saved_bitbuf_off = self.bitbuf;
                    let total_bits_off = off_entry & 0xFF;
                    self.bitbuf >>= total_bits_off;
                    self.bitsleft -= total_bits_off;
                    let mut offset = (off_entry >> 16) as usize;
                    let len_off = (off_entry >> 8) & 0xFF;
                    let extra_bits_off = total_bits_off - len_off;
                    if extra_bits_off > 0 {
                        offset +=
                            ((saved_bitbuf_off >> len_off) as usize) & ((1 << extra_bits_off) - 1);
                    }

                    if offset > *out_idx {
                        return DecompressResult::BadData;
                    } else {
                        let src = *out_idx - offset;
                        let dest = *out_idx;

                        unsafe {
                            let out_ptr = output.as_mut_ptr();
                            if offset < 8 {
                                let src_ptr = out_ptr.add(src);
                                let dest_ptr = out_ptr.add(dest);
                                if offset == 1 || offset == 2 || offset == 4 {
                                    let pattern = prepare_pattern(offset, src_ptr);
                                    let mut i = 0;
                                    while i + 32 <= length {
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i) as *mut u64,
                                            pattern,
                                        );
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i + 8) as *mut u64,
                                            pattern,
                                        );
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i + 16) as *mut u64,
                                            pattern,
                                        );
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i + 24) as *mut u64,
                                            pattern,
                                        );
                                        i += 32;
                                    }
                                    while i + 8 <= length {
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i) as *mut u64,
                                            pattern,
                                        );
                                        i += 8;
                                    }
                                    while i < length {
                                        *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                        i += 1;
                                    }
                                } else {
                                    let pattern = prepare_pattern(offset, src_ptr);
                                    let mut i = 0;
                                    while i + 8 <= length {
                                        std::ptr::write_unaligned(
                                            dest_ptr.add(i) as *mut u64,
                                            pattern,
                                        );
                                        i += offset;
                                    }
                                    while i < length {
                                        *dest_ptr.add(i) = *src_ptr.add(i);
                                        i += 1;
                                    }
                                }
                            } else {
                                let mut copied = 0;
                                while copied < length {
                                    let copy_len = min(offset, length - copied);
                                    std::ptr::copy_nonoverlapping(
                                        out_ptr.add(src + copied),
                                        out_ptr.add(dest + copied),
                                        copy_len,
                                    );
                                    copied += copy_len;
                                }
                            }
                        }
                        *out_idx += length;
                    }
                }
            }

            refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
            let mut entry = self.litlen_decode_table[(self.bitbuf as usize) & litlen_tablemask];
            if entry & HUFFDEC_EXCEPTIONAL != 0 {
                if entry & HUFFDEC_END_OF_BLOCK != 0 {
                    self.bitbuf >>= entry as u8;
                    self.bitsleft -= entry & 0xFF;
                    return DecompressResult::Success;
                }
                if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                    let main_bits = entry & 0xFF;
                    if self.bitsleft < main_bits {
                        return DecompressResult::ShortInput;
                    }
                    self.bitbuf >>= main_bits;
                    self.bitsleft -= main_bits;
                    let subtable_idx = (entry >> 16) as usize;
                    let subtable_bits = (entry >> 8) & 0x3F;
                    entry = self.litlen_decode_table
                        [subtable_idx + ((self.bitbuf as usize) & ((1 << subtable_bits) - 1))];
                }
            }
            let saved_bitbuf = self.bitbuf;
            let total_bits = entry & 0xFF;
            if self.bitsleft < total_bits {
                return DecompressResult::ShortInput;
            }
            self.bitbuf >>= total_bits;
            self.bitsleft -= total_bits;
            if entry & HUFFDEC_LITERAL != 0 {
                if *out_idx >= output.len() {
                    return DecompressResult::InsufficientSpace;
                }
                unsafe {
                    *output.get_unchecked_mut(*out_idx) = (entry >> 16) as u8;
                }
                *out_idx += 1;
            } else {
                let mut length = (entry >> 16) as usize;
                let len = (entry >> 8) & 0xFF;
                let extra_bits = total_bits - len;
                if extra_bits > 0 {
                    length += ((saved_bitbuf >> len) as usize) & ((1 << extra_bits) - 1);
                }
                refill_bits!(input, *in_idx, self.bitbuf, self.bitsleft);
                let mut entry = self.offset_decode_table
                    [(self.bitbuf as usize) & ((1 << OFFSET_TABLEBITS) - 1)];
                if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                    let main_bits = entry & 0xFF;
                    if self.bitsleft < main_bits {
                        self.state = DecompressorState::BlockBodyOffset {
                            length,
                            extra_bits: 0,
                        };
                        return DecompressResult::ShortInput;
                    }
                    self.bitbuf >>= main_bits;
                    self.bitsleft -= main_bits;
                    let subtable_idx = (entry >> 16) as usize;
                    let subtable_bits = (entry >> 8) & 0x3F;
                    entry = self.offset_decode_table
                        [subtable_idx + ((self.bitbuf as usize) & ((1 << subtable_bits) - 1))];
                }
                let saved_bitbuf = self.bitbuf;
                let total_bits = entry & 0xFF;
                if self.bitsleft < total_bits {
                    self.state = DecompressorState::BlockBodyOffset {
                        length,
                        extra_bits: 0,
                    };
                    return DecompressResult::ShortInput;
                }
                self.bitbuf >>= total_bits;
                self.bitsleft -= total_bits;
                let mut offset = (entry >> 16) as usize;
                let len = (entry >> 8) & 0xFF;
                let extra_bits = total_bits - len;
                if extra_bits > 0 {
                    offset += ((saved_bitbuf >> len) as usize) & ((1 << extra_bits) - 1);
                }
                if offset > *out_idx {
                    return DecompressResult::BadData;
                }
                let dest = *out_idx;
                let src = dest - offset;
                if dest + length > output.len() {
                    return DecompressResult::InsufficientSpace;
                }

                unsafe {
                    let out_ptr = output.as_mut_ptr();
                    if offset >= length {
                        std::ptr::copy_nonoverlapping(out_ptr.add(src), out_ptr.add(dest), length);
                    } else if offset == 1 {
                        let b = *out_ptr.add(src);
                        std::ptr::write_bytes(out_ptr.add(dest), b, length);
                    } else if offset < 8 {
                        let src_ptr = out_ptr.add(src);
                        let dest_ptr = out_ptr.add(dest);
                        if offset == 1 || offset == 2 || offset == 4 {
                            let pattern = prepare_pattern(offset, src_ptr);
                            let mut i = 0;
                            while i + 8 <= length {
                                std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                                i += 8;
                            }
                            while i < length {
                                *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                i += 1;
                            }
                        } else {
                            let mut copied = 0;
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        }
                    } else {
                        let mut copied = 0;
                        while copied < length {
                            let copy_len = std::cmp::min(offset, length - copied);
                            std::ptr::copy_nonoverlapping(
                                out_ptr.add(src + copied),
                                out_ptr.add(dest + copied),
                                copy_len,
                            );
                            copied += copy_len;
                        }
                    }
                }
                *out_idx += length;
            }
        }
    }

    pub fn decompress_zlib(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> (DecompressResult, usize, usize) {
        if input.len() < ZLIB_MIN_OVERHEAD {
            return (DecompressResult::ShortInput, 0, 0);
        }

        let hdr = u16::from_be_bytes([input[0], input[1]]);
        if hdr % 31 != 0 {
            return (DecompressResult::BadData, 0, 0);
        }
        if ((hdr >> 8) & 0xF) as u8 != ZLIB_CM_DEFLATE {
            return (DecompressResult::BadData, 0, 0);
        }
        if ((hdr >> 12) & 0xF) as u8 > ZLIB_CINFO_32K_WINDOW {
            return (DecompressResult::BadData, 0, 0);
        }
        if (hdr >> 5) & 1 != 0 {
            return (DecompressResult::BadData, 0, 0);
        }

        let (res, in_consumed, out_produced) =
            self.decompress(&input[2..input.len() - ZLIB_FOOTER_SIZE], output);

        if res != DecompressResult::Success {
            return (res, in_consumed + 2, out_produced);
        }

        let actual_adler = crate::adler32::adler32(1, &output[..out_produced]);
        let expected_adler = u32::from_be_bytes([
            input[2 + in_consumed],
            input[2 + in_consumed + 1],
            input[2 + in_consumed + 2],
            input[2 + in_consumed + 3],
        ]);

        if actual_adler != expected_adler {
            return (
                DecompressResult::BadData,
                in_consumed + 2 + ZLIB_FOOTER_SIZE,
                out_produced,
            );
        }

        (
            DecompressResult::Success,
            in_consumed + 2 + ZLIB_FOOTER_SIZE,
            out_produced,
        )
    }

    pub fn decompress_gzip(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> (DecompressResult, usize, usize) {
        if input.len() < GZIP_MIN_OVERHEAD {
            return (DecompressResult::ShortInput, 0, 0);
        }

        if input[0] != GZIP_ID1 || input[1] != GZIP_ID2 || input[2] != GZIP_CM_DEFLATE {
            return (DecompressResult::BadData, 0, 0);
        }

        let flg = input[3];
        if flg & GZIP_FRESERVED != 0 {
            return (DecompressResult::BadData, 0, 0);
        }

        let mut in_idx = 10;

        if flg & GZIP_FEXTRA != 0 {
            if in_idx + 2 > input.len() {
                return (DecompressResult::ShortInput, 0, 0);
            }
            let xlen = u16::from_le_bytes([input[in_idx], input[in_idx + 1]]) as usize;
            in_idx += 2 + xlen;
        }

        if flg & GZIP_FNAME != 0 {
            while in_idx < input.len() && input[in_idx] != 0 {
                in_idx += 1;
            }
            in_idx += 1;
        }

        if flg & GZIP_FCOMMENT != 0 {
            while in_idx < input.len() && input[in_idx] != 0 {
                in_idx += 1;
            }
            in_idx += 1;
        }

        if flg & GZIP_FHCRC != 0 {
            in_idx += 2;
        }

        if in_idx + GZIP_FOOTER_SIZE > input.len() {
            return (DecompressResult::ShortInput, 0, 0);
        }

        let (res, in_consumed, out_produced) =
            self.decompress(&input[in_idx..input.len() - GZIP_FOOTER_SIZE], output);

        if res != DecompressResult::Success {
            return (res, in_idx + in_consumed, out_produced);
        }

        let actual_crc = crate::crc32::crc32(0, &output[..out_produced]);
        let expected_crc = u32::from_le_bytes([
            input[in_idx + in_consumed],
            input[in_idx + in_consumed + 1],
            input[in_idx + in_consumed + 2],
            input[in_idx + in_consumed + 3],
        ]);

        if actual_crc != expected_crc {
            return (
                DecompressResult::BadData,
                in_idx + in_consumed + GZIP_FOOTER_SIZE,
                out_produced,
            );
        }

        let expected_isize = u32::from_le_bytes([
            input[in_idx + in_consumed + 4],
            input[in_idx + in_consumed + 5],
            input[in_idx + in_consumed + 6],
            input[in_idx + in_consumed + 7],
        ]);

        if (out_produced as u32) != expected_isize {
            return (
                DecompressResult::BadData,
                in_idx + in_consumed + GZIP_FOOTER_SIZE,
                out_produced,
            );
        }

        (
            DecompressResult::Success,
            in_idx + in_consumed + GZIP_FOOTER_SIZE,
            out_produced,
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn prepare_pattern(offset: usize, src_ptr: *const u8) -> u64 {
    unsafe {
        match offset {
            1 => {
                let b = *src_ptr as u64;
                b.wrapping_mul(0x0101010101010101)
            }
            2 => {
                let w = std::ptr::read_unaligned(src_ptr as *const u16) as u64;
                w | (w << 16) | (w << 32) | (w << 48)
            }
            3 => {
                // Optimization: Read u16 + u8 to avoid reading the uninitialized 4th byte.
                // Reading uninitialized memory is UB even if masked out later.
                let w = (src_ptr as *const u16).read_unaligned().to_le() as u64;
                let b = *src_ptr.add(2) as u64;
                let p = w | (b << 16);
                let p_le = p | (p << 24) | (p << 48);
                u64::from_le(p_le)
            }
            4 => {
                let w = std::ptr::read_unaligned(src_ptr as *const u32) as u64;
                w | (w << 32)
            }
            5 => {
                let d = (src_ptr as *const u32).read_unaligned().to_le() as u64;
                let b = *src_ptr.add(4) as u64;
                let p = d | (b << 32);
                let p_le = p | (p << 40);
                u64::from_le(p_le)
            }
            6 => {
                let d = (src_ptr as *const u32).read_unaligned().to_le() as u64;
                let w = (src_ptr.add(4) as *const u16).read_unaligned().to_le() as u64;
                let p = d | (w << 32);
                let p_le = p | (p << 48);
                u64::from_le(p_le)
            }
            7 => {
                let d = (src_ptr as *const u32).read_unaligned().to_le() as u64;
                let w = (src_ptr.add(4) as *const u16).read_unaligned().to_le() as u64;
                let b = *src_ptr.add(6) as u64;
                let p = d | (w << 32) | (b << 48);
                let p_le = p | (p << 56);
                u64::from_le(p_le)
            }
            _ => {
                let mut p_le = 0u64;
                for i in 0..offset {
                    p_le |= (*src_ptr.add(i) as u64) << (i * 8);
                }
                for i in 0..(8 - offset) {
                    p_le |= (*src_ptr.add(i) as u64) << ((offset + i) * 8);
                }
                u64::from_le(p_le)
            }
        }
    }
}

#[inline(always)]
fn make_decode_table_entry(decode_results: &[u32], sym: usize, len: u32) -> u32 {
    decode_results[sym] + (len << 8) + len
}

fn build_decode_table(
    decode_table: &mut [u32],
    lens: &[u8],
    num_syms: usize,
    decode_results: &[u32],
    mut table_bits: usize,
    max_codeword_len: usize,
    sorted_syms: &mut [u16],
    table_bits_ret: Option<&mut usize>,
) -> bool {
    let mut len_counts = [0u32; DEFLATE_MAX_CODEWORD_LEN + 1];
    let mut offsets = [0u32; DEFLATE_MAX_CODEWORD_LEN + 1];
    for &len in lens {
        if len as usize > DEFLATE_MAX_CODEWORD_LEN {
            return false;
        }
        len_counts[len as usize] += 1;
    }
    let mut actual_max_len = max_codeword_len;
    while actual_max_len > 1 && len_counts[actual_max_len] == 0 {
        actual_max_len -= 1;
    }
    if let Some(tb_ret) = table_bits_ret {
        table_bits = min(table_bits, actual_max_len);
        *tb_ret = table_bits;
    }
    offsets[0] = 0;
    offsets[1] = len_counts[0];
    let mut codespace_used = 0u32;
    for len in 1..actual_max_len {
        offsets[len + 1] = offsets[len] + len_counts[len];
        codespace_used = (codespace_used << 1) + len_counts[len];
    }
    codespace_used = (codespace_used << 1) + len_counts[actual_max_len];
    for (sym, &len) in lens.iter().enumerate() {
        if sym >= num_syms {
            break;
        }
        sorted_syms[offsets[len as usize] as usize] = sym as u16;
        offsets[len as usize] += 1;
    }
    if codespace_used > (1 << actual_max_len) {
        return false;
    }
    if codespace_used < (1 << actual_max_len) {
        let sym;
        if codespace_used == 0 {
            sym = 0;
        } else {
            if codespace_used != (1 << (actual_max_len - 1)) || len_counts[1] != 1 {
                return false;
            }
            sym = sorted_syms[offsets[0] as usize] as usize;
        }
        let entry = make_decode_table_entry(decode_results, sym, 1);
        for i in 0..(1 << table_bits) {
            decode_table[i] = entry;
        }
        return true;
    }
    let mut sym_ptr = offsets[0] as usize;
    let mut codeword = 0u32;
    let mut len = 1;
    while len <= DEFLATE_MAX_CODEWORD_LEN && len_counts[len] == 0 {
        len += 1;
    }
    if len > DEFLATE_MAX_CODEWORD_LEN {
        return true;
    }
    let mut cur_table_end: usize = 1 << len;
    let mut outer_iters = 0;
    while len <= table_bits {
        outer_iters += 1;
        if outer_iters > 100 {
            return false;
        }
        let mut count = len_counts[len];
        let mut inner_iters = 0;
        while count > 0 {
            inner_iters += 1;
            if inner_iters > 100000 {
                return false;
            }
            decode_table[codeword as usize] =
                make_decode_table_entry(decode_results, sorted_syms[sym_ptr] as usize, len as u32);
            sym_ptr += 1;
            if codeword == (cur_table_end as u32) - 1 {
                for _ in len..table_bits {
                    let size = cur_table_end;
                    for i in 0..size {
                        decode_table[size + i] = decode_table[i];
                    }
                    cur_table_end <<= 1;
                }
                return true;
            }
            let diff = codeword ^ ((cur_table_end as u32) - 1);
            if diff == 0 {
                codeword = 0;
            } else {
                let bit = 1 << bsr32(diff);
                codeword &= bit - 1;
                codeword |= bit;
            }
            count -= 1;
        }
        loop {
            len += 1;
            if len <= table_bits {
                let size = cur_table_end;
                for i in 0..size {
                    decode_table[size + i] = decode_table[i];
                }
                cur_table_end <<= 1;
            }
            if len > table_bits || len_counts[len] != 0 {
                break;
            }
        }
    }
    cur_table_end = 1 << table_bits;
    let mut subtable_prefix = !0u32;
    let mut subtable_start: usize = 0;
    loop {
        if (codeword & ((1 << table_bits) - 1)) != subtable_prefix {
            subtable_prefix = codeword & ((1 << table_bits) - 1);
            subtable_start = cur_table_end;
            let mut subtable_bits = len - table_bits;
            let mut sub_codespace_used = len_counts[len];
            while sub_codespace_used < (1 << subtable_bits) {
                subtable_bits += 1;
                sub_codespace_used = (sub_codespace_used << 1)
                    + if table_bits + subtable_bits <= DEFLATE_MAX_CODEWORD_LEN {
                        len_counts[table_bits + subtable_bits]
                    } else {
                        0
                    };
            }
            cur_table_end = subtable_start + (1 << subtable_bits);
            decode_table[subtable_prefix as usize] = ((subtable_start as u32) << 16)
                | HUFFDEC_EXCEPTIONAL
                | HUFFDEC_SUBTABLE_POINTER
                | ((subtable_bits as u32) << 8)
                | (table_bits as u32);
        }
        let entry = make_decode_table_entry(
            decode_results,
            sorted_syms[sym_ptr] as usize,
            (len - table_bits) as u32,
        );
        sym_ptr += 1;
        let mut i = subtable_start + (codeword >> table_bits) as usize;
        let stride = 1 << (len - table_bits);
        while i < cur_table_end {
            decode_table[i] = entry;
            i += stride;
        }
        if codeword == (1 << len) - 1 {
            return true;
        }
        let bit = 1 << bsr32(codeword ^ ((1 << len) - 1));
        codeword &= bit - 1;
        codeword |= bit;
        len_counts[len] -= 1;
        while len_counts[len] == 0 {
            len += 1;
            if len > DEFLATE_MAX_CODEWORD_LEN {
                return true;
            }
        }
    }
}
