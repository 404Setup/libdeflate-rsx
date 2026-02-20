#![allow(unsafe_op_in_unsafe_fn)]
use crate::decompress::tables::{
    HUFFDEC_END_OF_BLOCK, HUFFDEC_EXCEPTIONAL, HUFFDEC_LITERAL, HUFFDEC_SUBTABLE_POINTER,
    OFFSET_TABLEBITS,
};
use crate::decompress::{
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN,
    DEFLATE_BLOCKTYPE_UNCOMPRESSED, DecompressResult, Decompressor,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

static OFFSET9_MASKS: [u8; 144] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
];

// LCM(12, 16) = 48. 3 vectors.
static OFFSET12_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
];

// LCM(3, 16) = 48. 3 vectors.
static OFFSET3_MASKS: [u8; 48] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

// LCM(6, 16) = 48. 3 vectors.
static OFFSET6_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
    2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
];

// LCM(7, 16) = 112. 7 vectors.
static OFFSET7_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3,
    4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
    1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4,
    5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
];

// LCM(5, 16) = 80. 5 vectors.
static OFFSET5_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
    2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
    4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
];

// LCM(10, 16) = 80. 5 vectors.
static OFFSET10_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

// LCM(11, 16) = 176. 11 vectors.
static OFFSET11_MASKS: [u8; 176] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
];

// LCM(15, 16) = 240. 15 vectors.
static OFFSET15_MASKS: [u8; 240] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
];

// LCM(14, 16) = 112. 7 vectors.
static OFFSET14_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
];

// LCM(13, 16) = 208. 13 vectors.
static OFFSET13_MASKS: [u8; 208] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12,
];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,ssse3,sse4.1")]
pub unsafe fn decompress_bmi2(
    d: &mut Decompressor,
    input: &[u8],
    output: &mut [u8],
) -> (DecompressResult, usize, usize) {
    let mut out_idx = 0;
    let out_len = output.len();
    let mut in_idx = 0;
    let in_len = input.len();
    let mut bitbuf = 0u64;
    let mut bitsleft = 0u32;
    let mut is_final_block = false;

    while !is_final_block {
        refill_bits!(input, in_idx, bitbuf, bitsleft);

        is_final_block = (bitbuf & 1) != 0;
        let block_type = ((bitbuf >> 1) & 3) as u8;
        bitbuf >>= 3;
        bitsleft -= 3;

        match block_type {
            DEFLATE_BLOCKTYPE_UNCOMPRESSED => {
                let skip = bitsleft & 7;
                let _ = bitbuf >> skip;
                bitsleft -= skip;
                let unused_bytes = bitsleft / 8;
                in_idx -= unused_bytes as usize;
                bitbuf = 0;
                bitsleft = 0;
                if in_idx + 4 > in_len {
                    return (DecompressResult::BadData, 0, 0);
                }
                let len = u16::from_le_bytes([input[in_idx], input[in_idx + 1]]) as usize;
                let nlen = u16::from_le_bytes([input[in_idx + 2], input[in_idx + 3]]) as usize;
                in_idx += 4;
                if len != (!nlen & 0xFFFF) {
                    return (DecompressResult::BadData, 0, 0);
                }
                if out_idx + len > out_len {
                    return (DecompressResult::InsufficientSpace, 0, 0);
                }
                if in_idx + len > in_len {
                    return (DecompressResult::BadData, 0, 0);
                }
                std::ptr::copy_nonoverlapping(
                    input.as_ptr().add(in_idx),
                    output.as_mut_ptr().add(out_idx),
                    len,
                );
                in_idx += len;
                out_idx += len;
            }
            DEFLATE_BLOCKTYPE_STATIC_HUFFMAN | DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN => {
                if block_type == DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN {
                    d.bitbuf = bitbuf;
                    d.bitsleft = bitsleft;
                    let res = d.read_dynamic_huffman_header(input, &mut in_idx);
                    bitbuf = d.bitbuf;
                    bitsleft = d.bitsleft;
                    if res != DecompressResult::Success {
                        return (res, 0, 0);
                    }
                } else {
                    d.load_static_huffman_codes();
                }

                loop {
                    let mut eob_found = false;
                    unsafe {
                        let in_ptr_start = input.as_ptr();
                        let in_ptr_end = in_ptr_start.add(in_len);
                        let mut in_next = in_ptr_start.add(in_idx);
                        let out_ptr_start = output.as_mut_ptr();
                        let out_ptr_end = out_ptr_start.add(out_len);
                        let mut out_next = out_ptr_start.add(out_idx);

                        while in_next.add(15) <= in_ptr_end && out_next.add(258) <= out_ptr_end {
                            if bitsleft < 32 {
                                let word = (in_next as *const u64).read_unaligned();
                                let word = u64::from_le(word);
                                bitbuf |= word << bitsleft;
                                let consumed = (63 - bitsleft) >> 3;
                                in_next = in_next.add(consumed as usize);
                                bitsleft |= 56;
                            }

                            let table_idx = _bzhi_u64(bitbuf, d.litlen_tablebits as u32) as usize;
                            let mut entry = *d.litlen_decode_table.get_unchecked(table_idx);

                            if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                    bitbuf >>= entry as u8;
                                    bitsleft -= entry & 0xFF;
                                    eob_found = true;
                                    break;
                                }
                                if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                                    let saved_bitbuf = bitbuf;
                                    let saved_bitsleft = bitsleft;
                                    let main_bits = entry & 0xFF;
                                    bitbuf >>= main_bits;
                                    bitsleft -= main_bits;
                                    let subtable_idx = (entry >> 16) as usize;
                                    let subtable_bits = (entry >> 8) & 0x3F;
                                    let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                                    entry = *d
                                        .litlen_decode_table
                                        .get_unchecked(subtable_idx + sub_idx);
                                    if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                        if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                            bitbuf >>= entry as u8;
                                            bitsleft -= entry & 0xFF;
                                            break;
                                        }
                                        bitbuf = saved_bitbuf;
                                        bitsleft = saved_bitsleft;
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            let saved_bitbuf = bitbuf;
                            let total_bits = entry & 0xFF;
                            if bitsleft < total_bits {
                                break;
                            }
                            bitbuf >>= total_bits;
                            bitsleft -= total_bits;

                            if entry & HUFFDEC_LITERAL != 0 {
                                *out_next = (entry >> 16) as u8;
                                out_next = out_next.add(1);
                            } else {
                                let mut length = (entry >> 16) as usize;
                                let len = (entry >> 8) & 0xFF;
                                let extra_bits = total_bits - len;
                                if extra_bits > 0 {
                                    length += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                                }

                                if bitsleft < 32 {
                                    let word = (in_next as *const u64).read_unaligned();
                                    let word = u64::from_le(word);
                                    bitbuf |= word << bitsleft;
                                    let consumed = (63 - bitsleft) >> 3;
                                    in_next = in_next.add(consumed as usize);
                                    bitsleft |= 56;
                                }

                                let offset_idx =
                                    _bzhi_u64(bitbuf, OFFSET_TABLEBITS as u32) as usize;
                                let mut entry = *d.offset_decode_table.get_unchecked(offset_idx);

                                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                    if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                                        let main_bits = entry & 0xFF;
                                        bitbuf >>= main_bits;
                                        bitsleft -= main_bits;
                                        let subtable_idx = (entry >> 16) as usize;
                                        let subtable_bits = (entry >> 8) & 0x3F;
                                        let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                                        entry = *d
                                            .offset_decode_table
                                            .get_unchecked(subtable_idx + sub_idx);
                                    } else {
                                        break;
                                    }
                                }

                                let saved_bitbuf = bitbuf;
                                let total_bits = entry & 0xFF;
                                bitbuf >>= total_bits;
                                bitsleft -= total_bits;

                                let mut offset = (entry >> 16) as usize;
                                let len = (entry >> 8) & 0xFF;
                                let extra_bits = total_bits - len;
                                if extra_bits > 0 {
                                    offset += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                                }

                                let current_out_idx = out_next.offset_from(out_ptr_start) as usize;
                                if offset > current_out_idx {
                                    break;
                                }

                                let src = out_next.sub(offset);
                                if offset >= 16 {
                                    let v = _mm_loadu_si128(src as *const __m128i);
                                    _mm_storeu_si128(out_next as *mut __m128i, v);
                                    if length > 16 {
                                        if offset >= length {
                                            std::ptr::copy_nonoverlapping(
                                                src.add(16),
                                                out_next.add(16),
                                                length - 16,
                                            );
                                        } else {
                                            match offset {
                                                16 => {
                                                    let mut copied = 16;
                                                    while copied + 64 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 64;
                                                    }
                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                62 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Synthesize v3: src[48..64]. Bytes 48..62 are valid.
                                                    // Bytes 62..64 (2 bytes) wrap around to 0..2.
                                                    // We load src[48..64] (last 2 bytes are garbage/out_next).
                                                    let v3_raw = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v0 = v;
                                                    // Shift v0 (0..16) left by 14 bytes to place 0..2 at 14..16.
                                                    let v0_shifted = _mm_slli_si128(v0, 14);
                                                    // Blend: Keep first 7 words (14 bytes) of v3_raw, take last 1 word (2 bytes) of v0_shifted.
                                                    // Mask 0x80 (10000000) selects upper 1 word from v0_shifted.
                                                    let v3 =
                                                        _mm_blend_epi16(v3_raw, v0_shifted, 0x80);

                                                    let mut copied = 16;
                                                    let mut v0 = v;
                                                    let mut v1 = v1;
                                                    let mut v2 = v2;
                                                    let mut v3 = v3;

                                                    while copied + 64 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 2);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 2);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 2);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 2);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                        copied += 64;
                                                    }

                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 2);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            next_v0,
                                                        );
                                                        copied += 16;

                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 2);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 2);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 2);
                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                54 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Synthesize v3: src[48..64]. Bytes 48..54 are valid.
                                                    // Bytes 54..64 (10 bytes) wrap around to 0..10.
                                                    let v3_raw = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v0 = v;
                                                    // Shift v0 (0..16) left by 6 bytes to place 0..10 at 6..16.
                                                    let v0_shifted = _mm_slli_si128(v0, 6);
                                                    // Blend: Keep first 3 words (6 bytes) of v3_raw, take last 5 words (10 bytes) of v0_shifted.
                                                    // Mask 0xF8 (11111000) selects upper 5 words from v0_shifted.
                                                    let v3 =
                                                        _mm_blend_epi16(v3_raw, v0_shifted, 0xF8);

                                                    let mut copied = 16;
                                                    let mut v0 = v;
                                                    let mut v1 = v1;
                                                    let mut v2 = v2;
                                                    let mut v3 = v3;

                                                    while copied + 64 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 10);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 10);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 10);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 10);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                        copied += 64;
                                                    }

                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            next_v0,
                                                        );
                                                        copied += 16;

                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 10);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 10);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 10);
                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                46 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2_raw = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    let mut v0 = v;
                                                    let v0_shifted = _mm_slli_si128(v0, 14);
                                                    let mut v2 =
                                                        _mm_blend_epi16(v2_raw, v0_shifted, 0x80);
                                                    let mut v1 = v1;

                                                    let mut copied = 16;
                                                    while copied + 48 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 2);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 2);
                                                        let next_v2 =
                                                            _mm_alignr_epi8(next_v0, v2, 2);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        copied += 48;
                                                    }

                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        let next = _mm_alignr_epi8(v1, v0, 2);
                                                        v0 = v1;
                                                        v1 = v2;
                                                        v2 = next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                34 => {
                                                    let mut copied = 16;
                                                    let v1_init = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(18) as *const __m128i
                                                    );
                                                    let mut v0 = v;
                                                    let mut v1 = v1_init;
                                                    let mut v2 = _mm_alignr_epi8(v0, v_tail, 14);

                                                    while copied + 48 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
                                                        let next_v2 =
                                                            _mm_alignr_epi8(next_v0, v2, 14);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        copied += 48;
                                                    }

                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        let next_v = _mm_alignr_epi8(v1, v0, 14);
                                                        v0 = v1;
                                                        v1 = v2;
                                                        v2 = next_v;
                                                        copied += 16;
                                                    }

                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                50 => {
                                                    let mut copied = 16;
                                                    let mut v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let mut v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(34) as *const __m128i
                                                    );
                                                    let mut v0 = v;
                                                    let mut v3 = _mm_alignr_epi8(v0, v_tail, 14);

                                                    while copied + 64 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 14);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 14);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                        copied += 64;
                                                    }

                                                    loop {
                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 14);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            next_v0,
                                                        );
                                                        copied += 16;

                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 14);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 14);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 14);
                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                    }

                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                38 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(22) as *const __m128i
                                                    );
                                                    let mut v0 = v;
                                                    let mut v1 = v1;
                                                    let mut v2 = _mm_alignr_epi8(v0, v_tail, 10);

                                                    let mut copied = 16;
                                                    while copied + 48 <= length {
                                                        let new_v0 = _mm_alignr_epi8(v1, v0, 10);
                                                        let new_v1 = _mm_alignr_epi8(v2, v1, 10);
                                                        let new_v2 =
                                                            _mm_alignr_epi8(new_v0, v2, 10);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            new_v0,
                                                        );

                                                        v0 = new_v0;
                                                        v1 = new_v1;
                                                        v2 = new_v2;
                                                        copied += 48;
                                                    }
                                                    while copied + 16 <= length {
                                                        let new_v = _mm_alignr_epi8(v1, v0, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        v0 = v1;
                                                        v1 = v2;
                                                        v2 = new_v;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                42 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(26) as *const __m128i
                                                    );
                                                    let mut v0 = v;
                                                    let mut v1 = v1;
                                                    let mut v2 = _mm_alignr_epi8(v0, v_tail, 6);

                                                    let mut copied = 16;
                                                    while copied + 48 <= length {
                                                        let new_v0 = _mm_alignr_epi8(v1, v0, 6);
                                                        let new_v1 = _mm_alignr_epi8(v2, v1, 6);
                                                        let new_v2 = _mm_alignr_epi8(new_v0, v2, 6);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            new_v0,
                                                        );

                                                        v0 = new_v0;
                                                        v1 = new_v1;
                                                        v2 = new_v2;
                                                        copied += 48;
                                                    }
                                                    while copied + 16 <= length {
                                                        let new_v = _mm_alignr_epi8(v1, v0, 6);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        v0 = v1;
                                                        v1 = v2;
                                                        v2 = new_v;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                58 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Synthesize v3: src[48..64]. Bytes 48..58 are valid.
                                                    // Bytes 58..64 (6 bytes) wrap around to 0..6.
                                                    // We load src[48..64] (last 6 bytes are garbage/out_next).
                                                    let v3_raw = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v0 = v;
                                                    // Shift v0 (0..16) left by 10 bytes to place 0..6 at 10..16.
                                                    let v0_shifted = _mm_slli_si128(v0, 10);
                                                    // Blend: Keep first 5 words (10 bytes) of v3_raw, take last 3 words (6 bytes) of v0_shifted.
                                                    // Mask 0xE0 (11100000) selects upper 3 words from v0_shifted.
                                                    let v3 =
                                                        _mm_blend_epi16(v3_raw, v0_shifted, 0xE0);

                                                    let mut copied = 16;
                                                    let mut v0 = v;
                                                    let mut v1 = v1;
                                                    let mut v2 = v2;
                                                    let mut v3 = v3;

                                                    while copied + 64 <= length {
                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 6);
                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 6);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 6);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 6);

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            next_v0,
                                                        );

                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                        copied += 64;
                                                    }

                                                    while copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;
                                                        if copied + 16 > length {
                                                            break;
                                                        }

                                                        let next_v0 = _mm_alignr_epi8(v1, v0, 6);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            next_v0,
                                                        );
                                                        copied += 16;

                                                        let next_v1 = _mm_alignr_epi8(v2, v1, 6);
                                                        let next_v2 = _mm_alignr_epi8(v3, v2, 6);
                                                        let next_v3 =
                                                            _mm_alignr_epi8(next_v0, v3, 6);
                                                        v0 = next_v0;
                                                        v1 = next_v1;
                                                        v2 = next_v2;
                                                        v3 = next_v3;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                60 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Explicitly construct v3 from src[48..64] but overwriting the last 4 bytes
                                                    // (which overlap with out_next[0..4]) with valid data from v0.
                                                    // This avoids relying on the read-after-write behavior of overlapping pointers.
                                                    let v3_raw = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v0 = v;
                                                    // Shift v0 (0..16) left by 12 bytes to place 0..4 at 12..16
                                                    let v0_shifted = _mm_slli_si128(v0, 12);
                                                    // Blend: Mask 0xC0 selects upper 2 words (upper 4 bytes) from v0_shifted.
                                                    let v3 =
                                                        _mm_blend_epi16(v3_raw, v0_shifted, 0xC0);

                                                    let v4 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v7 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v8 = _mm_alignr_epi8(v5, v4, 4);
                                                    let v9 = _mm_alignr_epi8(v6, v5, 4);
                                                    let v10 = _mm_alignr_epi8(v7, v6, 4);
                                                    let v11 = _mm_alignr_epi8(v8, v7, 4);
                                                    let v12 = _mm_alignr_epi8(v9, v8, 4);
                                                    let v13 = _mm_alignr_epi8(v10, v9, 4);
                                                    let v14 = _mm_alignr_epi8(v11, v10, 4);

                                                    let mut copied = 16;
                                                    while copied + 240 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v9,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 144)
                                                                as *mut __m128i,
                                                            v10,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 160)
                                                                as *mut __m128i,
                                                            v11,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 176)
                                                                as *mut __m128i,
                                                            v12,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 192)
                                                                as *mut __m128i,
                                                            v13,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 208)
                                                                as *mut __m128i,
                                                            v14,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 224)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 240;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 240) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            7 => v7,
                                                            8 => v8,
                                                            9 => v9,
                                                            10 => v10,
                                                            11 => v11,
                                                            12 => v12,
                                                            13 => v13,
                                                            14 => v14,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                44 => {
                                                    let v0 = v;
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v_end = _mm_loadu_si128(
                                                        src.add(28) as *const __m128i
                                                    );
                                                    let v2 = _mm_alignr_epi8(v0, v_end, 4);
                                                    let v3 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v4 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v5 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v6 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v7 = _mm_alignr_epi8(v5, v4, 4);
                                                    let v8 = _mm_alignr_epi8(v6, v5, 4);
                                                    let v9 = _mm_alignr_epi8(v7, v6, 4);
                                                    let v10 = _mm_alignr_epi8(v8, v7, 4);

                                                    let mut copied = 16;
                                                    while copied + 176 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v9,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 144)
                                                                as *mut __m128i,
                                                            v10,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 160)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 176;
                                                    }
                                                    // Remainder loop. We know the next vector is v1 because copied starts at 16
                                                    // and the large loop steps by 176 (full cycle).
                                                    loop {
                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v4,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v5,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v6,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v7,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v8,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v9,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v10,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                36 => {
                                                    let mut copied = 16;
                                                    let v_part = _mm_loadu_si128(
                                                        src.add(20) as *const __m128i
                                                    );
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v0 = v;

                                                    let v2 = _mm_alignr_epi8(v0, v_part, 12);
                                                    let v3 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v4 = _mm_alignr_epi8(v0, v_part, 8);
                                                    let v5 = _mm_alignr_epi8(v1, v0, 8);
                                                    let v6 = _mm_alignr_epi8(v0, v_part, 4);
                                                    let v7 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v8 = v_part;

                                                    while copied + 144 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 144;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 144) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            7 => v7,
                                                            8 => v8,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                52 => {
                                                    let v0 = v;
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Load tail (36..52) which contains 48..52 at [12..16]
                                                    let v_tail = _mm_loadu_si128(
                                                        src.add(36) as *const __m128i
                                                    );
                                                    // v3: 48..52 ++ 0..12
                                                    let v3 = _mm_alignr_epi8(v0, v_tail, 12);

                                                    // The cycle uses shift 12 for all subsequent vectors
                                                    let v4 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 12);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 12);
                                                    let v7 = _mm_alignr_epi8(v4, v3, 12);
                                                    let v8 = _mm_alignr_epi8(v5, v4, 12);
                                                    let v9 = _mm_alignr_epi8(v6, v5, 12);
                                                    let v10 = _mm_alignr_epi8(v7, v6, 12);
                                                    let v11 = _mm_alignr_epi8(v8, v7, 12);
                                                    let v12 = _mm_alignr_epi8(v9, v8, 12);

                                                    let mut copied = 16;
                                                    while copied + 208 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v9,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 144)
                                                                as *mut __m128i,
                                                            v10,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 160)
                                                                as *mut __m128i,
                                                            v11,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 176)
                                                                as *mut __m128i,
                                                            v12,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 192)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 208;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 208) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            7 => v7,
                                                            8 => v8,
                                                            9 => v9,
                                                            10 => v10,
                                                            11 => v11,
                                                            12 => v12,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                56 => {
                                                    // LCM(56, 16) = 112.
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    // Optimize v3 construction to avoid store-forwarding stall.
                                                    // v3 should be [src[48..56], src[0..8]].
                                                    // src[0..8] is already in v0[0..8].
                                                    // We load src[48..56] using loadl_epi64.
                                                    let v0 = v;
                                                    let v3_low = _mm_loadl_epi64(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let v3 = _mm_unpacklo_epi64(v3_low, v0);

                                                    let v4 = _mm_alignr_epi8(v1, v0, 8);
                                                    let v5 = _mm_alignr_epi8(v2, v1, 8);
                                                    let v6 = _mm_alignr_epi8(v3, v2, 8);

                                                    let mut copied = 16;
                                                    while copied + 112 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 112;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 112) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                48 => {
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v3 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    let mut copied = 16;
                                                    while copied + 48 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 48;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 48) / 16;
                                                        let v_next = match idx {
                                                            1 => v2,
                                                            2 => v3,
                                                            _ => v,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                64 => {
                                                    let v1 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(32) as *const __m128i
                                                    );
                                                    let v3 = _mm_loadu_si128(
                                                        src.add(48) as *const __m128i
                                                    );
                                                    let mut copied = 16;

                                                    while copied + 64 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 64;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                17 => {
                                                    let mut copied = 16;
                                                    let c = *src.add(16);
                                                    let mut v_align =
                                                        _mm_insert_epi8(v, c as i32, 15);
                                                    let mut v_prev = v;

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 15);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                18 => {
                                                    let mut copied = 16;
                                                    let mut v_prev = v;
                                                    let c1 = *src.add(16);
                                                    let c2 = *src.add(17);
                                                    let mut v_align =
                                                        _mm_insert_epi8(v_prev, c1 as i32, 14);
                                                    v_align =
                                                        _mm_insert_epi8(v_align, c2 as i32, 15);

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 14);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    while copied < length {
                                                        let copy_len =
                                                            std::cmp::min(offset, length - copied);
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            copy_len,
                                                        );
                                                        copied += copy_len;
                                                    }
                                                }
                                                19 => {
                                                    let mut copied = 16;
                                                    let c1 = *src.add(16);
                                                    let c2 = *src.add(17);
                                                    let c3 = *src.add(18);
                                                    let mut v_align =
                                                        _mm_insert_epi8(v, c1 as i32, 13);
                                                    v_align =
                                                        _mm_insert_epi8(v_align, c2 as i32, 14);
                                                    v_align =
                                                        _mm_insert_epi8(v_align, c3 as i32, 15);
                                                    let mut v_prev = v;

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 13);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                20 => {
                                                    let mut copied = 16;
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32
                                                    );
                                                    let v_temp = _mm_cvtsi32_si128(val as i32);
                                                    let v_align = _mm_slli_si128(v_temp, 12);

                                                    let v0 = v;
                                                    let v1 = _mm_alignr_epi8(v0, v_align, 12);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 12);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 12);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 12);

                                                    while copied + 80 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 80;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 80) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                21 => {
                                                    let mut copied = 16;
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u64
                                                    );
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let mut v_align = _mm_slli_si128(v_temp, 11);
                                                    let mut v_prev = v;

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 11);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                22 => {
                                                    let mut copied = 16;
                                                    let v0 = v;
                                                    let v_align_low = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32,
                                                    );
                                                    let v_align_high = std::ptr::read_unaligned(
                                                        src.add(20) as *const u16,
                                                    );
                                                    let v_align_val = (v_align_low as u64)
                                                        | ((v_align_high as u64) << 32);
                                                    let v_tail = _mm_slli_si128(
                                                        _mm_cvtsi64_si128(v_align_val as i64),
                                                        10,
                                                    );

                                                    while copied + 176 <= length {
                                                        let v1 = _mm_alignr_epi8(v0, v_tail, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        let v2 = _mm_alignr_epi8(v1, v0, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        let v3 = _mm_alignr_epi8(v2, v1, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        let v4 = _mm_alignr_epi8(v3, v2, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        let v5 = _mm_alignr_epi8(v4, v3, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        let v6 = _mm_alignr_epi8(v5, v4, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        let v7 = _mm_alignr_epi8(v6, v5, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        let v8 = _mm_alignr_epi8(v7, v6, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        let v9 = _mm_alignr_epi8(v8, v7, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v9,
                                                        );
                                                        let v10 = _mm_alignr_epi8(v9, v8, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 144)
                                                                as *mut __m128i,
                                                            v10,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 160)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 176;
                                                    }

                                                    // Remainder loop.
                                                    loop {
                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v1 = _mm_alignr_epi8(v0, v_tail, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v2 = _mm_alignr_epi8(v1, v0, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v3 = _mm_alignr_epi8(v2, v1, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v3,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v4 = _mm_alignr_epi8(v3, v2, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v4,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v5 = _mm_alignr_epi8(v4, v3, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v5,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v6 = _mm_alignr_epi8(v5, v4, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v6,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v7 = _mm_alignr_epi8(v6, v5, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v7,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v8 = _mm_alignr_epi8(v7, v6, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v8,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v9 = _mm_alignr_epi8(v8, v7, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v9,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        let v10 = _mm_alignr_epi8(v9, v8, 10);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v10,
                                                        );
                                                        copied += 16;

                                                        if copied + 16 > length {
                                                            break;
                                                        }
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 16;
                                                    }

                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                23 => {
                                                    let mut copied = 16;
                                                    let v0 = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32
                                                    );
                                                    let v1 = std::ptr::read_unaligned(
                                                        src.add(19) as *const u32
                                                    );
                                                    let val = (v0 as u64) | ((v1 as u64) << 24);
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let mut v_align = _mm_slli_si128(v_temp, 9);
                                                    let mut v_prev = v;

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 9);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                24 => {
                                                    let mut copied = 16;
                                                    let v_part1 = std::ptr::read_unaligned(
                                                        src.add(16) as *const u32,
                                                    );
                                                    let v_part2 = std::ptr::read_unaligned(
                                                        src.add(20) as *const u32,
                                                    );
                                                    let val =
                                                        (v_part1 as u64) | ((v_part2 as u64) << 32);
                                                    let v_tail = _mm_cvtsi64_si128(val as i64);

                                                    let v0 = v;
                                                    let v1 = _mm_unpacklo_epi64(v_tail, v0);
                                                    let v2 = _mm_alignr_epi8(v_tail, v0, 8);

                                                    while copied + 48 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 48;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 48) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                25 => {
                                                    let mut copied = 16;
                                                    let val = std::ptr::read_unaligned(
                                                        src.add(16) as *const u64
                                                    );
                                                    let c = *src.add(24);
                                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                                    let v_temp =
                                                        _mm_insert_epi8(v_temp, c as i32, 8);
                                                    let mut v_align = _mm_slli_si128(v_temp, 7);
                                                    let mut v_prev = v;

                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 7);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                26 => {
                                                    // LCM(26, 16) = 208. 13 vectors.
                                                    // The pattern repeats every 26 bytes.
                                                    // We advance by 16 bytes each step.
                                                    // Next vector is constructed from the previous two shifted by 16 bytes.
                                                    // Since 26 % 16 = 10, the relative alignment shifts by 10 bytes each step.
                                                    // To align, we need to shift right by 16 - 10 = 6 bytes from the concatenated previous vectors.
                                                    // v_align (src[10..26]) acts as v_{-1}.
                                                    let mut copied = 16;
                                                    let v0 = v;
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(10) as *const __m128i
                                                    );

                                                    let v1 = _mm_alignr_epi8(v0, v_align, 6);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 6);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 6);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 6);
                                                    let v5 = _mm_alignr_epi8(v4, v3, 6);
                                                    let v6 = _mm_alignr_epi8(v5, v4, 6);
                                                    let v7 = _mm_alignr_epi8(v6, v5, 6);
                                                    let v8 = _mm_alignr_epi8(v7, v6, 6);
                                                    let v9 = _mm_alignr_epi8(v8, v7, 6);
                                                    let v10 = _mm_alignr_epi8(v9, v8, 6);
                                                    let v11 = _mm_alignr_epi8(v10, v9, 6);
                                                    let v12 = _mm_alignr_epi8(v11, v10, 6);

                                                    while copied + 208 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v7,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 112)
                                                                as *mut __m128i,
                                                            v8,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 128)
                                                                as *mut __m128i,
                                                            v9,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 144)
                                                                as *mut __m128i,
                                                            v10,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 160)
                                                                as *mut __m128i,
                                                            v11,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 176)
                                                                as *mut __m128i,
                                                            v12,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 192)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 208;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 208) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            7 => v7,
                                                            8 => v8,
                                                            9 => v9,
                                                            10 => v10,
                                                            11 => v11,
                                                            12 => v12,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                27 => {
                                                    let mut copied = 16;
                                                    let mut v_align = _mm_loadu_si128(
                                                        src.add(11) as *const __m128i
                                                    );
                                                    let mut v_prev = v;
                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 5);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                28 => {
                                                    let mut copied = 16;
                                                    let v_align = _mm_loadu_si128(
                                                        src.add(12) as *const __m128i
                                                    );

                                                    let v0 = v;
                                                    let v1 = _mm_alignr_epi8(v0, v_align, 4);
                                                    let v2 = _mm_alignr_epi8(v1, v0, 4);
                                                    let v3 = _mm_alignr_epi8(v2, v1, 4);
                                                    let v4 = _mm_alignr_epi8(v3, v2, 4);
                                                    let v5 = _mm_alignr_epi8(v4, v3, 4);
                                                    let v6 = _mm_alignr_epi8(v5, v4, 4);

                                                    while copied + 112 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v1,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v3,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v4,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 64)
                                                                as *mut __m128i,
                                                            v5,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 80)
                                                                as *mut __m128i,
                                                            v6,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 96)
                                                                as *mut __m128i,
                                                            v0,
                                                        );
                                                        copied += 112;
                                                    }
                                                    while copied + 16 <= length {
                                                        let idx = (copied % 112) / 16;
                                                        let v_next = match idx {
                                                            1 => v1,
                                                            2 => v2,
                                                            3 => v3,
                                                            4 => v4,
                                                            5 => v5,
                                                            6 => v6,
                                                            _ => v0,
                                                        };
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                29 => {
                                                    let mut copied = 16;
                                                    let mut v_align = _mm_loadu_si128(
                                                        src.add(13) as *const __m128i
                                                    );
                                                    let mut v_prev = v;
                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 3);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                30 => {
                                                    let mut copied = 16;
                                                    let mut v_align = _mm_loadu_si128(
                                                        src.add(14) as *const __m128i
                                                    );
                                                    let mut v_prev = v;
                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 2);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                31 => {
                                                    let mut copied = 16;
                                                    let mut v_align = _mm_loadu_si128(
                                                        src.add(15) as *const __m128i
                                                    );
                                                    let mut v_prev = v;
                                                    while copied + 16 <= length {
                                                        let v_next =
                                                            _mm_alignr_epi8(v_prev, v_align, 1);
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v_next,
                                                        );
                                                        v_align = v_prev;
                                                        v_prev = v_next;
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                32 => {
                                                    let v2 = _mm_loadu_si128(
                                                        src.add(16) as *const __m128i
                                                    );
                                                    let mut copied = 16;
                                                    while copied + 64 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 32)
                                                                as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 48)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 64;
                                                    }
                                                    while copied + 32 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(copied + 16)
                                                                as *mut __m128i,
                                                            v,
                                                        );
                                                        copied += 32;
                                                    }
                                                    if copied + 16 <= length {
                                                        _mm_storeu_si128(
                                                            out_next.add(copied) as *mut __m128i,
                                                            v2,
                                                        );
                                                        copied += 16;
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                40 => {
                                                    let mut copied = 16;
                                                    if length >= 40 {
                                                        let v_part2 = _mm_loadu_si128(
                                                            src.add(16) as *const __m128i
                                                        );
                                                        let v_part3 = _mm_loadu_si128(
                                                            src.add(24) as *const __m128i
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(16) as *mut __m128i,
                                                            v_part2,
                                                        );
                                                        _mm_storeu_si128(
                                                            out_next.add(24) as *mut __m128i,
                                                            v_part3,
                                                        );

                                                        let v0 = v;
                                                        let v1 = v_part2;
                                                        let v4 = v_part3;

                                                        let v2 = _mm_alignr_epi8(v0, v4, 8);
                                                        let v3 = _mm_alignr_epi8(v1, v0, 8);

                                                        copied = 40;
                                                        while copied + 80 <= length {
                                                            _mm_storeu_si128(
                                                                out_next.add(copied)
                                                                    as *mut __m128i,
                                                                v0,
                                                            );
                                                            _mm_storeu_si128(
                                                                out_next.add(copied + 16)
                                                                    as *mut __m128i,
                                                                v1,
                                                            );
                                                            _mm_storeu_si128(
                                                                out_next.add(copied + 32)
                                                                    as *mut __m128i,
                                                                v2,
                                                            );
                                                            _mm_storeu_si128(
                                                                out_next.add(copied + 48)
                                                                    as *mut __m128i,
                                                                v3,
                                                            );
                                                            _mm_storeu_si128(
                                                                out_next.add(copied + 64)
                                                                    as *mut __m128i,
                                                                v4,
                                                            );
                                                            copied += 80;
                                                        }
                                                        while copied + 16 <= length {
                                                            let idx = ((copied - 40) % 80) / 16;
                                                            let v_tail = match idx {
                                                                0 => v0,
                                                                1 => v1,
                                                                2 => v2,
                                                                3 => v3,
                                                                _ => v4,
                                                            };
                                                            _mm_storeu_si128(
                                                                out_next.add(copied)
                                                                    as *mut __m128i,
                                                                v_tail,
                                                            );
                                                            copied += 16;
                                                        }
                                                    }
                                                    if copied < length {
                                                        std::ptr::copy_nonoverlapping(
                                                            src.add(copied),
                                                            out_next.add(copied),
                                                            length - copied,
                                                        );
                                                    }
                                                }
                                                _ => {
                                                    let init = std::cmp::min(offset, length);
                                                    std::ptr::copy_nonoverlapping(
                                                        src, out_next, init,
                                                    );

                                                    let mut copied = init;
                                                    while copied < length {
                                                        let to_copy =
                                                            std::cmp::min(length - copied, copied);
                                                        std::ptr::copy_nonoverlapping(
                                                            out_next,
                                                            out_next.add(copied),
                                                            to_copy,
                                                        );
                                                        copied += to_copy;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else if offset >= length {
                                    std::ptr::copy_nonoverlapping(src, out_next, length);
                                } else {
                                    match offset {
                                        1 => {
                                            let b = *src;
                                            std::ptr::write_bytes(out_next, b, length);
                                        }
                                        2 | 4 => {
                                            let v_pattern = match offset {
                                                2 => _mm_set1_epi16(std::ptr::read_unaligned(
                                                    src as *const u16,
                                                )
                                                    as i16),
                                                4 => _mm_set1_epi32(std::ptr::read_unaligned(
                                                    src as *const u32,
                                                )
                                                    as i32),
                                                _ => std::hint::unreachable_unchecked(),
                                            };
                                            let mut i = 0;
                                            while i + 64 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 16) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 32) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 48) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 64;
                                            }
                                            while i + 32 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 16) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 32;
                                            }
                                            if i + 16 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 16;
                                            }
                                            let pattern = _mm_cvtsi128_si64(v_pattern) as u64;
                                            while i + 8 <= length {
                                                std::ptr::write_unaligned(
                                                    out_next.add(i) as *mut u64,
                                                    pattern,
                                                );
                                                i += 8;
                                            }
                                            while i < length {
                                                *out_next.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                                i += 1;
                                            }
                                        }
                                        3 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET3_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 48 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    copied += 48;
                                                }

                                                while copied + 16 <= length {
                                                    let idx = (copied % 48) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }

                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        6 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET6_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 48 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    copied += 48;
                                                }

                                                while copied + 16 <= length {
                                                    let idx = (copied % 48) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }

                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        5 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET5_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 80 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(3)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(4)),
                                                        ),
                                                    );
                                                    copied += 80;
                                                }

                                                while copied + 16 <= length {
                                                    let idx = (copied % 80) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }

                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        7 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET7_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 112 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    for i in 1..7 {
                                                        _mm_storeu_si128(
                                                            dest_ptr.add(copied + i * 16)
                                                                as *mut __m128i,
                                                            _mm_shuffle_epi8(
                                                                v_src,
                                                                _mm_loadu_si128(masks_ptr.add(i)),
                                                            ),
                                                        );
                                                    }
                                                    copied += 112;
                                                }

                                                while copied + 16 <= length {
                                                    let idx = (copied % 112) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }

                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        8 => {
                                            let val = std::ptr::read_unaligned(src as *const u64);
                                            let v_pattern = _mm_set1_epi64x(val as i64);
                                            _mm_storeu_si128(out_next as *mut __m128i, v_pattern);
                                            let mut i = 16;
                                            while i + 64 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 16) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 32) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 48) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 64;
                                            }
                                            while i + 32 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(i + 16) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 32;
                                            }
                                            if i + 16 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(i) as *mut __m128i,
                                                    v_pattern,
                                                );
                                                i += 16;
                                            }
                                            let pattern = _mm_cvtsi128_si64(v_pattern) as u64;
                                            while i + 8 <= length {
                                                std::ptr::write_unaligned(
                                                    out_next.add(i) as *mut u64,
                                                    pattern,
                                                );
                                                i += 8;
                                            }
                                            while i < length {
                                                *out_next.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                                i += 1;
                                            }
                                        }
                                        9 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;

                                            let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                            let masks_ptr =
                                                OFFSET9_MASKS.as_ptr() as *const __m128i;
                                            let v_base =
                                                _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                            _mm_storeu_si128(dest_ptr as *mut __m128i, v_base);
                                            let mut copied = 16;
                                            while copied + 144 <= length {
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied) as *mut __m128i,
                                                    v_base,
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 16) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(1)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 32) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(2)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 48) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(3)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 64) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(4)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 80) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(5)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 96) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(6)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 112) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(7)),
                                                    ),
                                                );
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + 128) as *mut __m128i,
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(8)),
                                                    ),
                                                );
                                                copied += 144;
                                            }

                                            while copied + 16 <= length {
                                                let idx = (copied % 144) / 16;
                                                let v = if idx == 0 {
                                                    v_base
                                                } else {
                                                    _mm_shuffle_epi8(
                                                        v_base,
                                                        _mm_loadu_si128(masks_ptr.add(idx)),
                                                    )
                                                };
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied) as *mut __m128i,
                                                    v,
                                                );
                                                copied += 16;
                                            }

                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        10 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET10_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 80 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(3)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(4)),
                                                        ),
                                                    );
                                                    copied += 80;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 80) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        11 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET11_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 176 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(3)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(4)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 80) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(5)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 96) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(6)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 112) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(7)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 128) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(8)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 144) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(9)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 160) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(10)),
                                                        ),
                                                    );
                                                    copied += 176;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 176) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        12 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET12_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 48 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(1)),
                                                        ),
                                                    );
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(2)),
                                                        ),
                                                    );
                                                    copied += 48;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 48) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        13 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET13_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 208 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    for i in 1..13 {
                                                        _mm_storeu_si128(
                                                            dest_ptr.add(copied + i * 16)
                                                                as *mut __m128i,
                                                            _mm_shuffle_epi8(
                                                                v_src,
                                                                _mm_loadu_si128(masks_ptr.add(i)),
                                                            ),
                                                        );
                                                    }
                                                    copied += 208;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 208) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        14 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET14_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 112 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    for i in 1..7 {
                                                        _mm_storeu_si128(
                                                            dest_ptr.add(copied + i * 16)
                                                                as *mut __m128i,
                                                            _mm_shuffle_epi8(
                                                                v_src,
                                                                _mm_loadu_si128(masks_ptr.add(i)),
                                                            ),
                                                        );
                                                    }
                                                    copied += 112;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 112) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        15 => {
                                            let dest_ptr = out_next;
                                            let src_ptr = src;
                                            let mut copied = 0;

                                            if length >= 16 {
                                                let v_src =
                                                    _mm_loadu_si128(src_ptr as *const __m128i);
                                                let masks_ptr =
                                                    OFFSET15_MASKS.as_ptr() as *const __m128i;
                                                let v_base = _mm_shuffle_epi8(
                                                    v_src,
                                                    _mm_loadu_si128(masks_ptr),
                                                );

                                                while copied + 240 <= length {
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v_base,
                                                    );
                                                    for i in 1..15 {
                                                        _mm_storeu_si128(
                                                            dest_ptr.add(copied + i * 16)
                                                                as *mut __m128i,
                                                            _mm_shuffle_epi8(
                                                                v_src,
                                                                _mm_loadu_si128(masks_ptr.add(i)),
                                                            ),
                                                        );
                                                    }
                                                    copied += 240;
                                                }
                                                while copied + 16 <= length {
                                                    let idx = (copied % 240) / 16;
                                                    let v = if idx == 0 {
                                                        v_base
                                                    } else {
                                                        _mm_shuffle_epi8(
                                                            v_src,
                                                            _mm_loadu_si128(masks_ptr.add(idx)),
                                                        )
                                                    };
                                                    _mm_storeu_si128(
                                                        dest_ptr.add(copied) as *mut __m128i,
                                                        v,
                                                    );
                                                    copied += 16;
                                                }
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        _ => {
                                            let mut copied = 0;
                                            while copied + 8 <= length {
                                                let val = std::ptr::read_unaligned(
                                                    src.add(copied) as *const u64
                                                );
                                                std::ptr::write_unaligned(
                                                    out_next.add(copied) as *mut u64,
                                                    val,
                                                );
                                                copied += 8;
                                            }
                                            while copied < length {
                                                *out_next.add(copied) = *src.add(copied);
                                                copied += 1;
                                            }
                                        }
                                    }
                                }
                                out_next = out_next.add(length);
                            }
                        }
                        in_idx = in_next.offset_from(in_ptr_start) as usize;
                        out_idx = out_next.offset_from(out_ptr_start) as usize;
                    }

                    if eob_found {
                        break;
                    }

                    d.bitbuf = bitbuf;
                    d.bitsleft = bitsleft;
                    d.state = crate::decompress::DecompressorState::BlockBody;

                    let res = d.decompress_huffman_block(input, &mut in_idx, output, &mut out_idx);

                    bitbuf = d.bitbuf;
                    bitsleft = d.bitsleft;

                    if res != DecompressResult::Success {
                        return (res, in_idx, out_idx);
                    }
                    break;
                }
            }
            _ => return (DecompressResult::BadData, 0, 0),
        }
    }
    (DecompressResult::Success, in_idx, out_idx)
}
