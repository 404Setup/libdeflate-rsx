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

const OFFSET9_MASKS: [u8; 144] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
];

// LCM(3, 16) = 48. 3 vectors.
const OFFSET3_MASKS: [u8; 48] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];
// LCM(5, 16) = 80. 5 vectors.
const OFFSET5_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
    2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
    4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
];
// LCM(6, 16) = 48. 3 vectors.
const OFFSET6_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
    2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
];
// LCM(7, 16) = 112. 7 vectors.
const OFFSET7_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3,
    4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
    1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4,
    5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
];
// LCM(12, 16) = 48. 3 vectors.
const OFFSET12_MASKS: [u8; 48] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
];

// LCM(10, 16) = 80. 5 vectors.
const OFFSET10_MASKS: [u8; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

// LCM(11, 16) = 176. 11 vectors.
const OFFSET11_MASKS: [u8; 176] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
];

// LCM(15, 16) = 240. 15 vectors.
const OFFSET15_MASKS: [u8; 240] = [
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
const OFFSET14_MASKS: [u8; 112] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
];

// LCM(13, 16) = 208. 13 vectors.
const OFFSET13_MASKS: [u8; 208] = [
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
#[target_feature(enable = "bmi2,ssse3")]
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
                                    // Optimization: Use SIMD to copy 16 bytes at once.
                                    // This replaces 2 scalar u64 loads/stores with 1 vector load/store.
                                    // Safe because offset >= 16 implies no destructive overlap for the first 16 bytes.
                                    let v = _mm_loadu_si128(src as *const __m128i);
                                    _mm_storeu_si128(out_next as *mut __m128i, v);
                                    if length > 16 {
                                        if offset >= length {
                                            std::ptr::copy_nonoverlapping(
                                                src.add(16),
                                                out_next.add(16),
                                                length - 16,
                                            );
                                        } else if offset == 17 {
                                            let mut copied = 16;
                                            // For offset 17, src[16] corresponds to dst[-1].
                                            // We need to synthesize the next vector from the previous vector and dst[-1].
                                            // v contains dst[-17..-2]. v[15] is dst[-2].
                                            // We need dst[-1]. It is safe to read because it was written before the loop.
                                            let c = *src.add(16);
                                            // Insert dst[-1] at index 15. The rest of v_align doesn't matter for alignr(..., 15).
                                            let mut v_align = _mm_insert_epi8(v, c as i32, 15);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                // v_next = alignr(v_prev, v_align, 15)
                                                // This effectively takes v_align[15] (which is dst[-1] or previous end)
                                                // and v_prev[0..14].
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 15);
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
                                        } else if offset == 32 {
                                            let v2 = _mm_loadu_si128(src.add(16) as *const __m128i);
                                            let mut copied = 16;
                                            while copied + 64 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(copied) as *mut __m128i,
                                                    v2,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 16) as *mut __m128i,
                                                    v,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 32) as *mut __m128i,
                                                    v2,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 48) as *mut __m128i,
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
                                                    out_next.add(copied + 16) as *mut __m128i,
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
                                        } else if offset == 27 {
                                            let mut copied = 16;
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(11) as *const __m128i);
                                            let mut v_prev = v;
                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 5);
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
                                        } else if offset == 29 {
                                            let mut copied = 16;
                                            // For offset 29, src[13] is dest[-16].
                                            // We load dest[-16..0] into v_align.
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(13) as *const __m128i);
                                            let mut v_prev = v; // Contains dest[-29..-14] (initially) which is written to dest[0..15]
                                            while copied + 16 <= length {
                                                // alignr(v_prev, v_align, 3) takes bytes 3..15 from v_align (dest[-13..-1])
                                                // and bytes 0..2 from v_prev (dest[0..2]).
                                                // Result matches dest[16..31] where dest[16] = dest[-13].
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 3);
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
                                        } else if offset == 30 {
                                            let mut copied = 16;
                                            // For offset 30, src[14] is dest[-16].
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(14) as *const __m128i);
                                            let mut v_prev = v;
                                            while copied + 16 <= length {
                                                // alignr(v_prev, v_align, 2) takes bytes 2..15 from v_align (dest[-14..-1])
                                                // and bytes 0..1 from v_prev (dest[0..1]).
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 2);
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
                                        } else if offset == 31 {
                                            let mut copied = 16;
                                            // For offset 31, src[15] is dest[-16].
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(15) as *const __m128i);
                                            let mut v_prev = v;
                                            while copied + 16 <= length {
                                                // alignr(v_prev, v_align, 1) takes bytes 1..15 from v_align (dest[-15..-1])
                                                // and byte 0 from v_prev (dest[0]).
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 1);
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
                                        } else if offset == 28 {
                                            let mut copied = 16;
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(12) as *const __m128i);
                                            let mut v_prev = v;
                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 4);
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
                                        } else if offset == 26 {
                                            let mut copied = 16;
                                            let mut v_align =
                                                _mm_loadu_si128(src.add(10) as *const __m128i);
                                            let mut v_prev = v;
                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 6);
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
                                        } else if offset == 25 {
                                            let mut copied = 16;
                                            // For offset 25, src[16] is dest[-9].
                                            // We need dest[-9..-1] (9 bytes) at indices 7..15 of v_align.
                                            // Load 8 bytes from src[16] (dest[-9..-2]) and 1 byte from src[24] (dest[-1]).
                                            // Avoids reading dest[0] (which would be STLF hazard).
                                            let val =
                                                std::ptr::read_unaligned(src.add(16) as *const u64);
                                            let c = *src.add(24);
                                            let v_temp = _mm_cvtsi64_si128(val as i64);
                                            let v_temp = _mm_insert_epi8(v_temp, c as i32, 8);
                                            // Shift left by 7 bytes. dest[-9] (byte 0) moves to byte 7. dest[-1] (byte 8) moves to byte 15.
                                            let mut v_align = _mm_slli_si128(v_temp, 7);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 7);
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
                                        } else if offset == 23 {
                                            let mut copied = 16;
                                            // src[16] is dest[-7]. We need dest[-7..-1] (7 bytes).
                                            // Avoid reading dest[0] by reading two overlapping u32s.
                                            // v0 at dest[-7..-4], v1 at dest[-4..-1].
                                            let v0 =
                                                std::ptr::read_unaligned(src.add(16) as *const u32);
                                            let v1 =
                                                std::ptr::read_unaligned(src.add(19) as *const u32);
                                            let val = (v0 as u64) | ((v1 as u64) << 24);
                                            let v_temp = _mm_cvtsi64_si128(val as i64);
                                            let mut v_align = _mm_slli_si128(v_temp, 9);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 9);
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
                                        } else if offset == 24 {
                                            let mut copied = 16;
                                            // src[16] is dest[-8]. We need dest[-8..-1] (8 bytes).
                                            // Avoid reading dest[0] by reading two u32s.
                                            // v0 at dest[-8..-5], v1 at dest[-4..-1].
                                            let v0 =
                                                std::ptr::read_unaligned(src.add(16) as *const u32);
                                            let v1 =
                                                std::ptr::read_unaligned(src.add(20) as *const u32);
                                            let val = (v0 as u64) | ((v1 as u64) << 32);
                                            let v_temp = _mm_cvtsi64_si128(val as i64);
                                            let mut v_align = _mm_slli_si128(v_temp, 8);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 8);
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
                                        } else if offset == 20 {
                                            let mut copied = 16;
                                            // For offset 20, src[16] is dest[-4].
                                            // We need dest[-4..0] at indices 12..16 of v_align (to be shifted in).
                                            // Load 4 bytes from src[16] (which is dest[-4..0]), put in vector, shift left by 12 bytes.
                                            let val =
                                                std::ptr::read_unaligned(src.add(16) as *const u32);
                                            let v_temp = _mm_cvtsi32_si128(val as i32);
                                            // Left shift by 12 bytes.
                                            let mut v_align = _mm_slli_si128(v_temp, 12);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 12);
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
                                        } else if offset == 21 {
                                            let mut copied = 16;
                                            // For offset 21, src[16] is dest[-5].
                                            // We need dest[-5..-1] at indices 11..15 of v_align (to be shifted in).
                                            // Load 8 bytes from src[16] (which is dest[-5..2]), put in vector, shift left by 11 bytes.
                                            let val =
                                                std::ptr::read_unaligned(src.add(16) as *const u64);
                                            let v_temp = _mm_cvtsi64_si128(val as i64);
                                            // Left shift by 11 bytes.
                                            let mut v_align = _mm_slli_si128(v_temp, 11);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 11);
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
                                        } else if offset == 19 {
                                            let mut copied = 16;
                                            // For offset 19, src[16] is dest[-3], src[17] is dest[-2], src[18] is dest[-1]
                                            let c1 = *src.add(16);
                                            let c2 = *src.add(17);
                                            let c3 = *src.add(18);
                                            let mut v_align = _mm_insert_epi8(v, c1 as i32, 13);
                                            v_align = _mm_insert_epi8(v_align, c2 as i32, 14);
                                            v_align = _mm_insert_epi8(v_align, c3 as i32, 15);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 13);
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
                                        } else if offset == 18 {
                                            let mut copied = 16;
                                            let mut v_prev = v;
                                            // For offset 18, src[16] is dest[-2], src[17] is dest[-1]
                                            let c1 = *src.add(16);
                                            let c2 = *src.add(17);
                                            let mut v_align =
                                                _mm_insert_epi8(v_prev, c1 as i32, 14);
                                            v_align = _mm_insert_epi8(v_align, c2 as i32, 15);

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 14);
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
                                        } else if offset == 22 {
                                            let mut copied = 16;
                                            // For offset 22, src[16] is dest[-6].
                                            // We need dest[-6..-1] at bytes 10..15 of v_align.
                                            // Load 6 bytes from src[16] (dest[-6..0]).
                                            // Avoid RAW stall on dest[0] by loading 4+2 bytes separately.
                                            let v0 =
                                                std::ptr::read_unaligned(src.add(16) as *const u32);
                                            let v1 =
                                                std::ptr::read_unaligned(src.add(20) as *const u16);
                                            let val = (v0 as u64) | ((v1 as u64) << 32);
                                            let v_temp = _mm_cvtsi64_si128(val as i64);
                                            let mut v_align = _mm_slli_si128(v_temp, 10);
                                            let mut v_prev = v;

                                            while copied + 16 <= length {
                                                let v_next = _mm_alignr_epi8(v_prev, v_align, 10);
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
                                        } else if offset == 16 {
                                            let mut copied = 16;
                                            while copied + 64 <= length {
                                                _mm_storeu_si128(
                                                    out_next.add(copied) as *mut __m128i,
                                                    v,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 16) as *mut __m128i,
                                                    v,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 32) as *mut __m128i,
                                                    v,
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 48) as *mut __m128i,
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
                                        } else {
                                            let mut copied = 16;
                                            // Optimization: Use 128-bit SIMD load/store loop for bulk copy.
                                            // This is safe because `offset >= 16` (guarded by outer `if`), meaning the
                                            // source and destination windows separated by `offset` do not overlap
                                            // destructively within a 16-byte chunk.
                                            // Reads from `src + copied` (which is `dst + copied - offset`)
                                            // are valid because we are at least 16 bytes into the match, and `offset >= 16`.
                                            while copied + 64 <= length {
                                                let v1 = _mm_loadu_si128(
                                                    src.add(copied) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied) as *mut __m128i,
                                                    v1,
                                                );
                                                let v2 = _mm_loadu_si128(
                                                    src.add(copied + 16) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 16) as *mut __m128i,
                                                    v2,
                                                );
                                                let v3 = _mm_loadu_si128(
                                                    src.add(copied + 32) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 32) as *mut __m128i,
                                                    v3,
                                                );
                                                let v4 = _mm_loadu_si128(
                                                    src.add(copied + 48) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 48) as *mut __m128i,
                                                    v4,
                                                );
                                                copied += 64;
                                            }
                                            while copied + 32 <= length {
                                                let v1 = _mm_loadu_si128(
                                                    src.add(copied) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied) as *mut __m128i,
                                                    v1,
                                                );
                                                let v2 = _mm_loadu_si128(
                                                    src.add(copied + 16) as *const __m128i
                                                );
                                                _mm_storeu_si128(
                                                    out_next.add(copied + 16) as *mut __m128i,
                                                    v2,
                                                );
                                                copied += 32;
                                            }
                                            while copied + 16 <= length {
                                                let v = _mm_loadu_si128(
                                                    src.add(copied) as *const __m128i
                                                );
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
                                    }
                                } else if offset >= length {
                                    std::ptr::copy_nonoverlapping(src, out_next, length);
                                } else if offset == 1 {
                                    let b = *src;
                                    std::ptr::write_bytes(out_next, b, length);
                                } else if offset < 8 {
                                    if offset == 1 || offset == 2 || offset == 4 {
                                        let v_pattern = match offset {
                                            1 => _mm_set1_epi8(*src as i8),
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
                                    } else if offset == 3 {
                                        let dest_ptr = out_next;
                                        let v0 = std::ptr::read_unaligned(src as *const u16) as u32;
                                        let v1 = std::ptr::read_unaligned(src.add(1) as *const u16)
                                            as u32;
                                        let val = v0 | (v1 << 8);
                                        let v_pat = _mm_cvtsi32_si128(val as i32);
                                        let masks_ptr = OFFSET3_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 48 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                            *dest_ptr.add(copied) = *src.add(copied);
                                            copied += 1;
                                        }
                                    } else if offset == 5 {
                                        let dest_ptr = out_next;
                                        let v0 = std::ptr::read_unaligned(src as *const u32) as u64;
                                        let v1 = std::ptr::read_unaligned(src.add(1) as *const u32)
                                            as u64;
                                        let val = v0 | (v1 << 8);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET5_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 80 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(2)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 48) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(3)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 64) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                            *dest_ptr.add(copied) = *src.add(copied);
                                            copied += 1;
                                        }
                                    } else if offset == 6 {
                                        let dest_ptr = out_next;
                                        let v0 = std::ptr::read_unaligned(src as *const u32) as u64;
                                        let v1 = std::ptr::read_unaligned(src.add(2) as *const u32)
                                            as u64;
                                        let val = v0 | (v1 << 16);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET6_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 48 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                            *dest_ptr.add(copied) = *src.add(copied);
                                            copied += 1;
                                        }
                                    } else {
                                        // offset == 7
                                        let dest_ptr = out_next;
                                        let v0 = std::ptr::read_unaligned(src as *const u32) as u64;
                                        let v1 = std::ptr::read_unaligned(src.add(3) as *const u32)
                                            as u64;
                                        let val = v0 | (v1 << 24);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET7_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 112 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(2)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 48) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(3)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 64) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(4)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 80) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(5)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 96) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(6)),
                                                ),
                                            );
                                            copied += 112;
                                        }
                                        while copied + 16 <= length {
                                            let idx = (copied % 112) / 16;
                                            let v = if idx == 0 {
                                                v_base
                                            } else {
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                            *dest_ptr.add(copied) = *src.add(copied);
                                            copied += 1;
                                        }
                                    }
                                } else if offset == 8 {
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
                                } else if offset == 9 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;

                                    let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                    let masks_ptr = OFFSET9_MASKS.as_ptr() as *const __m128i;
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
                                        _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                        copied += 16;
                                    }

                                    while copied < length {
                                        *dest_ptr.add(copied) = *src_ptr.add(copied);
                                        copied += 1;
                                    }
                                } else if offset == 13 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;
                                    let mut copied = 0;

                                    if length >= 16 {
                                        let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                        let masks_ptr = OFFSET13_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                        while copied + 208 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            for i in 1..13 {
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + i * 16) as *mut __m128i,
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
                                } else if offset == 11 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;

                                    let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                    let masks_ptr = OFFSET11_MASKS.as_ptr() as *const __m128i;
                                    let v_base =
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                    _mm_storeu_si128(dest_ptr as *mut __m128i, v_base);
                                    let mut copied = 16;

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
                                        _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                        copied += 16;
                                    }
                                    while copied < length {
                                        *dest_ptr.add(copied) = *src_ptr.add(copied);
                                        copied += 1;
                                    }
                                } else if offset == 15 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;
                                    let mut copied = 0;

                                    if length >= 16 {
                                        let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                        let masks_ptr = OFFSET15_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                        while copied + 240 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            for i in 1..15 {
                                                _mm_storeu_si128(
                                                    dest_ptr.add(copied + i * 16) as *mut __m128i,
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
                                } else if offset == 10 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;

                                    let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                    let masks_ptr = OFFSET10_MASKS.as_ptr() as *const __m128i;
                                    let v_base =
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                    _mm_storeu_si128(dest_ptr as *mut __m128i, v_base);
                                    let mut copied = 16;

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
                                        _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                        copied += 16;
                                    }
                                    while copied < length {
                                        *dest_ptr.add(copied) = *src_ptr.add(copied);
                                        copied += 1;
                                    }
                                } else if offset == 12 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;
                                    let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                    let masks_ptr = OFFSET12_MASKS.as_ptr() as *const __m128i;
                                    let v_base =
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                    _mm_storeu_si128(dest_ptr as *mut __m128i, v_base);
                                    let mut copied = 16;
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
                                        _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                        copied += 16;
                                    }
                                    while copied < length {
                                        *dest_ptr.add(copied) = *src_ptr.add(copied);
                                        copied += 1;
                                    }
                                } else if offset == 14 {
                                    let dest_ptr = out_next;
                                    let src_ptr = src;
                                    let mut copied = 0;

                                    if length >= 16 {
                                        let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                        let masks_ptr = OFFSET14_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                        while copied + 112 <= length {
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
                                } else {
                                    let mut copied = 0;
                                    while copied + 8 <= length {
                                        let val =
                                            std::ptr::read_unaligned(src.add(copied) as *const u64);
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
                                out_next = out_next.add(length);
                            }
                        }
                        in_idx = in_next.offset_from(in_ptr_start) as usize;
                        out_idx = out_next.offset_from(out_ptr_start) as usize;
                    }

                    refill_bits!(input, in_idx, bitbuf, bitsleft);

                    let table_idx = _bzhi_u64(bitbuf, d.litlen_tablebits as u32) as usize;
                    let mut entry = d.litlen_decode_table[table_idx];

                    if entry & HUFFDEC_EXCEPTIONAL != 0 {
                        if entry & HUFFDEC_END_OF_BLOCK != 0 {
                            bitbuf >>= entry as u8;
                            bitsleft -= entry & 0xFF;
                            break;
                        }
                        if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                            let main_bits = entry & 0xFF;
                            bitbuf >>= main_bits;
                            bitsleft -= main_bits;
                            let subtable_idx = (entry >> 16) as usize;
                            let subtable_bits = (entry >> 8) & 0x3F;
                            let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                            entry = d.litlen_decode_table[subtable_idx + sub_idx];
                            if entry & HUFFDEC_EXCEPTIONAL != 0 {
                                if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                    bitbuf >>= entry as u8;
                                    bitsleft -= entry & 0xFF;
                                    break;
                                }
                            }
                        }
                    }

                    let saved_bitbuf = bitbuf;
                    let total_bits = entry & 0xFF;
                    bitbuf >>= total_bits;
                    bitsleft -= total_bits;

                    if entry & HUFFDEC_LITERAL != 0 {
                        if out_idx >= out_len {
                            return (DecompressResult::InsufficientSpace, 0, 0);
                        }
                        output[out_idx] = (entry >> 16) as u8;
                        out_idx += 1;
                    } else {
                        let mut length = (entry >> 16) as usize;
                        let len = (entry >> 8) & 0xFF;
                        let extra_bits = total_bits - len;
                        if extra_bits > 0 {
                            length += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                        }

                        refill_bits!(input, in_idx, bitbuf, bitsleft);

                        let offset_idx = _bzhi_u64(bitbuf, OFFSET_TABLEBITS as u32) as usize;
                        let mut entry = d.offset_decode_table[offset_idx];

                        if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                            let main_bits = entry & 0xFF;
                            bitbuf >>= main_bits;
                            bitsleft -= main_bits;
                            let subtable_idx = (entry >> 16) as usize;
                            let subtable_bits = (entry >> 8) & 0x3F;
                            let sub_idx = _bzhi_u64(bitbuf, subtable_bits) as usize;
                            entry = d.offset_decode_table[subtable_idx + sub_idx];
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

                        if offset > out_idx {
                            return (DecompressResult::BadData, 0, 0);
                        }
                        let dest = out_idx;
                        let src = dest - offset;
                        if dest + length > out_len {
                            return (DecompressResult::InsufficientSpace, 0, 0);
                        }

                        let out_ptr = output.as_mut_ptr();

                        if offset >= 16 && dest + 16 <= out_len {
                            let v = _mm_loadu_si128(out_ptr.add(src) as *const __m128i);
                            _mm_storeu_si128(out_ptr.add(dest) as *mut __m128i, v);
                            if length > 16 {
                                if offset >= length {
                                    std::ptr::copy_nonoverlapping(
                                        out_ptr.add(src + 16),
                                        out_ptr.add(dest + 16),
                                        length - 16,
                                    );
                                } else if offset == 17 {
                                    let mut copied = 16;
                                    let mut v_prev = v;
                                    // For offset 17, src[16] is dest[-1]
                                    let c = *out_ptr.add(src + 16);
                                    let mut v_align = _mm_insert_epi8(v_prev, c as i32, 15);

                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 15);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }

                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else if offset == 20 {
                                    let mut copied = 16;
                                    let val = std::ptr::read_unaligned(
                                        out_ptr.add(src + 16) as *const u32
                                    );
                                    let v_temp = _mm_cvtsi32_si128(val as i32);
                                    let mut v_align = _mm_slli_si128(v_temp, 12);
                                    let mut v_prev = v;

                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 12);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }

                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else if offset == 21 {
                                    let mut copied = 16;
                                    // For offset 21, src[16] is dest[-5].
                                    // We need dest[-5..-1] at indices 11..15 of v_align (to be shifted in).
                                    // Load 8 bytes from src[16] (which is dest[-5..2]), put in vector, shift left by 11 bytes.
                                    let val = std::ptr::read_unaligned(
                                        out_ptr.add(src + 16) as *const u64
                                    );
                                    let v_temp = _mm_cvtsi64_si128(val as i64);
                                    // Left shift by 11 bytes.
                                    let mut v_align = _mm_slli_si128(v_temp, 11);
                                    let mut v_prev = v;

                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 11);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }

                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else if offset == 19 {
                                    let mut copied = 16;
                                    let mut v_prev = v;
                                    // For offset 19, src[16] is dest[-3], src[17] is dest[-2], src[18] is dest[-1]
                                    let c1 = *out_ptr.add(src + 16);
                                    let c2 = *out_ptr.add(src + 17);
                                    let c3 = *out_ptr.add(src + 18);
                                    let mut v_align = _mm_insert_epi8(v_prev, c1 as i32, 13);
                                    v_align = _mm_insert_epi8(v_align, c2 as i32, 14);
                                    v_align = _mm_insert_epi8(v_align, c3 as i32, 15);

                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 13);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }

                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else if offset == 18 {
                                    let mut copied = 16;
                                    let mut v_prev = v;
                                    // For offset 18, src[16] is dest[-2], src[17] is dest[-1]
                                    let c1 = *out_ptr.add(src + 16); // dest[-2]
                                    let c2 = *out_ptr.add(src + 17); // dest[-1]
                                    let mut v_align = _mm_insert_epi8(v_prev, c1 as i32, 14);
                                    v_align = _mm_insert_epi8(v_align, c2 as i32, 15);

                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 14);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }

                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else if offset == 26 {
                                    let mut copied = 16;
                                    let mut v_align =
                                        _mm_loadu_si128(out_ptr.add(src + 10) as *const __m128i);
                                    let mut v_prev = v;
                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 6);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }
                                    if copied < length {
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            length - copied,
                                        );
                                    }
                                } else if offset == 27 {
                                    let mut copied = 16;
                                    let mut v_align =
                                        _mm_loadu_si128(out_ptr.add(src + 11) as *const __m128i);
                                    let mut v_prev = v;
                                    while copied + 16 <= length {
                                        let v_next = _mm_alignr_epi8(v_prev, v_align, 5);
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v_next,
                                        );
                                        v_align = v_prev;
                                        v_prev = v_next;
                                        copied += 16;
                                    }
                                    if copied < length {
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            length - copied,
                                        );
                                    }
                                } else if offset == 16 {
                                    let mut copied = 16;
                                    while copied + 16 <= length {
                                        _mm_storeu_si128(
                                            out_ptr.add(dest + copied) as *mut __m128i,
                                            v,
                                        );
                                        copied += 16;
                                    }
                                    while copied < length {
                                        let copy_len = std::cmp::min(offset, length - copied);
                                        std::ptr::copy_nonoverlapping(
                                            out_ptr.add(src + copied),
                                            out_ptr.add(dest + copied),
                                            copy_len,
                                        );
                                        copied += copy_len;
                                    }
                                } else {
                                    let mut copied = 16;
                                    // Optimization for small offsets where repeated memcpy overhead is high.
                                    // Threshold 128 is heuristic.
                                    // Safe because offset >= 32 here (offsets 1-31 are handled by specialized blocks or outer checks).
                                    // With offset >= 32, we can copy 16 bytes at a time without destructive overlap.
                                    if offset < 128 {
                                        while copied + 16 <= length {
                                            let v = _mm_loadu_si128(
                                                out_ptr.add(src + copied) as *const __m128i
                                            );
                                            _mm_storeu_si128(
                                                out_ptr.add(dest + copied) as *mut __m128i,
                                                v,
                                            );
                                            copied += 16;
                                        }
                                        if copied < length {
                                            std::ptr::copy_nonoverlapping(
                                                out_ptr.add(src + copied),
                                                out_ptr.add(dest + copied),
                                                length - copied,
                                            );
                                        }
                                    } else {
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
                            }
                        } else if offset >= length {
                            std::ptr::copy_nonoverlapping(
                                out_ptr.add(src),
                                out_ptr.add(dest),
                                length,
                            );
                        } else if offset == 1 {
                            let b = *out_ptr.add(src);
                            std::ptr::write_bytes(out_ptr.add(dest), b, length);
                        } else if offset < 8 {
                            let src_ptr = out_ptr.add(src);
                            let dest_ptr = out_ptr.add(dest);

                            if offset == 2 {
                                let w = std::ptr::read_unaligned(src_ptr as *const u16) as u64;
                                let pattern = w | (w << 16) | (w << 32) | (w << 48);
                                let mut i = 0;
                                while i + 32 <= length {
                                    std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
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
                                    std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                                    i += 8;
                                }
                                while i < length {
                                    *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                    i += 1;
                                }
                            } else if offset == 4 {
                                let val = std::ptr::read_unaligned(src_ptr as *const u32) as i32;
                                let v_pattern = _mm_set1_epi32(val);
                                let mut i = 0;
                                while i + 32 <= length {
                                    _mm_storeu_si128(dest_ptr.add(i) as *mut __m128i, v_pattern);
                                    _mm_storeu_si128(
                                        dest_ptr.add(i + 16) as *mut __m128i,
                                        v_pattern,
                                    );
                                    i += 32;
                                }
                                if i + 16 <= length {
                                    _mm_storeu_si128(dest_ptr.add(i) as *mut __m128i, v_pattern);
                                    i += 16;
                                }
                                let pattern = _mm_cvtsi128_si64(v_pattern) as u64;
                                while i + 8 <= length {
                                    std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                                    i += 8;
                                }
                                while i < length {
                                    *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                    i += 1;
                                }
                            } else {
                                match offset {
                                    3 => {
                                        let v0 =
                                            std::ptr::read_unaligned(src_ptr as *const u16) as u32;
                                        let v1 =
                                            std::ptr::read_unaligned(src_ptr.add(1) as *const u16)
                                                as u32;
                                        let val = v0 | (v1 << 8);
                                        let v_pat = _mm_cvtsi32_si128(val as i32);
                                        let masks_ptr = OFFSET3_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 48 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                    5 => {
                                        let v0 =
                                            std::ptr::read_unaligned(src_ptr as *const u32) as u64;
                                        let v1 =
                                            std::ptr::read_unaligned(src_ptr.add(1) as *const u32)
                                                as u64;
                                        let val = v0 | (v1 << 8);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET5_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 80 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(2)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 48) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(3)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 64) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                    6 => {
                                        let v0 =
                                            std::ptr::read_unaligned(src_ptr as *const u32) as u64;
                                        let v1 =
                                            std::ptr::read_unaligned(src_ptr.add(2) as *const u32)
                                                as u64;
                                        let val = v0 | (v1 << 16);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET6_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 48 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                                    v_pat,
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
                                    7 => {
                                        let v0 =
                                            std::ptr::read_unaligned(src_ptr as *const u32) as u64;
                                        let v1 =
                                            std::ptr::read_unaligned(src_ptr.add(3) as *const u32)
                                                as u64;
                                        let val = v0 | (v1 << 24);
                                        let v_pat = _mm_cvtsi64_si128(val as i64);
                                        let masks_ptr = OFFSET7_MASKS.as_ptr() as *const __m128i;
                                        let v_base =
                                            _mm_shuffle_epi8(v_pat, _mm_loadu_si128(masks_ptr));

                                        let mut copied = 0;
                                        while copied + 112 <= length {
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied) as *mut __m128i,
                                                v_base,
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 16) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(1)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 32) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(2)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 48) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(3)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 64) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(4)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 80) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(5)),
                                                ),
                                            );
                                            _mm_storeu_si128(
                                                dest_ptr.add(copied + 96) as *mut __m128i,
                                                _mm_shuffle_epi8(
                                                    v_pat,
                                                    _mm_loadu_si128(masks_ptr.add(6)),
                                                ),
                                            );
                                            copied += 112;
                                        }
                                        while copied + 16 <= length {
                                            let idx = (copied % 112) / 16;
                                            let v = if idx == 0 {
                                                v_base
                                            } else {
                                                _mm_shuffle_epi8(
                                                    v_pat,
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
                                    _ => {
                                        let mut copied = 0;
                                        while copied + offset <= length {
                                            std::ptr::copy_nonoverlapping(
                                                src_ptr.add(copied),
                                                dest_ptr.add(copied),
                                                offset,
                                            );
                                            copied += offset;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                }
                            }
                        } else if offset == 8 {
                            let pattern = std::ptr::read_unaligned(out_ptr.add(src) as *const u64);
                            let dest_ptr = out_ptr.add(dest);
                            let mut i = 0;
                            while i + 32 <= length {
                                std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                                std::ptr::write_unaligned(dest_ptr.add(i + 8) as *mut u64, pattern);
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
                                std::ptr::write_unaligned(dest_ptr.add(i) as *mut u64, pattern);
                                i += 8;
                            }
                            while i < length {
                                *dest_ptr.add(i) = (pattern >> ((i & 7) * 8)) as u8;
                                i += 1;
                            }
                        } else if offset == 9 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET9_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 144 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(1))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(2))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(3))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(4))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 80) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(5))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 96) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(6))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 112) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(7))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 128) as *mut __m128i,
                                        _mm_shuffle_epi8(v_base, _mm_loadu_si128(masks_ptr.add(8))),
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
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }

                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 13 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET13_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 208 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    for i in 1..13 {
                                        _mm_storeu_si128(
                                            dest_ptr.add(copied + i * 16) as *mut __m128i,
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
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 11 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET11_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 176 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(1))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(2))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(3))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(4))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 80) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(5))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 96) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(6))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 112) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(7))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 128) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(8))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 144) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(9))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 160) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(10))),
                                    );
                                    copied += 176;
                                }
                                while copied + 16 <= length {
                                    let idx = (copied % 176) / 16;
                                    let v = if idx == 0 {
                                        v_base
                                    } else {
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 15 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET15_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 240 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    for i in 1..15 {
                                        _mm_storeu_si128(
                                            dest_ptr.add(copied + i * 16) as *mut __m128i,
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
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 10 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET10_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 80 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(1))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(2))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(3))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(4))),
                                    );
                                    copied += 80;
                                }
                                while copied + 16 <= length {
                                    let idx = (copied % 80) / 16;
                                    let v = if idx == 0 {
                                        v_base
                                    } else {
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 12 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET12_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 48 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(1))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(2))),
                                    );
                                    copied += 48;
                                }
                                while copied + 16 <= length {
                                    let idx = (copied % 48) / 16;
                                    let v = if idx == 0 {
                                        v_base
                                    } else {
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else if offset == 14 {
                            let dest_ptr = out_ptr.add(dest);
                            let src_ptr = out_ptr.add(src);
                            let mut copied = 0;

                            if length >= 16 {
                                let v_src = _mm_loadu_si128(src_ptr as *const __m128i);
                                let masks_ptr = OFFSET14_MASKS.as_ptr() as *const __m128i;
                                let v_base = _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr));

                                while copied + 112 <= length {
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v_base);
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 16) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(1))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 32) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(2))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 48) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(3))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 64) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(4))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 80) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(5))),
                                    );
                                    _mm_storeu_si128(
                                        dest_ptr.add(copied + 96) as *mut __m128i,
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(6))),
                                    );
                                    copied += 112;
                                }
                                while copied + 16 <= length {
                                    let idx = (copied % 112) / 16;
                                    let v = if idx == 0 {
                                        v_base
                                    } else {
                                        _mm_shuffle_epi8(v_src, _mm_loadu_si128(masks_ptr.add(idx)))
                                    };
                                    _mm_storeu_si128(dest_ptr.add(copied) as *mut __m128i, v);
                                    copied += 16;
                                }
                            }
                            while copied < length {
                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                copied += 1;
                            }
                        } else {
                            let mut copied = 0;
                            while copied + 8 <= length {
                                let val = std::ptr::read_unaligned(
                                    out_ptr.add(src + copied) as *const u64
                                );
                                std::ptr::write_unaligned(
                                    out_ptr.add(dest + copied) as *mut u64,
                                    val,
                                );
                                copied += 8;
                            }
                            while copied < length {
                                *out_ptr.add(dest + copied) = *out_ptr.add(src + copied);
                                copied += 1;
                            }
                        }
                        out_idx += length;
                    }
                }
            }
            _ => return (DecompressResult::BadData, 0, 0),
        }
    }
    (DecompressResult::Success, in_idx, out_idx)
}
