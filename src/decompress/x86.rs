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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
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
        while bitsleft < 32 && in_idx < in_len {
            bitbuf |= (input[in_idx] as u64) << bitsleft;
            in_idx += 1;
            bitsleft += 8;
        }

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
                    let res = d.read_dynamic_huffman_header(
                        input,
                        &mut in_idx,
                    );
                    bitbuf = d.bitbuf;
                    bitsleft = d.bitsleft;
                    if res != DecompressResult::Success {
                        return (res, 0, 0);
                    }
                } else {
                    d.load_static_huffman_codes();
                }

                loop {
                    while bitsleft < 32 && in_idx < in_len {
                        bitbuf |= (input[in_idx] as u64) << bitsleft;
                        in_idx += 1;
                        bitsleft += 8;
                    }

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

                        while bitsleft < 32 && in_idx < in_len {
                            bitbuf |= (input[in_idx] as u64) << bitsleft;
                            in_idx += 1;
                            bitsleft += 8;
                        }

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
                             let v1 = std::ptr::read_unaligned(out_ptr.add(src) as *const u64);
                             let v2 = std::ptr::read_unaligned(out_ptr.add(src + 8) as *const u64);
                             std::ptr::write_unaligned(out_ptr.add(dest) as *mut u64, v1);
                             std::ptr::write_unaligned(out_ptr.add(dest + 8) as *mut u64, v2);
                             if length > 16 {
                                 if offset >= length {
                                     std::ptr::copy_nonoverlapping(
                                         out_ptr.add(src + 16),
                                         out_ptr.add(dest + 16),
                                         length - 16,
                                     );
                                 } else {
                                     let mut copied = 16;
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
                        } else if offset >= length {
                            std::ptr::copy_nonoverlapping(
                                out_ptr.add(src),
                                out_ptr.add(dest),
                                length,
                            );
                        } else if offset == 1 {
                            let b = *out_ptr.add(src);
                            std::ptr::write_bytes(out_ptr.add(dest), b, length);
                        } else {
                            let mut copied = 0;
                            if offset >= 8 {
                                while copied + 8 <= length {
                                    let val = std::ptr::read_unaligned(out_ptr.add(src + copied) as *const u64);
                                    std::ptr::write_unaligned(out_ptr.add(dest + copied) as *mut u64, val);
                                    copied += 8;
                                }
                            } else if offset >= 4 {
                                while copied + 4 <= length {
                                    let val = std::ptr::read_unaligned(out_ptr.add(src + copied) as *const u32);
                                    std::ptr::write_unaligned(out_ptr.add(dest + copied) as *mut u32, val);
                                    copied += 4;
                                }
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