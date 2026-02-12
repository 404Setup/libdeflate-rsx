use crate::decompress::tables::{
    HUFFDEC_END_OF_BLOCK, HUFFDEC_EXCEPTIONAL, HUFFDEC_LITERAL, HUFFDEC_SUBTABLE_POINTER,
    OFFSET_TABLEBITS,
};
use crate::decompress::{
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN,
    DEFLATE_BLOCKTYPE_UNCOMPRESSED, DecompressResult, Decompressor, prepare_pattern,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! refill_bits {
    ($input:expr, $in_idx:expr, $bitbuf:expr, $bitsleft:expr) => {
        if $bitsleft < 32 {
            // Check if there are at least 8 bytes (64 bits) remaining in the input.
            // If so, we can perform a single unaligned 64-bit load.
            if $input.len().wrapping_sub($in_idx) >= 8 {
                // Read 64 bits from the input at the current index.
                let word = unsafe { ($input.as_ptr().add($in_idx) as *const u64).read_unaligned() };
                let word = u64::from_le(word);

                // Add the new bits to the bit buffer.
                // The new bits are shifted left by the number of bits already in the buffer.
                $bitbuf |= word << $bitsleft;

                // Calculate how many full bytes were consumed to fill the buffer up to 56 bits.
                // The logic guarantees the new bit count is between 56 and 63.
                let consumed = (63 - $bitsleft) >> 3;
                $in_idx += consumed as usize;

                // Update bitsleft. The logic `bitsleft |= 56` is a fast equivalent of
                // `bitsleft += consumed * 8` given the constraints.
                $bitsleft |= 56;
            } else {
                // Slow path: Read byte by byte if fewer than 8 bytes remain.
                while $bitsleft < 32 && $in_idx < $input.len() {
                    $bitbuf |= ($input[$in_idx] as u64) << $bitsleft;
                    $in_idx += 1;
                    $bitsleft += 8;
                }
            }
        }
    };
}

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

                let safe_in_limit = in_len.saturating_sub(15);
                let safe_out_limit = out_len.saturating_sub(258);

                'huffman_loop: loop {
                    // Fast inner loop: Run as long as we have plenty of input and output space.
                    // This allows skipping bounds checks on refill and most output writes.
                    while in_idx < safe_in_limit && out_idx < safe_out_limit {
                        if bitsleft < 32 {
                            let word = unsafe {
                                (input.as_ptr().add(in_idx) as *const u64).read_unaligned()
                            };
                            let word = u64::from_le(word);
                            bitbuf |= word << bitsleft;
                            let consumed = (63 - bitsleft) >> 3;
                            in_idx += consumed as usize;
                            bitsleft |= 56;
                        }

                        let table_idx = _bzhi_u64(bitbuf, d.litlen_tablebits as u32) as usize;
                        let mut entry = d.litlen_decode_table[table_idx];

                        if entry & HUFFDEC_EXCEPTIONAL != 0 {
                            if entry & HUFFDEC_END_OF_BLOCK != 0 {
                                bitbuf >>= entry as u8;
                                bitsleft -= entry & 0xFF;
                                break 'huffman_loop;
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
                            // Output bound check omitted due to loop condition
                            output[out_idx] = (entry >> 16) as u8;
                            out_idx += 1;
                        } else {
                            let mut length = (entry >> 16) as usize;
                            let len = (entry >> 8) & 0xFF;
                            let extra_bits = total_bits - len;
                            if extra_bits > 0 {
                                length += _bzhi_u64(saved_bitbuf >> len, extra_bits) as usize;
                            }

                            if bitsleft < 32 {
                                let word = unsafe {
                                    (input.as_ptr().add(in_idx) as *const u64).read_unaligned()
                                };
                                let word = u64::from_le(word);
                                bitbuf |= word << bitsleft;
                                let consumed = (63 - bitsleft) >> 3;
                                in_idx += consumed as usize;
                                bitsleft |= 56;
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
                            // Output bound check omitted due to loop condition

                            let out_ptr = output.as_mut_ptr();

                            if offset >= 16 {
                                // Safe to write 16 bytes because out_idx + 258 < out_len
                                // and match length < 258.
                                // Actually match length is unlimited? No, max match length is 258 in deflate.
                                let v1 = std::ptr::read_unaligned(out_ptr.add(src) as *const u64);
                                let v2 =
                                    std::ptr::read_unaligned(out_ptr.add(src + 8) as *const u64);
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
                            } else if offset < 8 {
                                let src_ptr = out_ptr.add(src);
                                let dest_ptr = out_ptr.add(dest);
                                if offset == 1 || offset == 2 || offset == 4 {
                                    let pattern = prepare_pattern(offset, src_ptr);
                                    let mut i = 0;
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
                                    match offset {
                                        3 => {
                                            let b0 = *src_ptr as u64;
                                            let b1 = *src_ptr.add(1) as u64;
                                            let b2 = *src_ptr.add(2) as u64;
                                            let pat0 = b0
                                                | (b1 << 8)
                                                | (b2 << 16)
                                                | (b0 << 24)
                                                | (b1 << 32)
                                                | (b2 << 40)
                                                | (b0 << 48)
                                                | (b1 << 56);
                                            let pat1 = b2
                                                | (b0 << 8)
                                                | (b1 << 16)
                                                | (b2 << 24)
                                                | (b0 << 32)
                                                | (b1 << 40)
                                                | (b2 << 48)
                                                | (b0 << 56);
                                            let pat2 = b1
                                                | (b2 << 8)
                                                | (b0 << 16)
                                                | (b1 << 24)
                                                | (b2 << 32)
                                                | (b0 << 40)
                                                | (b1 << 48)
                                                | (b2 << 56);

                                            let mut copied = 0;
                                            while copied + 24 <= length {
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied) as *mut u64,
                                                    pat0,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 8) as *mut u64,
                                                    pat1,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 16) as *mut u64,
                                                    pat2,
                                                );
                                                copied += 24;
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        5 => {
                                            let mut b = [0u64; 5];
                                            for i in 0..5 {
                                                b[i] = *src_ptr.add(i) as u64;
                                            }
                                            let pat0 = b[0]
                                                | (b[1] << 8)
                                                | (b[2] << 16)
                                                | (b[3] << 24)
                                                | (b[4] << 32)
                                                | (b[0] << 40)
                                                | (b[1] << 48)
                                                | (b[2] << 56);
                                            let pat1 = b[3]
                                                | (b[4] << 8)
                                                | (b[0] << 16)
                                                | (b[1] << 24)
                                                | (b[2] << 32)
                                                | (b[3] << 40)
                                                | (b[4] << 48)
                                                | (b[0] << 56);
                                            let pat2 = b[1]
                                                | (b[2] << 8)
                                                | (b[3] << 16)
                                                | (b[4] << 24)
                                                | (b[0] << 32)
                                                | (b[1] << 40)
                                                | (b[2] << 48)
                                                | (b[3] << 56);
                                            let pat3 = b[4]
                                                | (b[0] << 8)
                                                | (b[1] << 16)
                                                | (b[2] << 24)
                                                | (b[3] << 32)
                                                | (b[4] << 40)
                                                | (b[0] << 48)
                                                | (b[1] << 56);
                                            let pat4 = b[2]
                                                | (b[3] << 8)
                                                | (b[4] << 16)
                                                | (b[0] << 24)
                                                | (b[1] << 32)
                                                | (b[2] << 40)
                                                | (b[3] << 48)
                                                | (b[4] << 56);

                                            let mut copied = 0;
                                            while copied + 40 <= length {
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied) as *mut u64,
                                                    pat0,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 8) as *mut u64,
                                                    pat1,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 16) as *mut u64,
                                                    pat2,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 24) as *mut u64,
                                                    pat3,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 32) as *mut u64,
                                                    pat4,
                                                );
                                                copied += 40;
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        6 => {
                                            let mut b = [0u64; 6];
                                            for i in 0..6 {
                                                b[i] = *src_ptr.add(i) as u64;
                                            }
                                            let pat0 = b[0]
                                                | (b[1] << 8)
                                                | (b[2] << 16)
                                                | (b[3] << 24)
                                                | (b[4] << 32)
                                                | (b[5] << 40)
                                                | (b[0] << 48)
                                                | (b[1] << 56);
                                            let pat1 = b[2]
                                                | (b[3] << 8)
                                                | (b[4] << 16)
                                                | (b[5] << 24)
                                                | (b[0] << 32)
                                                | (b[1] << 40)
                                                | (b[2] << 48)
                                                | (b[3] << 56);
                                            let pat2 = b[4]
                                                | (b[5] << 8)
                                                | (b[0] << 16)
                                                | (b[1] << 24)
                                                | (b[2] << 32)
                                                | (b[3] << 40)
                                                | (b[4] << 48)
                                                | (b[5] << 56);

                                            let mut copied = 0;
                                            while copied + 24 <= length {
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied) as *mut u64,
                                                    pat0,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 8) as *mut u64,
                                                    pat1,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 16) as *mut u64,
                                                    pat2,
                                                );
                                                copied += 24;
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        7 => {
                                            let mut b = [0u64; 7];
                                            for i in 0..7 {
                                                b[i] = *src_ptr.add(i) as u64;
                                            }
                                            let pat0 = b[0]
                                                | (b[1] << 8)
                                                | (b[2] << 16)
                                                | (b[3] << 24)
                                                | (b[4] << 32)
                                                | (b[5] << 40)
                                                | (b[6] << 48)
                                                | (b[0] << 56);
                                            let pat1 = b[1]
                                                | (b[2] << 8)
                                                | (b[3] << 16)
                                                | (b[4] << 24)
                                                | (b[5] << 32)
                                                | (b[6] << 40)
                                                | (b[0] << 48)
                                                | (b[1] << 56);
                                            let pat2 = b[2]
                                                | (b[3] << 8)
                                                | (b[4] << 16)
                                                | (b[5] << 24)
                                                | (b[6] << 32)
                                                | (b[0] << 40)
                                                | (b[1] << 48)
                                                | (b[2] << 56);
                                            let pat3 = b[3]
                                                | (b[4] << 8)
                                                | (b[5] << 16)
                                                | (b[6] << 24)
                                                | (b[0] << 32)
                                                | (b[1] << 40)
                                                | (b[2] << 48)
                                                | (b[3] << 56);
                                            let pat4 = b[4]
                                                | (b[5] << 8)
                                                | (b[6] << 16)
                                                | (b[0] << 24)
                                                | (b[1] << 32)
                                                | (b[2] << 40)
                                                | (b[3] << 48)
                                                | (b[4] << 56);
                                            let pat5 = b[5]
                                                | (b[6] << 8)
                                                | (b[0] << 16)
                                                | (b[1] << 24)
                                                | (b[2] << 32)
                                                | (b[3] << 40)
                                                | (b[4] << 48)
                                                | (b[5] << 56);
                                            let pat6 = b[6]
                                                | (b[0] << 8)
                                                | (b[1] << 16)
                                                | (b[2] << 24)
                                                | (b[3] << 32)
                                                | (b[4] << 40)
                                                | (b[5] << 48)
                                                | (b[6] << 56);

                                            let mut copied = 0;
                                            while copied + 56 <= length {
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied) as *mut u64,
                                                    pat0,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 8) as *mut u64,
                                                    pat1,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 16) as *mut u64,
                                                    pat2,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 24) as *mut u64,
                                                    pat3,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 32) as *mut u64,
                                                    pat4,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 40) as *mut u64,
                                                    pat5,
                                                );
                                                std::ptr::write_unaligned(
                                                    dest_ptr.add(copied + 48) as *mut u64,
                                                    pat6,
                                                );
                                                copied += 56;
                                            }
                                            while copied < length {
                                                *dest_ptr.add(copied) = *src_ptr.add(copied);
                                                copied += 1;
                                            }
                                        }
                                        _ => {
                                            // Fallback/Safety
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
                                match offset {
                                    3 => {
                                        let b0 = *src_ptr as u64;
                                        let b1 = *src_ptr.add(1) as u64;
                                        let b2 = *src_ptr.add(2) as u64;
                                        let pat0 = b0 | (b1 << 8) | (b2 << 16) | (b0 << 24) | (b1 << 32) | (b2 << 40) | (b0 << 48) | (b1 << 56);
                                        let pat1 = b2 | (b0 << 8) | (b1 << 16) | (b2 << 24) | (b0 << 32) | (b1 << 40) | (b2 << 48) | (b0 << 56);
                                        let pat2 = b1 | (b2 << 8) | (b0 << 16) | (b1 << 24) | (b2 << 32) | (b0 << 40) | (b1 << 48) | (b2 << 56);

                                        let mut copied = 0;
                                        while copied + 24 <= length {
                                            std::ptr::write_unaligned(dest_ptr.add(copied) as *mut u64, pat0);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 8) as *mut u64, pat1);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 16) as *mut u64, pat2);
                                            copied += 24;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                    5 => {
                                        let mut b = [0u64; 5];
                                        for i in 0..5 { b[i] = *src_ptr.add(i) as u64; }
                                        let pat0 = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24) | (b[4] << 32) | (b[0] << 40) | (b[1] << 48) | (b[2] << 56);
                                        let pat1 = b[3] | (b[4] << 8) | (b[0] << 16) | (b[1] << 24) | (b[2] << 32) | (b[3] << 40) | (b[4] << 48) | (b[0] << 56);
                                        let pat2 = b[1] | (b[2] << 8) | (b[3] << 16) | (b[4] << 24) | (b[0] << 32) | (b[1] << 40) | (b[2] << 48) | (b[3] << 56);
                                        let pat3 = b[4] | (b[0] << 8) | (b[1] << 16) | (b[2] << 24) | (b[3] << 32) | (b[4] << 40) | (b[0] << 48) | (b[1] << 56);
                                        let pat4 = b[2] | (b[3] << 8) | (b[4] << 16) | (b[0] << 24) | (b[1] << 32) | (b[2] << 40) | (b[3] << 48) | (b[4] << 56);

                                        let mut copied = 0;
                                        while copied + 40 <= length {
                                            std::ptr::write_unaligned(dest_ptr.add(copied) as *mut u64, pat0);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 8) as *mut u64, pat1);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 16) as *mut u64, pat2);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 24) as *mut u64, pat3);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 32) as *mut u64, pat4);
                                            copied += 40;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                    6 => {
                                        let mut b = [0u64; 6];
                                        for i in 0..6 { b[i] = *src_ptr.add(i) as u64; }
                                        let pat0 = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24) | (b[4] << 32) | (b[5] << 40) | (b[0] << 48) | (b[1] << 56);
                                        let pat1 = b[2] | (b[3] << 8) | (b[4] << 16) | (b[5] << 24) | (b[0] << 32) | (b[1] << 40) | (b[2] << 48) | (b[3] << 56);
                                        let pat2 = b[4] | (b[5] << 8) | (b[0] << 16) | (b[1] << 24) | (b[2] << 32) | (b[3] << 40) | (b[4] << 48) | (b[5] << 56);

                                        let mut copied = 0;
                                        while copied + 24 <= length {
                                            std::ptr::write_unaligned(dest_ptr.add(copied) as *mut u64, pat0);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 8) as *mut u64, pat1);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 16) as *mut u64, pat2);
                                            copied += 24;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                    7 => {
                                        let mut b = [0u64; 7];
                                        for i in 0..7 { b[i] = *src_ptr.add(i) as u64; }
                                        let pat0 = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24) | (b[4] << 32) | (b[5] << 40) | (b[6] << 48) | (b[0] << 56);
                                        let pat1 = b[1] | (b[2] << 8) | (b[3] << 16) | (b[4] << 24) | (b[5] << 32) | (b[6] << 40) | (b[0] << 48) | (b[1] << 56);
                                        let pat2 = b[2] | (b[3] << 8) | (b[4] << 16) | (b[5] << 24) | (b[6] << 32) | (b[0] << 40) | (b[1] << 48) | (b[2] << 56);
                                        let pat3 = b[3] | (b[4] << 8) | (b[5] << 16) | (b[6] << 24) | (b[0] << 32) | (b[1] << 40) | (b[2] << 48) | (b[3] << 56);
                                        let pat4 = b[4] | (b[5] << 8) | (b[6] << 16) | (b[0] << 24) | (b[1] << 32) | (b[2] << 40) | (b[3] << 48) | (b[4] << 56);
                                        let pat5 = b[5] | (b[6] << 8) | (b[0] << 16) | (b[1] << 24) | (b[2] << 32) | (b[3] << 40) | (b[4] << 48) | (b[5] << 56);
                                        let pat6 = b[6] | (b[0] << 8) | (b[1] << 16) | (b[2] << 24) | (b[3] << 32) | (b[4] << 40) | (b[5] << 48) | (b[6] << 56);

                                        let mut copied = 0;
                                        while copied + 56 <= length {
                                            std::ptr::write_unaligned(dest_ptr.add(copied) as *mut u64, pat0);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 8) as *mut u64, pat1);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 16) as *mut u64, pat2);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 24) as *mut u64, pat3);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 32) as *mut u64, pat4);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 40) as *mut u64, pat5);
                                            std::ptr::write_unaligned(dest_ptr.add(copied + 48) as *mut u64, pat6);
                                            copied += 56;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                    _ => {
                                        // Fallback/Safety
                                        let mut copied = 0;
                                        while copied + offset <= length {
                                            std::ptr::copy_nonoverlapping(src_ptr.add(copied), dest_ptr.add(copied), offset);
                                            copied += offset;
                                        }
                                        while copied < length {
                                            *dest_ptr.add(copied) = *src_ptr.add(copied);
                                            copied += 1;
                                        }
                                    }
                                }
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
