#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::crc32_tables::*;

#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
pub unsafe fn crc32_x86_pclmulqdq(mut crc: u32, p: &[u8]) -> u32 {
    let mut len = p.len();
    let mut data = p;

    if len < 16 {
        // Optimize: For small inputs >= 8 bytes, the slicing-by-8 algorithm is significantly
        // faster than the byte-by-byte loop (crc32_slice1), as it processes 8 bytes at a time
        // using instruction-level parallelism. Benchmarks show a ~2x speedup for 8-15 bytes.
        if len >= 8 {
            return crate::crc32::crc32_slice8(crc, data);
        }
        return crate::crc32::crc32_slice1(crc, data);
    }

    let mults_128b = _mm_set_epi64x(CRC32_X95_MODG as i64, CRC32_X159_MODG as i64);
    let barrett_reduction_constants = _mm_set_epi64x(
        CRC32_BARRETT_CONSTANT_2 as i64,
        CRC32_BARRETT_CONSTANT_1 as i64,
    );

    let mut x0 = _mm_cvtsi32_si128(crc as i32);

    if len >= 64 {
        let mults_512b = _mm_set_epi64x(CRC32_X479_MODG as i64, CRC32_X543_MODG as i64);
        let v0 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        let v1 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
        let v2 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
        let v3 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);

        x0 = _mm_xor_si128(x0, v0);
        let mut x1 = v1;
        let mut x2 = v2;
        let mut x3 = v3;

        data = &data[64..];
        len -= 64;

        if len >= 64 {
            let mut x4 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let mut x5 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
            let mut x6 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
            let mut x7 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);

            data = &data[64..];
            len -= 64;

            if len >= 128 {
                let mults_1024b = _mm_set_epi64x(CRC32_X991_MODG as i64, CRC32_X1055_MODG as i64);
                while len >= 128 {
                    let v0 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
                    let v1 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
                    let v2 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
                    let v3 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);
                    let v4 = _mm_loadu_si128(data.as_ptr().add(64) as *const __m128i);
                    let v5 = _mm_loadu_si128(data.as_ptr().add(80) as *const __m128i);
                    let v6 = _mm_loadu_si128(data.as_ptr().add(96) as *const __m128i);
                    let v7 = _mm_loadu_si128(data.as_ptr().add(112) as *const __m128i);

                    let x0_low = _mm_clmulepi64_si128(x0, mults_1024b, 0x00);
                    let x0_high = _mm_clmulepi64_si128(x0, mults_1024b, 0x11);
                    x0 = _mm_xor_si128(v0, _mm_xor_si128(x0_low, x0_high));

                    let x1_low = _mm_clmulepi64_si128(x1, mults_1024b, 0x00);
                    let x1_high = _mm_clmulepi64_si128(x1, mults_1024b, 0x11);
                    x1 = _mm_xor_si128(v1, _mm_xor_si128(x1_low, x1_high));

                    let x2_low = _mm_clmulepi64_si128(x2, mults_1024b, 0x00);
                    let x2_high = _mm_clmulepi64_si128(x2, mults_1024b, 0x11);
                    x2 = _mm_xor_si128(v2, _mm_xor_si128(x2_low, x2_high));

                    let x3_low = _mm_clmulepi64_si128(x3, mults_1024b, 0x00);
                    let x3_high = _mm_clmulepi64_si128(x3, mults_1024b, 0x11);
                    x3 = _mm_xor_si128(v3, _mm_xor_si128(x3_low, x3_high));

                    let x4_low = _mm_clmulepi64_si128(x4, mults_1024b, 0x00);
                    let x4_high = _mm_clmulepi64_si128(x4, mults_1024b, 0x11);
                    x4 = _mm_xor_si128(v4, _mm_xor_si128(x4_low, x4_high));

                    let x5_low = _mm_clmulepi64_si128(x5, mults_1024b, 0x00);
                    let x5_high = _mm_clmulepi64_si128(x5, mults_1024b, 0x11);
                    x5 = _mm_xor_si128(v5, _mm_xor_si128(x5_low, x5_high));

                    let x6_low = _mm_clmulepi64_si128(x6, mults_1024b, 0x00);
                    let x6_high = _mm_clmulepi64_si128(x6, mults_1024b, 0x11);
                    x6 = _mm_xor_si128(v6, _mm_xor_si128(x6_low, x6_high));

                    let x7_low = _mm_clmulepi64_si128(x7, mults_1024b, 0x00);
                    let x7_high = _mm_clmulepi64_si128(x7, mults_1024b, 0x11);
                    x7 = _mm_xor_si128(v7, _mm_xor_si128(x7_low, x7_high));

                    data = &data[128..];
                    len -= 128;
                }
            }

            // Fold x0..x3 into x4..x7? No, x0..x7 are parallel.
            // We want to reduce 8 streams to 4.
            // x0 (pos 0), x4 (pos 64).
            // We want new x0 = fold(x0) ^ x4.
            // Distance is 64 bytes (512 bits). Use mults_512b.

            let x0_low = _mm_clmulepi64_si128(x0, mults_512b, 0x00);
            let x0_high = _mm_clmulepi64_si128(x0, mults_512b, 0x11);
            x0 = _mm_xor_si128(x4, _mm_xor_si128(x0_low, x0_high));

            let x1_low = _mm_clmulepi64_si128(x1, mults_512b, 0x00);
            let x1_high = _mm_clmulepi64_si128(x1, mults_512b, 0x11);
            x1 = _mm_xor_si128(x5, _mm_xor_si128(x1_low, x1_high));

            let x2_low = _mm_clmulepi64_si128(x2, mults_512b, 0x00);
            let x2_high = _mm_clmulepi64_si128(x2, mults_512b, 0x11);
            x2 = _mm_xor_si128(x6, _mm_xor_si128(x2_low, x2_high));

            let x3_low = _mm_clmulepi64_si128(x3, mults_512b, 0x00);
            let x3_high = _mm_clmulepi64_si128(x3, mults_512b, 0x11);
            x3 = _mm_xor_si128(x7, _mm_xor_si128(x3_low, x3_high));
        }

        while len >= 64 {
            let v0 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let v1 = _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i);
            let v2 = _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i);
            let v3 = _mm_loadu_si128(data.as_ptr().add(48) as *const __m128i);

            let x0_low = _mm_clmulepi64_si128(x0, mults_512b, 0x00);
            let x0_high = _mm_clmulepi64_si128(x0, mults_512b, 0x11);
            x0 = _mm_xor_si128(v0, _mm_xor_si128(x0_low, x0_high));

            let x1_low = _mm_clmulepi64_si128(x1, mults_512b, 0x00);
            let x1_high = _mm_clmulepi64_si128(x1, mults_512b, 0x11);
            x1 = _mm_xor_si128(v1, _mm_xor_si128(x1_low, x1_high));

            let x2_low = _mm_clmulepi64_si128(x2, mults_512b, 0x00);
            let x2_high = _mm_clmulepi64_si128(x2, mults_512b, 0x11);
            x2 = _mm_xor_si128(v2, _mm_xor_si128(x2_low, x2_high));

            let x3_low = _mm_clmulepi64_si128(x3, mults_512b, 0x00);
            let x3_high = _mm_clmulepi64_si128(x3, mults_512b, 0x11);
            x3 = _mm_xor_si128(v3, _mm_xor_si128(x3_low, x3_high));

            data = &data[64..];
            len -= 64;
        }

        let x0_low = _mm_clmulepi64_si128(x0, mults_128b, 0x00);
        let x0_high = _mm_clmulepi64_si128(x0, mults_128b, 0x11);
        x0 = _mm_xor_si128(x1, _mm_xor_si128(x0_low, x0_high));

        let x0_low = _mm_clmulepi64_si128(x0, mults_128b, 0x00);
        let x0_high = _mm_clmulepi64_si128(x0, mults_128b, 0x11);
        x0 = _mm_xor_si128(x2, _mm_xor_si128(x0_low, x0_high));

        let x0_low = _mm_clmulepi64_si128(x0, mults_128b, 0x00);
        let x0_high = _mm_clmulepi64_si128(x0, mults_128b, 0x11);
        x0 = _mm_xor_si128(x3, _mm_xor_si128(x0_low, x0_high));
    } else {
        let v0 = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        x0 = _mm_xor_si128(x0, v0);
        data = &data[16..];
        len -= 16;
    }

    while len >= 16 {
        let v_data = _mm_loadu_si128(data.as_ptr() as *const __m128i);

        let x0_low = _mm_clmulepi64_si128(x0, mults_128b, 0x00);
        let x0_high = _mm_clmulepi64_si128(x0, mults_128b, 0x11);

        x0 = _mm_xor_si128(v_data, _mm_xor_si128(x0_low, x0_high));

        data = &data[16..];
        len -= 16;
    }

    x0 = _mm_xor_si128(
        _mm_clmulepi64_si128(x0, mults_128b, 0x10),
        _mm_srli_si128(x0, 8),
    );

    let mut x1 = _mm_clmulepi64_si128(x0, barrett_reduction_constants, 0x00);
    x1 = _mm_clmulepi64_si128(x1, barrett_reduction_constants, 0x10);
    x0 = _mm_xor_si128(x0, x1);

    crc = _mm_extract_epi32(x0, 2) as u32;

    if len >= 8 {
        crc = crate::crc32::crc32_slice8(crc, data);
    } else if len > 0 {
        crc = crate::crc32::crc32_slice1(crc, data);
    }

    crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,vpclmulqdq")]
pub unsafe fn crc32_x86_vpclmulqdq_avx512_vl512(crc: u32, p: &[u8]) -> u32 {
    let mut len = p.len();
    let mut data = p;

    let mults_128b = _mm_set_epi64x(CRC32_X95_MODG as i64, CRC32_X159_MODG as i64);
    let barrett_reduction_constants = _mm_set_epi64x(
        CRC32_BARRETT_CONSTANT_2 as i64,
        CRC32_BARRETT_CONSTANT_1 as i64,
    );

    let mut x0 = _mm_cvtsi32_si128(crc as i32);

    if len < 512 {
        if len < 64 {
            if len < 16 {
                if len < 4 {
                    return crate::crc32::crc32_slice1(crc, data);
                }
                let mask = (1u32 << len) - 1;
                x0 = _mm_xor_si128(
                    x0,
                    _mm_maskz_loadu_epi8(mask as u16, data.as_ptr() as *const i8),
                );

                let shift_tab = [
                    0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
                    0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                ];
                let shifts = _mm_loadu_si128(shift_tab.as_ptr().add(len) as *const __m128i);
                x0 = _mm_shuffle_epi8(x0, shifts);
            } else {
                x0 = _mm_xor_si128(_mm_loadu_si128(data.as_ptr() as *const __m128i), x0);
                if len >= 32 {
                    x0 = fold_vec128(
                        x0,
                        _mm_loadu_si128(data.as_ptr().add(16) as *const __m128i),
                        mults_128b,
                    );
                    if len >= 48 {
                        x0 = fold_vec128(
                            x0,
                            _mm_loadu_si128(data.as_ptr().add(32) as *const __m128i),
                            mults_128b,
                        );
                    }
                }
                data = &data[len & !15..];
                len &= 15;
                if len > 0 {
                    x0 = fold_lessthan16bytes(x0, data, len, mults_128b);
                }
            }
        } else {
            let mut v0 = _mm512_zextsi128_si512(_mm_xor_si128(
                _mm_loadu_si128(data.as_ptr() as *const __m128i),
                x0,
            ));
            if len >= 128 {
                let mut v1 = _mm512_loadu_si512(data.as_ptr().add(64) as *const _);
                let mults_1v = _mm512_set_epi64(
                    CRC32_X479_MODG as i64,
                    CRC32_X543_MODG as i64,
                    CRC32_X479_MODG as i64,
                    CRC32_X543_MODG as i64,
                    CRC32_X479_MODG as i64,
                    CRC32_X543_MODG as i64,
                    CRC32_X479_MODG as i64,
                    CRC32_X543_MODG as i64,
                );

                if len >= 256 {
                    let v2 = _mm512_loadu_si512(data.as_ptr().add(128) as *const _);
                    let v3 = _mm512_loadu_si512(data.as_ptr().add(192) as *const _);

                    data = &data[256..];
                    len -= 256;

                    if len >= 128 {
                        let mults_2v = _mm512_set_epi64(
                            CRC32_X991_MODG as i64,
                            CRC32_X1055_MODG as i64,
                            CRC32_X991_MODG as i64,
                            CRC32_X1055_MODG as i64,
                            CRC32_X991_MODG as i64,
                            CRC32_X1055_MODG as i64,
                            CRC32_X991_MODG as i64,
                            CRC32_X1055_MODG as i64,
                        );
                        v0 = fold_vec512(v0, v2, mults_2v);
                        v1 = fold_vec512(v1, v3, mults_2v);

                        if len >= 64 {
                            v0 = fold_vec512(
                                v0,
                                _mm512_loadu_si512(data.as_ptr() as *const _),
                                mults_2v,
                            );
                            v1 = fold_vec512(
                                v1,
                                _mm512_loadu_si512(data.as_ptr().add(64) as *const _),
                                mults_2v,
                            );
                            data = &data[128..];
                            len -= 128;
                        }
                    }

                    v0 = fold_vec512(v0, v1, mults_1v);
                    if len >= 64 {
                        v0 = fold_vec512(
                            v0,
                            _mm512_loadu_si512(data.as_ptr() as *const _),
                            mults_1v,
                        );
                        data = &data[64..];
                        len -= 64;
                    }
                } else {
                    data = &data[128..];
                    len -= 128;
                    v0 = fold_vec512(v0, v1, mults_1v);
                    if len >= 64 {
                        v0 = fold_vec512(
                            v0,
                            _mm512_loadu_si512(data.as_ptr() as *const _),
                            mults_1v,
                        );
                        data = &data[64..];
                        len -= 64;
                    }
                }
            } else {
                data = &data[64..];
                len -= 64;
            }

            let mults_256b = _mm256_set_epi64x(
                CRC32_X223_MODG as i64,
                CRC32_X287_MODG as i64,
                CRC32_X223_MODG as i64,
                CRC32_X287_MODG as i64,
            );
            let mut y0 = fold_vec256(
                _mm512_extracti64x4_epi64(v0, 0),
                _mm512_extracti64x4_epi64(v0, 1),
                mults_256b,
            );
            if len >= 32 {
                y0 = fold_vec256(
                    y0,
                    _mm256_loadu_si256(data.as_ptr() as *const _),
                    mults_256b,
                );
                data = &data[32..];
                len -= 32;
            }
            x0 = fold_vec128(
                _mm256_extracti128_si256(y0, 0),
                _mm256_extracti128_si256(y0, 1),
                mults_128b,
            );
            if len >= 16 {
                x0 = fold_vec128(x0, _mm_loadu_si128(data.as_ptr() as *const _), mults_128b);
                data = &data[16..];
                len -= 16;
            }
            if len > 0 {
                x0 = fold_lessthan16bytes(x0, data, len, mults_128b);
            }
        }
    } else {
        let mut v0;

        if len > 65536 && (data.as_ptr() as usize & 63) != 0 {
            let align_offset = (data.as_ptr() as usize) & 63;
            let align = 64 - align_offset;

            len -= align;
            x0 = _mm_xor_si128(_mm_loadu_si128(data.as_ptr() as *const __m128i), x0);
            data = &data[16..];
            let mut align_rem = align;
            if (align_rem & 15) != 0 {
                let chunk = align_rem & 15;
                x0 = fold_lessthan16bytes(x0, data, chunk, mults_128b);
                data = &data[chunk..];
                align_rem &= !15;
            }
            while align_rem > 0 {
                x0 = fold_vec128(
                    x0,
                    _mm_loadu_si128(data.as_ptr() as *const __m128i),
                    mults_128b,
                );
                data = &data[16..];
                align_rem -= 16;
            }
            v0 = _mm512_zextsi128_si512(x0);
            v0 = _mm512_xor_si512(_mm512_load_si512(data.as_ptr() as *const _), v0);
        } else {
            v0 = _mm512_zextsi128_si512(_mm_xor_si128(
                _mm_loadu_si128(data.as_ptr() as *const __m128i),
                x0,
            ));
            v0 = _mm512_xor_si512(_mm512_loadu_si512(data.as_ptr().add(16) as *const _), v0);
        }

        let mut v1 = _mm512_loadu_si512(data.as_ptr().add(64) as *const _);
        let mut v2 = _mm512_loadu_si512(data.as_ptr().add(128) as *const _);
        let mut v3 = _mm512_loadu_si512(data.as_ptr().add(192) as *const _);
        let mut v4 = _mm512_loadu_si512(data.as_ptr().add(256) as *const _);
        let mut v5 = _mm512_loadu_si512(data.as_ptr().add(320) as *const _);
        let mut v6 = _mm512_loadu_si512(data.as_ptr().add(384) as *const _);
        let mut v7 = _mm512_loadu_si512(data.as_ptr().add(448) as *const _);
        data = &data[512..];
        len -= 512;

        let mults_8v = _mm512_set_epi64(
            CRC32_X4063_MODG as i64,
            CRC32_X4127_MODG as i64,
            CRC32_X4063_MODG as i64,
            CRC32_X4127_MODG as i64,
            CRC32_X4063_MODG as i64,
            CRC32_X4127_MODG as i64,
            CRC32_X4063_MODG as i64,
            CRC32_X4127_MODG as i64,
        );

        while len >= 512 {
            v0 = fold_vec512(v0, _mm512_loadu_si512(data.as_ptr() as *const _), mults_8v);
            v1 = fold_vec512(
                v1,
                _mm512_loadu_si512(data.as_ptr().add(64) as *const _),
                mults_8v,
            );
            v2 = fold_vec512(
                v2,
                _mm512_loadu_si512(data.as_ptr().add(128) as *const _),
                mults_8v,
            );
            v3 = fold_vec512(
                v3,
                _mm512_loadu_si512(data.as_ptr().add(192) as *const _),
                mults_8v,
            );
            v4 = fold_vec512(
                v4,
                _mm512_loadu_si512(data.as_ptr().add(256) as *const _),
                mults_8v,
            );
            v5 = fold_vec512(
                v5,
                _mm512_loadu_si512(data.as_ptr().add(320) as *const _),
                mults_8v,
            );
            v6 = fold_vec512(
                v6,
                _mm512_loadu_si512(data.as_ptr().add(384) as *const _),
                mults_8v,
            );
            v7 = fold_vec512(
                v7,
                _mm512_loadu_si512(data.as_ptr().add(448) as *const _),
                mults_8v,
            );
            data = &data[512..];
            len -= 512;
        }

        let mults_4v = _mm512_set_epi64(
            CRC32_X2015_MODG as i64,
            CRC32_X2079_MODG as i64,
            CRC32_X2015_MODG as i64,
            CRC32_X2079_MODG as i64,
            CRC32_X2015_MODG as i64,
            CRC32_X2079_MODG as i64,
            CRC32_X2015_MODG as i64,
            CRC32_X2079_MODG as i64,
        );

        v0 = fold_vec512(v0, v4, mults_4v);
        v1 = fold_vec512(v1, v5, mults_4v);
        v2 = fold_vec512(v2, v6, mults_4v);
        v3 = fold_vec512(v3, v7, mults_4v);

        if len >= 256 {
            v0 = fold_vec512(v0, _mm512_loadu_si512(data.as_ptr() as *const _), mults_4v);
            v1 = fold_vec512(
                v1,
                _mm512_loadu_si512(data.as_ptr().add(64) as *const _),
                mults_4v,
            );
            v2 = fold_vec512(
                v2,
                _mm512_loadu_si512(data.as_ptr().add(128) as *const _),
                mults_4v,
            );
            v3 = fold_vec512(
                v3,
                _mm512_loadu_si512(data.as_ptr().add(192) as *const _),
                mults_4v,
            );
            data = &data[256..];
            len -= 256;
        }

        let mults_2v = _mm512_set_epi64(
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
        );

        v0 = fold_vec512(v0, v2, mults_2v);
        v1 = fold_vec512(v1, v3, mults_2v);

        if len >= 128 {
            v0 = fold_vec512(v0, _mm512_loadu_si512(data.as_ptr() as *const _), mults_2v);
            v1 = fold_vec512(
                v1,
                _mm512_loadu_si512(data.as_ptr().add(64) as *const _),
                mults_2v,
            );
            data = &data[128..];
            len -= 128;
        }

        let mults_1v = _mm512_set_epi64(
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
        );

        v0 = fold_vec512(v0, v1, mults_1v);

        if len >= 64 {
            v0 = fold_vec512(v0, _mm512_loadu_si512(data.as_ptr() as *const _), mults_1v);
            data = &data[64..];
            len -= 64;
        }

        let mults_256b = _mm256_set_epi64x(
            CRC32_X223_MODG as i64,
            CRC32_X287_MODG as i64,
            CRC32_X223_MODG as i64,
            CRC32_X287_MODG as i64,
        );
        let mut y0 = fold_vec256(
            _mm512_extracti64x4_epi64(v0, 0),
            _mm512_extracti64x4_epi64(v0, 1),
            mults_256b,
        );
        if len >= 32 {
            y0 = fold_vec256(
                y0,
                _mm256_loadu_si256(data.as_ptr() as *const _),
                mults_256b,
            );
            data = &data[32..];
            len -= 32;
        }
        x0 = fold_vec128(
            _mm256_extracti128_si256(y0, 0),
            _mm256_extracti128_si256(y0, 1),
            mults_128b,
        );
        if len >= 16 {
            x0 = fold_vec128(x0, _mm_loadu_si128(data.as_ptr() as *const _), mults_128b);
            data = &data[16..];
            len -= 16;
        }
        if len > 0 {
            x0 = fold_lessthan16bytes(x0, data, len, mults_128b);
        }
    }

    x0 = _mm_xor_si128(
        _mm_clmulepi64_si128(x0, mults_128b, 0x10),
        _mm_srli_si128(x0, 8),
    );
    let mut x1 = _mm_clmulepi64_si128(x0, barrett_reduction_constants, 0x00);
    x1 = _mm_clmulepi64_si128(x1, barrett_reduction_constants, 0x10);
    x0 = _mm_xor_si128(x0, x1);

    let mut res = _mm_extract_epi32(x0, 2) as u32;

    if len >= 8 {
        res = crate::crc32::crc32_slice8(res, data);
    } else if len > 0 {
        res = crate::crc32::crc32_slice1(res, data);
    }

    res
}

#[inline(always)]
unsafe fn fold_vec128(dst: __m128i, src: __m128i, mults: __m128i) -> __m128i {
    let t1 = _mm_clmulepi64_si128(dst, mults, 0x00);
    let t2 = _mm_clmulepi64_si128(dst, mults, 0x11);
    _mm_xor_si128(src, _mm_xor_si128(t1, t2))
}

#[inline(always)]
unsafe fn fold_vec256(dst: __m256i, src: __m256i, mults: __m256i) -> __m256i {
    #[cfg(target_feature = "avx512f")]
    {
        _mm256_ternarylogic_epi32(
            _mm256_clmulepi64_epi128(dst, mults, 0x00),
            _mm256_clmulepi64_epi128(dst, mults, 0x11),
            src,
            0x96,
        )
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        _mm256_xor_si256(
            _mm256_xor_si256(src, _mm256_clmulepi64_epi128(dst, mults, 0x00)),
            _mm256_clmulepi64_epi128(dst, mults, 0x11),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,vpclmulqdq")]
unsafe fn fold_vec512(dst: __m512i, src: __m512i, mults: __m512i) -> __m512i {
    _mm512_ternarylogic_epi32(
        _mm512_clmulepi64_epi128(dst, mults, 0x00),
        _mm512_clmulepi64_epi128(dst, mults, 0x11),
        src,
        0x96,
    )
}

#[cfg(target_arch = "x86_64")]
unsafe fn fold_lessthan16bytes(x: __m128i, p: &[u8], len: usize, mults: __m128i) -> __m128i {
    let shift_tab = [
        0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
        0x0e, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff,
    ];
    let lshift = _mm_loadu_si128(shift_tab.as_ptr().add(len) as *const __m128i);
    let rshift = _mm_loadu_si128(shift_tab.as_ptr().add(len + 16) as *const __m128i);

    let x0 = _mm_shuffle_epi8(x, lshift);
    let x1 = _mm_blendv_epi8(
        _mm_shuffle_epi8(x, rshift),
        _mm_loadu_si128(p.as_ptr().offset((len as isize) - 16) as *const __m128i),
        rshift,
    );
    fold_vec128(x0, x1, mults)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,vpclmulqdq")]
pub unsafe fn crc32_x86_vpclmulqdq_avx2(crc: u32, p: &[u8]) -> u32 {
    let mut len = p.len();
    let mut data = p;

    let mults_128b = _mm_set_epi64x(CRC32_X95_MODG as i64, CRC32_X159_MODG as i64);
    let barrett_reduction_constants = _mm_set_epi64x(
        CRC32_BARRETT_CONSTANT_2 as i64,
        CRC32_BARRETT_CONSTANT_1 as i64,
    );

    let mut x0 = _mm_cvtsi32_si128(crc as i32);

    if len >= 128 {
        let mut v0 = _mm256_loadu_si256(data.as_ptr() as *const _);
        let mut v1 = _mm256_loadu_si256(data.as_ptr().add(32) as *const _);
        let mut v2 = _mm256_loadu_si256(data.as_ptr().add(64) as *const _);
        let mut v3 = _mm256_loadu_si256(data.as_ptr().add(96) as *const _);

        let x0_256 = _mm256_set_m128i(_mm_setzero_si128(), x0);
        v0 = _mm256_xor_si256(v0, x0_256);

        data = &data[128..];
        len -= 128;

        let mults_4v = _mm256_set_epi64x(
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
            CRC32_X991_MODG as i64,
            CRC32_X1055_MODG as i64,
        );

        while len >= 128 {
            v0 = fold_vec256(v0, _mm256_loadu_si256(data.as_ptr() as *const _), mults_4v);
            v1 = fold_vec256(
                v1,
                _mm256_loadu_si256(data.as_ptr().add(32) as *const _),
                mults_4v,
            );
            v2 = fold_vec256(
                v2,
                _mm256_loadu_si256(data.as_ptr().add(64) as *const _),
                mults_4v,
            );
            v3 = fold_vec256(
                v3,
                _mm256_loadu_si256(data.as_ptr().add(96) as *const _),
                mults_4v,
            );
            data = &data[128..];
            len -= 128;
        }

        let mults_2v = _mm256_set_epi64x(
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
            CRC32_X479_MODG as i64,
            CRC32_X543_MODG as i64,
        );
        v0 = fold_vec256(v0, v2, mults_2v);
        v1 = fold_vec256(v1, v3, mults_2v);

        let mults_1v = _mm256_set_epi64x(
            CRC32_X223_MODG as i64,
            CRC32_X287_MODG as i64,
            CRC32_X223_MODG as i64,
            CRC32_X287_MODG as i64,
        );
        v0 = fold_vec256(v0, v1, mults_1v);

        x0 = fold_vec128(
            _mm256_extracti128_si256(v0, 0),
            _mm256_extracti128_si256(v0, 1),
            mults_128b,
        );
    } else {
        if len >= 16 {
            x0 = _mm_xor_si128(_mm_loadu_si128(data.as_ptr() as *const __m128i), x0);
            data = &data[16..];
            len -= 16;
        } else {
            // Optimize: Use slicing-by-8 for small tails >= 8 bytes.
            if len >= 8 {
                return crate::crc32::crc32_slice8(crc, data);
            }
            return crate::crc32::crc32_slice1(crc, data);
        }
    }

    while len >= 16 {
        x0 = fold_vec128(
            x0,
            _mm_loadu_si128(data.as_ptr() as *const __m128i),
            mults_128b,
        );
        data = &data[16..];
        len -= 16;
    }

    x0 = _mm_xor_si128(
        _mm_clmulepi64_si128(x0, mults_128b, 0x10),
        _mm_srli_si128(x0, 8),
    );
    let mut x1 = _mm_clmulepi64_si128(x0, barrett_reduction_constants, 0x00);
    x1 = _mm_clmulepi64_si128(x1, barrett_reduction_constants, 0x10);
    x0 = _mm_xor_si128(x0, x1);

    let mut res = _mm_extract_epi32(x0, 2) as u32;

    if len >= 8 {
        res = crate::crc32::crc32_slice8(res, data);
    } else if len > 0 {
        res = crate::crc32::crc32_slice1(res, data);
    }

    res
}
