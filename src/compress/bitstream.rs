use std::mem::MaybeUninit;

pub struct Bitstream<'a> {
    pub output: &'a mut [MaybeUninit<u8>],
    pub out_idx: usize,
    pub bitbuf: u64,
    pub bitcount: u32,
}

impl<'a> Bitstream<'a> {
    pub fn new(output: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            output,
            out_idx: 0,
            bitbuf: 0,
            bitcount: 0,
        }
    }

    #[inline(always)]
    pub fn write_bits(&mut self, bits: u32, count: u32) -> bool {
        if count == 0 {
            return true;
        }
        // Use u64 to handle count=32 case without branching (1u32 << 32 overflows)
        let mask = ((1u64 << count) - 1) as u32;
        unsafe { self.write_bits_unchecked(bits & mask, count) }
    }

    /// Writes up to 32 bits without checking count or masking bits.
    ///
    /// # Safety
    ///
    /// * `count` must be > 0.
    /// * `bits` must not have any bits set above `count`.
    #[inline(always)]
    pub unsafe fn write_bits_upto_32(&mut self, bits: u32, count: u32) -> bool {
        self.write_bits_unchecked(bits, count)
    }

    /// Writes bits assuming sufficient buffer space (at least 8 bytes at current `out_idx`).
    ///
    /// # Safety
    ///
    /// * `count` must be > 0.
    /// * `bits` must not have any bits set above `count`.
    /// * `self.out_idx + 8 <= self.output.len()`.
    #[inline(always)]
    pub unsafe fn write_bits_unchecked_fast(&mut self, bits: u32, count: u32) {
        debug_assert!(count > 0);
        debug_assert!(count <= 32);
        debug_assert!(self.out_idx + 8 <= self.output.len());

        let bitcount = self.bitcount;
        let new_bitcount = bitcount + count;

        if new_bitcount >= 32 {
            let bitbuf = self.bitbuf | ((bits as u64) << bitcount);
            std::ptr::write_unaligned(
                self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                bitbuf.to_le(),
            );
            self.out_idx += 4;
            self.bitbuf = bitbuf >> 32;
            self.bitcount = new_bitcount - 32;
        } else {
            self.bitbuf |= (bits as u64) << bitcount;
            self.bitcount = new_bitcount;
        }
    }

    /// Writes up to 60 bits assuming sufficient buffer space (at least 8 bytes at current `out_idx`).
    ///
    /// # Safety
    ///
    /// * `count` must be > 0 and <= 60.
    /// * `bits` must not have any bits set above `count`.
    /// * `self.out_idx + 8 <= self.output.len()`.
    #[inline(always)]
    pub unsafe fn write_bits_unchecked_fast_64(&mut self, bits: u64, count: u32) {
        debug_assert!(count > 0);
        debug_assert!(count <= 60);
        debug_assert!(self.out_idx + 8 <= self.output.len());

        let bitcount = self.bitcount;
        let new_bitcount = bitcount + count;

        if new_bitcount >= 64 {
            let bitbuf_low = self.bitbuf | (bits << bitcount);
            let bitbuf_high = bits >> (64 - bitcount);

            std::ptr::write_unaligned(
                self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                bitbuf_low.to_le(),
            );
            self.out_idx += 8;
            self.bitbuf = bitbuf_high;
            self.bitcount = new_bitcount - 64;
        } else {
            let bitbuf = self.bitbuf | (bits << bitcount);
            if new_bitcount >= 32 {
                std::ptr::write_unaligned(
                    self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                    bitbuf.to_le(),
                );
                self.out_idx += 4;
                self.bitbuf = bitbuf >> 32;
                self.bitcount = new_bitcount - 32;
            } else {
                self.bitbuf = bitbuf;
                self.bitcount = new_bitcount;
            }
        }
    }

    /// Writes bits without checking count or masking bits.
    ///
    /// # Safety
    ///
    /// * `count` must be > 0.
    /// * `bits` must not have any bits set above `count` (i.e., `bits & !((1 << count) - 1) == 0`).
    #[inline(always)]
    pub unsafe fn write_bits_unchecked(&mut self, bits: u32, count: u32) -> bool {
        debug_assert!(count > 0);
        // Optimization: Flush every 32 bits.
        // This ensures that `self.bitcount` (max 31 before add) + count (max 32) <= 63,
        // preventing u64 bitbuf overflow.
        debug_assert!(count <= 32);

        let bitcount = self.bitcount;
        let new_bitcount = bitcount + count;

        // Flush when we have at least 4 bytes (32 bits).
        // This reduces store frequency compared to flushing at 4 bytes (32 bits) with byte-wise writes,
        // but more frequent than 48 bits. However, it simplifies logic for 32-bit writes.
        if new_bitcount >= 32 {
            let bitbuf = self.bitbuf | ((bits as u64) << bitcount);

            // Optimization: Write 64 bits (8 bytes) at once if buffer space allows.
            // This is safe even if we only have 32 bits of valid data because we only advance `out_idx` by 4.
            // The extra 4 bytes written are speculative and will be overwritten by the next write.
            // This avoids truncation to u32 and allows using full register width stores on 64-bit systems.
            if self.out_idx + 8 <= self.output.len() {
                unsafe {
                    std::ptr::write_unaligned(
                        self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                        bitbuf.to_le(),
                    );
                }
                self.out_idx += 4;
                self.bitbuf = bitbuf >> 32;
                self.bitcount = new_bitcount - 32;
                return true;
            }

            // Optimization: Write 32 bits at once if buffer space allows.
            // Using u32 write avoids 8-byte boundary check and reduces memory bandwidth compared to u64 blind write.
            if self.out_idx + 4 <= self.output.len() {
                unsafe {
                    std::ptr::write_unaligned(
                        self.output.as_mut_ptr().add(self.out_idx) as *mut u32,
                        (bitbuf as u32).to_le(),
                    );
                }
                self.out_idx += 4;
                self.bitbuf = bitbuf >> 32;
                self.bitcount = new_bitcount - 32;
                return true;
            }

            self.bitbuf = bitbuf;
            self.bitcount = new_bitcount;

            while self.bitcount >= 8 {
                if self.out_idx >= self.output.len() {
                    return false;
                }
                unsafe {
                    self.output
                        .get_unchecked_mut(self.out_idx)
                        .write((self.bitbuf & 0xFF) as u8);
                }
                self.out_idx += 1;
                self.bitbuf >>= 8;
                self.bitcount -= 8;
            }
        } else {
            self.bitbuf |= (bits as u64) << bitcount;
            self.bitcount = new_bitcount;
        }
        true
    }

    pub fn flush(&mut self) -> (bool, u32) {
        while self.bitcount >= 8 {
            if self.out_idx >= self.output.len() {
                return (false, 0);
            }
            unsafe {
                self.output
                    .get_unchecked_mut(self.out_idx)
                    .write((self.bitbuf & 0xFF) as u8);
            }
            self.out_idx += 1;
            self.bitbuf >>= 8;
            self.bitcount -= 8;
        }

        let mut valid_bits = 0;
        if self.bitcount > 0 {
            if self.out_idx >= self.output.len() {
                return (false, 0);
            }

            self.output[self.out_idx].write((self.bitbuf & 0xFF) as u8);
            self.out_idx += 1;
            valid_bits = self.bitcount;
            self.bitbuf = 0;
            self.bitcount = 0;
        }
        (true, valid_bits)
    }
}
