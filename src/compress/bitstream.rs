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

    /// Writes bits without checking count or masking bits.
    ///
    /// # Safety
    ///
    /// * `count` must be > 0.
    /// * `bits` must not have any bits set above `count` (i.e., `bits & !((1 << count) - 1) == 0`).
    #[inline(always)]
    pub unsafe fn write_bits_unchecked(&mut self, bits: u32, count: u32) -> bool {
        debug_assert!(count > 0);
        // Optimization assumes we never write more than 16 bits at a time.
        // This ensures that `self.bitcount` (max 47 before add) + count (max 16) <= 63,
        // preventing u64 bitbuf overflow.
        debug_assert!(count <= 16);

        self.bitbuf |= (bits as u64) << self.bitcount;
        self.bitcount += count;

        // Flush when we have at least 6 bytes (48 bits).
        // This reduces store frequency compared to flushing at 4 bytes (32 bits).
        if self.bitcount >= 48 {
            if self.out_idx + 8 <= self.output.len() {
                unsafe {
                    std::ptr::write_unaligned(
                        self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                        self.bitbuf.to_le(),
                    );
                }
                let bytes_written = (self.bitcount >> 3) as usize;
                self.out_idx += bytes_written;
                self.bitbuf >>= bytes_written * 8;
                self.bitcount &= 7;
                return true;
            }
        }

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
        true
    }

    pub fn flush(&mut self) -> (bool, u32) {
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
