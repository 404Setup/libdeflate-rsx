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
    pub fn write_bits(&mut self, mut bits: u32, mut count: u32) -> bool {
        // Security/Robustness: Handle count > 16 by splitting writes.
        // The underlying write_bits_unchecked supports max 16 bits at a time.
        // This prevents potential assertion failures in debug builds and data corruption in release builds.
        while count > 16 {
            // Safety: count is 16, and we mask bits to 16 bits, so the requirement
            // `bits & !((1 << count) - 1) == 0` is satisfied.
            unsafe {
                if !self.write_bits_unchecked(bits & 0xFFFF, 16) {
                    return false;
                }
            }
            bits >>= 16;
            count -= 16;
        }
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
        if count <= 16 {
            return self.write_bits_unchecked(bits, count);
        }
        if !self.write_bits_unchecked(bits & 0xFFFF, 16) {
            return false;
        }
        self.write_bits_unchecked(bits >> 16, count - 16)
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
        // We flush a constant 6 bytes (48 bits) to enable constant shifts and avoids variable logic.
        if self.bitcount >= 48 {
            if self.out_idx + 8 <= self.output.len() {
                unsafe {
                    std::ptr::write_unaligned(
                        self.output.as_mut_ptr().add(self.out_idx) as *mut u64,
                        self.bitbuf.to_le(),
                    );
                }
                self.out_idx += 6;
                self.bitbuf >>= 48;
                self.bitcount -= 48;
                return true;
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
