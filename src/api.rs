use crate::compress::{CompressResult, Compressor as InternalCompressor, FlushMode};
use crate::decompress::Decompressor as InternalDecompressor;
use std::io::{self};

pub struct Compressor {
    inner: InternalCompressor,
}

impl Compressor {
    pub fn new(level: i32) -> io::Result<Self> {
        if !(0..=12).contains(&level) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Compression level must be between 0 and 12",
            ));
        }
        Ok(Self {
            inner: InternalCompressor::new(level as usize),
        })
    }

    pub fn compress_deflate(&mut self, data: &[u8]) -> io::Result<Vec<u8>> {
        let bound = self.deflate_compress_bound(data.len());
        self.compress_helper(data, bound, |c, data, out| {
            let (res, size, _) = c.compress(data, out, FlushMode::Finish);
            (res, size)
        })
    }

    pub fn compress_deflate_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.compress_into_helper(data, output, "Insufficient space", |c, data, out| {
            let (res, size, _) = c.compress(data, out, FlushMode::Finish);
            (res, size)
        })
    }

    pub fn compress_zlib(&mut self, data: &[u8]) -> io::Result<Vec<u8>> {
        let bound = self.zlib_compress_bound(data.len());
        self.compress_helper(data, bound, |c, data, out| c.compress_zlib(data, out))
    }

    pub fn compress_zlib_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.compress_into_helper(data, output, "Compression failed", |c, data, out| {
            c.compress_zlib(data, out)
        })
    }

    pub fn compress_gzip(&mut self, data: &[u8]) -> io::Result<Vec<u8>> {
        let bound = self.gzip_compress_bound(data.len());
        self.compress_helper(data, bound, |c, data, out| c.compress_gzip(data, out))
    }

    pub fn compress_gzip_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.compress_into_helper(data, output, "Compression failed", |c, data, out| {
            c.compress_gzip(data, out)
        })
    }

    pub fn deflate_compress_bound(&mut self, size: usize) -> usize {
        InternalCompressor::deflate_compress_bound(size)
    }

    pub fn zlib_compress_bound(&mut self, size: usize) -> usize {
        InternalCompressor::zlib_compress_bound(size)
    }

    pub fn gzip_compress_bound(&mut self, size: usize) -> usize {
        InternalCompressor::gzip_compress_bound(size)
    }

    fn compress_helper<F>(&mut self, data: &[u8], bound: usize, f: F) -> io::Result<Vec<u8>>
    where
        F: FnOnce(
            &mut InternalCompressor,
            &[u8],
            &mut [std::mem::MaybeUninit<u8>],
        ) -> (CompressResult, usize),
    {
        let mut output = Vec::new();
        output.try_reserve_exact(bound).map_err(io::Error::other)?;

        // Use spare_capacity_mut to avoid zero-initialization.
        // Since len is 0, this returns the entire capacity as MaybeUninit.
        // We slice it to 'bound' to match the requested size.
        let out_uninit = output.spare_capacity_mut();
        let out_uninit = &mut out_uninit[..bound];

        let (res, size) = f(&mut self.inner, data, out_uninit);
        match res {
            CompressResult::Success => {
                unsafe {
                    output.set_len(size);
                }
                Ok(output)
            }
            CompressResult::InsufficientSpace => Err(io::Error::other("Insufficient space")),
        }
    }

    fn compress_into_helper<F>(
        &mut self,
        data: &[u8],
        output: &mut [u8],
        error_msg: &str,
        f: F,
    ) -> io::Result<usize>
    where
        F: FnOnce(
            &mut InternalCompressor,
            &[u8],
            &mut [std::mem::MaybeUninit<u8>],
        ) -> (CompressResult, usize),
    {
        if is_overlapping(data, output) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Input and output buffers overlap",
            ));
        }
        let out_uninit = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                output.len(),
            )
        };
        let (res, size) = f(&mut self.inner, data, out_uninit);
        if res == CompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::other(error_msg))
        }
    }
}

pub struct Decompressor {
    inner: InternalDecompressor,
    max_memory_limit: usize,
    limit_ratio: usize,
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            inner: InternalDecompressor::new(),
            max_memory_limit: usize::MAX,
            limit_ratio: 2000,
        }
    }

    pub fn set_max_memory_limit(&mut self, limit: usize) {
        self.max_memory_limit = limit;
    }

    pub fn set_limit_ratio(&mut self, ratio: usize) {
        self.limit_ratio = ratio;
    }

    pub fn decompress_deflate(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        self.decompress_helper(data, expected_size, |d, data, out| unsafe {
            d.decompress_uninit(data, out)
        })
    }

    pub fn decompress_deflate_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.decompress_into_helper(data, output, |d, data, out| unsafe {
            d.decompress_uninit(data, out)
        })
    }

    pub fn decompress_zlib(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        self.decompress_helper(data, expected_size, |d, data, out| unsafe {
            d.decompress_zlib_uninit(data, out)
        })
    }

    pub fn decompress_zlib_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.decompress_into_helper(data, output, |d, data, out| unsafe {
            d.decompress_zlib_uninit(data, out)
        })
    }

    pub fn decompress_gzip(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        self.decompress_helper(data, expected_size, |d, data, out| unsafe {
            d.decompress_gzip_uninit(data, out)
        })
    }

    pub fn decompress_gzip_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.decompress_into_helper(data, output, |d, data, out| unsafe {
            d.decompress_gzip_uninit(data, out)
        })
    }

    fn decompress_helper<F>(
        &mut self,
        data: &[u8],
        expected_size: usize,
        f: F,
    ) -> io::Result<Vec<u8>>
    where
        F: FnOnce(
            &mut InternalDecompressor,
            &[u8],
            &mut [std::mem::MaybeUninit<u8>],
        ) -> (crate::decompress::DecompressResult, usize, usize),
    {
        // Security check: prevent massive allocations for small inputs (Zip bomb prevention)
        // Max compression ratio for Deflate is ~1032:1. We use a generous limit of 2000:1 + overhead.
        // This prevents allocating GBs of memory for small inputs.
        let limit = data
            .len()
            .saturating_mul(self.limit_ratio)
            .saturating_add(4096);
        if expected_size > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Expected size {} exceeds safety limit for input size {}",
                    expected_size,
                    data.len()
                ),
            ));
        }

        if expected_size > self.max_memory_limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Expected size {} exceeds maximum memory limit {}",
                    expected_size, self.max_memory_limit
                ),
            ));
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(expected_size)
            .map_err(io::Error::other)?;

        // Use spare_capacity_mut to avoid zero-initialization.
        let out_uninit = output.spare_capacity_mut();
        // Ensure we only use the expected size
        let out_uninit = &mut out_uninit[..expected_size];

        let (res, _, size) = f(&mut self.inner, data, out_uninit);
        if res == crate::decompress::DecompressResult::Success {
            unsafe {
                output.set_len(size);
            }
            Ok(output)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    fn decompress_into_helper<F>(
        &mut self,
        data: &[u8],
        output: &mut [u8],
        f: F,
    ) -> io::Result<usize>
    where
        F: FnOnce(
            &mut InternalDecompressor,
            &[u8],
            &mut [std::mem::MaybeUninit<u8>],
        ) -> (crate::decompress::DecompressResult, usize, usize),
    {
        if is_overlapping(data, output) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Input and output buffers overlap",
            ));
        }
        // Safe to cast &mut [u8] to &mut [MaybeUninit<u8>]
        let out_uninit = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr() as *mut std::mem::MaybeUninit<u8>,
                output.len(),
            )
        };
        let (res, _, size) = f(&mut self.inner, data, out_uninit);
        if res == crate::decompress::DecompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }
}

fn is_overlapping(s1: &[u8], s2: &[u8]) -> bool {
    let p1 = s1.as_ptr() as usize;
    let len1 = s1.len();
    let p2 = s2.as_ptr() as usize;
    let len2 = s2.len();

    let end1 = p1.saturating_add(len1);
    let end2 = p2.saturating_add(len2);

    use std::cmp::{max, min};
    max(p1, p2) < min(end1, end2)
}
