use crate::compress::{CompressResult, Compressor as InternalCompressor, FlushMode};
use crate::decompress::Decompressor as InternalDecompressor;
use std::io::{self};

pub struct Compressor {
    inner: InternalCompressor,
}

impl Compressor {
    pub fn new(level: i32) -> io::Result<Self> {
        if level < 0 || level > 12 {
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
        let mut output = Vec::with_capacity(bound);
        unsafe {
            output.set_len(bound);
        }

        let (res, size, _) = self.inner.compress(data, &mut output, FlushMode::Finish);

        match res {
            CompressResult::Success => {
                unsafe {
                    output.set_len(size);
                }
                Ok(output)
            }
            CompressResult::InsufficientSpace => {
                Err(io::Error::new(io::ErrorKind::Other, "Insufficient space"))
            }
        }
    }

    pub fn compress_deflate_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, size, _) = self.inner.compress(data, output, FlushMode::Finish);
        if res == CompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Insufficient space"))
        }
    }

    pub fn compress_zlib(&mut self, data: &[u8]) -> io::Result<Vec<u8>> {
        let bound = self.zlib_compress_bound(data.len());
        let mut output = vec![0u8; bound];
        let (res, size) = self.inner.compress_zlib(data, &mut output);
        if res == CompressResult::Success {
            output.truncate(size);
            Ok(output)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
        }
    }

    pub fn compress_zlib_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, size) = self.inner.compress_zlib(data, output);
        if res == CompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
        }
    }

    pub fn compress_gzip(&mut self, data: &[u8]) -> io::Result<Vec<u8>> {
        let bound = self.gzip_compress_bound(data.len());
        let mut output = vec![0u8; bound];
        let (res, size) = self.inner.compress_gzip(data, &mut output);
        if res == CompressResult::Success {
            output.truncate(size);
            Ok(output)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
        }
    }

    pub fn compress_gzip_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, size) = self.inner.compress_gzip(data, output);
        if res == CompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
        }
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
}

pub struct Decompressor {
    inner: InternalDecompressor,
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            inner: InternalDecompressor::new(),
        }
    }

    pub fn decompress_deflate(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(expected_size);
        unsafe {
            output.set_len(expected_size);
        }

        let (res, _, size) = self.inner.decompress(data, &mut output);

        if res == crate::decompress::DecompressResult::Success {
            output.truncate(size);
            Ok(output)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    pub fn decompress_deflate_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, _, size) = self.inner.decompress(data, output);
        if res == crate::decompress::DecompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    pub fn decompress_zlib(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(expected_size);
        unsafe {
            output.set_len(expected_size);
        }

        let (res, _, size) = self.inner.decompress_zlib(data, &mut output);

        if res == crate::decompress::DecompressResult::Success {
            output.truncate(size);
            Ok(output)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    pub fn decompress_zlib_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, _, size) = self.inner.decompress_zlib(data, output);
        if res == crate::decompress::DecompressResult::Success {
            Ok(size)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    pub fn decompress_gzip(&mut self, data: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(expected_size);
        unsafe {
            output.set_len(expected_size);
        }

        let (res, _, size) = self.inner.decompress_gzip(data, &mut output);

        if res == crate::decompress::DecompressResult::Success {
            output.truncate(size);
            Ok(output)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Decompression failed",
            ))
        }
    }

    pub fn decompress_gzip_into(&mut self, data: &[u8], output: &mut [u8]) -> io::Result<usize> {
        let (res, _, size) = self.inner.decompress_gzip(data, output);
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
