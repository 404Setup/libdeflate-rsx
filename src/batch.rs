use crate::compress::{CompressResult, Compressor};
use crate::decompress::{DecompressResult, Decompressor};
use rayon::prelude::*;

pub struct BatchCompressor {
    level: usize,
}

impl BatchCompressor {
    pub fn new(level: usize) -> Self {
        Self { level }
    }

    pub fn compress_batch(&self, inputs: &[&[u8]]) -> Vec<Vec<u8>> {
        inputs
            .par_iter()
            .map_init(
                || (Compressor::new(self.level), Vec::new()),
                |(compressor, buffer), &input| {
                    let bound = Compressor::deflate_compress_bound(input.len());
                    buffer.clear();
                    buffer.reserve(bound);
                    let buf_uninit = buffer.spare_capacity_mut();
                    // Safe because we reserved enough capacity
                    let buf_slice = &mut buf_uninit[..bound];

                    let (res, size, _) =
                        compressor.compress(input, buf_slice, crate::compress::FlushMode::Finish);
                    if res == CompressResult::Success {
                        unsafe {
                            buffer.set_len(size);
                        }
                        buffer.to_vec()
                    } else {
                        Vec::new()
                    }
                },
            )
            .collect()
    }
}

pub struct BatchDecompressor;

impl BatchDecompressor {
    pub fn new() -> Self {
        Self
    }

    pub fn decompress_batch(
        &self,
        inputs: &[&[u8]],
        max_out_sizes: &[usize],
    ) -> Vec<Option<Vec<u8>>> {
        inputs
            .par_iter()
            .zip(max_out_sizes.par_iter())
            .map_init(Decompressor::new, |decompressor, (&input, &max_size)| {
                let mut output = vec![0u8; max_size];
                let (res, _, size) = decompressor.decompress(input, &mut output);
                if res == DecompressResult::Success {
                    output.truncate(size);
                    Some(output)
                } else {
                    None
                }
            })
            .collect()
    }
}
