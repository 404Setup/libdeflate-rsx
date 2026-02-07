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
            .map(|&input| {
                let mut compressor = Compressor::new(self.level);
                let bound = Compressor::deflate_compress_bound(input.len());
                let mut output = vec![0u8; bound];
                let (res, size, _) = compressor.compress(input, &mut output, crate::compress::FlushMode::Finish);
                if res == CompressResult::Success {
                    output.truncate(size);
                    output
                } else {
                    Vec::new()
                }
            })
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
            .map(|(&input, &max_size)| {
                let mut decompressor = Decompressor::new();
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
