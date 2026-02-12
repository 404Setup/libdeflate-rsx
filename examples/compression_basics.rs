use libdeflate::{Compressor, Decompressor};
use std::io;

fn main() -> io::Result<()> {
    let data = b"The quick brown fox jumps over the lazy dog. ".repeat(10);
    println!("Original size: {}", data.len());

    let mut compressor = Compressor::new(6)?;

    let compressed_data = compressor.compress_deflate(&data)?;
    println!("Compressed size: {}", compressed_data.len());

    let mut decompressor = Decompressor::new();

    let decompressed_data = decompressor.decompress_deflate(&compressed_data, data.len())?;

    assert_eq!(data.as_slice(), decompressed_data.as_slice());
    println!("Decompression successful!");

    Ok(())
}
