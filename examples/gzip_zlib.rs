use libdeflate::{Compressor, Decompressor};
use std::io;

fn main() -> io::Result<()> {
    let data = b"Compression using Gzip and Zlib wrappers.".repeat(20);
    let mut compressor = Compressor::new(9)?;
    let mut decompressor = Decompressor::new();

    println!("--- Zlib ---");
    let zlib_data = compressor.compress_zlib(&data)?;
    println!("Zlib Compressed size: {}", zlib_data.len());

    let zlib_out = decompressor.decompress_zlib(&zlib_data, data.len())?;
    assert_eq!(data.as_slice(), zlib_out.as_slice());
    println!("Zlib Decompression successful");

    println!("--- Gzip ---");
    let gzip_data = compressor.compress_gzip(&data)?;
    println!("Gzip Compressed size: {}", gzip_data.len());

    let gzip_out = decompressor.decompress_gzip(&gzip_data, data.len())?;
    assert_eq!(data.as_slice(), gzip_out.as_slice());
    println!("Gzip Decompression successful");

    Ok(())
}
