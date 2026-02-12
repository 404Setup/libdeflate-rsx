use libdeflate::stream::{DeflateDecoder, DeflateEncoder};
use std::io::{Read, Write};

fn main() -> std::io::Result<()> {
    let data = b"Stream compression is useful for large files or network streams. ".repeat(50);
    println!("Original data size: {}", data.len());

    let mut encoded_data = Vec::new();

    let mut encoder = DeflateEncoder::new(&mut encoded_data, 6);

    encoder.write_all(&data[..data.len() / 2])?;
    encoder.write_all(&data[data.len() / 2..])?;

    encoder.finish()?;

    println!("Encoded data size: {}", encoded_data.len());

    let mut decoded_data = Vec::new();
    let mut decoder = DeflateDecoder::new(encoded_data.as_slice());

    decoder.read_to_end(&mut decoded_data)?;

    println!("Decoded data size: {}", decoded_data.len());

    assert_eq!(data.as_slice(), decoded_data.as_slice());
    println!("Stream round-trip successful!");

    Ok(())
}
