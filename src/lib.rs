#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![allow(unsafe_op_in_unsafe_fn)]
pub mod adler32;
pub mod api;
pub mod batch;
pub mod common;
pub mod compress;
pub mod crc32;
pub mod crc32_tables;
pub mod decompress;
pub mod stream;

pub use adler32::adler32;
pub use api::{Compressor, Decompressor};
pub use crc32::crc32;