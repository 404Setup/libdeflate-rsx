pub const LIBDEFLATE_VERSION_MAJOR: u32 = 1;
pub const LIBDEFLATE_VERSION_MINOR: u32 = 25;
pub const LIBDEFLATE_VERSION_STRING: &str = "1.25";


pub const DEFLATE_BLOCKTYPE_UNCOMPRESSED: u8 = 0;
pub const DEFLATE_BLOCKTYPE_STATIC_HUFFMAN: u8 = 1;
pub const DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN: u8 = 2;

pub const DEFLATE_MIN_MATCH_LEN: usize = 3;
pub const DEFLATE_MAX_MATCH_LEN: usize = 258;

pub const DEFLATE_MAX_MATCH_OFFSET: usize = 32768;
pub const DEFLATE_WINDOW_ORDER: usize = 15;

pub const DEFLATE_NUM_PRECODE_SYMS: usize = 19;
pub const DEFLATE_NUM_LITLEN_SYMS: usize = 288;
pub const DEFLATE_NUM_OFFSET_SYMS: usize = 32;

pub const DEFLATE_MAX_NUM_SYMS: usize = 288;

pub const DEFLATE_NUM_LITERALS: usize = 256;
pub const DEFLATE_END_OF_BLOCK: usize = 256;
pub const DEFLATE_FIRST_LEN_SYM: usize = 257;

pub const DEFLATE_MAX_PRE_CODEWORD_LEN: usize = 7;
pub const DEFLATE_MAX_LITLEN_CODEWORD_LEN: usize = 15;
pub const DEFLATE_MAX_OFFSET_CODEWORD_LEN: usize = 15;

pub const DEFLATE_MAX_CODEWORD_LEN: usize = 15;

pub const DEFLATE_MAX_LENS_OVERRUN: usize = 137;

pub const DEFLATE_MAX_EXTRA_LENGTH_BITS: usize = 5;
pub const DEFLATE_MAX_EXTRA_OFFSET_BITS: usize = 13;

pub const ZLIB_MIN_HEADER_SIZE: usize = 2;
pub const ZLIB_FOOTER_SIZE: usize = 4;
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_MIN_HEADER_SIZE + ZLIB_FOOTER_SIZE;

pub const ZLIB_CM_DEFLATE: u8 = 8;
pub const ZLIB_CINFO_32K_WINDOW: u8 = 7;

pub const ZLIB_FASTEST_COMPRESSION: u8 = 0;
pub const ZLIB_FAST_COMPRESSION: u8 = 1;
pub const ZLIB_DEFAULT_COMPRESSION: u8 = 2;
pub const ZLIB_SLOWEST_COMPRESSION: u8 = 3;

pub const GZIP_MIN_HEADER_SIZE: usize = 10;
pub const GZIP_FOOTER_SIZE: usize = 8;
pub const GZIP_MIN_OVERHEAD: usize = GZIP_MIN_HEADER_SIZE + GZIP_FOOTER_SIZE;

pub const GZIP_ID1: u8 = 0x1F;
pub const GZIP_ID2: u8 = 0x8B;
pub const GZIP_CM_DEFLATE: u8 = 8;

pub const GZIP_FTEXT: u8 = 0x01;
pub const GZIP_FHCRC: u8 = 0x02;
pub const GZIP_FEXTRA: u8 = 0x04;
pub const GZIP_FNAME: u8 = 0x08;
pub const GZIP_FCOMMENT: u8 = 0x10;
pub const GZIP_FRESERVED: u8 = 0xE0;

pub const GZIP_MTIME_UNAVAILABLE: u32 = 0;
pub const GZIP_XFL_SLOWEST_COMPRESSION: u8 = 0x02;
pub const GZIP_XFL_FASTEST_COMPRESSION: u8 = 0x04;
pub const GZIP_OS_UNKNOWN: u8 = 255;

pub const MIN_BLOCK_LENGTH: usize = 5000;
pub const SOFT_MAX_BLOCK_LENGTH: usize = 300000;
pub const SEQ_STORE_LENGTH: usize = 50000;
