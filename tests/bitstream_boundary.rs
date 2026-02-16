use libdeflate::compress::bitstream::Bitstream;
use std::mem::MaybeUninit;

#[test]
fn test_write_bits_boundary_exact_4_bytes() {
    let mut output = vec![MaybeUninit::uninit(); 4];
    {
        let mut bs = Bitstream::new(&mut output);
        let val = 0x12345678;
        assert!(bs.write_bits(val, 32));
    }

    unsafe {
        let ptr = output.as_ptr() as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, 4);
        assert_eq!(bytes, 0x12345678u32.to_le_bytes());
    }
}

#[test]
fn test_write_bits_boundary_7_bytes() {
    let mut output = vec![MaybeUninit::uninit(); 7];
    {
        let mut bs = Bitstream::new(&mut output);
        let val = 0x87654321;
        assert!(bs.write_bits(val, 32));
    }

    unsafe {
        let ptr = output.as_ptr() as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, 4);
        assert_eq!(bytes, 0x87654321u32.to_le_bytes());
    }
}

#[test]
fn test_write_bits_multiple_writes_boundary() {
    // Total 12 bytes buffer.
    // Write 4 bytes (32 bits). Remaining 8 bytes.
    // Write 4 bytes (32 bits). Remaining 4 bytes. (Current fast path fails here? No, 4+8=12 <= 12. Fits.)
    // Write 4 bytes (32 bits). Remaining 0 bytes. (Current fast path fails here. 8+8=16 > 12.)
    let mut output = vec![MaybeUninit::uninit(); 12];
    {
        let mut bs = Bitstream::new(&mut output);
        bs.write_bits(0xAAAAAAAA, 32);
        bs.write_bits(0xBBBBBBBB, 32);
        bs.write_bits(0xCCCCCCCC, 32);
    }

    unsafe {
        let ptr = output.as_ptr() as *const u32;
        let vals = std::slice::from_raw_parts(ptr, 3);
        assert_eq!(vals[0].to_le(), 0xAAAAAAAA);
        assert_eq!(vals[1].to_le(), 0xBBBBBBBB);
        assert_eq!(vals[2].to_le(), 0xCCCCCCCC);
    }
}
