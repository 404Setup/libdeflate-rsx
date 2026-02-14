use libdeflate::compress::bitstream::Bitstream;
use std::mem::MaybeUninit;

#[test]
fn test_write_bits_large_count() {
    let mut output = vec![MaybeUninit::uninit(); 100];
    let output_len;
    {
        let mut bs = Bitstream::new(&mut output);

        // Write 20 bits: 0xAAAAA (10101010101010101010)
        let val = 0xAAAAA;
        let count = 20;

        // This should now succeed for count > 16 due to the fix.
        assert!(bs.write_bits(val, count));

        let (res, valid_bits) = bs.flush();
        assert!(res);
        assert_eq!(valid_bits, 4);
        output_len = bs.out_idx;
    }

    let out_slice = unsafe { std::slice::from_raw_parts(output.as_ptr() as *const u8, output_len) };

    println!("Output: {:?}", out_slice);

    assert_eq!(out_slice.len(), 3);
    assert_eq!(out_slice[0], 0xAA);
    assert_eq!(out_slice[1], 0xAA);
    assert_eq!(out_slice[2] & 0x0F, 0x0A);
}
