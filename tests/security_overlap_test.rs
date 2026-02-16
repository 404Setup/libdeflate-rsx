
#[cfg(test)]
mod tests {
    use libdeflate::{Compressor, Decompressor};
    use std::io;

    #[test]
    fn test_compress_overlap() {
        let mut compressor = Compressor::new(1).unwrap();
        let mut buffer = vec![0u8; 1024];
        // Fill buffer with some data to compress
        for i in 0..512 {
            buffer[i] = (i % 256) as u8;
        }

        // Create overlapping slices using unsafe
        let ptr = buffer.as_mut_ptr();
        let input_slice = unsafe { std::slice::from_raw_parts(ptr, 512) };
        let output_slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(10), 512) };

        let res = compressor.compress_deflate_into(input_slice, output_slice);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(err.to_string(), "Input and output buffers overlap");
    }

    #[test]
    fn test_decompress_overlap() {
        let mut decompressor = Decompressor::new();
        let mut buffer = vec![0u8; 1024];

        // We don't strictly need valid compressed data to test the overlap check,
        // because the check happens before decompression logic.
        // But to be safe against future reordering, we can put some dummy data.

        let ptr = buffer.as_mut_ptr();
        // Input: 0..100
        let input_slice = unsafe { std::slice::from_raw_parts(ptr, 100) };
        // Output: 50..150 (Overlap 50..100)
        let output_slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(50), 100) };

        let res = decompressor.decompress_deflate_into(input_slice, output_slice);
        assert!(res.is_err(), "Result should be Err, got {:?}", res);
        let err = res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(err.to_string(), "Input and output buffers overlap");
    }
}
