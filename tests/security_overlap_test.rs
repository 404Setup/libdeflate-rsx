#[cfg(test)]
mod tests {
    use libdeflate::{Compressor, Decompressor};
    use std::io;

    #[test]
    fn test_compress_overlap() {
        let mut compressor = Compressor::new(1).unwrap();
        let mut buffer = vec![0u8; 1024];

        // Helper to test overlap for a specific format
        let mut check_overlap = |input_range: std::ops::Range<usize>,
                                 output_range: std::ops::Range<usize>,
                                 method: &str| {
            let ptr = buffer.as_mut_ptr();
            let input_slice = unsafe {
                std::slice::from_raw_parts(ptr.add(input_range.start), input_range.len())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(ptr.add(output_range.start), output_range.len())
            };

            let res = match method {
                "deflate" => compressor.compress_deflate_into(input_slice, output_slice),
                "zlib" => compressor.compress_zlib_into(input_slice, output_slice),
                "gzip" => compressor.compress_gzip_into(input_slice, output_slice),
                _ => panic!("Unknown method"),
            };

            assert!(
                res.is_err(),
                "Expected error for overlap with method {}",
                method
            );
            let err = res.unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(err.to_string(), "Input and output buffers overlap");
        };

        // Scenarios:
        // 1. Partial overlap (start)
        // Input: [0, 100), Output: [50, 150) -> Overlap [50, 100)
        check_overlap(0..100, 50..150, "deflate");
        check_overlap(0..100, 50..150, "zlib");
        check_overlap(0..100, 50..150, "gzip");

        // 2. Partial overlap (end)
        // Input: [50, 150), Output: [0, 100) -> Overlap [50, 100)
        check_overlap(50..150, 0..100, "deflate");
        check_overlap(50..150, 0..100, "zlib");
        check_overlap(50..150, 0..100, "gzip");

        // 3. Inclusion (input inside output)
        // Input: [50, 100), Output: [0, 150) -> Overlap [50, 100)
        check_overlap(50..100, 0..150, "deflate");
        check_overlap(50..100, 0..150, "zlib");
        check_overlap(50..100, 0..150, "gzip");

        // 4. Inclusion (output inside input)
        // Input: [0, 150), Output: [50, 100) -> Overlap [50, 100)
        check_overlap(0..150, 50..100, "deflate");
        check_overlap(0..150, 50..100, "zlib");
        check_overlap(0..150, 50..100, "gzip");

        // 5. Exact match
        // Input: [0, 100), Output: [0, 100) -> Overlap [0, 100)
        check_overlap(0..100, 0..100, "deflate");
        check_overlap(0..100, 0..100, "zlib");
        check_overlap(0..100, 0..100, "gzip");
    }

    #[test]
    fn test_decompress_overlap() {
        let mut decompressor = Decompressor::new();
        let mut buffer = vec![0u8; 1024];

        // Helper to test overlap for a specific format
        let mut check_overlap = |input_range: std::ops::Range<usize>,
                                 output_range: std::ops::Range<usize>,
                                 method: &str| {
            let ptr = buffer.as_mut_ptr();
            let input_slice = unsafe {
                std::slice::from_raw_parts(ptr.add(input_range.start), input_range.len())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(ptr.add(output_range.start), output_range.len())
            };

            let res = match method {
                "deflate" => decompressor.decompress_deflate_into(input_slice, output_slice),
                "zlib" => decompressor.decompress_zlib_into(input_slice, output_slice),
                "gzip" => decompressor.decompress_gzip_into(input_slice, output_slice),
                _ => panic!("Unknown method"),
            };

            assert!(
                res.is_err(),
                "Expected error for overlap with method {}",
                method
            );
            let err = res.unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(err.to_string(), "Input and output buffers overlap");
        };

        // Scenarios:
        // 1. Partial overlap (start)
        check_overlap(0..100, 50..150, "deflate");
        check_overlap(0..100, 50..150, "zlib");
        check_overlap(0..100, 50..150, "gzip");

        // 2. Partial overlap (end)
        check_overlap(50..150, 0..100, "deflate");
        check_overlap(50..150, 0..100, "zlib");
        check_overlap(50..150, 0..100, "gzip");

        // 3. Inclusion (input inside output)
        check_overlap(50..100, 0..150, "deflate");
        check_overlap(50..100, 0..150, "zlib");
        check_overlap(50..100, 0..150, "gzip");

        // 4. Inclusion (output inside input)
        check_overlap(0..150, 50..100, "deflate");
        check_overlap(0..150, 50..100, "zlib");
        check_overlap(0..150, 50..100, "gzip");

        // 5. Exact match
        check_overlap(0..100, 0..100, "deflate");
        check_overlap(0..100, 0..100, "zlib");
        check_overlap(0..100, 0..100, "gzip");
    }

    #[test]
    fn test_no_overlap() {
        let mut compressor = Compressor::new(1).unwrap();
        let mut buffer = vec![0u8; 1024];

        let ptr = buffer.as_mut_ptr();

        // Input: [0, 100)
        // Output: [100, 200) -> No overlap (Touching at 100)
        let input_slice = unsafe { std::slice::from_raw_parts(ptr, 100) };
        let output_slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(100), 100) };

        let res = compressor.compress_deflate_into(input_slice, output_slice);
        if let Err(e) = res {
            assert_ne!(e.to_string(), "Input and output buffers overlap");
        }

        // Touching on the other side
        // Input: [100, 200)
        // Output: [0, 100) -> No overlap (Touching at 100)
        let input_slice = unsafe { std::slice::from_raw_parts(ptr.add(100), 100) };
        let output_slice = unsafe { std::slice::from_raw_parts_mut(ptr, 100) };

        let res = compressor.compress_deflate_into(input_slice, output_slice);
        if let Err(e) = res {
            assert_ne!(e.to_string(), "Input and output buffers overlap");
        }
    }
}
