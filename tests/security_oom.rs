use libdeflate::{Compressor, Decompressor};

#[test]
fn test_oom_no_panic() {
    // 1. Prepare a `Compressor`
    let mut compressor = Compressor::new(1).unwrap();

    // 2. Create a minimal valid compressed stream (small input)
    let input_data = vec![b'A'; 100];
    let valid_compressed = compressor.compress_deflate(&input_data).unwrap();

    // 3. Construct a large input buffer to satisfy the security limit check.
    // Limit is roughly data.len() * 2000.
    // To request an 8GB allocation, we need input size >= 8GB / 2000 = 4MB.
    // We use 5MB to be safe.
    let fake_input_len = 5 * 1024 * 1024;
    let mut big_compressed = vec![0u8; fake_input_len];

    // Copy valid stream to the beginning
    big_compressed[..valid_compressed.len()].copy_from_slice(&valid_compressed);

    // 4. Request a huge allocation (8GB)
    // On most test environments, this will likely fail (OOM).
    // On beefy machines, it might succeed.
    // Crucially, it MUST NOT panic.
    let target_size = 8usize.saturating_mul(1024 * 1024 * 1024); // 8GB

    // Skip test on 32-bit targets if 8GB cannot be addressed
    if std::mem::size_of::<usize>() < 8 && target_size > usize::MAX {
        return;
    }

    let mut decompressor = Decompressor::new();
    let result = decompressor.decompress_deflate(&big_compressed, target_size);

    match result {
        Ok(_) => {
            println!("Allocation succeeded (unexpected but valid for high-memory systems)");
        }
        Err(e) => {
            println!("Got error as expected: {:?}", e);
        }
    }
}
