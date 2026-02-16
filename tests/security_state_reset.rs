
use libdeflate::compress::{Compressor, FlushMode};
use libdeflate::decompress::{Decompressor, DecompressResult, DecompressorState};

#[test]
fn test_state_corruption_after_one_shot_decompress() {
    // 1. Create a dynamic block
    // Use random-ish data to ensure it's not too small and uses dynamic codes
    let mut compressor = Compressor::new(6);
    let mut data = Vec::with_capacity(10000);
    for i in 0..10000 {
        data.push((i % 251) as u8); // Pseudo-random pattern
    }

    let mut compressed = vec![std::mem::MaybeUninit::uninit(); 20000];
    let (res, size, _) = compressor.compress(&data, &mut compressed, FlushMode::Finish);
    assert_eq!(res, libdeflate::compress::CompressResult::Success);
    println!("Compressed size: {}", size);

    let compressed_data = unsafe {
        std::slice::from_raw_parts(compressed.as_ptr() as *const u8, size)
    };

    // 2. Split into two parts.
    // Ensure split point is within the data but not at the very end
    if size < 200 {
        panic!("Compressed data too small ({}) to split reliably for this test", size);
    }
    let split_point = size / 2;
    let (part1, part2) = compressed_data.split_at(split_point);

    let mut decompressor = Decompressor::new();
    let mut output = vec![0u8; 20000];
    let mut out_idx = 0;

    // 3. Start streaming part 1
    let (res1, in1, out1) = decompressor.decompress_streaming(part1, &mut output, &mut out_idx);

    // We expect ShortInput because we didn't provide full data
    println!("Streaming part 1 result: {:?}, consumed: {}, produced: {}", res1, in1, out1);

    match decompressor.state {
        DecompressorState::BlockBody | DecompressorState::BlockBodyOffset { .. } => {
            println!("State after part1: BlockBody (Correct)");
        },
        state => {
            println!("State after part1: {:?}", state);
            // If we are not in BlockBody, we might have finished a block?
            // With 10000 bytes, likely one block.
        }
    }

    // 4. Perform one-shot decompression on different data
    let other_data = vec![66u8; 500]; // "BBBB..."
    let mut other_compressed = vec![std::mem::MaybeUninit::uninit(); 1000];
    let mut other_comp = Compressor::new(6);
    let (_, other_size, _) = other_comp.compress(&other_data, &mut other_compressed, FlushMode::Finish);
    let other_compressed_slice = unsafe {
        std::slice::from_raw_parts(other_compressed.as_ptr() as *const u8, other_size)
    };

    let mut other_out = vec![0u8; 1000];
    let res_other = decompressor.decompress(other_compressed_slice, &mut other_out);
    println!("One-shot result: {:?}", res_other);

    // 5. Check state
    // Vulnerable: State is still BlockBody (from step 3).
    // Secure: State is reset to Start.
    println!("State after one-shot: {:?}", decompressor.state);

    if decompressor.state != DecompressorState::Start {
        panic!("VULNERABILITY: Decompressor state was not reset after one-shot decompression! Internal state corrupted.");
    }
}
