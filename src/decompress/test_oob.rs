use crate::decompress::prepare_pattern;

#[test]
fn test_prepare_pattern_boundary_safety() {
    // This test simulates a condition where the source buffer for pattern preparation
    // is very close to the end of the accessible memory.

    // Offset 5 case
    // Pattern source: 0, 1, 2, 3, 4
    // Expected result (u64): 0, 1, 2, 3, 4, 0, 1, 2
    let buffer_5 = vec![0u8, 1, 2, 3, 4];
    let src_ptr_5 = buffer_5.as_ptr();
    unsafe {
        let p = prepare_pattern(5, src_ptr_5);
        let p_bytes = p.to_le_bytes();
        assert_eq!(p_bytes, [0, 1, 2, 3, 4, 0, 1, 2]);
    }

    // Offset 6 case
    // Pattern source: 0, 1, 2, 3, 4, 5
    // Expected result (u64): 0, 1, 2, 3, 4, 5, 0, 1
    let buffer_6 = vec![0u8, 1, 2, 3, 4, 5];
    let src_ptr_6 = buffer_6.as_ptr();
    unsafe {
        let p = prepare_pattern(6, src_ptr_6);
        let p_bytes = p.to_le_bytes();
        assert_eq!(p_bytes, [0, 1, 2, 3, 4, 5, 0, 1]);
    }
}
