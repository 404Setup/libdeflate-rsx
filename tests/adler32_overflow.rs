use libdeflate;
use libdeflater;

#[test]
fn test_adler32_overflow_check() {
    let size = 100000;
    let data = vec![0xFF; size];

    // Reference implementation (C binding)
    let expected = libdeflater::adler32(&data);

    // Our implementation
    let actual = libdeflate::adler32(1, &data);

    assert_eq!(
        actual, expected,
        "Adler32 mismatch for size {} with 0xFF",
        size
    );
}

#[test]
fn test_adler32_overflow_check_large() {
    let size = 1_000_000;
    let data = vec![0xFF; size];

    // Reference implementation (C binding)
    let expected = libdeflater::adler32(&data);

    // Our implementation
    let actual = libdeflate::adler32(1, &data);

    assert_eq!(
        actual, expected,
        "Adler32 mismatch for size {} with 0xFF",
        size
    );
}
