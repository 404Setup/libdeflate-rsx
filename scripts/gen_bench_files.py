import os
import random

def get_random_data(size):
    base_pattern = bytearray()
    for i in range(100):
        val = ((i * 1234567) ^ (i * 987654)) & 0xFF
        base_pattern.append(val)
    return base_pattern

def generate_file(filename, target_size):
    pattern = get_random_data(100)
    write_pattern(filename, target_size, pattern)

def write_pattern(filename, target_size, pattern):
    print(f"Generating {filename} ({target_size} bytes)...")
    with open(filename, 'wb') as f:
        bytes_written = 0
        chunk_size = 1024 * 1024
        large_chunk = pattern * (chunk_size // len(pattern) + 1)
        large_chunk = large_chunk[:chunk_size]

        while bytes_written < target_size:
            remaining = target_size - bytes_written
            write_amt = min(remaining, len(large_chunk))
            f.write(large_chunk[:write_amt])
            bytes_written += write_amt

def main():
    if not os.path.exists("bench_data"):
        os.makedirs("bench_data")
    
    sizes = {
        "XXS": 1024,             # 1KB
        "XS": 4096,              # 4KB
        "S": 65536,              # 64KB
        "M": 1 * 1024 * 1024,    # 1MB
        "L": 16 * 1024 * 1024,   # 16MB
        "XL": 64 * 1024 * 1024   # 64MB
    }
    
    for name, size in sizes.items():
        generate_file(f"bench_data/data_{name}.bin", size)

    patterns = {
        1: b"1",
        2: b"12",
        3: b"123",
        4: b"1234",
        5: b"12345",
        7: b"1234567",
        8: b"12345678",
        9: b"123456789",
        10: b"1234567890",
        11: b"12345678901",
        12: b"123456789012",
        13: b"1234567890123",
        14: b"12345678901234",
        15: b"123456789012345",
        16: b"1234567890123456",
        17: b"12345678901234567",
        18: b"123456789012345678",
        19: b"1234567890123456789",
        20: b"ABCDEFGHIJKLMNOPQRST",
        21: b"ABCDEFGHIJKLMNOPQRSTU",
        22: b"ABCDEFGHIJKLMNOPQRSTUV",
        23: b"ABCDEFGHIJKLMNOPQRSTUVW",
        24: b"ABCDEFGHIJKLMNOPQRSTUVWX",
        25: b"ABCDEFGHIJKLMNOPQRSTUVWXY",
        26: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        27: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0",
        28: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ01",
        29: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012",
        30: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
        31: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
        32: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    }

    for offset, pattern in patterns.items():
        write_pattern(f"bench_data/data_offset{offset}.bin", 1024 * 1024, pattern)

    # Small match variants
    # For offset N, we want matches of length N (or close to N).
    # Pattern P (len N) + Pattern P (len N) + X.
    # Match len N, offset N.
    generate_offset_small("bench_data/data_offset3_small.bin", 1024 * 1024, b"123")
    generate_offset_small("bench_data/data_offset9_small.bin", 1024 * 1024, b"123456789")

def generate_offset_small(filename, target_size, pattern):
    print(f"Generating {filename} ({target_size} bytes)...")

    with open(filename, 'wb') as f:
        bytes_written = 0
        while bytes_written < target_size:
            f.write(pattern)
            f.write(pattern)
            # Break match with random literal
            f.write(bytes([random.randint(0, 255)]))
            bytes_written += len(pattern) * 2 + 1

if __name__ == "__main__":
    main()
