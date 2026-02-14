import os
import random

def get_random_data(size):
    base_pattern = bytearray()
    for i in range(100):
        val = ((i * 1234567) ^ (i * 987654)) & 0xFF
        base_pattern.append(val)
    return base_pattern

def generate_file(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = get_random_data(100)
    
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

    generate_offset8("bench_data/data_offset8.bin", 1024 * 1024)
    generate_offset3("bench_data/data_offset3.bin", 1024 * 1024)
    generate_offset5("bench_data/data_offset5.bin", 1024 * 1024)
    generate_offset1("bench_data/data_offset1.bin", 1024 * 1024)
    generate_offset2("bench_data/data_offset2.bin", 1024 * 1024)
    generate_offset4("bench_data/data_offset4.bin", 1024 * 1024)
    generate_offset9("bench_data/data_offset9.bin", 1024 * 1024)
    generate_offset10("bench_data/data_offset10.bin", 1024 * 1024)
    generate_offset11("bench_data/data_offset11.bin", 1024 * 1024)
    generate_offset12("bench_data/data_offset12.bin", 1024 * 1024)
    generate_offset15("bench_data/data_offset15.bin", 1024 * 1024)

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

def generate_offset1(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"1"
    write_pattern(filename, target_size, pattern)

def generate_offset12(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"123456789012"
    write_pattern(filename, target_size, pattern)

def generate_offset2(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12"
    write_pattern(filename, target_size, pattern)

def generate_offset4(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"1234"
    write_pattern(filename, target_size, pattern)

def generate_offset3(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"123"
    write_pattern(filename, target_size, pattern)

def generate_offset8(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12345678"
    write_pattern(filename, target_size, pattern)

def generate_offset5(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12345"
    write_pattern(filename, target_size, pattern)

def generate_offset9(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"123456789"
    write_pattern(filename, target_size, pattern)

def generate_offset10(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"1234567890"
    write_pattern(filename, target_size, pattern)

def generate_offset11(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12345678901"
    write_pattern(filename, target_size, pattern)

def generate_offset15(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"123456789012345"
    write_pattern(filename, target_size, pattern)

def write_pattern(filename, target_size, pattern):
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

if __name__ == "__main__":
    main()
