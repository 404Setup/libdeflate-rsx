import os

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

def generate_offset1(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"1"

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

def generate_offset2(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12"

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

def generate_offset4(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"1234"

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

def generate_offset3(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"123"

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

def generate_offset8(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12345678"

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

def generate_offset5(filename, target_size):
    print(f"Generating {filename} ({target_size} bytes)...")
    pattern = b"12345"

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
