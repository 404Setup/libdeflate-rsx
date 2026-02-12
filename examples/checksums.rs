use libdeflate::{adler32, crc32};

fn main() {
    let data = b"The quick brown fox jumps over the lazy dog";

    let adler = adler32(1, data);
    println!("Adler-32: {:08X}", adler);

    let crc = crc32(0, data);
    println!("CRC-32:   {:08X}", crc);

    let data_part1 = b"The quick ";
    let data_part2 = b"brown fox";

    let crc_running = crc32(0, data_part1);
    let crc_final = crc32(crc_running, data_part2);

    println!("Running CRC-32 (partial): {:08X}", crc_final);
}
