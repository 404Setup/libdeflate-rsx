use libdeflate::stream::DeflateEncoder;
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct SharedWriter {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut data = self.data.lock().unwrap();
        data.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_with_buffer_size() {
    let writer_data = Arc::new(Mutex::new(Vec::new()));
    let writer = SharedWriter {
        data: writer_data.clone(),
    };

    let buffer_size = 100;
    // Buffer size 100, we write 150 bytes.
    let mut encoder = DeflateEncoder::new(writer, 1).with_buffer_size(buffer_size);

    let data = vec![0u8; 150];
    encoder.write_all(&data).unwrap();

    // The buffer size (100) is exceeded by 150 bytes, so flush_buffer(false) should be called.
    // flush_buffer compresses and writes to the underlying writer.
    let compressed_len = writer_data.lock().unwrap().len();
    assert!(compressed_len > 0, "Encoder should have flushed when buffer limit was exceeded");

    // Finish the stream
    encoder.finish().unwrap();

    let final_len = writer_data.lock().unwrap().len();
    assert!(final_len > compressed_len, "Finish should write more data (footer/final block)");
}

#[test]
fn test_default_buffer_size() {
    let writer_data = Arc::new(Mutex::new(Vec::new()));
    let writer = SharedWriter {
        data: writer_data.clone(),
    };

    // Default buffer size is usually large (e.g. 1MB).
    let mut encoder = DeflateEncoder::new(writer, 1);

    let data = vec![0u8; 150];
    encoder.write_all(&data).unwrap();

    // Should not have flushed yet as 150 < default buffer size
    let compressed_len = writer_data.lock().unwrap().len();
    assert_eq!(compressed_len, 0, "Encoder should NOT have flushed with default buffer size");

    encoder.finish().unwrap();
    assert!(writer_data.lock().unwrap().len() > 0);
}
