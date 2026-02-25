use libdeflate::stream::{DeflateDecoder, DeflateEncoder};
use std::io::{Cursor, Read, Write};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct FlushTrackingWriter {
    data: Arc<Mutex<Vec<u8>>>,
    flush_count: Arc<Mutex<usize>>,
}

impl Write for FlushTrackingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.data.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        *self.flush_count.lock().unwrap() += 1;
        Ok(())
    }
}

struct ErrorFlushWriter;

impl Write for ErrorFlushWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "flush error",
        ))
    }
}

#[test]
fn test_stream_round_trip() {
    let mut data = Vec::with_capacity(10000);
    for i in 0..10000 {
        data.push((i % 256) as u8);
    }

    let mut encoder = DeflateEncoder::new(Vec::new(), 6);
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder = DeflateDecoder::new(Cursor::new(compressed));
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_stream_small_chunks() {
    let mut data = Vec::with_capacity(10000);
    for i in 0..10000 {
        data.push((i % 256) as u8);
    }

    let mut encoder = DeflateEncoder::new(Vec::new(), 6);
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder = DeflateDecoder::new(Cursor::new(compressed));
    let mut decompressed = Vec::new();
    let mut buf = [0u8; 10];
    loop {
        let n = decoder.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buf[..n]);
    }

    assert_eq!(data, decompressed);
}

#[test]
fn test_encoder_flush() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let flush_count = Arc::new(Mutex::new(0));
    let writer = FlushTrackingWriter {
        data: data.clone(),
        flush_count: flush_count.clone(),
    };

    let mut encoder = DeflateEncoder::new(writer, 6);
    encoder.write_all(b"Hello World").unwrap();
    encoder.flush().unwrap();

    // Verify data was written (compressed)
    assert!(!data.lock().unwrap().is_empty());

    // Verify flush was called on the underlying writer
    assert_eq!(*flush_count.lock().unwrap(), 1);
}

#[test]
fn test_encoder_flush_error() {
    let writer = ErrorFlushWriter;
    let mut encoder = DeflateEncoder::new(writer, 6);
    encoder.write_all(b"Hello World").unwrap();

    // flush() should fail because the underlying writer returns an error
    assert!(encoder.flush().is_err());
}
