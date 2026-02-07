use crate::compress::{CompressResult, Compressor};
use crate::decompress::{DecompressResult, Decompressor, DecompressorState};
use rayon::prelude::*;
use std::cmp::min;
use std::io::{self, Read, Write};

pub struct DeflateEncoder<W: Write + Send> {
    writer: Option<W>,
    buffer: Vec<u8>,
    buffer_size: usize,
    level: usize,
}

impl<W: Write + Send> DeflateEncoder<W> {
    pub fn new(writer: W, level: usize) -> Self {
        Self {
            writer: Some(writer),
            buffer: Vec::with_capacity(1024 * 1024),
            buffer_size: 1024 * 1024,
            level,
        }
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self.buffer.reserve(size);
        self
    }

    fn flush_buffer(&mut self, final_block: bool) -> io::Result<()> {
        if self.buffer.is_empty() && !final_block {
            return Ok(());
        }

        let chunk_size = 64 * 1024;
        if self.buffer.len() > chunk_size {
            let chunks: Vec<&[u8]> = self.buffer.chunks(chunk_size).collect();
            let num_chunks = chunks.len();
            
            let compressed_chunks: Vec<io::Result<Vec<u8>>> = chunks
                .par_iter()
                .enumerate()
                .map_init(
                    || Compressor::new(self.level),
                    |compressor, (i, &chunk)| {
                        let bound = Compressor::deflate_compress_bound(chunk.len());
                        let mut output = vec![0u8; bound];
                        let mode = if final_block && i == num_chunks - 1 {
                             crate::compress::FlushMode::Finish
                        } else {
                             crate::compress::FlushMode::Sync
                        };
                        let (res, size, _) = compressor.compress(chunk, &mut output, mode);
                        if res == CompressResult::Success {
                            output.truncate(size);
                            Ok(output)
                        } else {
                            Err(io::Error::new(io::ErrorKind::Other, "Compression failed"))
                        }
                    },
                )
                .collect();

            if let Some(writer) = &mut self.writer {
                for chunk_res in compressed_chunks {
                    let chunk_data = chunk_res?;
                    writer.write_all(&chunk_data)?;
                }
            }
        } else {
            let mut compressor = Compressor::new(self.level);
            let bound = Compressor::deflate_compress_bound(self.buffer.len());
            let mut output = vec![0u8; bound];
            let mode = if final_block {
                 crate::compress::FlushMode::Finish
            } else {
                 crate::compress::FlushMode::Sync
            };
            let (res, size, _) = compressor.compress(&self.buffer, &mut output, mode);
            if res == CompressResult::Success {
                if let Some(writer) = &mut self.writer {
                    writer.write_all(&output[..size])?;
                }
            } else {
                return Err(io::Error::new(io::ErrorKind::Other, "Compression failed"));
            }
        }

        self.buffer.clear();
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.flush_buffer(true)?;
        Ok(self.writer.take().unwrap())
    }
}

impl<W: Write + Send> Write for DeflateEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        if self.buffer.len() >= self.buffer_size {
            self.flush_buffer(false)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer(false)?;
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }
}

impl<W: Write + Send> Drop for DeflateEncoder<W> {
    fn drop(&mut self) {
        if self.writer.is_some() {
            let _ = self.flush_buffer(true);
        }
    }
}

pub struct DeflateDecoder<R: Read> {
    inner: R,
    decompressor: Decompressor,
    input_buffer: Vec<u8>,
    input_pos: usize,
    input_cap: usize,
    window: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
    done: bool,
}

impl<R: Read> DeflateDecoder<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            decompressor: Decompressor::new(),
            input_buffer: vec![0; 32 * 1024],
            input_pos: 0,
            input_cap: 0,
            window: vec![0; 64 * 1024],
            read_pos: 0,
            write_pos: 0,
            done: false,
        }
    }
}

impl<R: Read> Read for DeflateDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_pos < self.write_pos {
             let count = min(buf.len(), self.write_pos - self.read_pos);
             buf[..count].copy_from_slice(&self.window[self.read_pos..self.read_pos + count]);
             self.read_pos += count;
             return Ok(count);
        }
        
        if self.done {
            return Ok(0);
        }

        loop {
             if self.write_pos >= 64 * 1024 {
                 if self.read_pos >= 32 * 1024 {
                     self.window.copy_within(self.read_pos - 32 * 1024..self.write_pos, 32 * 1024 - (self.read_pos - 32 * 1024));

                     let amount_to_keep = 32 * 1024;
                     let shift = self.write_pos - amount_to_keep;
                     self.window.copy_within(shift..self.write_pos, 0);
                     self.write_pos = amount_to_keep;
                     self.read_pos -= shift;
                 }
             }

             let mut output_full = false;
             if self.input_pos < self.input_cap {
                 let input = &self.input_buffer[self.input_pos..self.input_cap];
                 let (res, in_consumed) = {
                     let (res, inc, _outc) = self.decompressor.decompress_streaming(input, &mut self.window, &mut self.write_pos);
                     (res, inc)
                 };
                 
                 self.input_pos += in_consumed;

                 if let DecompressorState::Done = self.decompressor.state {
                      self.done = true;
                      if self.read_pos < self.write_pos {
                           let count = min(buf.len(), self.write_pos - self.read_pos);
                           buf[..count].copy_from_slice(&self.window[self.read_pos..self.read_pos + count]);
                           self.read_pos += count;
                           return Ok(count);
                      }
                      return Ok(0);
                 }
                 
                 if self.read_pos < self.write_pos {
                      let count = min(buf.len(), self.write_pos - self.read_pos);
                      buf[..count].copy_from_slice(&self.window[self.read_pos..self.read_pos + count]);
                      self.read_pos += count;
                      return Ok(count);
                 }

                 match res {
                     DecompressResult::ShortInput => {
                     }
                     DecompressResult::InsufficientSpace => {
                         output_full = true;
                     }
                     DecompressResult::BadData => {
                         return Err(io::Error::new(io::ErrorKind::InvalidData, "deflate decompression failed"));
                     }
                     _ => {}
                 }
             }

             if !output_full {
                 if self.input_pos > 0 {
                  self.input_buffer.copy_within(self.input_pos..self.input_cap, 0);
                  self.input_cap -= self.input_pos;
                  self.input_pos = 0;
             }
             if self.input_cap == self.input_buffer.len() {
                  if self.input_buffer.len() < 1024 * 1024 {
                      self.input_buffer.resize(self.input_buffer.len() * 2, 0);
                  } else {
                      return Err(io::Error::new(io::ErrorKind::Other, "input buffer full"));
                  }
             }
             
             let n = self.inner.read(&mut self.input_buffer[self.input_cap..])?;
             if n == 0 {
                 if self.done {
                     return Ok(0);
                 }
                 if self.input_pos < self.input_cap {
                     return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
                 }
                 if !self.done && self.decompressor.state != DecompressorState::Start {
                      return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF"));
                 }
                 return Ok(0);
             }
                 self.input_cap += n;
             }
        }
    }
}
