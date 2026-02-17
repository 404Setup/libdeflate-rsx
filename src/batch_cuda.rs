#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

pub struct CudaBatchCompressor {
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    has_kernel: bool,
    #[allow(dead_code)]
    level: usize,
}

impl CudaBatchCompressor {
    pub fn new(level: usize) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        {
            let device = CudaDevice::new(0)?;

            // Load CUDA kernel
            let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/compress.ptx"));
            let ptx = Ptx::from_src(ptx_src);
            device.load_ptx(ptx, "compress", &["compress_kernel"])?;

            let has_kernel = true;
            Ok(Self {
                device,
                has_kernel,
                level,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA feature not enabled".into())
        }
    }

    pub fn compress_batch(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        {
            if !self.has_kernel {
                return Err("CUDA compression kernel not initialized".into());
            }

            if inputs.is_empty() {
                return Ok(Vec::new());
            }

            // 1. Calculate sizes and flatten inputs
            let total_input_size: usize = inputs.iter().map(|i| i.len()).sum();
            let mut flat_input = Vec::with_capacity(total_input_size);
            let mut input_offsets = Vec::with_capacity(inputs.len() + 1);

            let mut current_offset = 0;
            input_offsets.push(current_offset as u64);
            for input in inputs {
                flat_input.extend_from_slice(input);
                current_offset += input.len();
                input_offsets.push(current_offset as u64);
            }

            // 2. Calculate output offsets
            let mut output_offsets = Vec::with_capacity(inputs.len());
            let mut current_out_offset = 0;
            for input in inputs {
                output_offsets.push(current_out_offset as u64);
                let bound = crate::compress::Compressor::deflate_compress_bound(input.len());
                current_out_offset += bound;
            }
            let total_output_bound = current_out_offset;

            // 3. Transfer data to GPU
            let dev_input = self.device.htod_copy(flat_input)?;
            let dev_input_offsets = self.device.htod_copy(input_offsets)?;
            let dev_output_offsets = self.device.htod_copy(output_offsets.clone())?;

            // 4. Allocate output buffers on GPU
            let mut dev_output = unsafe { self.device.alloc::<u8>(total_output_bound)? };
            let mut dev_output_sizes = self.device.alloc_zeros::<u64>(inputs.len())?;

            // 5. Launch Kernel
            let launch_config = LaunchConfig {
                grid_dim: (inputs.len() as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            let kernel = self.device.get_func("compress", "compress_kernel")
                .ok_or("Kernel not found")?;

            unsafe { kernel.launch(
                launch_config,
                (
                    &dev_input,
                    &dev_input_offsets,
                    &mut dev_output,
                    &dev_output_offsets,
                    &mut dev_output_sizes,
                    inputs.len() as i32
                )
            ) }?;

            // 6. Retrieve results
            let output_sizes = self.device.dtoh_sync_copy(&dev_output_sizes)?;

            let mut results = Vec::with_capacity(inputs.len());
            for (i, &size) in output_sizes.iter().enumerate() {
                let offset = output_offsets[i] as usize;
                let size = size as usize;

                let slice = dev_output.slice(offset..offset+size);
                let host_data = self.device.dtoh_sync_copy(&slice)?;
                results.push(host_data);
            }

            Ok(results)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA feature not enabled".into())
        }
    }
}
