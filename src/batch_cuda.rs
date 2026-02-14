#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
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
            // TODO: Load CUDA kernel here
            let has_kernel = false;
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
            // Flattening inputs into a single buffer reduces memory transfer overhead.
            let total_input_size: usize = inputs.iter().map(|i| i.len()).sum();
            let mut flat_input = Vec::with_capacity(total_input_size);
            let mut offsets = Vec::with_capacity(inputs.len() + 1);

            let mut current_offset = 0;
            offsets.push(current_offset as u64);
            for input in inputs {
                flat_input.extend_from_slice(input);
                current_offset += input.len();
                offsets.push(current_offset as u64);
            }

            // 2. Check GPU memory availability
            // Ensure there is enough free memory on the device before allocation.
            // Estimate memory needed: input + offsets + output + output_sizes
            // Output bound: similar to input size (conservative estimate)
            let output_bound = crate::compress::Compressor::deflate_compress_bound(total_input_size);

            // Note: Explicit memory check removed as CudaDevice::mem_get_info is not directly available/verified.
            // Allocation will fail with OOM error if insufficient memory.

            // 3. Transfer data to GPU
            // Use htod_copy (Host to Device) to allocate and copy data.
            // Using pinned memory for host buffers (if supported/implemented) would be faster,
            // but standard Vec is used here for simplicity.
            // htod_copy consumes the vector.
            let _dev_input = self.device.htod_copy(flat_input)?;
            let _dev_offsets = self.device.htod_copy(offsets)?;

            // 4. Allocate output buffers on GPU
            // Allocate a single large output buffer to store compressed data for all streams.
            // Also allocate an array to store the size of each compressed stream.
            // cudarc's alloc is safe and handles cleanup on drop.
            // Using alloc instead of alloc_zeros for performance since kernel will overwrite.
            let mut _dev_output = unsafe { self.device.alloc::<u8>(output_bound)? };
            let mut _dev_out_sizes = self.device.alloc_zeros::<u64>(inputs.len())?;

            // 5. Launch Kernel (Placeholder)
            // Ideally, we would load a PTX module and launch a kernel here.
            // For now, return an error to trigger CPU fallback.

            Err("CUDA compression kernel not implemented yet".into())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA feature not enabled".into())
        }
    }
}
