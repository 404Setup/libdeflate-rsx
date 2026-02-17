extern "C" __global__ void compress_kernel(
    const unsigned char* __restrict__ inputs,
    const unsigned long long* __restrict__ input_offsets,
    unsigned char* __restrict__ outputs,
    const unsigned long long* __restrict__ output_offsets,
    unsigned long long* __restrict__ output_sizes,
    int num_inputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    unsigned long long start = input_offsets[idx];
    unsigned long long end = input_offsets[idx+1];
    unsigned long long len = end - start;

    // Output offset for this specific stream
    unsigned long long my_out_offset = output_offsets[idx];
    unsigned char* out_ptr = outputs + my_out_offset;

    const unsigned char* in_ptr = inputs + start;

    unsigned long long bytes_left = len;
    unsigned long long out_pos = 0;

    // If input is empty, emit an empty stored block
    if (len == 0) {
        out_ptr[out_pos++] = 1; // BFINAL=1, BTYPE=00
        out_ptr[out_pos++] = 0; // LEN=0 (low)
        out_ptr[out_pos++] = 0; // LEN=0 (high)
        out_ptr[out_pos++] = 0xFF; // NLEN=0xFFFF (low)
        out_ptr[out_pos++] = 0xFF; // NLEN=0xFFFF (high)
        output_sizes[idx] = out_pos;
        return;
    }

    // Process input in 65535 byte chunks (max stored block size)
    while (bytes_left > 0) {
        unsigned int chunk_len = (bytes_left > 65535) ? 65535 : (unsigned int)bytes_left;
        int last = (bytes_left == chunk_len) ? 1 : 0;

        // DEFLATE stored block header:
        // Bit 0: BFINAL (1 if last)
        // Bit 1-2: BTYPE (00 for stored)
        // Bits 3-7: Padding to byte boundary (0)
        out_ptr[out_pos++] = (unsigned char)(last);

        // LEN (2 bytes, little endian)
        out_ptr[out_pos++] = (unsigned char)(chunk_len & 0xFF);
        out_ptr[out_pos++] = (unsigned char)((chunk_len >> 8) & 0xFF);

        // NLEN (2 bytes, one's complement of LEN)
        unsigned int nlen = ~chunk_len;
        out_ptr[out_pos++] = (unsigned char)(nlen & 0xFF);
        out_ptr[out_pos++] = (unsigned char)((nlen >> 8) & 0xFF);

        // Data copy
        for (unsigned int i = 0; i < chunk_len; i++) {
            out_ptr[out_pos++] = in_ptr[i];
        }

        in_ptr += chunk_len;
        bytes_left -= chunk_len;
    }

    output_sizes[idx] = out_pos;
}
