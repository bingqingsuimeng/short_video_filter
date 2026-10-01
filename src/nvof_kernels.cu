// nvof_kernels.cu -- CUDA kernels for the NVOF2 (OFA hardware optical flow)
// "global-motion sub-interval interval sampling" rule (--dedup-of, opt-in,
// default OFF). Loaded only when --dedup-of is enabled; the default path never
// touches this module, so behavior stays bit-identical to HEAD without the
// flag. Companion file of src/nvof.py (which owns the NVOF session and the
// cudart stream these kernels are launched on, primary-context driven).
//
//   rgb2gray_u8       RGB interleaved decode-pool frame (H,W,3) -> GRAY8
//                     written straight into an NVOF GRAYSCALE8 input buffer
//                     (row stride dst_stride). Fixed-point weights replicate
//                     cv2.cvtColor(x, COLOR_RGB2GRAY):
//                     (R*4899 + G*9617 + B*1868 + 8192) >> 14.
//   of_sig_reduce_u8  NVOF grid flow vectors (int16 S10.5, full-res px, gh
//                     rows x gw cells, NVOF row stride stride_bytes) -> 7
//                     double partial sums accumulated per thread
//                     in registers, block-reduced through shared memory, then
//                     atomicAdd'ed into out[7] (host zeroes it first):
//                       out[0] n        (all grid cells, incl. "bad" cells)
//                       out[1] sum_mag  (mag over all cells; bad cells = 0)
//                       out[2] cnt_m    (cells with mag >= 0.3 px@256)
//                       out[3] sum_m    (mag over m)
//                       out[4] sumsq_m  (mag*mag over m, float-product then
//                                        widened, mirroring numpy f32 std)
//                       out[5] sum_fx_m (fx over m)
//                       out[6] sum_fy_m (fy over m)
//                     Host turns these into fm/mr/dc/mcv with the exact
//                     post7/post8 formulas (src/nvof.signal_from_sums).
//                     Per frame only 56 B cross the PCIe bus.
//
// Build (RTX 3060 / sm_86; -fmad=false keeps float ops unfused):
//   nvcc -cubin -arch=sm_86 -fmad=false nvof_kernels.cu -o nvof_kernels.cubin
//
// NOTE: ASCII-only comments -- MSVC (cp936) mangles UTF-8 comments and may
// swallow the following newline, breaking the code (observed).

// grid: (ceil(W/16), ceil(H/16))  block: (16,16,1)
extern "C" __global__ void rgb2gray_u8(
    const unsigned char* __restrict__ src,   // RGB interleaved, H*W*3
    unsigned char* __restrict__ dst,         // NVOF GRAY8 input buffer
    int H, int W, int dst_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    long si = (long)y * W * 3 + x * 3;
    // cv2 RGB2GRAY fixed point: R2Y=4899, G2Y=9617, B2Y=1868, bias 1<<13,
    // shift 14 (saturate not needed: max (255*16384)>>14 = 255).
    int g = (src[si + 0] * 4899 + src[si + 1] * 9617 + src[si + 2] * 1868
             + 8192) >> 14;
    dst[(long)y * dst_stride + x] = (unsigned char)g;
}

// grid: (ceil(gw/16), ceil(gh/16))  block: (16,16,1)
extern "C" __global__ void of_sig_reduce_u8(
    const short* __restrict__ flow,          // gh rows, S10.5 full-res px
    int gw, int gh, int stride_bytes,        // NVOF output row stride (bytes)
    float scale,                             // 256 / max(H,W) (post8 same)
    double* __restrict__ out)                // out[7], zeroed by host
{
    __shared__ double sh[7][256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
    if (gx < gw && gy < gh) {
        // row stride is the NVOF buffer stride (e.g. 1536 B for gw=270),
        // NOT gw*4 -- compact addressing would read past the buffer.
        const short* row = flow + (((long)stride_bytes >> 1) * gy);
        long i = (long)gx * 2;
        float fx = (float)row[i] / 32.0f;        // S10.5 -> full-res px
        float fy = (float)row[i + 1] / 32.0f;
        // bad-cell hygiene (post8 same rule): |v|>512 px (half of the S10.5
        // range) -> invalid cell, contributes n (denominator) but 0 magnitude.
        if (fabsf(fx) > 512.0f || fabsf(fy) > 512.0f) { fx = 0.f; fy = 0.f; }
        fx = __fmul_rn(fx, scale);               // -> 256-thumb px
        fy = __fmul_rn(fy, scale);
        float mag = hypotf(fx, fy);
        s0 = 1.0;
        s1 = (double)mag;
        if (mag >= 0.3f) {
            s2 = 1.0;
            s3 = (double)mag;
            s4 = (double)(mag * mag);
            s5 = (double)fx;
            s6 = (double)fy;
        }
    }
    sh[0][tid] = s0; sh[1][tid] = s1; sh[2][tid] = s2; sh[3][tid] = s3;
    sh[4][tid] = s4; sh[5][tid] = s5; sh[6][tid] = s6;
    __syncthreads();
    #pragma unroll
    for (int c = 0; c < 7; ++c) {
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) sh[c][tid] += sh[c][tid + st];
            __syncthreads();
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int c = 0; c < 7; ++c) atomicAdd(&out[c], sh[c][0]);
    }
}
