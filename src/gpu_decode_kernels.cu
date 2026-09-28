// gpu_decode_kernels.cu -- CUDA kernels for the VRAM-direct zero-copy decode
// arm (--decode pynvvc-gpu). Fully independent of src/preproc_kernel.cu (that
// file is NOT modified); the existing letterbox_bgr2rgb kernel is still reused
// for the 768->640 step. This file only adds the three glue stages between an
// NVDEC device frame and the existing pipeline:
//
//   area_fast_u8        INTER_AREA integer-ratio downsample (e.g. 2160x3840
//                       -> 432x768, 5x). Bit-exact replica of OpenCV
//                       resizeAreaFast_<uchar,int>:
//                       sum (exact int) * (1.f/area) -> RNE -> clamp.
//   area_generic_u8     INTER_AREA non-integer ratio (e.g. 1900x3378 ->
//                       432x768), table-driven. Bit-exact replica of
//                       resizeArea_ + computeResizeAreaTab (tables built on
//                       host in double precision, uploaded). Float add/mul
//                       order matches cv2 exactly; no FMA anywhere
//                       (explicit __fmul_rn/__fadd_rn).
//   head_letterbox_u8   head-gate 640 center letterbox (pad=114), fixed-point
//                       INTER_LINEAR. Bit-exact replica of resizeGeneric_<
//                       HResizeLinear<uchar,int,short,2048>,
//                       VResizeLinear<...,FixedPtCast<int,uchar,22>>>. Writes
//                       the fp32 blob (value/255, IEEE division) directly.
//   rgb2bgr_inplace_u8  in-place RGB->BGR channel swap (before D2H of face
//                       frames).
//   copy_u8             plain D2D copy (fallback for memcpy_dtod_async).
//
// Build (RTX 3060 / sm_86; -fmad=false keeps float ops unfused, like MSVC):
//   nvcc -cubin -arch=sm_86 -fmad=false gpu_decode_kernels.cu -o gpu_decode_kernels.cubin
//
// Bit-exactness spec source: _test/post3_e2_resample_ref.py (Python reference
// vs cv2: byte-identical on 32 real GT frames across the 4 benchmark videos;
// see data/e2e_benchmark_2026-09-28.md section 10).
//
// NOTE: ASCII-only comments -- MSVC (cp936) mangles UTF-8 comments and may
// swallow the following newline, breaking the code (observed).

// RNE + saturate: equivalent to OpenCV saturate_cast<uchar>(float) = cvRound
// followed by clamp.
__device__ __forceinline__ unsigned char sat_u8_rn(float v) {
    int iv = __float2int_rn(v);
    return (unsigned char)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
}

// ---------------- 1) INTER_AREA fast (integer ratio) ----------------
// grid: (ceil(nw/16), ceil(nh/16), batch)  block: (16,16,1)
// Host guarantees W == nw*kx && H == nh*ky (exact ratio => cv2 always takes
// the fast path, no border region exists).
// oswap: 0 = straight channel order; 1 = swap channels on output
//        (device frame is RGB; output BGR matches the host INTER_AREA result
//        byte-for-byte, so the original engine + original preproc kernel stay
//        bit-identical to the host arm).
extern "C" __global__ void area_fast_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int H, int W, int nh, int nw, int kx, int ky, int oswap)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= nw || dy >= nh) return;
    long sb = (long)H * W * 3, db = (long)nh * nw * 3;
    const unsigned char* S = src + sb * blockIdx.z;
    unsigned char* D = dst + db * blockIdx.z;
    int area = kx * ky;
    float scale = 1.f / (float)area;          // cv2: float scale = 1.f/(area)
    int sy0 = dy * ky;
    int sum0 = 0, sum1 = 0, sum2 = 0;         // exact int accumulation
    for (int j = 0; j < ky; ++j) {
        const unsigned char* row = S + (long)(sy0 + j) * W * 3;
        int sx = dx * kx * 3;
        for (int i = 0; i < kx; ++i) {
            sum0 += row[sx + 0];
            sum1 += row[sx + 1];
            sum2 += row[sx + 2];
            sx += 3;
        }
    }
    long o = (long)dy * nw * 3 + dx * 3;
    if (oswap) {
        D[o + 0] = sat_u8_rn(__fmul_rn((float)sum2, scale));
        D[o + 1] = sat_u8_rn(__fmul_rn((float)sum1, scale));
        D[o + 2] = sat_u8_rn(__fmul_rn((float)sum0, scale));
    } else {
        D[o + 0] = sat_u8_rn(__fmul_rn((float)sum0, scale));
        D[o + 1] = sat_u8_rn(__fmul_rn((float)sum1, scale));
        D[o + 2] = sat_u8_rn(__fmul_rn((float)sum2, scale));
    }
}

// ---------------- 2) INTER_AREA generic (table driven) ----------------
// Tables are built on host in double precision (src/gpu_decode.py _area_tab):
//   x_ofs[nw+1]       per-dx entry range [x_ofs[dx], x_ofs[dx+1])
//   x_si[], x_alpha[] (source col si, weight f32), si ascending within each dx
//   y_ofs[nh+1], y_si[], y_alpha[]  same for rows
extern "C" __global__ void area_generic_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int H, int W, int nh, int nw, int oswap,
    const int* __restrict__ x_ofs,
    const int* __restrict__ x_si,
    const float* __restrict__ x_alpha,
    const int* __restrict__ y_ofs,
    const int* __restrict__ y_si,
    const float* __restrict__ y_alpha)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= nw || dy >= nh) return;
    long sb = (long)H * W * 3, db = (long)nh * nw * 3;
    const unsigned char* S = src + sb * blockIdx.z;
    unsigned char* D = dst + db * blockIdx.z;

    // ResizeArea_Invoker: per (sy, beta) row: buf = sum_x alpha*S (ascending,
    // no FMA); first row: sum = beta*buf (mul), then sum += beta*buf (mul+add).
    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;
    bool first = true;
    int j0 = y_ofs[dy], j1 = y_ofs[dy + 1];
    for (int j = j0; j < j1; ++j) {
        const unsigned char* row = S + (long)y_si[j] * W * 3;
        float beta = y_alpha[j];
        float b0 = 0.f, b1 = 0.f, b2 = 0.f;
        int k0 = x_ofs[dx], k1 = x_ofs[dx + 1];
        for (int k = k0; k < k1; ++k) {
            float a = x_alpha[k];
            const unsigned char* p = row + (long)x_si[k] * 3;
            b0 = __fadd_rn(b0, __fmul_rn((float)p[0], a));
            b1 = __fadd_rn(b1, __fmul_rn((float)p[1], a));
            b2 = __fadd_rn(b2, __fmul_rn((float)p[2], a));
        }
        if (first) {
            sum0 = __fmul_rn(beta, b0);
            sum1 = __fmul_rn(beta, b1);
            sum2 = __fmul_rn(beta, b2);
            first = false;
        } else {
            sum0 = __fadd_rn(sum0, __fmul_rn(beta, b0));
            sum1 = __fadd_rn(sum1, __fmul_rn(beta, b1));
            sum2 = __fadd_rn(sum2, __fmul_rn(beta, b2));
        }
    }
    long o = (long)dy * nw * 3 + dx * 3;
    if (oswap) {
        D[o + 0] = sat_u8_rn(sum2);
        D[o + 1] = sat_u8_rn(sum1);
        D[o + 2] = sat_u8_rn(sum0);
    } else {
        D[o + 0] = sat_u8_rn(sum0);
        D[o + 1] = sat_u8_rn(sum1);
        D[o + 2] = sat_u8_rn(sum2);
    }
}

// ---------------- 3) head-gate 640 letterbox (fixed-point INTER_LINEAR) ----
// grid: (640/16, 640/16, batch)  block: (16,16,1)
// Input: pool frame (H,W,3); bswap=0: RGB, 1: BGR (i.e. after the in-place
//        swap -- equals the host cvtColor'd channel order).
// Output: fp32 blob CHW (batch,3,S,S), value = canvas_u8 / 255 (IEEE divide).
// Host guarantees sx+1 <= W-1 and sy+1 <= H-1 (downscale geometry, no border).
extern "C" __global__ void head_letterbox_u8(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int H, int W, int S, int nh, int nw, int pt, int pl,
    int bswap)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= S || oy >= S) return;
    long sb = (long)H * W * 3;
    const unsigned char* Sf = src + sb * blockIdx.z;

    double scale_x = (double)W / (double)nw;   // = 1./inv_scale_x, double
    double scale_y = (double)H / (double)nh;
    unsigned char v0, v1, v2;
    if (ox < pl || ox >= pl + nw || oy < pt || oy >= pt + nh) {
        v0 = v1 = v2 = 114;                    // pad value (channel-equal)
    } else {
        int dx = ox - pl, dy = oy - pt;
        // alpha build (same formula as resizeGeneric_):
        // fx = (float)((dx+0.5)*scale_x - 0.5)
        float fx = __double2float_rn(((dx + 0.5) * scale_x - 0.5));
        int sx = (int)floorf(fx);
        fx = __fsub_rn(fx, (float)sx);
        int a0 = __float2int_rn(__fmul_rn(__fsub_rn(1.0f, fx), 2048.0f));
        int a1 = __float2int_rn(__fmul_rn(fx, 2048.0f));
        float fy = __double2float_rn(((dy + 0.5) * scale_y - 0.5));
        int sy = (int)floorf(fy);
        fy = __fsub_rn(fy, (float)sy);
        int b0 = __float2int_rn(__fmul_rn(__fsub_rn(1.0f, fy), 2048.0f));
        int b1 = __float2int_rn(__fmul_rn(fy, 2048.0f));
        // horizontal (exact int): H = S[sx]*a0 + S[sx+1]*a1, per source row
        const unsigned char* r0 = Sf + (long)sy * W * 3;
        const unsigned char* r1 = Sf + (long)(sy + 1) * W * 3;
        int s3 = sx * 3;
        v0 = v1 = v2 = 0;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            int cin = bswap ? (2 - c) : c;     // out channel c <- in channel cin
            int h0 = (int)r0[s3 + cin] * a0 + (int)r0[s3 + 3 + cin] * a1;
            int h1 = (int)r1[s3 + cin] * a0 + (int)r1[s3 + 3 + cin] * a1;
            int res = (((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2;
            unsigned char vv = (unsigned char)(res < 0 ? 0 : (res > 255 ? 255 : res));
            if (c == 0) v0 = vv; else if (c == 1) v1 = vv; else v2 = vv;
        }
    }
    long base = (long)blockIdx.z * 3 * S * S;
    dst[base + 0 * (long)S * S + oy * S + ox] = __fdiv_rn((float)v0, 255.0f);
    dst[base + 1 * (long)S * S + oy * S + ox] = __fdiv_rn((float)v1, 255.0f);
    dst[base + 2 * (long)S * S + oy * S + ox] = __fdiv_rn((float)v2, 255.0f);
}

// ---------------- 4) in-place RGB->BGR (before D2H of face frames) ---------
// grid: (nblocks,)  block: (256,)  n = pixel count within one frame
extern "C" __global__ void rgb2bgr_inplace_u8(unsigned char* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x) {
        unsigned char t = p[i * 3];
        p[i * 3] = p[i * 3 + 2];
        p[i * 3 + 2] = t;
    }
}

// ---------------- 5) plain D2D copy (memcpy_dtod_async fallback) -----------
extern "C" __global__ void copy_u8(const unsigned char* src,
                                   unsigned char* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}
