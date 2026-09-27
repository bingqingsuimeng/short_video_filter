// Top-left letterbox + (v-127.5)/128 + BGR->RGB, fused in one kernel.
// src: (B,H,W,3) BGR uint8 contiguous (HWC)   dst: (B,3,640,640) float32 CHW
// Geometry matches cv2.resize bilinear: sx = (ox+0.5)*W/nw - 0.5
extern "C"
__global__ void letterbox_bgr2rgb(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int H, int W, int nw, int nh,
    float rx, float ry)
{
    const int S = 640;
    const float PAD = -0.99609375f;  // (0 - 127.5) / 128

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (ox >= S || oy >= S) return;

    float r, g, bl;
    if (ox >= nw || oy >= nh) {
        r = g = bl = PAD;
    } else {
        float sx = fminf(fmaxf((ox + 0.5f) * rx - 0.5f, 0.0f), (float)W - 1.0f);
        float sy = fminf(fmaxf((oy + 0.5f) * ry - 0.5f, 0.0f), (float)H - 1.0f);
        int x0 = (int)sx, y0 = (int)sy;
        int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
        float fx = sx - (float)x0, fy = sy - (float)y0;
        const unsigned char* s = src + (long)b * H * W * 3;   // HWC batch stride
        #define C3(px, py, c) s[(((long)(py) * W + (px)) * 3 + (c))]
        float v[3];
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            float c00 = (float)C3(x0, y0, c);
            float c10 = (float)C3(x1, y0, c);
            float c01 = (float)C3(x0, y1, c);
            float c11 = (float)C3(x1, y1, c);
            float top = c00 + (c10 - c00) * fx;
            float bot = c01 + (c11 - c01) * fx;
            v[c] = (top + (bot - top) * fy) / 128.0f - 0.99609375f;
        }
        #undef C3
        r = v[2]; g = v[1]; bl = v[0];   // BGR -> RGB
    }
    long base = ((long)b * 3 * S + oy) * S + ox;   // channel 0 offset
    dst[base]             = r;    // RGB ch0 = source ch2 (R)
    dst[base + 1L * S * S] = g;
    dst[base + 2L * S * S] = bl;  // RGB ch2 = source ch0 (B)
}
