"""Benchmark: CUDA EP vs TensorRT EP for SCRFD"""
import time, numpy as np, onnxruntime as ort, pipeline_v6

print('Providers:', ort.get_available_providers())

# Dynamic lookup - avoid hardcoding class name
Det = getattr(pipeline_v6, [c for c in dir(pipeline_v6) if 'Fast' in c][0])

# CUDA EP baseline
det_cuda = Det('/mnt/cfs-baidu/public/huitao.wang/work/kk/scrfd_500m.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('CUDA EP:', det_cuda.sess.get_providers())

# TensorRT EP (default, auto FP16)
det_trt = Det('/mnt/cfs-baidu/public/huitao.wang/work/kk/scrfd_500m.onnx')
print('TRT EP: ', det_trt.sess.get_providers())

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
canvas, scale = det_trt.letterbox(img)

N = 100
print('TRT engine building (first run)...', end=' ', flush=True)
for _ in range(5):
    det_trt.detect_single(canvas)
print('done')

t0 = time.perf_counter()
for _ in range(N):
    det_trt.detect_single(canvas)
print('TRT EP full detect_single: %.2fms' % ((time.perf_counter()-t0)/N*1000))

t0 = time.perf_counter()
for _ in range(N):
    det_cuda.detect_single(canvas)
print('CUDA EP full detect_single: %.2fms' % ((time.perf_counter()-t0)/N*1000))
