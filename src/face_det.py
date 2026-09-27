# -*- coding: utf-8 -*-
"""
SCRFD-500m face detector on TensorRT (batched) + 5-keypoint pose estimation.

Features
--------
- Batched GPU inference (feeds up to MAX_BATCH images per forward pass)
- Letterbox preprocessing to a fixed 640x640 input
- Returns per-image: list of (box, score, kps) where kps is 5x2
- estimate_pose() -> (yaw, pitch, roll) in degrees via cv2.solvePnP

Requires the engine produced by build_engine.py.
"""
import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (initializes CUDA + context)
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray


INPUT_SIZE = 640
# Upper bound for the raw-frame staging buffers. Raw frames are uploaded to the
# GPU verbatim for the letterbox kernel, so the buffers must cover the largest
# frame we will ever see. 4K (3840x2160) is the practical ceiling for Douyin /
# social videos; the buffers are pre-allocated once at this size and reused
# across videos of any (smaller) resolution, so switching video resolution
# never triggers a cudaMalloc / GC.
MAX_FRAME_H = 2160
MAX_FRAME_W = 3840
_ROOT = os.path.dirname(os.path.abspath(__file__))
CUBIN_PATH = os.path.join(_ROOT, "preproc_kernel.cubin")
_CUDA_DLL = r"C:\Windows\System32\nvcuda.dll"  # driver-provided; CUDA 13.1 ships no stub

# ctypes handles to the CUDA driver API (pycuda 2026.1's module-load bindings
# segfault on this machine, so the letterbox kernel is loaded directly)
_cudll = None


def _get_cudll():
    global _cudll
    if _cudll is None:
        lib = ctypes.CDLL(_CUDA_DLL)
        lib.cuModuleLoadData.restype = ctypes.c_int
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleGetFunction.restype = ctypes.c_int
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                            ctypes.c_void_p, ctypes.c_char_p]
        lib.cuLaunchKernel.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_void_p)]
        _cudll = lib
    return _cudll
# strides and per-stride anchor counts for a 640x640 input (2 anchors/grid point)
STRIDES = [8, 16, 32]
ANCHORS = [(INPUT_SIZE // s) * (INPUT_SIZE // s) * 2 for s in STRIDES]  # [12800, 3200, 800]


def _build_anchors(size, stride, num_anchors=2):
    """Anchor centers matching the reference implementation ordering."""
    h = w = size // stride
    centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
    return centers


_ANCHOR_CACHE = {s: _build_anchors(INPUT_SIZE, s) for s in STRIDES}

# per-worker-thread canvas for letterbox (avoids per-image allocation)
_tls = threading.local()


def _nms(boxes, scores, thresh=0.4):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class SCRFDTRTDetector:
    """Batched SCRFD detector with 5-keypoint output on TensorRT."""

    def __init__(self, engine_path, max_batch=32, conf_thres=0.5, nms_thres=0.4,
                 max_workers=8, max_frame_h=MAX_FRAME_H, max_frame_w=MAX_FRAME_W):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.max_batch = max_batch
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # staging-buffer ceiling (pre-allocated once, reused across resolutions)
        self.max_frame_h = max_frame_h
        self.max_frame_w = max_frame_w

        self.input_name = self.engine.get_tensor_name(0)
        # outputs in ONNX order: score(s8,s16,s32), bbox(s8,s16,s32), kps(s8,s16,s32)
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(1, self.engine.num_io_tensors)
        ]
        self._alloc_buffers()
        self.stream = drv.Stream()
        # bind the (fixed) device buffers once; only input shape changes per batch
        self.ctx.set_input_shape(self.input_name, (1, 3, INPUT_SIZE, INPUT_SIZE))
        self.ctx.infer_shapes()
        self._shape_b = 1
        self.ctx.set_tensor_address(self.input_name, self._ptr(self.in_dev))
        for i, name in enumerate(self.output_names):
            self.ctx.set_tensor_address(name, self._ptr(self.out_dev[i]))
        # persistent thread pool for CPU preprocessing (cv2/numpy release the GIL)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        # GPU letterbox kernel (lazy-loaded on first use)
        self._cu_fn = None
        # Raw-frame staging. Frame i of a (h,w) chunk is packed at
        # [i*h*w*3 : (i+1)*h*w*3] with row stride w*3 — exactly the layout the
        # letterbox kernel reads, so no repack.
        #
        #   _raw_dev    : ONE flat device buffer, pre-allocated at max res
        #                 (see MAX_FRAME_H/W) and reused for every chunk. The
        #                 consumer is single-threaded and each run() finishes
        #                 its H2D+infer+D2H (synchronized) before the next, so a
        #                 single device buffer is race-free. Pre-allocating at
        #                 max res means a video-resolution switch never triggers
        #                 a cudaMalloc.
        #   _host_ring  : a ring of 4 flat HOST buffers (ping-pong). The pipeline
        #                 producer packs chunk i+1 while the consumer H2D-s
        #                 chunk i, so they must NOT share a buffer. The producer
        #                 leads the consumer by up to 3 chunks (queue depth 2 +
        #                 the one pack that happens before its blocking put), so
        #                 while chunk c's H2D reads ring[c%N], the producer may be
        #                 writing ring[(c+1)%N..(c+3)%N]. N must keep offsets 1,
        #                 2, 3 all != 0 (mod N) -> N=4. Grow-only to the largest
        #                 res seen (no per-chunk malloc).
        self._raw_dev = None
        self._raw_dev_cap = 0
        self._host_ring = []
        self._host_cap = 0
        self._host_idx = 0
        self._host_ring_n = 4
        print(f"[SCRFD-TRT] engine loaded: {os.path.basename(engine_path)} "
              f"max_batch={max_batch} conf={conf_thres}")

    # ---------------- GPU preprocessing (letterbox kernel) ----------------
    def _load_preproc_kernel(self):
        lib = _get_cudll()
        with open(CUBIN_PATH, "rb") as f:
            image = f.read()
        mod = ctypes.c_void_p()
        if lib.cuModuleLoadData(ctypes.byref(mod), image) != 0:
            raise RuntimeError("cuModuleLoadData failed")
        fn = ctypes.c_void_p()
        if lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"letterbox_bgr2rgb") != 0:
            raise RuntimeError("cuModuleGetFunction failed")
        self._cu_mod = mod  # keep alive
        self._cu_fn = fn

    def _launch_letterbox(self, raw_ptr, out_ptr, b, h, w, nw, nh):
        if self._cu_fn is None:
            self._load_preproc_kernel()
        args = (ctypes.c_void_p(raw_ptr), ctypes.c_void_p(out_ptr),
                ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(nw), ctypes.c_int(nh),
                ctypes.c_float(w / nw), ctypes.c_float(h / nh))
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        ret = _get_cudll().cuLaunchKernel(self._cu_fn,
                                          INPUT_SIZE // 16, INPUT_SIZE // 16, b,
                                          16, 16, 1, 0, self.stream.handle, ptrs, None)
        if ret != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {ret}")

    # ---------------- memory ----------------
    @staticmethod
    def _ptr(arr):
        # GPUArray exposes __cuda_array_interface__; TRT accepts the int device address
        return arr.__cuda_array_interface__["data"][0]

    def _alloc_buffers(self):
        b = self.max_batch
        self.in_dev = GPUArray((b, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self.out_dev = []
        for i in range(9):
            if i < 3:      # scores: (b, anchors, 1)
                shape = (b, ANCHORS[i], 1)
            elif i < 6:    # bbox: (b, anchors, 4)
                shape = (b, ANCHORS[i - 3], 4)
            else:          # kps: (b, anchors, 10)
                shape = (b, ANCHORS[i - 6], 10)
            self.out_dev.append(GPUArray(shape, dtype=np.float32))

    # ---------------- preprocess (threaded: cv2.resize & numpy release GIL) ----
    def _letterbox(self, img, size=INPUT_SIZE):
        """Top-left letterbox to size x size (matches reference impl).
        Returns blob [3,size,size] and scale. Thread-safe: each worker thread
        reuses its own thread-local canvases (no per-image allocation).
        NOTE: cv2.resize silently no-ops when dst dtype != src dtype in this
        OpenCV build, so the resize targets the uint8 canvas and the
        uint8->float32 conversion happens in the multiply below."""
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        if not hasattr(_tls, "u8"):
            _tls.u8 = np.empty((size, size, 3), dtype=np.uint8)
            _tls.f32 = np.empty((size, size, 3), dtype=np.float32)
        u8, f32 = _tls.u8, _tls.f32
        cv2.resize(img, (nw, nh), dst=u8[:nh, :nw])
        f32[nh:, :] = 0
        f32[:nh, nw:] = 0
        # (v/128 - 127.5/128) in-place, then BGR->RGB in-place (padding is
        # channel-equal, so the swap leaves it untouched)
        np.multiply(u8[:nh, :nw], 1.0 / 128.0, out=f32[:nh, :nw])
        np.subtract(f32, 127.5 / 128.0, out=f32)
        cv2.cvtColor(f32, cv2.COLOR_BGR2RGB, dst=f32)
        return f32.transpose(2, 0, 1), scale           # CHW view, all-positive strides

    def _preprocess_batch(self, images, workers=0):
        """Fill ONE pre-allocated batch buffer in parallel (single big copy,
        no intermediate stack)."""
        b = len(images)
        arr = np.empty((b, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        geo = [0.0] * b

        def fill(i, im):
            blob, scale = self._letterbox(im)
            arr[i] = blob
            geo[i] = scale

        if workers and b > 1:
            list(self._pool.map(lambda p: fill(*p), enumerate(images)))
        else:
            for i, im in enumerate(images):
                fill(i, im)
        return arr, geo

    # ---------------- postprocess ----------------
    def _postprocess(self, outs, b, threshold):
        """outs: list of 9 numpy arrays (already batched). Returns per-image (dets, kpss)."""
        score_arrs = [outs[i].reshape(b, ANCHORS[i], 1) for i in range(3)]
        bbox_arrs = [outs[i].reshape(b, ANCHORS[i - 3], 4) for i in range(3, 6)]
        kps_arrs = [outs[i].reshape(b, ANCHORS[i - 6], 10) for i in range(6, 9)]

        results = []
        for img in range(b):
            boxes_list, scores_list, kps_list = [], [], []
            for idx, stride in enumerate(STRIDES):
                s_blob = score_arrs[idx][img].reshape(-1)
                pos = np.where(s_blob >= threshold)[0]
                if pos.size == 0:
                    continue
                anchors = _ANCHOR_CACHE[stride][pos]
                cx, cy = anchors[:, 0], anchors[:, 1]
                d = bbox_arrs[idx][img][pos]  # (n,4) dx1,dy1,dx2,dy2
                x1 = cx - d[:, 0] * stride
                y1 = cy - d[:, 1] * stride
                x2 = cx + d[:, 2] * stride
                y2 = cy + d[:, 3] * stride
                boxes_list.append(np.stack([x1, y1, x2, y2], axis=1))
                scores_list.append(s_blob[pos])
                k = kps_arrs[idx][img][pos]  # (n,10)
                kpx = k[:, 0::2] * stride + cx[:, None]
                kpy = k[:, 1::2] * stride + cy[:, None]
                kp = np.stack([kpx, kpy], axis=2).reshape(-1, 10)
                kps_list.append(kp)

            if not boxes_list:
                results.append((np.zeros((0, 4)), np.zeros((0, 10)), np.zeros((0,))))
                continue
            boxes = np.concatenate(boxes_list)
            scores = np.concatenate(scores_list)
            kpss = np.concatenate(kps_list)
            keep = _nms(boxes, scores, self.nms_thres)
            # boxes (n,4) + score as 5th column -> (n,5) [x1,y1,x2,y2,score]
            results.append((boxes[keep], kpss[keep], scores[keep]))
        return results

    # ---------------- inference ----------------
    def _trt_pass(self, b, geo, threshold, all_dets, all_kpss, rescale=1.0):
        """TRT execute + D2H + postprocess (assumes input buffer already filled
        on self.stream). All transfers enqueued async in-order; one sync at the
        end. Main thread only holds the GIL for short enqueue calls — the sync
        releases it (verified empirically), so a CPU producer can overlap.

        rescale: full/scaled ratio (1.0 when no downscale). The letterbox `scale`
        maps 640-space -> the (possibly downscaled) frame that was fed in;
        rescale then maps that down to the ORIGINAL input frame coords."""
        if b != self._shape_b:
            self.ctx.set_input_shape(self.input_name, (b, 3, INPUT_SIZE, INPUT_SIZE))
            self.ctx.infer_shapes()
            self._shape_b = b

        self.ctx.execute_async_v3(self.stream.handle)

        # read outputs (D2H into fresh host buffers, in-order after the kernels)
        outs = []
        for i in range(9):
            if i < 3:
                n = b * ANCHORS[i] * 1
            elif i < 6:
                n = b * ANCHORS[i - 3] * 4
            else:
                n = b * ANCHORS[i - 6] * 10
            host = np.empty(n, dtype=np.float32)
            drv.memcpy_dtoh_async(host, self._ptr(self.out_dev[i]), self.stream)
            outs.append(host)
        self.stream.synchronize()

        per = self._postprocess(outs, b, threshold)
        # top-left letterbox: frame coord = letterbox coord / scale; then map the
        # (possibly downscaled) frame back to the original input coords.
        inv = rescale  # full/scaled
        for (dets, kpss, scores), scale in zip(per, geo):
            n = len(dets)
            m = (1.0 / scale) * inv
            xy = dets * m
            dets = np.concatenate([xy, scores.reshape(-1, 1)], axis=1) if n else np.zeros((0, 5))
            kpss = (kpss.reshape(-1, 2) * m).reshape(n, 5, 2) if n else kpss.reshape(0, 5, 2)
            all_dets.append(dets)
            all_kpss.append(kpss)

    def _run_chunk(self, arr, geo, b, threshold, all_dets, all_kpss, rescale=1.0):
        """CPU-preprocessed path: H2D the 640x640 blob, then TRT pass."""
        drv.memcpy_htod_async(self._ptr(self.in_dev), arr, self.stream)
        self._trt_pass(b, geo, threshold, all_dets, all_kpss, rescale)

    def _ensure_dev(self, b, h, w):
        """Pre-allocate (once) the single flat device raw buffer at max res,
        then reuse it for every chunk. Grows only if a frame exceeds the cap or
        the batch exceeds max_batch — never on a mere resolution switch."""
        H = max(self.max_frame_h, h)
        W = max(self.max_frame_w, w)
        need = max(self.max_batch, b) * H * W * 3
        if self._raw_dev is None or self._raw_dev_cap < need:
            if self._raw_dev is not None:
                del self._raw_dev  # drop old block explicitly (no double-residency)
                self._raw_dev = None
            self._raw_dev = GPUArray((need,), dtype=np.uint8)
            self._raw_dev_cap = need

    def _ensure_host_ring(self, b, h, w):
        """Ensure the 3-buffer host ping-pong ring is large enough (grow-only).
        Allocated at the largest resolution seen so far (capped at the 4K max);
        re-allocating only happens on first sight of a bigger frame, which lands
        at the start of a run — never mid-pipeline for a fixed-resolution video."""
        need = max(self.max_batch, b) * h * w * 3
        if not self._host_ring or self._host_cap < need:
            self._host_ring = [np.empty(need, dtype=np.uint8)
                               for _ in range(self._host_ring_n)]
            self._host_cap = need
            self._host_idx = 0

    def _run_chunk_gpu(self, raw, b, h, w, threshold, all_dets, all_kpss,
                       rescale=1.0):
        """GPU-preprocessed path: H2D the flat-packed raw bytes, letterbox
        kernel writes directly into the TRT input, then TRT pass."""
        self._ensure_dev(b, h, w)
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        drv.memcpy_htod_async(self._ptr(self._raw_dev), raw, self.stream)
        self._launch_letterbox(self._ptr(self._raw_dev), self._ptr(self.in_dev),
                               b, h, w, nw, nh)
        geo = [scale] * b
        self._trt_pass(b, geo, threshold, all_dets, all_kpss, rescale)

    def _pack_raw(self, chunk, h, w):
        """Copy HWC uint8 frames contiguously into the NEXT buffer of the host
        ping-pong ring (no per-chunk np.stack / malloc; ring makes it safe for
        the pipeline producer to overlap the consumer's in-flight H2D). Frame i
        occupies [i*h*w*3 : (i+1)*h*w*3] with row stride w*3 — exactly the
        layout the letterbox kernel reads. Returns a contiguous 1-D view."""
        b = len(chunk)
        self._ensure_host_ring(b, h, w)
        n = h * w * 3
        buf = self._host_ring[self._host_idx % self._host_ring_n]
        self._host_idx += 1
        for i in range(b):
            buf[i * n:(i + 1) * n] = chunk[i].ravel()
        return buf[:b * n]

    def _prepare_chunk(self, chunk, target_long):
        """Optionally downscale a chunk to long-side `target_long` (downscale
        only — never upscale), in the thread pool so it overlaps the GPU.
        Returns (chunk, h, w, rescale) where rescale = full/scaled so boxes /
        keypoints can be mapped back to the ORIGINAL input frame coords. If no
        downscale is needed, returns the chunk unchanged with rescale=1.0."""
        h, w = chunk[0].shape[:2]
        if target_long and max(h, w) > target_long:
            sf = target_long / max(h, w)
            nw, nh = int(round(w * sf)), int(round(h * sf))
            if len(chunk) > 1:
                chunk = list(self._pool.map(
                    lambda x: cv2.resize(x, (nw, nh),
                                         interpolation=cv2.INTER_AREA), chunk))
            else:
                chunk = [cv2.resize(chunk[0], (nw, nh),
                                    interpolation=cv2.INTER_AREA)]
            return chunk, nh, nw, 1.0 / sf
        return chunk, h, w, 1.0

    def _pipeline(self, make_work, n_works, run):
        """Producer thread feeds make_work(i) into a bounded queue while the
        main thread executes run(item) — overlaps CPU prep of work i+1 with
        GPU execution of work i. Returns when all n_works are done."""
        import queue
        q = queue.Queue(maxsize=2)

        def producer():
            for i in range(n_works):
                q.put(make_work(i))
            q.put(None)  # sentinel

        prod = threading.Thread(target=producer, daemon=True)
        prod.start()
        for _ in range(n_works):
            run(q.get())
        prod.join()

    def detect(self, images, threshold=None, workers=8, pipeline=True,
               gpu_preprocess=None, target_long=None):
        """
        images: list of HWC BGR uint8 images.
        threshold: face confidence cutoff (default self.conf_thres).
        workers: CPU threads for letterbox preprocessing (GIL is released by
                 cv2.resize / numpy, so >1 gives real parallelism).
        pipeline: if True and there are >=2 chunks, a producer thread prepares
                 batch i+1 while the main thread drives the GPU on batch i.
        gpu_preprocess: letterbox/normalize on GPU via the CUDA kernel (needs
                 preproc_kernel.cubin). None = use it if the cubin exists.
                 Frames are processed in consecutive same-size runs (a whole
                 video is one run). Falls back to CPU letterbox per run.
        target_long: if set, frames are downscaled to this long-side (never
                 upscaled) BEFORE detection, in the producer pool so it overlaps
                 the GPU. This shrinks the PCIe upload and is the main lever
                 once the chain is bandwidth-bound. Transparent to the caller:
                 boxes/keypoints are still returned in the ORIGINAL frame coords.
                 Default None (no downscale). 768 is a good value for 1080p/4K.
        Returns: list (len=images) of (dets, kpss)
            dets : (n,5) float32 [x1,y1,x2,y2,score] in original frame coords
            kpss : (n,5,2) float32  5 keypoints (x,y) in original frame coords
        """
        if threshold is None:
            threshold = self.conf_thres
        if gpu_preprocess is None:
            gpu_preprocess = os.path.exists(CUBIN_PATH)
        n = len(images)
        all_dets, all_kpss = [], []
        if n == 0:
            return all_dets, all_kpss

        if gpu_preprocess:
            # group into consecutive runs of identical (h,w)
            runs = []
            for im in images:
                key = (im.shape[0], im.shape[1])
                if runs and runs[-1][0] == key:
                    runs[-1][1].append(im)
                else:
                    runs.append((key, [im]))
            for (h, w), run in runs:
                self._detect_run(run, threshold, pipeline, gpu_preprocess=True,
                                 target_long=target_long,
                                 all_dets=all_dets, all_kpss=all_kpss)
            return all_dets, all_kpss

        self._detect_run(images, threshold, pipeline, gpu_preprocess=False,
                         workers=workers, target_long=target_long,
                         all_dets=all_dets, all_kpss=all_kpss)
        return all_dets, all_kpss

    def _detect_run(self, images, threshold, pipeline, gpu_preprocess,
                    workers=8, target_long=None, all_dets=None, all_kpss=None):
        n = len(images)
        n_chunks = (n + self.max_batch - 1) // self.max_batch
        if not (pipeline and n_chunks >= 2):
            for start in range(0, n, self.max_batch):
                chunk = images[start:start + self.max_batch]
                chunk, h, w, rescale = self._prepare_chunk(chunk, target_long)
                b = len(chunk)
                if gpu_preprocess:
                    raw = self._pack_raw(chunk, h, w)
                    self._run_chunk_gpu(raw, b, h, w,
                                        threshold, all_dets, all_kpss, rescale)
                else:
                    arr, geo = self._preprocess_batch(chunk, workers=workers)
                    self._run_chunk(arr, geo, b, threshold,
                                    all_dets, all_kpss, rescale)
            return

        if gpu_preprocess:
            def make_work(i):
                chunk = images[i * self.max_batch:(i + 1) * self.max_batch]
                chunk, h, w, rescale = self._prepare_chunk(chunk, target_long)
                return self._pack_raw(chunk, h, w), len(chunk), h, w, rescale

            def run(item):
                raw, b, h, w, rescale = item
                self._run_chunk_gpu(raw, b, h, w,
                                    threshold, all_dets, all_kpss, rescale)
        else:
            def make_work(i):
                chunk = images[i * self.max_batch:(i + 1) * self.max_batch]
                chunk, h, w, rescale = self._prepare_chunk(chunk, target_long)
                arr, geo = self._preprocess_batch(chunk, workers=workers)
                return arr, geo, rescale
            run = lambda ag: self._run_chunk(
                ag[0], ag[1], ag[0].shape[0],
                threshold, all_dets, all_kpss, ag[2])
        self._pipeline(make_work, n_chunks, run)


# ---------------- pose estimation ----------------
# Canonical 3D face model (x right, y down, z toward viewer).
# Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
_CANON_3D = np.array([
    [-1.0, -0.2, 0.0],
    [ 1.0, -0.2, 0.0],
    [ 0.0,  0.5, 1.2],
    [-0.7,  1.2, 0.3],
    [ 0.7,  1.2, 0.3],
], dtype=np.float64)


def estimate_pose(kps, box):
    """
    Estimate (yaw, pitch, roll) in degrees from 5 2D keypoints + face box.
    Returns None if solvePnP fails.
    """
    kps = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    f = float(np.hypot(bw, bh))          # focal ~ face diagonal
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cam = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        _CANON_3D, kps, cam, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # ZYX extraction
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))     # about Z (in-plane)
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))        # about X (nod up/down)
    yaw = np.degrees(np.arctan2(R[2, 1], R[2, 2]))      # about Y (turn left/right)
    return float(yaw), float(pitch), float(roll)


# ---------------- self test ----------------
if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    if len(sys.argv) > 1:
        engine = sys.argv[1]
    else:
        # 默认引擎优先级: FP16 动态 batch → FP32 动态 batch，按存在性逐级回退
        _candidates = (
            os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps_batch32_fp16.engine"),
            os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps_batch32.engine"),
        )
        engine = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
    img_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "_test", "frame_zhao_yi_lin_010.jpg")

    det = SCRFDTRTDetector(engine)
    img = cv2.imread(img_path)
    dets, kpss = det.detect([img], threshold=0.5)

    # fail loudly if the sample frame (a known portrait) yields zero faces —
    # a broken preprocess pipeline can produce "clean" empty results silently
    if len(dets[0]) == 0:
        print("!! SELF-TEST FAILED: no face detected in sample frame "
              "(expected >=1) — preprocessing/inference is broken !!")
        sys.exit(1)

    print(f"\nimage {img_path}  size={img.shape[1]}x{img.shape[0]}")
    for box, kps in zip(dets[0], kpss[0]):
        y, p, r = estimate_pose(kps, box[:4])
        print(f"  box={np.round(box).astype(int).tolist()}  "
              f"yaw={y:7.1f}  pitch={p:7.1f}  roll={r:7.1f}")

    # annotate + save
    out = img.copy()
    for box, kps in zip(dets[0], kpss[0]):
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (kx, ky) in kps:
            cv2.circle(out, (int(kx), int(ky)), 2, (0, 0, 255), -1)
    save = os.path.join(root, "_test", "out_detect.jpg")
    cv2.imwrite(save, out)
    print(f"\nannotated -> {save}")

    # throughput quick check
    import time
    imgs = [img] * 32
    t0 = time.time()
    for _ in range(10):
        det.detect(imgs, threshold=0.5)
    dt = (time.time() - t0) / 10
    print(f"throughput: batch32 in {dt*1000:.1f} ms  -> {32/dt:.0f} img/s")
