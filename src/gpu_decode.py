# -*- coding: utf-8 -*-
"""
gpu_decode.py — 显存直通零拷贝解码臂(--decode pynvvc-gpu)的构件。

数据流(NVDEC device 帧 → 现管线, 主路径免全帧 D2H/H2D):
  NVDEC RGB device 帧(解码器与推理共享同一条 CUDA 流, 自动保序)
    → D2D 拷入自有显存池(_PynvvcGpuSource, 唯一一次整帧设备侧拷贝)
    → area_fast_u8 / area_generic_u8: RGB→BGR 768 长边 INTER_AREA 下采样,
      与 cv2.resize(INTER_AREA) 逐字节一致(规格: _test/post3_e2_resample_ref.py)
    → 现役 preproc_kernel letterbox_bgr2rgb(768→640, 原核原引擎未改)
      → SCRFD 检测结果与宿主臂逐位一致
    → 人脸帧 rgb2bgr_inplace_u8(池内 RGB→BGR) → 同步 D2H 落宿主
      → pose68 / gaze / imwrite 走原宿主代码(与宿主臂同输入同结果)
    → head 闸门帧: head_letterbox_u8(定点 INTER_LINEAR 640 居中 letterbox,
      与宿主 cv2.resize 画布逐字节一致)直接写 fp32 blob → 原宿主 head 引擎

等价性策略: 不引入翻转引擎, 而是 GPU 上逐位复刻宿主预处理(OpenCV 的
INTER_AREA / INTER_LINEAR 精确配方), 使送入各引擎的 blob 与宿主臂逐字节
相同 → 判定结果结构性一致(连阈值边缘帧都不会漂)。

pycuda 精简版适配: drv.Stream() + stream.handle(int) 传 execute_async_v3;
设备指针用 GPUArray.__cuda_array_interface__["data"][0]; D2D 用
drv.memcpy_dtod_async。
"""
import ctypes
import os
import queue
import threading
import time

import numpy as np
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray

from src.face_det import _get_cudll, INPUT_SIZE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUBIN_PATH = os.path.join(_ROOT, "src", "gpu_decode_kernels.cubin")
_HEAD_S = 640
_DBL_EPS = 2.220446049250313e-16   # DBL_EPSILON


# ---------------------------------------------------------------------------
# INTER_AREA 精确表构建(宿主, double 精度; OpenCV computeResizeAreaTab 逐式复刻)
# ---------------------------------------------------------------------------
def _area_tab(ssize, dsize, scale):
    """computeResizeAreaTab: 每 dx 一组 [(si, alpha_f32)](si 升序)。"""
    groups = []
    for dx in range(dsize):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cellWidth = min(scale, ssize - fsx1)
        sx1 = int(np.ceil(fsx1))
        sx2 = int(np.floor(fsx2))
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        ents = []
        if sx1 - fsx1 > 1e-3:
            ents.append((sx1 - 1, np.float32((sx1 - fsx1) / cellWidth)))
        for sx in range(sx1, sx2):
            ents.append((sx, np.float32(1.0 / cellWidth)))
        if fsx2 - sx2 > 1e-3:
            ents.append((sx2, np.float32(min(min(fsx2 - sx2, 1.0), cellWidth)
                                         / cellWidth)))
        groups.append(ents)
    return groups


def area_path(W, H, nw, nh):
    """cv2.resize 路径判定: ("fast", kx, ky) 或 ("generic", 0, 0)。
    与 resize() 一致: scale_x = 1./inv_scale_x (inv = dsize/ssize, double)。"""
    scale_x = 1.0 / (np.float64(nw) / np.float64(W))
    scale_y = 1.0 / (np.float64(nh) / np.float64(H))
    kx = int(round(scale_x))          # saturate_cast<int> = cvRound(RNE)
    ky = int(round(scale_y))
    if (abs(scale_x - kx) < _DBL_EPS and abs(scale_y - ky) < _DBL_EPS
            and nw * kx == W and nh * ky == H):
        return "fast", kx, ky
    return "generic", 0, 0


class AreaTables:
    """一次几何构建, 整视频复用。fast: 记 (kx, ky); generic: 上传 6 张表。"""

    def __init__(self, W, H, nw, nh):
        kind, kx, ky = area_path(W, H, nw, nh)
        self.kind = kind
        self.kx, self.ky = kx, ky
        self._dev = []
        if kind == "generic":
            scale_x = 1.0 / (np.float64(nw) / np.float64(W))
            scale_y = 1.0 / (np.float64(nh) / np.float64(H))
            xt = _area_tab(W, nw, scale_x)
            yt = _area_tab(H, nh, scale_y)
            x_ofs = np.zeros(nw + 1, np.int32)
            y_ofs = np.zeros(nh + 1, np.int32)
            xs, xa, ys, ya = [], [], [], []
            for dx, ents in enumerate(xt):
                x_ofs[dx] = len(xs)
                for si, al in ents:
                    xs.append(si)
                    xa.append(al)
            x_ofs[nw] = len(xs)
            for dy, ents in enumerate(yt):
                y_ofs[dy] = len(ys)
                for si, al in ents:
                    ys.append(si)
                    ya.append(al)
            y_ofs[nh] = len(ys)
            host = (x_ofs, np.asarray(xs, np.int32), np.asarray(xa, np.float32),
                    y_ofs, np.asarray(ys, np.int32), np.asarray(ya, np.float32))
            for arr in host:
                ga = GPUArray(arr.shape, dtype=arr.dtype)
                drv.memcpy_htod(ga.__cuda_array_interface__["data"][0], arr)
                self._dev.append(ga)
        self.ptrs = [a.__cuda_array_interface__["data"][0] for a in self._dev]


# ---------------------------------------------------------------------------
# kernel 加载 + 发射(ctypes 路线, 与 face_det 同款)
# ---------------------------------------------------------------------------
class GpuKernels:
    def __init__(self, cubin_path=CUBIN_PATH):
        lib = _get_cudll()
        self._lib = lib
        with open(cubin_path, "rb") as f:
            image = f.read()
        mod = ctypes.c_void_p()
        if lib.cuModuleLoadData(ctypes.byref(mod), image) != 0:
            raise RuntimeError("cuModuleLoadData failed (gpu_decode_kernels)")
        self._mod = mod          # 持引用防 GC
        self.fn = {}
        for name in (b"area_fast_u8", b"area_generic_u8", b"head_letterbox_u8",
                     b"rgb2bgr_inplace_u8", b"copy_u8"):
            fn = ctypes.c_void_p()
            if lib.cuModuleGetFunction(ctypes.byref(fn), mod, name) != 0:
                raise RuntimeError(f"cuModuleGetFunction failed: {name!r}")
            self.fn[name.decode()] = fn

    @staticmethod
    def _args(*vals):
        return (ctypes.c_void_p * len(vals))(
            *[ctypes.addressof(v) for v in vals])

    def _launch(self, fn, grid, block, args, stream):
        ret = self._lib.cuLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            0, stream.handle, self._args(*args), None)
        if ret != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {ret}")

    def area_batch(self, stream, src_ptr, dst_ptr, b, H, W, nh, nw, tab,
                   oswap=1):
        """b 帧连续槽 [src_ptr .. +b*H*W*3) → b 个 (nh,nw,3) 槽。
        oswap=1: 输入 RGB → 输出 BGR(与宿主 cv2.resize 结果逐字节一致)。"""
        grid = ((nw + 15) // 16, (nh + 15) // 16, b)
        ci, cp = ctypes.c_int, ctypes.c_void_p
        if tab.kind == "fast":
            self._launch(self.fn["area_fast_u8"], grid, (16, 16, 1),
                         (cp(src_ptr), cp(dst_ptr), ci(H), ci(W), ci(nh),
                          ci(nw), ci(tab.kx), ci(tab.ky), ci(oswap)), stream)
        else:
            x_ofs, x_si, x_alpha, y_ofs, y_si, y_alpha = tab.ptrs
            self._launch(self.fn["area_generic_u8"], grid, (16, 16, 1),
                         (cp(src_ptr), cp(dst_ptr), ci(H), ci(W), ci(nh),
                          ci(nw), ci(oswap),
                          cp(x_ofs), cp(x_si), cp(x_alpha),
                          cp(y_ofs), cp(y_si), cp(y_alpha)), stream)

    def head_letterbox(self, stream, src_ptr, dst_ptr, b, H, W,
                       nh, nw, pt, pl, bswap=1):
        """b 帧 → b 份 (3,640,640) fp32 blob(值/255; bswap=1: BGR 入 RGB 出,
        即宿主 cvtColor 后的通道序)。"""
        grid = (_HEAD_S // 16, _HEAD_S // 16, b)
        ci, cp = ctypes.c_int, ctypes.c_void_p
        self._launch(self.fn["head_letterbox_u8"], grid, (16, 16, 1),
                     (cp(src_ptr), cp(dst_ptr), ci(H), ci(W), ci(_HEAD_S),
                      ci(nh), ci(nw), ci(pt), ci(pl), ci(bswap)), stream)

    def swap_inplace(self, stream, ptr, npix):
        self._launch(self.fn["rgb2bgr_inplace_u8"], (1024, 1, 1), (256, 1, 1),
                     (ctypes.c_void_p(ptr), ctypes.c_int(npix)), stream)


# ---------------------------------------------------------------------------
# 显存帧池
# ---------------------------------------------------------------------------
class DeviceFramePool:
    """连续槽位显存池: 槽 k 位于 base + k*h*w*3。grow-only, 跨视频复用。"""

    def __init__(self):
        self.ga = None
        self.cap_slots = 0
        self.frame_bytes = 0

    def ensure(self, n_slots, h, w):
        if self.ga is not None and self.cap_slots >= n_slots \
                and self.frame_bytes == h * w * 3:
            return
        if self.ga is not None:
            del self.ga
        self.ga = GPUArray((n_slots * h * w * 3,), dtype=np.uint8)
        self.cap_slots = n_slots
        self.frame_bytes = h * w * 3

    @property
    def base(self):
        return self.ga.__cuda_array_interface__["data"][0]

    def slot(self, k=0):
        """槽 k 的物理地址(不取模): 槽号必须落在 [0, cap_slots) 内。
        块内连续槽由 area/head kernel 直接按连续物理地址消费, 故由
        _PynvvcGpuSource.read_block 保证每块不跨池尾回绕(放不下则回到 0)。"""
        return self.base + k * self.frame_bytes


# ---------------------------------------------------------------------------
# NVDEC device 帧源(选帧规则与 _PynvvcSource 完全一致)
# ---------------------------------------------------------------------------
def _create_threaded_decoder(video, buffer_size, ctx, stream):
    """ThreadedDecoder 工厂(独立函数便于测试注入初始化失败)。
    位置参数序: (encSource, bufferSize, gpuid, cudaContext, cudaStream,
    useDeviceMemory, maxWidth, maxHeight, needScannedStreamMetadata,
    decoderCacheSize, outputColorType) —— 与 CreateSimpleDecoder 同风格,
    cudaContext/cudaStream 传 int handle(探针1 P4/T2 验证)。"""
    import PyNvVideoCodec as pnv
    return pnv.CreateThreadedDecoder(
        video, int(buffer_size), 0, int(ctx.handle), int(stream.handle),
        True, 0, 0, 0, 0, pnv.OutputColorType.RGB)


class _PynvvcGpuSource:
    """pynvvc useDeviceMemory=True 帧源: 选中帧 D2D 拷入显存池, 不落宿主。
    read_block(n) → (start_slot, m): m(≥n, 末批溢出可到 n+7) 个选中帧落在
    池槽 [start_slot, start_slot+m)。池容量须 ≥ n + 8: 一块最多写
    start + n + 7, 放不下则整块回到槽 0(保证块内物理连续且不回绕 ——
    area/head kernel 按连续物理地址消费; 单线程消费, 旧块先读完再写新块)。
    waited 累计(NVDEC 拉帧 + D2D 入队)供 decode 段计时。
    EOF/断流语义与 _PynvvcSource 相同(干净 EOF → m=0; 提前断流 → 抛错回退)。"""

    def __init__(self, video, w, h, fps, max_fps, stream, pool,
                 decoder="simple", buffer_size=16, dstream=None):
        import filter_video as fv
        self.waited = 0.0
        self._closed = False
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        self.stream = stream
        # D 轮(A 项)流分流: 解码器(NVDEC 硬解 + YUV→RGB 转换 kernel)与帧
        # D2D 入池走 dstream(独立流), 消费 kernel(area/letterbox/TRT/换序/
        # 落地 D2H/head)留在主流 stream —— 转换 kernel(~0.31ms/帧)不再与
        # 推理 kernel 在同一流上串行(§13.5: 共享流拉长解码窗口, 099
        # +0.56s / 154 +1.65s / 214 +0.13s)。dstream=None → 退化为旧单流。
        self.dstream = stream if dstream is None else dstream
        # 出口事件护栏(每源一对句柄, 跨块复用): CUDA wait 捕获「wait 调用
        # 时刻最近一次 record」, 同线程 record→wait 程序序 ⇒ 本块 wait 恒
        # 绑定本块 record(下一块的 re-record 不影响已入队 wait)。
        self._ev_d = drv.Event()
        self.pool = pool
        sd = fv._pynvvc_stride_delta(self.fps, max_fps)
        if sd is None:
            raise fv._PynvvcDecodeError(
                f"src_fps={self.fps:g} 不在已标定集合 {sorted(fv._PYNVVC_ALIGN)}")
        self.stride, self.delta = sd
        self._next_want = self.delta
        self._given = 0
        self._slot = 0
        self._done = False
        try:
            import PyNvVideoCodec as pnv
            ctx = drv.Context.get_current()
            if decoder == "threaded":
                # 内部 C++ 线程做 NVDEC 解码(不占 GIL), get_batch_frames 只
                # 是环形队列弹出 —— 解码∥推理重叠的前提(探针1 P3b)。
                # bufferSize: 解码先行帧数(16×4K RGB ≈ 400MB 显存);批量
                # get_batch_frames(n) 要求 n ≤ bufferSize(探针2 T3)。
                self.dec = _create_threaded_decoder(video, buffer_size,
                                                    ctx, self.dstream)
            else:
                self.dec = pnv.CreateSimpleDecoder(
                    video, 0, int(ctx.handle), int(self.dstream.handle), True,
                    0, 0, 0, 0, pnv.OutputColorType.RGB)
            md = self.dec.get_stream_metadata()
            self.num_frames = int(getattr(md, "num_frames", 0) or 0)
            mw = int(getattr(md, "width", 0) or 0)
            mh = int(getattr(md, "height", 0) or 0)
        except Exception as e:
            self.dec = None
            raise fv._PynvvcDecodeError(
                f"pynvvc({decoder}) 建解码器失败: {e!r}") from e
        if (mw, mh) != (self.w, self.h):
            self.dec = None
            raise fv._PynvvcDecodeError(
                f"pynvvc 解码尺寸 {mw}x{mh} != ffprobe {self.w}x{self.h}"
                f"（旋转/元数据差异）")
        self._nbytes = self.h * self.w * 3

    def _expected_frames(self):
        if self.num_frames <= 0:
            return 0
        if self.stride == 1 and self.delta == 0:
            return self.num_frames
        return max(0, (self.num_frames - self.delta - 1) // self.stride + 1)

    def read_block(self, n, slot_base=None):
        """拉取 n 个选中帧(EOF 不足则拉到多少算多少)。
        返回 (start_slot, m)。池容量 ≥ n+8: 一块最多写 start+n+7 个槽,
        尾部放不下整块则回到槽 0(块内物理连续、不跨池尾回绕);
        批边界溢出帧(≤8)不会覆盖本块未消费槽位(单线程消费,
        上一块在下次 read_block 前已处理完)。
        slot_base: 显式指定本块起始槽(重叠臂按块序把块钉在
        (k%2)*(n+8) 的半池基址上, 两块一轮回 —— 连续排布会被下一块
        覆写上一块的溢出槽, 见 _OverlappedGpuSource 文档); None=旧行为
        (游标连续排布, 池尾回绕)。"""
        import filter_video as fv
        if self._done:
            return self._slot, 0
        t1 = time.time()
        cap = self.pool.cap_slots
        if slot_base is not None:
            self._slot = int(slot_base)
        elif cap and self._slot + n + 8 > cap:
            self._slot = 0
        start = self._slot
        copied = 0
        try:
            while copied < n:
                frames = self.dec.get_batch_frames(8)
                if not frames:
                    self._done = True
                    break
                for f in frames:
                    j = self._given
                    self._given += 1
                    if j == self._next_want:
                        src_ptr = int(f.cuda()[0].dataptr)
                        drv.memcpy_dtod_async(self.pool.slot(self._slot),
                                              src_ptr, self._nbytes,
                                              self.dstream)
                        self._slot += 1
                        copied += 1
                        self._next_want += self.stride
                del frames
            # D 轮(A 项)出口护栏: 主流后续消费 kernel(area/letterbox/TRT/
            # 换序/落地 D2H/head)等本块全部 D2D 在 dstream 上完成 —— 两流
            # 后「解码器写帧完成 → 消费读帧」的 happens-before 由该事件
            # 钉死(替代旧同流程序序; §14 有完整保序证明)。串行臂同线程
            # record→wait; 重叠臂由 producer 线程代入队(q_put 在 wait 之后
            # ⇒ consumer 的 kernel 必排在 wait 之后, 无竞态窗口)。
            self._ev_d.record(self.dstream)
            self.stream.wait_for_event(self._ev_d)
        except Exception as e:
            self.waited += time.time() - t1
            raise fv._PynvvcDecodeError(
                f"pynvvc(device) 解码中断: {e!r}") from e
        self.waited += time.time() - t1
        if self._done and self.num_frames > 0 \
                and self._given < self._expected_frames() - 1:
            raise fv._PynvvcDecodeError(
                f"pynvvc 提前断流: 解出 {self._given} 帧 < 期望 "
                f"{self._expected_frames()}（源 {self.num_frames} 帧）")
        return start, copied

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.dec = None          # 释放 NVDEC 会话（不支持重放, 用完即弃）


# ---------------------------------------------------------------------------
# 解码∥推理重叠包装(阶梯1: ThreadedDecoder 原生线程重叠)
# ---------------------------------------------------------------------------
_OVL_EOF = object()      # producer → consumer 的结束哨兵
OVL_SETS = 2             # 槽组数=信号量 token 数(C 轮实测 2→3 无收益回退,
                         # 见 data/e2e_benchmark_2026-09-28.md §12)


class _OverlappedGpuSource:
    """_PynvvcGpuSource 的解码∥推理重叠包装(--overlap auto/on)。

    背景(A 轮基线): device 臂的解码段 8.60s(3171 源帧 ≈2.7ms/帧)与推理
    (~4.5s)完全串行 —— _PipelinedSource(§9)的线程级重叠被 GIL 锁死。
    探针1(post4_probe_threaded.py)推翻了 §9.2 对 ThreadedDecoder 的结论:
    §9.2 用「纯 Python 轮询循环」测 GIL 占用, 而真实推理主循环的墙钟大头
    在释放 GIL 的 C 调用里(cv2/numpy/TRT execute/synchronize) ——
      P2  后台 SimpleDecoder(host) 拉帧: 解码线程被 GIL 饿死(47s 解不完
          583 帧) —— 无内部线程, 每次解码 C 调用都与主线程争 GIL;
      P3b 后台 ThreadedDecoder(device) 拉帧: 解码全速(2.6ms/帧, 单独跑
          2.31), 主线程 detect_batch it/s 保持基线 96~97% —— 内部 C++
          解码线程不占 GIL, Python 层 get_batch_frames 只是环形队列弹出。
    ⇒ 重叠成立且必须用 ThreadedDecoder。

    槽位/保序协议(消费代码 _run_pass_gpu 一行不改):
      - 池容量 OVL_SETS×(chunk+8): 第 k 块钉在槽基址 (k%OVL_SETS)*(chunk+8),
        OVL_SETS 块一轮回(read_block(slot_base=...)); 不能连续排布 —— 上一块
        的批边界溢出槽(≤7 个)会被下一块的 D2D 覆写, 溢出槽的时序不受信号量
        保护;
      - 信号量 OVL_SETS 个 token: producer 解码第 k 块前 acquire; consumer
        在下一次 read_block 进入时 release 上一块的 token —— 此时上一块的
        全部 GPU 工作(detect/换序/落地 D2H/head)已入队或同步完成, producer
        第 k+OVL_SETS 块(同槽基址)的 D2D 在同一条流上必然排在其后
        (C 轮实测 2→3: 非首视频 e2e 变化 ≤0.04s=噪声, 首视频多付 ~0.55s
         一次性分配 —— producer 未被 token 饿住, 维持 2, 见 §12);
      - 流分流(D 轮 A 项): 解码器(NVDEC+转换 kernel)与 producer D2D 走
        dstream(独立流), consumer kernel 留主流 stream —— 转换 kernel 不再
        与推理 kernel 同流串行(§13.5 解码窗口拉长 099 +0.56s/154 +1.65s)。
        两流后的保序(§14 完整证明):
          ① D2D(j) → 消费读帧: read_block 出口 ev_d.record(dstream) →
             stream.wait_for_event(ev_d) 在 q_put(j) 之前入队 ⇒ consumer
             块 j 的 kernel 在流上恒排在该 wait 后 ⇒ 执行序晚于 D2D(j) 完成;
          ② 消费读帧 → D2D(j+OVL_SETS) 覆写同组槽: token 协议 + 块内同步链
             (scrfd_block_detect/_trt_pass、落地 D2H、head_slots_detect 均
             stream.synchronize()) ⇒ pop(j-1) 释放 token 时块 j-1(及所有更早
             块)的读槽 kernel 已设备侧完成, producer 解码第 j 块(≥NVDEC 一帧
             延迟)远在其后 —— 与旧单流设计同一论证, 事件只加不减;
          ③ 解码器内部帧缓冲回收: 写它的转换 kernel 与读它的 D2D 同在
             dstream, 同流程序序保护(§10.5 enqueue-order 余量论证原样成立)。
        探针2 T2 实测: 32 选帧横跨多个 bufferSize=16 回收-覆写窗口,
        生产者式 D2D(不 sync, 帧即拉即释放)与同步参考逐字节一致。
      - 指针生命周期: ThreadedDecoder 的 device 帧不因 Python 持引用而保活
        (探针1 P5: 持有首批再拉 6 批, 8 槽全部被覆写) —— 与串行臂同款
        「批内立即 D2D」策略恰好是唯一正确姿势。

    EOF/断流/异常语义与 _PynvvcGpuSource 一致(EOF=空列表, 探针3; end() 后
    再拉帧会永久挂死, 探针3 —— 故 close 顺序: 停 producer → join → 才
    释放解码器, producer 正常退出时已自行释放)。初始化(建解码器)在调用方
    线程同步完成, 失败就地暴露 → _run_pass_gpu 回退 A 轮串行臂; 解码中途
    错误由 producer 存下、consumer read_block 原样重抛 → process() 逐级
    回退链(宿主 pynvvc → 管道 → 落盘)不变。"""

    def __init__(self, video, w, h, fps, max_fps, stream, pool, chunk,
                 buffer_size=16, first_chunk=16, dstream=None):
        self._chunk = int(chunk)
        self._first = int(first_chunk)   # 首块小批量: 消费端提前启动(削 ramp)
        self._half = self._chunk + 8     # 槽组大小(OVL_SETS 组一轮换)
        self._q = queue.Queue(maxsize=4)
        self._sem = threading.Semaphore(OVL_SETS)
        self._stop = threading.Event()
        self._err = None
        self._waited_done = 0.0
        self._pending = None      # consumer 手里未释放 token 的块序号
        self._closed = False
        # 内层源在调用方线程构建(主线程有 CUDA context): 初始化失败同步抛出
        self._src = _PynvvcGpuSource(video, w, h, fps, max_fps, stream, pool,
                                     decoder="threaded",
                                     buffer_size=buffer_size, dstream=dstream)
        self._ctx = drv.Context.get_current()
        self._th = threading.Thread(target=self._produce, daemon=True,
                                    name="decode-producer-overlap")
        self._th.start()

    # -- producer 线程 --
    def _produce(self):
        src = self._src
        pushed = False
        try:
            try:
                self._ctx.push()     # 本线程 D2D 需 context current(探针2 T1)
                pushed = True
            except Exception:
                pushed = False
            k = 0
            while not self._stop.is_set():
                if not self._sem_acquire():
                    break            # stop 已置位
                if self._stop.is_set():
                    self._sem.release()
                    break
                try:
                    s0, m = src.read_block(
                        self._chunk if k else self._first,
                        slot_base=(k % OVL_SETS) * self._half)
                except BaseException as e:
                    self._err = e
                    self._q_put(_OVL_EOF)
                    break
                if m == 0:
                    self._q_put(_OVL_EOF)
                    break
                k += 1
                if not self._q_put((s0, m, k - 1)):
                    break
        finally:
            self._waited_done = float(getattr(src, "waited", 0.0))
            try:
                src.close()          # 释放 NVDEC 会话(ctx 在本线程 current)
            except Exception:
                pass
            if pushed:
                try:
                    self._ctx.pop()
                except Exception:
                    pass

    def _sem_acquire(self):
        while not self._stop.is_set():
            if self._sem.acquire(timeout=0.2):
                return True
        return False

    def _q_put(self, item):
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    # -- consumer 接口(与 _PynvvcGpuSource.read_block 同契约) --
    def read_block(self, n):
        if self._pending is not None:
            # 上一块在 pop 时即释放其 token: 块 k 的判定链(含 head 同步)在本
            # 轮迭代内完成, 之后才回来 pop 块 k+1 —— producer 领先
            # ≤ OVL_SETS-1 块, D2D(j) 覆写的第 j%OVL_SETS 组上一用户是块
            # j-OVL_SETS(≤ j-2), 其判定链已在 consumer pop(j-1) 前同步完成,
            # 流上 D2D 必然排在其后, 无覆写竞态
            self._pending = None
            self._sem.release()
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if not self._th.is_alive():
                    if self._err is not None:
                        raise self._err
                    return 0, 0
                continue
            if item is _OVL_EOF:
                if self._err is not None:
                    raise self._err
                return 0, 0
            s0, m, idx = item
            self._pending = idx
            return s0, m

    @property
    def waited(self):
        """producer 线程内累计的解码等待(NVDEC 拉帧 + D2D 入队)。"""
        if self._src is not None:
            return float(getattr(self._src, "waited", 0.0))
        return self._waited_done

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._sem.release()          # 解锁可能阻塞在 acquire 上的 producer
        deadline = time.time() + 5.0
        while self._th.is_alive() and time.time() < deadline:
            try:
                self._q.get_nowait()  # 排空队列, 解锁阻塞在 put 上的 producer
            except queue.Empty:
                self._th.join(timeout=0.05)
        self._th.join(timeout=2.0)
        if self._th.is_alive():
            # 极端路径: producer 仍阻塞在解码器内部 C 调用 —— 此时不能触碰
            # 解码器(end() 后再拉帧会挂死), 留给 daemon 线程随进程退出
            print("  WARNING: 重叠解码 producer 线程未在超时内退出"
                  "(NVDEC 会话随进程退出释放)", flush=True)
        elif self._src is not None:
            self._src.close()        # 兜底(producer 正常退出时已自行 close)
        self._src = None


# ---------------------------------------------------------------------------
# 块级 SCRFD / head 的 device 直通
# ---------------------------------------------------------------------------
def scrfd_block_detect(kernels, det, stream, pool, start, m, H, W,
                       small_ga, nh7, nw7, rescale, tables, conf):
    """显存池槽 [start, start+m) 批量检测(16 一批, 与宿主臂 det.detect 的
    分批一致)。返回 (dets, kpss) 两条 m 长列表(格式与 det.detect 相同)。
    前置: pool 槽内为 RGB 帧; small_ga 容量 ≥ det.max_batch 个 768 槽。"""
    all_dets, all_kpss = [], []
    small_ptr = small_ga.__cuda_array_interface__["data"][0]
    scale = min(INPUT_SIZE / nw7, INPUT_SIZE / nh7)
    nw, nh = int(round(nw7 * scale)), int(round(nh7 * scale))
    for s0 in range(start, start + m, det.max_batch):
        b = min(det.max_batch, start + m - s0)
        kernels.area_batch(stream, pool.slot(s0), small_ptr, b, H, W,
                           nh7, nw7, tables, oswap=1)
        det._launch_letterbox(small_ptr, det._ptr(det.in_dev),
                              b, nh7, nw7, nw, nh)
        det._trt_pass(b, [scale] * b, conf, all_dets, all_kpss, rescale)
    return all_dets, all_kpss


def head_slots_detect(kernels, head, stream, pool, slots, H, W):
    """显存池槽位(已换 BGR)直接 letterbox 写入 head.in_dev 并批量执行。
    返回每槽 list[(x1,y1,x2,y2,conf)](与 HeadGate.detect_batch 同格式)。"""
    from src.head_gate import _OUT_C, _OUT_HW
    results = []
    scale = min(_HEAD_S / W, _HEAD_S / H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    pt, pl = (_HEAD_S - nh) // 2, (_HEAD_S - nw) // 2
    in_ptr = head._ptr(head.in_dev)
    blob_bytes = 3 * _HEAD_S * _HEAD_S * 4
    for g0 in range(0, len(slots), head.max_batch):
        group = slots[g0:g0 + head.max_batch]
        m = len(group)
        for i, s in enumerate(group):
            kernels.head_letterbox(stream, pool.slot(s),
                                   in_ptr + i * blob_bytes, 1, H, W,
                                   nh, nw, pt, pl, bswap=1)
        if head._dynamic:
            head.ctx.set_input_shape(head.in_name, (m, 3, _HEAD_S, _HEAD_S))
        head.ctx.execute_async_v3(stream.handle)
        host = np.empty(m * _OUT_C * _OUT_HW, dtype=np.float32)
        drv.memcpy_dtoh_async(host, head._ptr(head.out_dev), stream)
        stream.synchronize()
        raw = host.reshape(m, _OUT_C, _OUT_HW)
        for i in range(m):
            results.append(head._decode_one(raw[i], scale, pt, pl, H, W))
    return results
