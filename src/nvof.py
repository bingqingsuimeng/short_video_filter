# -*- coding: utf-8 -*-
"""
nvof.py — NVIDIA OFA(NVOF2)硬件光流生产封装(--dedup-of, opt-in 默认关)。

从 _test/post8_nvof_probe.py 提炼(该脚本已在本机跑通 7 视频全流程), 所有
结构体一律照官方 SDK 5.0.7 头文件(E:/output/nvof_probe/sdk/, NvOFInterface/
= API 5.0, NVIDIAOpticalFlowSDK/= API 2.0, 按 NvOFGetMaxSupportedApiVersion
自动选), 不猜任何布局。

沿用 post8 的关键环境解法(实测本机 driver 591.86 + WDDM):
  * ctypes 直调 nvcuda driver API 的 cuMemAlloc/cuMemcpy*/cuEvent* 一律
    rc=201(INVALID_CONTEXT) —— NVOF 会话挂 primary context, 拷贝/流/事件
    全走 cudart(PyNvVideoCodec 自带 cudart64_12.dll, 同一地址空间);
  * 输入 buffer 由 rgb2gray kernel(device→device)直写(生产 GPU 臂解码池
    是 RGB, 经 src/nvof_kernels.cubin 的 rgb2gray_u8 转成 GRAY8 直接写进
    NVOF 输入 buffer 的行 stride 内, 零 H2D); 对账/分析场景仍保留
    upload_nv12(post8 的 NV12 直喂路径)与 download()(矢量 D2H);
  * 信号归约: of_sig_reduce_u8 kernel 在 device 上把 grid 矢量归约成
    7 个 double 部分和, 每帧只 D2H 56B, 宿主按 post7/post8 同公式换算
    fm/mr/dc/mcv(避免 ~1MB/帧 矢量 D2H + 宿主 numpy 2-5ms)。

失败面收敛: DLL 加载 / NvOFGetMaxSupportedApiVersion / NvOFAPICreateInstanceCuda /
CreateInstance / nvOFInit / nvOFExecute / 任何 cudart 调用失败 → 统一抛
NvOFError(可捕获), 由调用方(filter_video.py)打 WARNING 后禁用 OF 规则
继续跑(纯 MAD 行为), 不回退 Farneback 慢路径。

测试钩子: 环境变量 SVF_NVOF_FAIL=1 → NvofEngine.__init__ 强制抛 NvOFError
(回退演练用, 不触碰任何真实 API)。
"""
import contextlib
import ctypes
import os
from ctypes import (POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_size_t, c_uint32, c_uint64, c_void_p, WINFUNCTYPE,
                    WinDLL)

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
CUBIN_PATH = os.path.join(_ROOT, "nvof_kernels.cubin")

# ---- 官方头文件常量(nvOpticalFlowCommon.h) ----
NV_OF_SUCCESS = 0
NV_OF_BUFFER_FORMAT_GRAYSCALE8 = 1
NV_OF_BUFFER_FORMAT_NV12 = 2
NV_OF_BUFFER_FORMAT_SHORT2 = 5
NV_OF_BUFFER_USAGE_INPUT = 1
NV_OF_BUFFER_USAGE_OUTPUT = 2
NV_OF_MODE_OPTICALFLOW = 1
NV_OF_PERF_LEVEL_MEDIUM, NV_OF_PERF_LEVEL_FAST = 10, 20
NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2
CAPS_OUT_GRID, CAPS_W_MIN, CAPS_W_H_MIN, CAPS_W_MAX, CAPS_H_MAX = 0, 4, 5, 6, 7
CUDA_MEMCPY_H2D, CUDA_MEMCPY_D2H, CUDA_MEMCPY_D2D = 1, 2, 3

STATUS_NAMES = ["SUCCESS", "ERR_OF_NOT_AVAILABLE", "ERR_UNSUPPORTED_DEVICE",
                "ERR_DEVICE_DOES_NOT_EXIST", "ERR_INVALID_PTR", "ERR_INVALID_PARAM",
                "ERR_INVALID_CALL", "ERR_INVALID_VERSION", "ERR_OUT_OF_MEMORY",
                "ERR_NOT_INITIALIZED", "ERR_UNSUPPORTED_FEATURE", "ERR_GENERIC"]

# 阈值默认口径(post8 compare 结论): dc>=0.70 & mr>=0.50 & mcv<=0.70
OF_DC_DEFAULT, OF_MR_DEFAULT, OF_MCV_DEFAULT = 0.70, 0.50, 0.70


def st_name(s):
    return STATUS_NAMES[s] if 0 <= s < len(STATUS_NAMES) else f"UNK_{s}"


class NvOFError(RuntimeError):
    """OFA 任一环节失败统一抛此异常(调用方 catch 后禁用 OF 规则)。"""


def _ck(rc, what):
    if rc != 0:
        raise NvOFError(f"{what} failed rc={rc}")


def _find_cudart():
    """cudart64_12.dll 路径: 优先 PyNvVideoCodec 包内(post8 实测可用),
    否则 System32。找不到返回 None(WinDLL(None) 走默认搜索)。"""
    try:
        import PyNvVideoCodec
        p = os.path.join(os.path.dirname(PyNvVideoCodec.__file__),
                         "cudart64_12.dll")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                     "System32", "cudart64_12.dll")
    return p if os.path.exists(p) else None


# ---------------- cudart(内存/流/事件全走这里, post8 解法) ----------------
class GpuRt:
    """cudart 最小封装 + primary context 句柄(NVOF 会话与拷贝同地址空间)。
    进程级单例(见 _gpu_rt())。"""

    def __init__(self, device_id=0):
        path = _find_cudart()
        try:
            self.rt = WinDLL(path) if path else WinDLL("cudart64_12")
        except OSError as e:
            raise NvOFError(f"cudart64_12.dll 加载失败({path}): {e!r}") from e
        rt = self.rt
        rt.cudaSetDevice.argtypes = [c_int]
        rt.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        rt.cudaFree.argtypes = [c_void_p]
        rt.cudaMemsetAsync.argtypes = [c_void_p, c_int, c_size_t, c_void_p]
        rt.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int,
                                       c_void_p]
        rt.cudaMemcpy2DAsync.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t,
                                         c_size_t, c_size_t, c_int, c_void_p]
        rt.cudaHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, c_int]
        rt.cudaStreamCreate.argtypes = [POINTER(c_void_p)]
        rt.cudaStreamSynchronize.argtypes = [c_void_p]
        # 注意: 不能调 cudaSetDevice —— 实测它会把调用线程的 current ctx
        # 直接【替换】成 primary(不是 push), 把调用方(pyautoinit/TRT)的 ctx
        # 踢出线程, 之后其 kernel launch 全部 rc=400 且 ctx 进 sticky error。
        # 单卡机器 runtime 默认 device 0, 无需 setDevice。
        self.dev_id = device_id
        # primary context 句柄(与 cudart 运行时同一上下文)
        try:
            drv = WinDLL("nvcuda")
        except OSError as e:
            raise NvOFError(f"nvcuda.dll 加载失败: {e!r}") from e
        drv.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        drv.cuCtxPushCurrent.argtypes = [c_void_p]
        drv.cuCtxPopCurrent.argtypes = [POINTER(c_void_p)]
        self.drv = drv
        self.pctx = c_void_p()
        _ck(drv.cuInit(0), "cuInit")
        _ck(drv.cuDevicePrimaryCtxRetain(byref(self.pctx), device_id),
            "cuDevicePrimaryCtxRetain")
        # cudart 遵循「调用线程 current context」: 若调用方(pyautoinit/TRT)
        # 已把别的 ctx 设为 current, 此处创建的流/分配会落到别的 ctx, 之后
        # NVOF/本模块 kernel(primary)launch 该流句柄 → rc=400 INVALID_HANDLE。
        # 因此资源创建必须显式在 primary ctx 下完成(_ctx_enter/_ctx_exit)。
        self._ctx_enter()
        try:
            self.stream = c_void_p()
            _ck(rt.cudaStreamCreate(byref(self.stream)), "cudaStreamCreate")
        finally:
            self._ctx_exit()

    def _ctx_enter(self):
        """保存调用线程 current ctx 并切到 primary。
        注意不能用 cuCtxPushCurrent/cuCtxPopCurrent: 实测(591.86 WDDM)
        cudart 调用(cudaStreamCreate 等)之后再 cuCtxPopCurrent 会变成
        no-op(rc=0 但 current 不恢复), 栈永久停在 primary, 调用方(pyautoinit)
        的后续 kernel launch 全部落到错误 ctx(rc=400)。cuCtxSetCurrent 是
        绝对设置, 不依赖浮点栈, save/restore 可靠。"""
        self.drv.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        self._prev_ctx = c_void_p()
        _ck(self.drv.cuCtxGetCurrent(byref(self._prev_ctx)), "cuCtxGetCurrent")
        _ck(self.drv.cuCtxSetCurrent(self.pctx), "cuCtxSetCurrent")

    def _ctx_exit(self):
        _ck(self.drv.cuCtxSetCurrent(self._prev_ctx), "cuCtxSetCurrent restore")

    @property
    def stream_int(self):
        """cudart 流句柄的 int 值(可传给 driver cuLaunchKernel 的 stream)。"""
        return self.stream.value

    def cuda_malloc(self, nbytes):
        with self.nvof_ctx():
            p = c_void_p()
            _ck(self.rt.cudaMalloc(byref(p), c_size_t(nbytes)), "cudaMalloc")
        return p.value

    def memset_async(self, dev, val, nbytes):
        with self.nvof_ctx():
            _ck(self.rt.cudaMemsetAsync(c_void_p(dev), val, c_size_t(nbytes),
                                        self.stream), "cudaMemsetAsync")

    def memcpy_d2h_async(self, dst_host_bytes, src_dev, nbytes):
        with self.nvof_ctx():
            _ck(self.rt.cudaMemcpyAsync(c_void_p(dst_host_bytes),
                                        c_void_p(src_dev), c_size_t(nbytes),
                                        CUDA_MEMCPY_D2H, self.stream),
                "cudaMemcpyAsync D2H")

    def host_alloc(self, nbytes):
        with self.nvof_ctx():
            p = c_void_p()
            _ck(self.rt.cudaHostAlloc(byref(p), c_size_t(nbytes), 0),
                "cudaHostAlloc")
        return p.value

    @contextlib.contextmanager
    def nvof_ctx(self):
        """NVOF/本模块 kernel(driver API)调用期间保证 primary ctx current
        (save/restore 语义, 见 _ctx_enter 注释; prev 存局部变量 → 可嵌套)。"""
        drv = self.drv
        drv.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        prev = c_void_p()
        _ck(drv.cuCtxGetCurrent(byref(prev)), "cuCtxGetCurrent")
        _ck(drv.cuCtxSetCurrent(self.pctx), "cuCtxSetCurrent")
        try:
            yield
        finally:
            _ck(drv.cuCtxSetCurrent(prev), "cuCtxSetCurrent restore")

    def h2d(self, dst_dev, dst_pitch, src_host, src_pitch, width, height,
            row_offset=0):
        with self.nvof_ctx():
            _ck(self.rt.cudaMemcpy2DAsync(
                c_void_p(dst_dev + row_offset * dst_pitch), dst_pitch,
                src_host, src_pitch, width, height, CUDA_MEMCPY_H2D,
                self.stream), "cudaMemcpy2DAsync H2D")

    def d2h(self, dst_host, src_dev, src_pitch, width, height):
        with self.nvof_ctx():
            _ck(self.rt.cudaMemcpy2DAsync(dst_host, width, c_void_p(src_dev),
                                          src_pitch, width, height,
                                          CUDA_MEMCPY_D2H, self.stream),
                "cudaMemcpy2DAsync D2H")

    def d2d2d(self, dst_dev, dst_pitch, src_dev, src_pitch, width, height,
              row_offset=0):
        with self.nvof_ctx():
            _ck(self.rt.cudaMemcpy2DAsync(
                c_void_p(dst_dev + row_offset * dst_pitch), dst_pitch,
                c_void_p(src_dev), src_pitch, width, height, CUDA_MEMCPY_D2D,
                self.stream), "cudaMemcpy2DAsync D2D")

    def sync(self):
        with self.nvof_ctx():
            _ck(self.rt.cudaStreamSynchronize(self.stream),
                "cudaStreamSynchronize")


_GPU_RT = None


def gpu_rt():
    global _GPU_RT
    if _GPU_RT is None:
        _GPU_RT = GpuRt()
    return _GPU_RT


# ---------------- NVOF API(官方头文件映射, 与 post8 一致) ----------------
W = WINFUNCTYPE
FN_SIGS = [  # NV_OF_CUDA_API_FUNCTION_LIST 12 个函数指针(nvOpticalFlowCuda.h)
    W(c_int, c_void_p, POINTER(c_void_p)),                    # nvCreateOpticalFlowCuda
    W(c_int, c_void_p, c_void_p),                             # nvOFInit
    W(c_int, c_void_p, c_void_p, c_int, POINTER(c_void_p)),   # nvOFCreateGPUBufferCuda
    W(c_void_p, c_void_p),                                    # nvOFGPUBufferGetCUarray
    W(c_uint64, c_void_p),                                    # nvOFGPUBufferGetCUdeviceptr
    W(c_int, c_void_p, c_void_p),                             # nvOFGPUBufferGetStrideInfo
    W(c_int, c_void_p, c_void_p, c_void_p),                   # nvOFSetIOCudaStreams
    W(c_int, c_void_p, c_void_p, c_void_p),                   # nvOFExecute
    W(c_int, c_void_p),                                       # nvOFDestroyGPUBufferCuda
    W(c_int, c_void_p),                                       # nvOFDestroy
    W(c_int, c_void_p, c_char_p, POINTER(c_uint32)),          # nvOFGetLastError
    W(c_int, c_void_p, c_int, POINTER(c_uint32), POINTER(c_uint32)),  # nvOFGetCaps
]


class NV_OF_CUDA_API_FUNCTION_LIST(Structure):
    _fields_ = [(n, f) for n, f in zip(
        ["nvCreateOpticalFlowCuda", "nvOFInit", "nvOFCreateGPUBufferCuda",
         "nvOFGPUBufferGetCUarray", "nvOFGPUBufferGetCUdeviceptr",
         "nvOFGPUBufferGetStrideInfo", "nvOFSetIOCudaStreams", "nvOFExecute",
         "nvOFDestroyGPUBufferCuda", "nvOFDestroy", "nvOFGetLastError",
         "nvOFGetCaps"], FN_SIGS)]


class NV_OF_BUFFER_DESCRIPTOR(Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("bufferUsage", c_int), ("bufferFormat", c_int)]


class NV_OF_BUFFER_STRIDE_INFO(Structure):
    _fields_ = [("stride", (ctypes.c_uint32 * 2) * 3), ("numPlanes", ctypes.c_uint32)]


class NV_OF_INIT_PARAMS_V5(Structure):
    # nvOpticalFlowCommon.h(NvOFInterface, API 5.0)NV_OF_INIT_PARAMS 逐字段
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("outGridSize", c_int), ("hintGridSize", c_int),
                ("mode", c_int), ("perfLevel", c_int),
                ("enableExternalHints", c_int), ("enableOutputCost", c_int),
                ("hPrivData", c_void_p), ("disparityRange", c_int),
                ("enableRoi", c_int), ("predDirection", c_int),
                ("enableGlobalFlow", c_int), ("inputBufferFormat", c_int)]


class NV_OF_INIT_PARAMS_V2(Structure):
    # NVIDIAOpticalFlowSDK/nvOpticalFlowCommon.h(API 2.0)NV_OF_INIT_PARAMS
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("outGridSize", c_int), ("hintGridSize", c_int),
                ("mode", c_int), ("perfLevel", c_int),
                ("enableExternalHints", c_int), ("enableOutputCost", c_int),
                ("hPrivData", c_void_p), ("disparityRange", c_int),
                ("enableRoi", c_int)]


class NV_OF_EXECUTE_INPUT_PARAMS(Structure):
    _fields_ = [("inputFrame", c_void_p), ("referenceFrame", c_void_p),
                ("externalHints", c_void_p), ("disableTemporalHints", c_int),
                ("padding", ctypes.c_uint32), ("hPrivData", c_void_p),
                ("padding2", ctypes.c_uint32), ("numRois", ctypes.c_uint32),
                ("roiData", c_void_p)]


class NV_OF_EXECUTE_OUTPUT_PARAMS_V5(Structure):
    _fields_ = [("outputBuffer", c_void_p), ("outputCostBuffer", c_void_p),
                ("hPrivData", c_void_p), ("bwdOutputBuffer", c_void_p),
                ("bwdOutputCostBuffer", c_void_p), ("globalFlowBuffer", c_void_p)]


class NV_OF_EXECUTE_OUTPUT_PARAMS_V2(Structure):
    _fields_ = [("outputBuffer", c_void_p), ("outputCostBuffer", c_void_p),
                ("hPrivData", c_void_p)]


class NvOfApi:
    """nvofapi64.dll + 12 函数表(按驱动支持的最高版本选结构体布局)。"""

    def __init__(self):
        try:
            self.dll = WinDLL("nvofapi64")
        except OSError as e:
            raise NvOFError(f"nvofapi64.dll 加载失败: {e!r}") from e
        self.dll_path = "C:/Windows/System32/nvofapi64.dll"
        self.dll.NvOFGetMaxSupportedApiVersion.argtypes = [POINTER(ctypes.c_uint)]
        self.dll.NvOFGetMaxSupportedApiVersion.restype = c_int
        v = ctypes.c_uint(0)
        _ck(self.dll.NvOFGetMaxSupportedApiVersion(byref(v)),
            "NvOFGetMaxSupportedApiVersion")
        self.max_ver = v.value
        self.api_ver = 0x50 if self.max_ver >= 0x50 else 0x20
        self.is_v5 = self.api_ver >= 0x50
        self.fn = NV_OF_CUDA_API_FUNCTION_LIST()
        self.dll.NvOFAPICreateInstanceCuda.argtypes = [
            ctypes.c_uint, POINTER(NV_OF_CUDA_API_FUNCTION_LIST)]
        self.dll.NvOFAPICreateInstanceCuda.restype = c_int
        _ck(self.dll.NvOFAPICreateInstanceCuda(self.api_ver, byref(self.fn)),
            "NvOFAPICreateInstanceCuda")

    def ck(self, rc, what, handle=None):
        if rc == NV_OF_SUCCESS:
            return
        msg = ""
        if handle is not None:
            buf = ctypes.create_string_buffer(80)
            n = ctypes.c_uint32(80)
            try:
                if self.fn.nvOFGetLastError(handle, buf, byref(n)) == 0:
                    msg = f": {buf.value.decode(errors='replace')}"
            except Exception:
                pass
        raise NvOFError(f"NVOF {what} failed rc={rc}({st_name(rc)}){msg}")


_NV_OF_API = None


def nvof_api():
    global _NV_OF_API
    if _NV_OF_API is None:
        _NV_OF_API = NvOfApi()
    return _NV_OF_API


class GpuBuffer:
    def __init__(self, api, hof, desc):
        self.h = c_void_p()
        api.ck(api.fn.nvOFCreateGPUBufferCuda(hof, byref(desc),
                                              NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                              byref(self.h)), "CreateGPUBufferCuda",
               hof)
        self.dev = int(api.fn.nvOFGPUBufferGetCUdeviceptr(self.h))
        si = NV_OF_BUFFER_STRIDE_INFO()
        api.ck(api.fn.nvOFGPUBufferGetStrideInfo(self.h, byref(si)),
               "GetStrideInfo", hof)
        self.stride_x = si.stride[0][0]
        self.stride_y = si.stride[0][1]   # 官方 C++ 用它作 UV 平面的行偏移
        self.num_planes = si.numPlanes


class NvOfSession:
    """一个 OFA 会话 = 1 组 (w,h,in_fmt)。2 输入 ping-pong + 1 输出。
    会话挂 primary context(与 cudart 拷贝/本模块 kernel 同一地址空间同一条流,
    读写自然保序)。"""

    def __init__(self, gpu, w, h, in_fmt=NV_OF_BUFFER_FORMAT_GRAYSCALE8,
                 out_grid=4, perf_level=NV_OF_PERF_LEVEL_FAST, disable_th=0):
        self.gpu, self.api, self.w, self.h = gpu, nvof_api(), w, h
        self.in_fmt, self.out_grid, self.dth = in_fmt, out_grid, disable_th
        self.hof = c_void_p()
        with gpu.nvof_ctx():
            self.api.ck(self.api.fn.nvCreateOpticalFlowCuda(gpu.pctx,
                                                            byref(self.hof)),
                        "CreateOpticalFlowCuda")
        self.caps = {}
        for key, cap in [("out_grid", CAPS_OUT_GRID), ("w_min", CAPS_W_MIN),
                         ("h_min", CAPS_W_H_MIN), ("w_max", CAPS_W_MAX),
                         ("h_max", CAPS_H_MAX)]:
            vals = (ctypes.c_uint32 * 8)()
            n = ctypes.c_uint32(8)
            with gpu.nvof_ctx():
                rc = self.api.fn.nvOFGetCaps(self.hof, cap, vals, byref(n))
            self.caps[key] = list(vals[:n.value]) if rc == 0 else None
        with gpu.nvof_ctx():
            self.api.ck(self.api.fn.nvOFSetIOCudaStreams(self.hof,
                                                         gpu.stream,
                                                         gpu.stream),
                        "SetIOCudaStreams", self.hof)
        ip = (NV_OF_INIT_PARAMS_V5() if self.api.is_v5
              else NV_OF_INIT_PARAMS_V2())
        ip.width, ip.height = w, h
        ip.outGridSize, ip.hintGridSize = out_grid, out_grid
        ip.mode = NV_OF_MODE_OPTICALFLOW
        ip.perfLevel = perf_level
        ip.enableExternalHints, ip.enableOutputCost = 0, 0
        ip.hPrivData, ip.disparityRange, ip.enableRoi = None, 0, 0
        if self.api.is_v5:
            ip.predDirection, ip.enableGlobalFlow = 0, 0
            ip.inputBufferFormat = in_fmt
        with gpu.nvof_ctx():
            self.api.ck(self.api.fn.nvOFInit(self.hof, byref(ip)), "nvOFInit",
                        self.hof)
        self.gw, self.gh = (w + out_grid - 1) // out_grid, (h + out_grid - 1) // out_grid
        in_desc = NV_OF_BUFFER_DESCRIPTOR(w, h, NV_OF_BUFFER_USAGE_INPUT, in_fmt)
        out_desc = NV_OF_BUFFER_DESCRIPTOR(self.gw, self.gh,
                                           NV_OF_BUFFER_USAGE_OUTPUT,
                                           NV_OF_BUFFER_FORMAT_SHORT2)
        # buffer 创建也必须挂 primary ctx(与 NVOF 会话同 ctx)
        with gpu.nvof_ctx():
            self.in_buf = [GpuBuffer(self.api, self.hof, in_desc)
                           for _ in range(2)]
            self.out_buf = GpuBuffer(self.api, self.hof, out_desc)
        self.out_host = np.empty(self.gh * self.gw * 4, dtype=np.uint8)
        self.in_params = NV_OF_EXECUTE_INPUT_PARAMS()
        self.out_params = (NV_OF_EXECUTE_OUTPUT_PARAMS_V5() if self.api.is_v5
                           else NV_OF_EXECUTE_OUTPUT_PARAMS_V2())
        self.out_params.outputBuffer = self.out_buf.h

    # ---- NV12 直喂(post8 对账路径: Y 平面 + UV 平面 H2D) ----
    def upload_nv12(self, slot, y_ptr, y_bytes, y_h, uv_ptr, uv_bytes, uv_h):
        b = self.in_buf[slot]
        self.gpu.h2d(b.dev, b.stride_x, y_ptr, y_bytes, y_bytes, y_h)
        self.gpu.h2d(b.dev, b.stride_x, uv_ptr, uv_bytes, uv_bytes, uv_h,
                     row_offset=b.stride_y)

    # ---- GRAY8 device 直写/直拷(生产路径: kernel 写 / D2D 拷) ----
    def input_dev(self, slot):
        """输入 buffer 槽 (device 指针, 行 stride): rgb2gray kernel 直写用。"""
        b = self.in_buf[slot]
        return b.dev, b.stride_x

    def upload_gray_dev(self, slot, src_dev, src_stride, w, h):
        """GRAY8 device→device 拷进输入 buffer(处理行 stride)。"""
        b = self.in_buf[slot]
        self.gpu.d2d2d(b.dev, b.stride_x, src_dev, src_stride, w, h)

    def execute(self, slot_prev, slot_cur):
        self.in_params.inputFrame = self.in_buf[slot_prev].h
        self.in_params.referenceFrame = self.in_buf[slot_cur].h
        self.in_params.disableTemporalHints = self.dth
        self.in_params.numRois, self.in_params.roiData = 0, None
        self.in_params.externalHints = None
        with self.gpu.nvof_ctx():
            rc = self.api.fn.nvOFExecute(self.hof, byref(self.in_params),
                                         byref(self.out_params))
        if rc != NV_OF_SUCCESS:
            self.api.ck(rc, "nvOFExecute", self.hof)

    def download(self):
        """输出矢量 D2H + 流同步, 返回 (gh, gw, 2) int16(S10.5 全分辨率 px)。"""
        b = self.out_buf
        self.gpu.d2h(self.out_host.ctypes.data, b.dev, b.stride_x,
                     self.gw * 4, self.gh)
        self.gpu.sync()
        return self.out_host.view(np.int16).reshape(self.gh, self.gw, 2)

    def destroy(self):
        try:
            for b in self.in_buf:
                self.api.fn.nvOFDestroyGPUBufferCuda(b.h)
            self.api.fn.nvOFDestroyGPUBufferCuda(self.out_buf.h)
            self.api.fn.nvOFDestroy(self.hof)
        except Exception:
            pass


# ---------------- 本模块 kernel(nvof_kernels.cubin) ----------------
class OfKernels:
    """nvof_kernels.cubin: rgb2gray_u8(解码池 RGB → NVOF GRAY8 输入 buffer)
    + of_sig_reduce_u8(grid 矢量 → 7 double 部分和)。在 primary ctx 加载
    (post8 同一地址空间), 发射到 cudart 流(NVOF IO 流, 读写自然保序)。"""

    def __init__(self):
        if not os.path.exists(CUBIN_PATH):
            raise NvOFError(f"nvof_kernels.cubin 不存在({CUBIN_PATH}), "
                            f"先用 nvcc 编译(见文件头注释)")
        gpu = gpu_rt()
        lib = ctypes.CDLL(r"C:\Windows\System32\nvcuda.dll")
        lib.cuModuleLoadData.restype = ctypes.c_int
        lib.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_char_p]
        lib.cuModuleGetFunction.restype = ctypes.c_int
        lib.cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p]
        lib.cuLaunchKernel.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [c_void_p,
                                       ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, c_void_p,
                                       POINTER(c_void_p), c_void_p]
        self.lib = lib
        with gpu.nvof_ctx():
            with open(CUBIN_PATH, "rb") as f:
                image = f.read()
            mod = c_void_p()
            if lib.cuModuleLoadData(byref(mod), image) != 0:
                raise NvOFError("cuModuleLoadData failed (nvof_kernels)")
            self._mod = mod
            self.fn = {}
            for name in (b"rgb2gray_u8", b"of_sig_reduce_u8"):
                fn = c_void_p()
                if lib.cuModuleGetFunction(byref(fn), mod, name) != 0:
                    raise NvOFError(f"cuModuleGetFunction failed: {name!r}")
                self.fn[name.decode()] = fn

    @staticmethod
    def _args(*vals):
        """cuLaunchKernel 参数包: 每槽 = 参数值的地址。vals 临时 ctypes 对象
        必须随数组存活(挂 _keep 防 GC), 否则 launch 时读到悬空地址 →
        非法参数/IllegalAddress(实测踩坑)。"""
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.addressof(v) for v in vals])
        arr._keep = vals
        return arr

    def rgb2gray(self, gpu, src_rgb_dev, dst_dev, dst_stride, H, W):
        """单帧 RGB(H*W*3)→GRAY8 写 NVOF 输入 buffer(行距 dst_stride)。"""
        ci, cp = ctypes.c_int, c_void_p
        grid = ((W + 15) // 16, (H + 15) // 16, 1)
        with gpu.nvof_ctx():
            rc = self.lib.cuLaunchKernel(
                self.fn["rgb2gray_u8"], grid[0], grid[1], grid[2],
                16, 16, 1, 0, gpu.stream_int,
                self._args(cp(src_rgb_dev), cp(dst_dev), ci(H), ci(W),
                           ci(dst_stride)), None)
        if rc != 0:
            raise NvOFError(f"cuLaunchKernel(rgb2gray_u8) failed: {rc}")

    def sig_reduce(self, gpu, flow_dev, gw, gh, stride_bytes, scale, out_dev):
        """grid 矢量 → out_dev[7] double 部分和(atomicAdd; 调用方先清零)。
        stride_bytes = NVOF 输出 buffer 行字节数(非紧凑 gw*4)。"""
        ci, cf, cp = ctypes.c_int, ctypes.c_float, c_void_p
        grid = ((gw + 15) // 16, (gh + 15) // 16, 1)
        with gpu.nvof_ctx():
            rc = self.lib.cuLaunchKernel(
                self.fn["of_sig_reduce_u8"], grid[0], grid[1], grid[2],
                16, 16, 1, 0, gpu.stream_int,
                self._args(cp(flow_dev), ci(gw), ci(gh), ci(stride_bytes),
                           cf(scale), cp(out_dev)), None)
        if rc != 0:
            raise NvOFError(f"cuLaunchKernel(of_sig_reduce_u8) failed: {rc}")


def signal_from_flow(vx_raw, vy_raw, scale):
    """宿主 numpy 信号(post8 同公式, 对账/参考实现; 生产路径走 GPU 归约)。
    vx_raw/vy_raw: grid 矢量 int16(S10.5 全分辨率 px); scale = 256/长边。"""
    fx = vx_raw.astype(np.float32) / 32.0
    fy = vy_raw.astype(np.float32) / 32.0
    bad = (np.abs(fx) > 512.0) | (np.abs(fy) > 512.0)
    fx[bad] = 0.0
    fy[bad] = 0.0
    fx = fx * scale
    fy = fy * scale
    mag = np.hypot(fx, fy)
    fm = float(mag.mean())
    mr = float((mag >= 0.3).mean())
    m = mag >= 0.3
    if int(m.sum()) >= 4:
        mm = mag[m]
        dc = float(np.hypot(fx[m].mean(), fy[m].mean()) / (mm.mean() + 1e-9))
        mcv = float(mm.std() / (mm.mean() + 1e-9))
    else:
        dc, mcv = 0.0, 0.0
    return fm, mr, dc, mcv


def signal_from_sums(sums):
    """GPU 归约部分和(out_dev D2H 的 7 个 double)→ fm/mr/dc/mcv(post8 同公式,
    累加在 device 上用 double 完成, 精度不低于宿主 float32 pairwise)。"""
    n, sum_mag, cnt_m, sum_m, sumsq_m, sum_fx, sum_fy = [float(x) for x in sums]
    if n <= 0:
        return 0.0, 0.0, 0.0, 0.0
    fm = sum_mag / n
    mr = cnt_m / n
    if cnt_m >= 4:
        mm = sum_m / cnt_m
        dc = float(np.hypot(sum_fx / cnt_m, sum_fy / cnt_m) / (mm + 1e-9))
        var = max(0.0, sumsq_m / cnt_m - mm * mm)
        mcv = float(np.sqrt(var) / (mm + 1e-9))
    else:
        dc, mcv = 0.0, 0.0
    return fm, mr, dc, mcv


class NvofEngine:
    """生产门面: 会话按 (w,h) 缓存(换尺寸销毁重建, 毫秒级), 每帧一步
    compute(rgb device 帧 → fm/mr/dc/mcv)。全部操作在调用线程(主线程,
    primary ctx push/pop 包裹)的 cudart 流上完成。"""

    def __init__(self):
        if os.environ.get("SVF_NVOF_FAIL") == "1":
            raise NvOFError("SVF_NVOF_FAIL=1 强制失败(回退演练钩子)")
        self.gpu = gpu_rt()
        self.api = nvof_api()
        self.kern = OfKernels()
        self._sess = None
        self._wh = None
        self._prev_slot = None
        self._out_dev = self.gpu.cuda_malloc(7 * 8)
        self._out_host_ptr = self.gpu.host_alloc(7 * 8)
        self._out_host = (ctypes.c_double * 7).from_address(self._out_host_ptr)

    def ensure(self, w, h):
        if self._sess is not None and self._wh == (w, h):
            return
        if self._sess is not None:
            self._sess.destroy()
            self._sess = None
        self._sess = NvOfSession(self.gpu, w, h,
                                 NV_OF_BUFFER_FORMAT_GRAYSCALE8, 4,
                                 NV_OF_PERF_LEVEL_FAST)
        self._wh = (w, h)
        self._prev_slot = None

    def compute_gray(self, gray_dev, gray_stride, w, h):
        """GRAY8 device 帧输入版本(供对账/分析脚本)。返回 (fm,mr,dc,mcv);
        首帧(无 prev)返回 (0,0,0,0) 并只落 slot。"""
        self.ensure(w, h)
        s = self._sess
        if self._prev_slot is None:
            s.upload_gray_dev(0, gray_dev, gray_stride, w, h)
            self.gpu.sync()
            self._prev_slot = 0
            return 0.0, 0.0, 0.0, 0.0
        cur = 1 - self._prev_slot
        s.upload_gray_dev(cur, gray_dev, gray_stride, w, h)
        s.execute(self._prev_slot, cur)
        self._prev_slot = cur
        return self._reduce_signal(s)

    def compute_rgb(self, rgb_dev, w, h):
        """生产入口: 解码池 RGB device 帧 → rgb2gray kernel 直写 NVOF 输入
        buffer → execute → GPU 归约 → 56B D2H → 宿主换算四标量。
        任一环节失败抛 NvOFError(调用方禁用 OF 规则)。"""
        self.ensure(w, h)
        s = self._sess
        if self._prev_slot is None:
            dev, stride = s.input_dev(0)
            self.kern.rgb2gray(self.gpu, rgb_dev, dev, stride, h, w)
            self.gpu.sync()
            self._prev_slot = 0
            return 0.0, 0.0, 0.0, 0.0
        cur = 1 - self._prev_slot
        dev, stride = s.input_dev(cur)
        self.kern.rgb2gray(self.gpu, rgb_dev, dev, stride, h, w)
        s.execute(self._prev_slot, cur)
        self._prev_slot = cur
        return self._reduce_signal(s)

    def _reduce_signal(self, s):
        self.gpu.memset_async(self._out_dev, 0, 7 * 8)
        self.kern.sig_reduce(self.gpu, s.out_buf.dev, s.gw, s.gh,
                             s.out_buf.stride_x, 256.0 / float(max(s.w, s.h)),
                             self._out_dev)
        self.gpu.memcpy_d2h_async(self._out_host_ptr, self._out_dev, 7 * 8)
        self.gpu.sync()
        return signal_from_sums(self._out_host)

    def reset(self):
        """跨视频复用: 清 ping-pong 状态(下一帧重新当首帧 prev)。
        会话与 buffer 按 (w,h) 保留复用, 不销毁。"""
        self._prev_slot = None

    def close(self):
        if self._sess is not None:
            self._sess.destroy()
            self._sess = None
