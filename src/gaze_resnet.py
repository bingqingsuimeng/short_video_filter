# gaze_resnet.py — ResNet34 全脸注视角估计（Stage2 眼神闸门的第二候选实现，
# 与 face_gaze.py 的 MediaPipe 虹膜偏移法做 A/B 对比）。
#
# 模型来源：github.com/yakhyo/gaze-estimation（MobileGaze，Gaze360 训练，
# 已并入 github.com/yakhyo/uniface 的 uniface.gaze.MobileGaze）。
# 推理逻辑移植自其 ONNX 版，但运行后端改为 pycuda + TensorRT
# （onnxruntime 的 CUDA/TRT EP 在本机不可用：缺 cublasLt64_12.dll，
#   TRT EP 静默回退 CPU，故弃用 onnxruntime）。
# 引擎用 build_gaze_engine.py 从本地 resnet34_gaze.onnx 构建。
#
# 输入：人脸裁剪（BGR，detector bbox 原样裁剪，不扩展）
#       → BGR2RGB → resize 448x448 → /255 → ImageNet 归一化 → NCHW
# 输出：yaw / pitch 各 90 个角度 bin 的 logits
#       → softmax 概率 × bin 序号加权平均 → ×4° − 180 → ±180° 连续角
# 角度约定（uniface 文档）：
#   pitch：+ = 往上看，- = 往下看
#   yaw  ：+ = 往右看（画面右侧），- = 往左看
# 精度：Gaze360 测试集 MAE 11.33°（ResNet34）。文档的 attention 判定：
#   |pitch| < 15° 且 |yaw| < 15° 视为「看镜头」。
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (初始化 CUDA + context)
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray

INPUT_SIZE = 448
MAX_BATCH = 32       # 动态引擎 profile 上限(构建脚本 --dyn 同值)
_PIPE_GROUP = 32     # 固定 batch=1 引擎流水化 enqueue 的在途深度(每 G 帧一次 sync)
_PREP_WORKERS = 8    # 预处理线程池(cv2 释放 GIL, 与 head letterbox workers 同款)

_GAZE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "gaze")
_FP16_MODEL = os.path.join(_GAZE_DIR, "resnet34_gaze_fp16.engine")
_FP32_MODEL = os.path.join(_GAZE_DIR, "resnet34_gaze.engine")
_DEFAULT_MODELS = (_FP16_MODEL, _FP32_MODEL)


def _default_model_path():
    """默认引擎优先级: FP16 → FP32，按存在性逐级回退。"""
    for p in _DEFAULT_MODELS:
        if os.path.exists(p):
            return p
    return _DEFAULT_MODELS[0]


# 兼容旧引用（指向默认解析函数，不再是固定 FP32 路径）
DEFAULT_MODEL = _default_model_path()


class GazeResNet:
    BINS = 90
    BINWIDTH = 4.0
    ANGLE_OFFSET = 180.0
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet RGB
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path=None, providers=None):
        """model_path: 序列化 TRT 引擎路径；providers 参数仅为兼容旧调用，忽略。"""
        model_path = model_path or _default_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到 TRT 引擎 {model_path}，请先运行 build_gaze_engine.py 构建")
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        # 引擎张量名（onnx 签名：input [1,3,448,448] -> yaw/pitch 各 [1,90]；
        # --dyn 引擎 input [batch,3,448,448]（profile min=1/opt=16/max=32））
        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [self.engine.get_tensor_name(1),
                             self.engine.get_tensor_name(2)]
        assert self.output_names == ["yaw", "pitch"], \
            f"输出节点不符: {self.output_names}"
        in_shape = self.engine.get_tensor_shape(self.input_name)
        b0 = in_shape[0] if in_shape else 1
        # 动态 batch 引擎(--dyn): get_tensor_shape 的动态维返回 -1
        self._dynamic = not (isinstance(b0, int) and b0 > 0)
        if self._dynamic:
            self.max_batch = MAX_BATCH
        else:
            if b0 != 1:
                raise ValueError(
                    f"gaze 引擎 batch={b0} 非 1 且非动态: 批量路径不适用")
            self.max_batch = 1
        self.input_size = (INPUT_SIZE, INPUT_SIZE)             # (w, h)
        self.idx = np.arange(self.BINS, dtype=np.float32)
        self.ep = "TensorRT"
        self._shape_b = 0
        self.in_dev = GPUArray((self.max_batch, 3, INPUT_SIZE, INPUT_SIZE),
                               dtype=np.float32)
        self.out_dev = [GPUArray((self.max_batch, self.BINS), dtype=np.float32)
                        for _ in range(2)]                     # [yaw, pitch]
        self.stream = drv.Stream()
        self.ctx.set_tensor_address(self.input_name, self._ptr(self.in_dev))
        for name, dev in zip(self.output_names, self.out_dev):
            self.ctx.set_tensor_address(name, self._ptr(dev))
        print(f"[GazeResNet-TRT] engine loaded: {os.path.basename(model_path)} "
              f"input={self.input_name} outputs={self.output_names} "
              f"max_batch={self.max_batch}")

    @staticmethod
    def _ptr(arr):
        return arr.__cuda_array_interface__["data"][0]

    def _preprocess(self, crop):
        """BGR 裁剪 → BGR2RGB → resize 448×448 → /255 → (x-mean)/std → NCHW fp32。"""
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32) / 255.0
        img = ((img - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32)

    def _prep(self, frame, bbox):
        """(frame, bbox) → NCHW blob(裁剪过小/无效 → None)。裁剪与预处理式与
        estimate 原实现逐行一致。"""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return self._preprocess(frame[y1:y2, x1:x2])

    def _trt_pass(self, blob):
        """单次推理：上传 NCHW blob → 执行 → 拉回 yaw/pitch logits (1,90) 各一。"""
        if self._dynamic and self._shape_b != 1:
            self.ctx.set_input_shape(self.input_name, (1, 3, INPUT_SIZE, INPUT_SIZE))
            self._shape_b = 1
        drv.memcpy_htod_async(self._ptr(self.in_dev), blob, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        outs = []
        for dev in self.out_dev:
            host = np.empty(self.BINS, dtype=np.float32)
            drv.memcpy_dtoh_async(host, self._ptr(dev), self.stream)
            outs.append(host.reshape(1, self.BINS))           # 与 onnx 输出 (1,90) 一致
        self.stream.synchronize()
        return outs

    def _decode(self, logits):
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return float(p[0] @ self.idx) * self.BINWIDTH - self.ANGLE_OFFSET

    def estimate(self, frame, bbox):
        """frame: 原分辨率 BGR；bbox: (x1,y1,x2,y2) 原分辨率。
        返回 (pitch_deg, yaw_deg)；裁剪过小返回 None。"""
        blob = self._prep(frame, bbox)
        if blob is None:
            return None
        o_yaw, o_pitch = self._trt_pass(blob)
        return self._decode(o_pitch), self._decode(o_yaw)     # (pitch, yaw)

    # ---- 批量推理(C 轮 gaze 段减负): 判定等价(与逐帧 estimate 同值) ----
    def _infer_group_dynamic(self, blobs):
        """动态 batch 引擎: n 个 blob 一次 H2D/execute/D2H + 一次 sync。
        返回 [(yaw_logits(1,90), pitch_logits(1,90))] 与输入同序。"""
        n = len(blobs)
        if self._shape_b != n:
            self.ctx.set_input_shape(self.input_name,
                                     (n, 3, INPUT_SIZE, INPUT_SIZE))
            self._shape_b = n
        blob = np.ascontiguousarray(np.concatenate(blobs, axis=0))
        drv.memcpy_htod_async(self._ptr(self.in_dev), blob, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        hosts = []
        for dev in self.out_dev:
            host = np.empty((n, self.BINS), dtype=np.float32)
            drv.memcpy_dtoh_async(host, self._ptr(dev), self.stream)
            hosts.append(host)
        self.stream.synchronize()
        return [(hosts[0][j:j + 1], hosts[1][j:j + 1]) for j in range(n)]

    def _pipe_slabs(self):
        """流水化后端的常驻 pinned slab(懒建): 输入 _PIPE_GROUP 槽 × (3,448,448)
        fp32 + 输出 _PIPE_GROUP 槽 × (2,90)。在途帧各占独立槽, 覆写要等下一
        组 sync 之后(组内槽不复用), 无竞态。分配失败返回 None(回退逐帧
        np.empty 输出, 逐字节同)。"""
        slab = getattr(self, "_pipe_cache", None)
        if slab is None:
            try:
                in_pin = drv.pagelocked_empty(
                    (_PIPE_GROUP, 3, INPUT_SIZE, INPUT_SIZE), np.float32)
                out_pin = drv.pagelocked_empty(
                    (_PIPE_GROUP, 2, self.BINS), np.float32)
                slab = self._pipe_cache = (in_pin, out_pin)
            except Exception:
                slab = self._pipe_cache = False
        return slab if slab else None

    def _infer_group_pipelined(self, blobs):
        """固定 batch=1 引擎: 逐帧 H2D→execute→D2H 连续入队(_PIPE_GROUP 帧一批,
        批末一次 sync), 替代逐帧 sync。同引擎同 blob 在同一条流上顺序执行,
        每帧计算与逐帧同步版逐位一致(无批量算子, 无批组合效应)。
        blob 先拷入 pinned 输入槽(H2D 真异步, 免 pageable 暂存阻塞)。"""
        slab = self._pipe_slabs()
        hosts = []          # (yaw_host, pitch_host) —— slab 缺席时的回退缓冲
        for k, blob in enumerate(blobs):
            j = k % _PIPE_GROUP
            if slab is not None:
                in_pin, out_pin = slab
                np.copyto(in_pin[j], blob[0])     # 宿主侧进 pinned 槽
                hsrc = in_pin[j]                  # H2D 从 pinned 槽(真异步)
                yaw_h = out_pin[j, 0].reshape(1, self.BINS)
                pit_h = out_pin[j, 1].reshape(1, self.BINS)
            else:
                # pinned 分配失败回退: 可分页 blob 直传(数值逐字节同, 仅慢)
                hsrc = blob
                yaw_h = np.empty((1, self.BINS), np.float32)
                pit_h = np.empty((1, self.BINS), np.float32)
            drv.memcpy_htod_async(self._ptr(self.in_dev), hsrc, self.stream)
            self.ctx.execute_async_v3(self.stream.handle)
            drv.memcpy_dtoh_async(yaw_h, self._ptr(self.out_dev[0]), self.stream)
            drv.memcpy_dtoh_async(pit_h, self._ptr(self.out_dev[1]), self.stream)
            hosts.append((yaw_h, pit_h))
            if len(hosts) == _PIPE_GROUP or k == len(blobs) - 1:
                self.stream.synchronize()
                for yaw_h, pit_h in hosts:
                    yield (yaw_h, pit_h)
                hosts = []

    def estimate_batch(self, items, workers=_PREP_WORKERS):
        """批量眼神推理: items=[(frame BGR, bbox)]，返回与逐帧 estimate 同构的
        [(pitch_deg, yaw_deg) or None]（顺序一致、数值逐位一致 —— 预处理式与
        _decode 均为原逐帧实现, 差异只在推理的入队节奏）。

        后端按引擎自动选择:
          - 动态 batch 引擎(--dyn 构建): 按 max_batch 分组, 组内单次
            H2D/execute/D2H + 一次 sync（head 闸门同款）；
          - 固定 batch=1 引擎: 流水化 enqueue（_PIPE_GROUP 帧在途, 组末一次
            sync）—— CPU 预处理第 k+1 帧与 GPU 执行第 k 帧重叠。
        workers>1: 预处理走线程池（cv2 释放 GIL; 每帧独立 blob 无共享状态,
        与 head letterbox 线程池同款），blob 逐位一致。"""
        n = len(items)
        out = [None] * n
        if n == 0:
            return out
        # 1) 预处理（与 estimate/_prep 逐行同式）
        if workers and workers > 1 and n > 1:
            blobs = [None] * n
            with ThreadPoolExecutor(max_workers=min(int(workers), n)) as ex:
                futs = [ex.submit(self._prep, fr, bb) for fr, bb in items]
                for i, f in enumerate(futs):
                    blobs[i] = f.result()
        else:
            blobs = [self._prep(fr, bb) for fr, bb in items]
        valid = [i for i, b in enumerate(blobs) if b is not None]
        if not valid:
            return out
        # 2) 分组推理（组内结果与输入同序; 逐帧 _decode 复用原实现保位级一致）
        if self._dynamic:
            groups = [valid[g0:g0 + self.max_batch]
                      for g0 in range(0, len(valid), self.max_batch)]
        else:
            groups = [valid[g0:g0 + _PIPE_GROUP]
                      for g0 in range(0, len(valid), _PIPE_GROUP)]
        for grp in groups:
            gb = [blobs[i] for i in grp]
            if self._dynamic:
                res = self._infer_group_dynamic(gb)
            else:
                res = list(self._infer_group_pipelined(gb))
            for j, i in enumerate(grp):
                o_yaw, o_pitch = res[j]
                out[i] = (self._decode(o_pitch), self._decode(o_yaw))
        return out
