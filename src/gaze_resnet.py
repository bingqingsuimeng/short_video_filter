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

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (初始化 CUDA + context)
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray

INPUT_SIZE = 448

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "gaze", "resnet34_gaze.engine")


class GazeResNet:
    BINS = 90
    BINWIDTH = 4.0
    ANGLE_OFFSET = 180.0
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet RGB
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path=None, providers=None):
        """model_path: 序列化 TRT 引擎路径；providers 参数仅为兼容旧调用，忽略。"""
        model_path = model_path or DEFAULT_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到 TRT 引擎 {model_path}，请先运行 build_gaze_engine.py 构建")
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        # 引擎张量名（onnx 签名：input [1,3,448,448] -> yaw/pitch 各 [1,90]）
        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [self.engine.get_tensor_name(1),
                             self.engine.get_tensor_name(2)]
        assert self.output_names == ["yaw", "pitch"], \
            f"输出节点不符: {self.output_names}"
        self.input_size = (INPUT_SIZE, INPUT_SIZE)             # (w, h)
        self.idx = np.arange(self.BINS, dtype=np.float32)
        self.ep = "TensorRT"
        self.in_dev = GPUArray((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self.out_dev = [GPUArray((1, self.BINS), dtype=np.float32)
                        for _ in range(2)]                     # [yaw, pitch]
        self.stream = drv.Stream()
        self.ctx.set_tensor_address(self.input_name, self._ptr(self.in_dev))
        for name, dev in zip(self.output_names, self.out_dev):
            self.ctx.set_tensor_address(name, self._ptr(dev))
        print(f"[GazeResNet-TRT] engine loaded: {os.path.basename(model_path)} "
              f"input={self.input_name} outputs={self.output_names}")

    @staticmethod
    def _ptr(arr):
        return arr.__cuda_array_interface__["data"][0]

    def _preprocess(self, crop):
        """BGR 裁剪 → BGR2RGB → resize 448×448 → /255 → (x-mean)/std → NCHW fp32。"""
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32) / 255.0
        img = ((img - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32)

    def _trt_pass(self, blob):
        """单次推理：上传 NCHW blob → 执行 → 拉回 yaw/pitch logits (1,90) 各一。"""
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
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        blob = self._preprocess(frame[y1:y2, x1:x2])
        o_yaw, o_pitch = self._trt_pass(blob)
        return self._decode(o_pitch), self._decode(o_yaw)     # (pitch, yaw)
