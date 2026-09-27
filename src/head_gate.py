# -*- coding: utf-8 -*-
"""
head_gate.py — Stage1.5 人头检测闸门（head2 单类 head，TensorRT）

对「其它闸门全过、且单人脸」的帧跑 head2 人头检测：
画面里 NMS 后 head 总数 >= 2（含主角的头）→ 判 multi_head 丢弃。
自拍单帧主角只有 1 个头，冒出第 2 个头即画面里还有别人。

引擎（默认优先动态 batch 引擎）:
  models/head/model_dyn.engine  输入 images (batch,3,640,640) fp32 NCHW RGB,
                          profile min=1/opt=16/max=32（TRT 11 无全局 FP16 flag → FP32）
  models/head/model.engine      旧固定 batch=1 引擎（dyn 缺失时自动回退）
  输出: (batch,5,8400) fp32 = cx, cy, w, h (640 输入像素), conf (已 sigmoid)
预处理: BGR→RGB → 保持宽高比居中 letterbox 到 640×640 (pad 值 114) → /255 → NCHW
解码:   conf>=阈值 → cxcywh→xyxy → 减 pad 除 scale 映射回原图 → NMS(IoU 0.6)

批量推理: detect_batch(frames) 一次 execute 跑满一个 batch（超 max 自动切片
多次），每图各自 letterbox 的 scale/pad 各自还原坐标；detect(frame) 即
detect_batch([frame])[0]，视频逐帧路径行为不变。

推理核心移植自 _test/head2_final.py / _test/head2_test.py（已验证），
pycuda 精简版适配: drv.Stream() + stream.handle(int) 传 execute_async_v3，
设备指针用 GPUArray.__cuda_array_interface__["data"][0]。
"""
import os

import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray

_S = 640
_NMS_IOU = 0.6
_OUT_C, _OUT_HW = 5, 8400   # 输出 (batch, 5, 8400)
_HEAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "head")
_DYN_ENGINE = os.path.join(_HEAD_DIR, "model_dyn.engine")
_LEGACY_ENGINE = os.path.join(_HEAD_DIR, "model.engine")


def _default_engine_path():
    """优先动态 batch 引擎 model_dyn.engine，缺失回退固定 batch=1 的 model.engine。"""
    return _DYN_ENGINE if os.path.exists(_DYN_ENGINE) else _LEGACY_ENGINE


# 兼容旧引用
_DEFAULT_ENGINE = _LEGACY_ENGINE


def _nms(boxes, scores, thresh=0.6):
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


def _letterbox_rgb(img):
    """BGR uint8 HWC → (blob(1,3,640,640) fp32 连续, scale, pad_top, pad_left)"""
    h, w = img.shape[:2]
    scale = min(_S / w, _S / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pt, pl = (_S - nh) // 2, (_S - nw) // 2
    canvas = np.full((_S, _S, 3), 114, dtype=np.uint8)
    cv2.resize(img, (nw, nh), dst=canvas[pt:pt + nh, pl:pl + nw])
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)   # 引擎要 RGB 通道序
    blob = np.ascontiguousarray((canvas.astype(np.float32) / 255.0)
                                .transpose(2, 0, 1)[None])
    return blob, scale, pt, pl


class HeadGate:
    """head2 人头检测闸门。加载失败在 __init__ 里直接 raise，由上层 try/except
    捕获后关闭闸门。detect_batch(frames) 返回每张一个
    list[(x1,y1,x2,y2,conf)]（原图坐标, NMS IoU 0.6, conf 降序）。"""

    def __init__(self, model_path=None, conf_thresh=0.30):
        engine_path = model_path or _default_engine_path()
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"head2 engine not found: {engine_path}")
        self.conf_thresh = float(conf_thresh)
        logger = trt.Logger(trt.Logger.Severity.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        in_shape = self.engine.get_tensor_shape(self.in_name)
        out_shape = self.engine.get_tensor_shape(self.out_name)
        assert tuple(in_shape[1:]) == (3, _S, _S), in_shape
        assert tuple(out_shape[1:]) == (_OUT_C, _OUT_HW), out_shape
        # 引擎 profile 的 max batch：动态引擎=32，固定引擎=1
        # （TRT 11 绑定: engine.get_tensor_profile_shape(name, 0) -> [min,opt,max]）
        try:
            if self.engine.num_optimization_profiles > 0:
                _min_d, _opt_d, max_d = self.engine.get_tensor_profile_shape(
                    self.in_name, 0)
                self.max_batch = max(1, int(max_d[0]))
            else:
                raise ValueError("no optimization profile")
        except Exception:
            b = int(in_shape[0])
            self.max_batch = 1 if b in (-1, 0) else b
        # 动态引擎每次 execute 前须 set_input_shape；固定引擎不能/不必设
        self._dynamic = (in_shape[0] in (-1, 0)) or (self.max_batch > 1)
        # 按 max batch 一次性分配设备/主机缓冲；实际 n<max 时只用前 n 份
        # （C 连续分配，头部 n 个元素即所批数据）
        self.in_dev = GPUArray((self.max_batch, 3, _S, _S), dtype=np.float32)
        self.out_dev = GPUArray(
            (self.max_batch, _OUT_C, _OUT_HW), dtype=np.float32)
        self.host_out = np.empty(
            (self.max_batch, _OUT_C, _OUT_HW), dtype=np.float32)
        self.stream = drv.Stream()
        self.ctx.set_tensor_address(self.in_name, self._ptr(self.in_dev))
        self.ctx.set_tensor_address(self.out_name, self._ptr(self.out_dev))
        print(f"[head_gate] head2 engine ok: {os.path.basename(engine_path)} "
              f"conf={self.conf_thresh} max_batch={self.max_batch}")

    @staticmethod
    def _ptr(a):
        return a.__cuda_array_interface__["data"][0]

    def detect_batch(self, frames_bgr):
        """frames_bgr: list[HWC BGR uint8]。返回与输入等长的 list，每项
        list[(x1,y1,x2,y2,conf)] 原图坐标（每图用自己的 letterbox scale/pad
        还原 + NMS IoU 0.6，conf 降序）。内部按引擎 max_batch 分批，超量切片
        多次 execute。"""
        n = len(frames_bgr)
        results = []
        for start in range(0, n, self.max_batch):
            chunk = frames_bgr[start:start + self.max_batch]
            m = len(chunk)
            blobs = np.empty((m, 3, _S, _S), dtype=np.float32)
            params = []
            for i, fr in enumerate(chunk):
                b, scale, pt, pl = _letterbox_rgb(fr)
                blobs[i] = b[0]
                params.append((scale, pt, pl, fr.shape[0], fr.shape[1]))
            if self._dynamic:
                self.ctx.set_input_shape(self.in_name, (m, 3, _S, _S))
            drv.memcpy_htod_async(self._ptr(self.in_dev), blobs, self.stream)
            self.ctx.execute_async_v3(self.stream.handle)
            drv.memcpy_dtoh_async(self.host_out.reshape(-1)[:m * _OUT_C * _OUT_HW],
                                  self._ptr(self.out_dev), self.stream)
            self.stream.synchronize()
            # 必须 copy: host_out 是复用 D2H 缓冲区, 返回视图会被下一次推理覆盖
            raw = self.host_out[:m].copy()
            for i, (scale, pt, pl, ih, iw) in enumerate(params):
                results.append(self._decode_one(raw[i], scale, pt, pl, ih, iw))
        return results

    def _decode_one(self, raw, scale, pt, pl, img_h, img_w):
        """单图 raw (5,8400) → list[(x1,y1,x2,y2,conf)] 原图坐标。
        用该图自己的 scale/pad 还原: (cxcywh 640 输入系) 减 pad 除 scale。"""
        cx, cy, bw, bh, sc = raw[0], raw[1], raw[2], raw[3], raw[4]
        m = sc >= self.conf_thresh
        if not m.any():
            return []
        xs, ys, ws, hs, ss = cx[m], cy[m], bw[m], bh[m], sc[m]
        xyxy = np.stack([xs - ws / 2 - pl, ys - hs / 2 - pt,
                         xs + ws / 2 - pl, ys + hs / 2 - pt], axis=1) / scale
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, img_w)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, img_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, img_h)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, img_h)
        keep = _nms(xyxy, ss, _NMS_IOU)
        out = np.concatenate([xyxy[keep], ss[keep, None]], axis=1)
        out = out[np.argsort(-out[:, 4])]   # conf 降序
        return [(float(x1), float(y1), float(x2), float(y2), float(s))
                for (x1, y1, x2, y2, s) in out]

    def detect(self, frame_bgr):
        """frame_bgr: HWC BGR uint8。单帧入口（视频逐帧路径沿用），
        等价 detect_batch([frame])[0]。返回 list[(x1,y1,x2,y2,conf)] 原图坐标。"""
        return self.detect_batch([frame_bgr])[0]
