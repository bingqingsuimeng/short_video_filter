# -*- coding: utf-8 -*-
"""
眼神方向闸门（级联第二级）：MediaPipe FaceMesh 虹膜中心偏移。

只在 Stage1（角度规则）通过的帧上跑——那些帧头基本正，FaceMesh 检测率高，
虹膜偏移信号干净。整帧直接跑 FaceMesh 检测率低（4K 小脸 40% 漏检），
所以先按 SCRFD 框裁剪 + 放大到长边 640 再跑。

信号：两眼虹膜中心相对眼眶中点(内外眼角中点)的偏移，归一化到眼宽。
正视镜头 ≈ 0；瞟走时横向偏移。mag = |偏移向量|。
实测(buding_018, 93 帧人工标注): 正视帧 mag p90=0.098 / max=0.109,
明显瞟走帧 0.128+。阈值取 0.12。
"""
import os

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except ImportError:
    _HAS_MP = False

CROP_TARGET = 640    # 裁剪后长边
CROP_PAD = 0.6       # SCRFD 框外扩比例

# FaceMesh 478 点（refine_landmarks=True）
LO_OUT, LO_IN, LO_IRIS = 33, 133, 468    # 画面左眼: 外角/内角/虹膜中心
RO_OUT, RO_IN, RO_IRIS = 263, 362, 473   # 画面右眼


class GazeGate:
    def __init__(self, min_det_conf=0.3):
        if not _HAS_MP:
            raise ImportError("mediapipe not installed (pip install mediapipe==0.10.14)")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True,
            min_detection_confidence=min_det_conf)

    @staticmethod
    def _crop(frame, bbox, target=CROP_TARGET, pad=CROP_PAD):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if max(w, h) < 16:
            return None
        pw, ph = int(w * pad), int(h * pad)
        H, W = frame.shape[:2]
        cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
        cx2, cy2 = min(W, x2 + pw), min(H, y2 + ph)
        crop = frame[cy1:cy2, cx1:cx2]
        sc = target / max(crop.shape[:2])
        if abs(sc - 1.0) > 1e-3:
            crop = cv2.resize(crop, (0, 0), fx=sc, fy=sc,
                              interpolation=cv2.INTER_LINEAR)
        return crop

    def estimate(self, frame, bbox):
        """(frame BGR, bbox [x1,y1,x2,y2]) -> (mag, dx, dy) 或 None。
        mag: 虹膜偏移幅度(归一化眼宽)；dx: 横向(正=看向画面右)；
        dy: 纵向(本视频/坐标系中负值≈眼珠略偏上，为常规基线)。"""
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        res = self.mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        out = []
        for out_c, in_c, iris in [(LO_OUT, LO_IN, LO_IRIS),
                                  (RO_OUT, RO_IN, RO_IRIS)]:
            a = np.array([pts[out_c].x, pts[out_c].y])
            b = np.array([pts[in_c].x, pts[in_c].y])
            c = np.array([pts[iris].x, pts[iris].y])
            w = float(np.hypot(*(a - b))) + 1e-9
            ctr = (a + b) / 2.0
            out.append(((c[0] - ctr[0]) / w, (c[1] - ctr[1]) / w))
        dx = float(np.mean([o[0] for o in out]))
        dy = float(np.mean([o[1] for o in out]))
        return float(np.hypot(dx, dy)), dx, dy
