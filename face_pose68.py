# -*- coding: utf-8 -*-
"""
1k3d68 68-point 3D face landmark model (TensorRT) + head pose (pitch, yaw, roll).

Port of insightface's official pipeline (insightface/model_zoo/landmark.py,
antelopev2 pack) so the numbers match insightface `face['pose']` exactly:

  crop      : square centred on the face bbox, side = 1.5*max(w,h), no rotation
              (face_align.transform(img, center, 192, scale, 0))
  normalize : RAW [0,255] RGB (this onnx has its input norm baked in — see _to_blob)
  output    : (b, 3309); the LAST 68*3 values are the 68 3D landmarks
              (x,y) -> (x+1)*96, z -> z*96  (crop pixel coords)
              -> original frame coords via the inverse affine of the crop
  pose      : least-squares affine 3D->3D from the mean-shape template
              (meanshape_68.pkl) to the per-frame points, decomposed to
              (pitch, yaw, roll) in degrees.

Only the last 68 points are used; the first 1035 points of the 3309-dim
output are discarded (insightface does the same).

Requires the engine built with:
  trtexec --onnx=1k3d68.onnx --saveEngine=1k3d68_dyn.engine \
    --minShapes=data:1x3x192x192 --optShapes=data:16x3x192x192 \
    --maxShapes=data:32x3x192x192
"""
import math
import os
import pickle

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (initializes CUDA + context)
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray

INPUT_SIZE = 192
_ROOT = os.path.dirname(os.path.abspath(__file__))
MEAN_SHAPE_PATH = os.path.join(_ROOT, "meanshape_68.pkl")


# ---------------- pose math (verbatim from insightface/utils/transform.py) ----
def estimate_affine_matrix_3d23d(X, Y):
    """Least-squares affine 3D->3D: Y = P @ [X, 1].  X: (n,3) fixed, Y: (n,3) moving."""
    X_homo = np.hstack((X, np.ones((X.shape[0], 1))))
    P = np.linalg.lstsq(X_homo, Y)[0].T  # (3,4)
    return P


def P2sRt(P):
    """Decompose affine camera matrix P into (s, R, t)."""
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    """Euler angles (pitch, yaw, roll) in degrees from a rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy < 1e-6:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    else:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    return x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi


def trans_points3d(pts, M):
    """Map (n,3) points through a 2x3 affine M; z scaled by the similarity scale."""
    scale = math.sqrt(M[0][0] ** 2 + M[0][1] ** 2)
    new_pts = np.zeros_like(pts)
    homo = np.hstack([pts[:, 0:2], np.ones((pts.shape[0], 1))])
    new_pts[:, 0:2] = homo @ M.T
    new_pts[:, 2] = pts[:, 2] * scale
    return new_pts


class FacePose68:
    """Batched 1k3d68 head pose on TensorRT.

    estimate(frame, bbox)          -> (pitch, yaw, roll) degrees or None
    estimate_batch([(frame, bbox)]) -> list of poses (one TRT call per <=max_batch)
    """

    def __init__(self, engine_path, max_batch=32):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.max_batch = max_batch
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.in_dev = GPUArray((max_batch, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self.out_dev = GPUArray((max_batch, 3309), dtype=np.float32)
        self.stream = drv.Stream()
        self._shape_b = 1
        self.ctx.set_input_shape(self.input_name, (1, 3, INPUT_SIZE, INPUT_SIZE))
        self.ctx.infer_shapes()
        self.ctx.set_tensor_address(self.input_name, self._ptr(self.in_dev))
        self.ctx.set_tensor_address(self.output_name, self._ptr(self.out_dev))
        with open(MEAN_SHAPE_PATH, "rb") as f:
            self.mean_lmk = np.asarray(pickle.load(f), dtype=np.float64)
        print(f"[Pose68-TRT] engine loaded: {os.path.basename(engine_path)} "
              f"max_batch={max_batch}")

    @staticmethod
    def _ptr(arr):
        return arr.__cuda_array_interface__["data"][0]

    # ---- crop, exactly like insightface face_align.transform(center, 192, s, 0)
    @staticmethod
    def align_crop(frame, bbox):
        """bbox: [x1,y1,x2,y2] in frame coords (float).
        Returns (crop 192x192 BGR, M_frame 2x3 frame->crop) or (None, None)."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if max(w, h) < 8:
            return None, None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(w, h) * 1.5
        s = INPUT_SIZE / side
        M = np.array([[s, 0.0, INPUT_SIZE / 2 - cx * s],
                      [0.0, s, INPUT_SIZE / 2 - cy * s]], dtype=np.float32)
        crop = cv2.warpAffine(frame, M, (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)
        return crop, M

    @staticmethod
    def _to_blob(crop):
        """BGR uint8 192x192 -> float32 CHW in RAW [0,255] (RGB order).

        This 1k3d68.onnx is an mxnet export whose input normalization is baked
        in, so it expects RAW pixel values (insightface's mxnet branch:
        input_mean=0, input_std=1). Verified empirically against the SCRFD
        5-keypoints: raw input spreads the 68 points across the face (kps error
        ~5-7px); any (x-127.5)/128-style normalization collapses them to the
        nose (error ~30-46px, pose garbage). Channel order is immaterial here
        (raw BGR ~ raw RGB); RGB is used to match insightface's swapRB=True."""
        f = crop.astype(np.float32)
        cv2.cvtColor(f, cv2.COLOR_BGR2RGB, dst=f)
        return f.transpose(2, 0, 1)

    def _trt_pass(self, b):
        if b != self._shape_b:
            self.ctx.set_input_shape(self.input_name,
                                     (b, 3, INPUT_SIZE, INPUT_SIZE))
            self.ctx.infer_shapes()
            self._shape_b = b
        self.ctx.execute_async_v3(self.stream.handle)
        host = np.empty(b * 3309, dtype=np.float32)
        drv.memcpy_dtoh_async(host, self._ptr(self.out_dev), self.stream)
        self.stream.synchronize()
        return host.reshape(b, 3309)

    def estimate_batch(self, items):
        """items: list of (frame BGR, bbox). Returns list of ((yaw, pitch, roll),
        ear) — pose in degrees + Eye Aspect Ratio (min of both eyes, ~0.30-0.39
        open, <0.25 closed) — or None where the crop was invalid / fit degenerate."""
        poses = [None] * len(items)
        for start in range(0, len(items), self.max_batch):
            sub = items[start:start + self.max_batch]
            b = len(sub)
            blobs = np.empty((b, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            Ms = [None] * b
            for i, (frame, bbox) in enumerate(sub):
                crop, M = self.align_crop(frame, bbox)
                if crop is None:
                    continue
                Ms[i] = M
                blobs[i] = self._to_blob(crop)
            # only infer the valid ones (keep a compact list)
            valid = [i for i in range(b) if Ms[i] is not None]
            if not valid:
                continue
            vb = len(valid)
            drv.memcpy_htod_async(self._ptr(self.in_dev),
                                  np.ascontiguousarray(blobs[valid]), self.stream)
            out = self._trt_pass(vb)
            half = INPUT_SIZE // 2
            for j, i in enumerate(valid):
                # last 68 points of the 1103 (3309/3) predicted points
                pts = out[j].reshape(-1, 3)[-68:].astype(np.float64)
                pts[:, 0:2] = (pts[:, 0:2] + 1) * half
                pts[:, 2] *= half
                IM = cv2.invertAffineTransform(Ms[i])  # crop -> frame
                pts = trans_points3d(pts, IM)
                try:
                    P = estimate_affine_matrix_3d23d(self.mean_lmk, pts)
                    s, R, _ = P2sRt(P)
                    if s < 1e-9:
                        continue
                    poses[start + i] = (matrix2angle(R), self.eye_aspect(pts))
                except Exception:
                    continue
        return poses

    def estimate(self, frame, bbox):
        return self.estimate_batch([(frame, bbox)])[0]

    @staticmethod
    def eye_aspect(pts):
        """68 个 2D 点 (68,2) -> min(EAR左眼, EAR右眼)。

        iBUG: 36-41 = 画面左眼, 42-47 = 画面右眼。
        EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)：睁眼 ~0.30-0.39，闭眼 <0.2。
        取两眼最小值——任一眼闭即压低。
        """
        el = (np.linalg.norm(pts[37] - pts[41]) + np.linalg.norm(pts[38] - pts[40])) \
             / (2.0 * np.linalg.norm(pts[36] - pts[39]) + 1e-9)
        er = (np.linalg.norm(pts[43] - pts[47]) + np.linalg.norm(pts[44] - pts[46])) \
             / (2.0 * np.linalg.norm(pts[42] - pts[45]) + 1e-9)
        return float(min(el, er))

    def landmarks2d(self, frame, bbox):
        """单脸 68 个 2D 关键点（原图像素坐标），返回 (68,2) float64 或 None。

        走与 estimate 完全相同的裁剪/归一化/TRT/逆仿射链路，只取 x,y——
        用于可视化核对关键点落点。
        """
        crop, M = self.align_crop(frame, bbox)
        if crop is None:
            return None
        arr = np.ascontiguousarray(self._to_blob(crop))[np.newaxis, ...]  # (1,3,192,192)
        drv.memcpy_htod_async(self._ptr(self.in_dev), arr, self.stream)
        out = self._trt_pass(1)
        pts = out[0].reshape(-1, 3)[-68:].astype(np.float64)
        pts[:, 0:2] = (pts[:, 0:2] + 1.0) * 96.0
        pts[:, 2] = pts[:, 2] * 96.0
        IM = cv2.invertAffineTransform(M)
        pts = trans_points3d(pts, IM)
        return pts[:, 0:2]


# ---------------- self test ----------------
if __name__ == "__main__":
    import sys
    import time
    root = os.path.dirname(os.path.abspath(__file__))
    engine = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "1k3d68_dyn.engine")
    det_engine = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "scrfd_500m_bnkps_batch32.engine")
    img_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "_test", "frame_zhao_yi_lin_010.jpg")

    from face_det import SCRFDTRTDetector, estimate_pose

    det = SCRFDTRTDetector(det_engine, max_batch=4, conf_thres=0.5)
    p68 = FacePose68(engine)
    img = cv2.imread(img_path)
    dets, kpss = det.detect([img], threshold=0.5)
    if len(dets[0]) == 0:
        print("!! SELF-TEST FAILED: no face detected !!")
        sys.exit(1)

    print(f"\nimage {img_path}")
    for k, (box, kps) in enumerate(zip(dets[0], kpss[0])):
        box4 = box[:4]
        # 5-point solvePnP pose (old, noisy) for comparison
        old = estimate_pose(kps, box4)
        new = p68.estimate(img, box4)
        (nyaw, npitch, nroll), n_ear = new
        print(f"  box={np.round(box4).astype(int).tolist()}")
        print(f"    5pt  solvePnP : yaw={old[0]:7.2f} pitch={old[1]:7.2f} roll={old[2]:7.2f}")
        print(f"    68pt 3D model: yaw={nyaw:7.2f} pitch={npitch:7.2f} roll={nroll:7.2f}  ear={n_ear:.3f}")

        # validation: predicted 68 3D pts projected to 2D should land near the
        # SCRFD 5 keypoints (same frame coords) -> catches crop/normalization/
        # channel bugs
        crop, M = FacePose68.align_crop(img, box4)
        blob = FacePose68._to_blob(crop)[None]
        drv.memcpy_htod_async(p68._ptr(p68.in_dev), np.ascontiguousarray(blob), p68.stream)
        out = p68._trt_pass(1)
        pts = out[0].reshape(-1, 3)[-68:].astype(np.float64)
        pts[:, 0:2] = (pts[:, 0:2] + 1) * (INPUT_SIZE // 2)
        pts[:, 2] *= INPUT_SIZE // 2
        pts = trans_points3d(pts, cv2.invertAffineTransform(M))
        # 68-pt order: nose tip=30, right eye 36-41 (person's right = image LEFT),
        # left eye 42-47 (image RIGHT), mouth corners 48/54.
        # SCRFD kps: 0 left_eye (image-left), 1 right_eye (image-right), 2 nose,
        # 3/4 mouth.  -> pair by image side: pt38<->kp0, pt44<->kp1.
        pairs = [(30, 2), (38, 0), (44, 1), (48, 3), (54, 4)]
        errs = []
        for pi, ki in pairs:
            dx = pts[pi, 0] - kps[ki, 0]
            dy = pts[pi, 1] - kps[ki, 1]
            errs.append(math.hypot(dx, dy))
        print(f"    2D 2D check (68pt proj vs SCRFD kps): "
              f"nose={errs[0]:.1f}px eyesL/R={errs[1]:.1f}/{errs[2]:.1f} "
              f"mouth={errs[3]:.1f}/{errs[4]:.1f}  (all < ~10px = OK)")

    # throughput
    n = 64
    t0 = time.time()
    p68.estimate_batch([(img, dets[0][0, :4])] * n)
    dt = (time.time() - t0) / n
    print(f"\nthroughput: {n} faces in {n/dt:.0f} face/s ({dt*1000:.2f} ms/face)")
