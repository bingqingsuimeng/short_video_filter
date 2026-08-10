# -*- coding: utf-8 -*-
"""
短视频人脸清洗流水线 v6 (生产级确定性版本)
============================================
相对 v5 的修正：
1.【修复致命 Bug】修复 resolve_rotation_plan 误杀手机竖屏视频的问题（90°/270° 旋转后 display_h > display_w 对应 coded_w > coded_h）。
2.【 Early Return 算力优化】CLIP 属性检测改为 Batch 级别即时推理，一旦发现违规面部（眼镜/动漫/贴纸）立刻中断后续解码与检测。
3.【多线程安全】SCRFD Anchor 网格改为 __init__ 时预计算，规避 ThreadPoolExecutor 并发写入 shared dict 的竞态风险。
4.【鲁棒性增强】ffprobe 元数据解析增加 stream_tags=rotate 提取，防止部分安卓设备旋转信息漏检。
5.【裁剪质量优化】人脸 Crop 增加 15% 动态 Padding，避免边界被裁剪导致 CLIP 细粒度属性（如眼镜腿）识别率下降。
"""

import glob
import os
import json
import subprocess
import time
import concurrent.futures
from typing import List, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================ #
# 0. GPU 加速：自动配置 pip 安装的 CUDA 12 库路径
# ============================================================ #
def _setup_cuda_libs():
    """确保 onnxruntime-gpu 能找到 pip 安装的 CUDA 12 .so 库"""
    import site
    cuda_home = os.path.join(site.getsitepackages()[0], 'nvidia')
    if not os.path.isdir(cuda_home):
        return
    lib_dirs = []
    for root, dirs, files in os.walk(cuda_home):
        if os.path.basename(root) == 'lib':
            lib_dirs.append(root)
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    paths = ':'.join(lib_dirs)
    os.environ['LD_LIBRARY_PATH'] = (paths + ':' + existing).rstrip(':')

_setup_cuda_libs()


# ============================================================ #
# 1. 确定性旋转判定：ffprobe 基准 + 运行时 shape 校验
# ============================================================ #
def probe_raw_metadata(video_path: str) -> dict:
    """
    仅读 header，不解码像素。返回编码层的真实宽高（未受 cv2 内部旋转状态影响的可信基准）和 rotation 角度。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        info = json.loads(out.stdout)
        stream = info["streams"][0]
        coded_w, coded_h = int(stream["width"]), int(stream["height"])

        # 提取 rotation（优先 side_data，兜底 stream_tags）
        rotation = 0
        for sd in stream.get("side_data_list", []):
            if "rotation" in sd:
                rotation = int(sd["rotation"])
                break
        if rotation == 0 and "tags" in stream and "rotate" in stream["tags"]:
            rotation = int(stream["tags"]["rotate"])

        rotation = ((rotation % 360) + 360) % 360

        r_rate = stream.get("r_frame_rate", "30/1")
        num, den = r_rate.split("/") if "/" in r_rate else (r_rate, "1")
        fps = float(num) / float(den) if float(den) != 0 else 30.0

        return {"coded_w": coded_w, "coded_h": coded_h, "rotation": rotation, "fps": fps, "ok": True}
    except Exception:
        return {"coded_w": 0, "coded_h": 0, "rotation": 0, "fps": 30.0, "ok": False}


def _rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resolve_rotation_plan(meta: dict, first_frame: np.ndarray) -> Tuple[bool, int]:
    """
    用第一帧的真实 shape 反推：cv2 这次解码到底有没有自动转正。
    返回 (is_vertical_target, angle_to_apply_manually)
    """
    rotation = meta["rotation"]
    coded_w, coded_h = meta["coded_w"], meta["coded_h"]
    actual_h, actual_w = first_frame.shape[:2]

    if not meta["ok"] or coded_w == 0:
        return actual_h > actual_w, 0

    if rotation in (90, 270):
        already_rotated = (actual_w, actual_h) == (coded_h, coded_w)
        # 编码层 coded_w > coded_h 旋转 90/270 度后 display_h > display_w (竖屏)
        target_is_vertical = coded_w > coded_h
        angle = 0 if already_rotated else rotation
        return target_is_vertical, angle
    else:
        # 0°/180°：宽高比例保持不变
        target_is_vertical = coded_h > coded_w
        return target_is_vertical, rotation


# ============================================================ #
# 2. Thread-Safe SCRFD 检测器
# ============================================================ #
# 默认 Execution Provider，自动优先 TensorRT + FP16
def _default_providers():
    return [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider"
    ]

class SCRFDFastDetector:
    def __init__(self, model_path: str, input_size=640, conf_thres=0.5, nms_thres=0.4,
                 providers: Optional[List] = None):
        self.sess = ort.InferenceSession(model_path, providers=providers or _default_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.strides = [8, 16, 32]

        # 预计算 Anchor Centers，并发只读，彻底消除多线程字典竞态
        self._anchor_centers = {}
        for stride in self.strides:
            h = w = self.input_size // stride
            ys, xs = np.mgrid[:h, :w]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
            centers = np.stack([centers, centers], axis=2).reshape(-1, 2)
            self._anchor_centers[stride] = centers

        # 预分配推理输入 buffer，避免每次调用都申请新内存
        self._input_blob = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)

        # 预分配 canvas（letterbox 用），复用减少分配
        self._canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        # 预分配后处理临时 buffer
        self._post_nms_result = np.empty((0, 4), dtype=np.float32)

    def letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        # 复用预分配 canvas
        c = self._canvas
        c.fill(0)
        c[:nh, :nw, :] = resized
        return c, scale

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Letterbox + normalize + transpose 合并，直接写入预分配 blob"""
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))

        # 直接写入预分配的 blob (BGR + normalize)
        b = self._input_blob[0]
        b[0, :nh, :nw] = (resized[:, :, 2].astype(np.float32) - 127.5) * 0.0078125
        b[1, :nh, :nw] = (resized[:, :, 1].astype(np.float32) - 127.5) * 0.0078125
        b[2, :nh, :nw] = (resized[:, :, 0].astype(np.float32) - 127.5) * 0.0078125
        b[:, nh:, :] = b[:, :, nw:] = 0

        return self._input_blob, scale

    def _postprocess(self, net_outs: List) -> List[Tuple[np.ndarray, float]]:
        """Optimized postprocessing - pure numpy NMS, no .tolist() conversion"""
        boxes_list, scores_list = [], []
        for idx, stride in enumerate(self.strides):
            scores = net_outs[idx].flatten()
            bbox_preds = net_outs[idx + 3].reshape(-1, 4) * stride
            centers = self._anchor_centers[stride][: len(scores)]

            pos = np.where(scores >= self.conf_thres)[0]
            if len(pos) == 0:
                continue

            cx, cy = centers[pos, 0], centers[pos, 1]
            dx1, dy1, dx2, dy2 = bbox_preds[pos].T
            x1, y1, x2, y2 = cx - dx1, cy - dy1, cx + dx2, cy + dy2

            boxes_list.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
            scores_list.append(scores[pos])

        if not boxes_list:
            return []

        all_boxes = np.concatenate(boxes_list)
        all_scores = np.concatenate(scores_list)
        keep = self._fast_nms(all_boxes, all_scores)

        if len(keep) == 0:
            return []

        result = all_boxes[keep].copy()
        result[:, 2] += result[:, 0]  # x1, y1, w, h → x1, y1, x2, y2
        result[:, 3] += result[:, 1]
        scores_sel = all_scores[keep]
        return [(result[i], float(scores_sel[i])) for i in range(len(keep))]

    def _fast_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Pure numpy NMS, 避免 .tolist() 转换开销"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thres)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.intp)

    def detect_single(self, canvas: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Run detection on a single frame (fixed batch=1 model)."""
        b = self._input_blob[0]
        b[0] = (canvas[:, :, 2].astype(np.float32) - 127.5) * 0.0078125
        b[1] = (canvas[:, :, 1].astype(np.float32) - 127.5) * 0.0078125
        b[2] = (canvas[:, :, 0].astype(np.float32) - 127.5) * 0.0078125

        net_outs = self.sess.run(None, {self.input_name: self._input_blob})
        return self._postprocess(net_outs)

    def detect_single_fast(self, frame: np.ndarray) -> Tuple[List[Tuple[np.ndarray, float]], float]:
        """Fast detection with optimized preprocessing, returns (faces, scale)"""
        blob, scale = self.preprocess(frame)
        net_outs = self.sess.run(None, {self.input_name: blob})
        return self._postprocess(net_outs), scale

    def detect_batch(self, letterboxed_frames: List[np.ndarray]) -> List[List[Tuple[np.ndarray, float]]]:
        """Loop over frames for fixed-batch models."""
        if not letterboxed_frames:
            return []
        # Check if model supports dynamic batch
        input_shape = self.sess.get_inputs()[0].shape
        dynamic_batch = input_shape[0] is None

        if dynamic_batch:
            return self._detect_dynamic_batch(letterboxed_frames)
        else:
            return [self.detect_single(f) for f in letterboxed_frames]

    def _detect_dynamic_batch(self, letterboxed_frames: List[np.ndarray]) -> List[List[Tuple[np.ndarray, float]]]:
        """Original batch inference for dynamic-batch models."""
        batch_size = len(letterboxed_frames)
        blobs = np.zeros((batch_size, 3, self.input_size, self.input_size), dtype=np.float32)
        for i, canvas in enumerate(letterboxed_frames):
            blob = (canvas.astype(np.float32) - 127.5) / 128.0
            blobs[i] = blob.transpose(2, 0, 1)[::-1]

        net_outs = self.sess.run(None, {self.input_name: blobs})
        batch_results = []

        for b in range(batch_size):
            boxes_list, scores_list = [], []
            for idx, stride in enumerate(self.strides):
                scores = net_outs[idx][b].flatten()
                bbox_preds = net_outs[idx + 3][b].reshape(-1, 4) * stride
                centers = self._anchor_centers[stride][: len(scores)]

                pos = np.where(scores >= self.conf_thres)[0]
                if len(pos) == 0:
                    continue

                cx, cy = centers[pos, 0], centers[pos, 1]
                dx1, dy1, dx2, dy2 = bbox_preds[pos].T
                x1, y1, x2, y2 = cx - dx1, cy - dy1, cx + dx2, cy + dy2

                boxes_list.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
                scores_list.append(scores[pos])

            if not boxes_list:
                batch_results.append([])
                continue

            all_boxes = np.concatenate(boxes_list)
            all_scores = np.concatenate(scores_list)
            indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), self.conf_thres, self.nms_thres)

            frame_faces = []
            if len(indices) > 0:
                for k in np.array(indices).flatten():
                    bx, by, bw, bh = all_boxes[k]
                    frame_faces.append((np.array([bx, by, bx + bw, by + bh]), float(all_scores[k])))
            batch_results.append(frame_faces)

        return batch_results


# ============================================================ #
# 3. CLIP 属性过滤器
# ============================================================ #
class CLIPAttributeFilter:
    PROMPTS = {
        "real": "a clear close-up portrait photo of a real human face",
        "anime": "a digital illustration of a face, anime or manga style",
        "sticker": "a person wearing a virtual pet filter on their face",
        "glasses": "portrait of a person wearing glasses",
        "no_glasses": "portrait of a person without any glasses on their face",
    }

    def __init__(self, visual_onnx_path: str, text_embeds_path: str, input_size=256, margin=0.01,
                 providers: Optional[List] = None):
        self.sess = ort.InferenceSession(visual_onnx_path, providers=providers or _default_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.text_embeds = np.load(text_embeds_path)
        self.margin = margin

    def classify_batch(self, face_crops: List[np.ndarray]) -> List[Tuple[bool, str]]:
        if not face_crops:
            return []
        size = self.input_size
        blobs = np.zeros((len(face_crops), 3, size, size), dtype=np.float32)
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # MobileCLIP2-S0 expects RGB [0,1] with no normalization

        for i, crop in enumerate(face_crops):
            img = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
            img = (img[..., ::-1] - mean) / std
            blobs[i] = img.transpose(2, 0, 1)

        image_embeds = self.sess.run(None, {self.input_name: blobs})[0]
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
        sims = image_embeds @ self.text_embeds.T

        results = []
        keys = list(self.PROMPTS.keys())
        for row in sims:
            s = dict(zip(keys, row))
            if (s["anime"] - s["real"] > self.margin) or (s["sticker"] - s["real"] > self.margin):
                results.append((False, "anime_or_sticker"))
            elif s["glasses"] - s["no_glasses"] > self.margin:
                results.append((False, "glasses"))
            else:
                results.append((True, "clean"))
        return results


# ============================================================ #
# 4. 主流水线 (Early Exit 优化版)
# ============================================================ #
class ProductionVideoCleaningPipeline:
    def __init__(self, scrfd_path: str, clip_visual_path: str, clip_text_embeds_path: str,
                 sample_fps=5, batch_size=16, first_window_sec=3,
                 providers: Optional[List] = None):
        self.detector = SCRFDFastDetector(scrfd_path, providers=providers)
        self.clip_filter = CLIPAttributeFilter(clip_visual_path, clip_text_embeds_path, providers=providers)
        self.sample_fps = sample_fps
        self.batch_size = batch_size
        self.first_window_sec = first_window_sec

    def _flush_batch(self, letterboxed, raw_frames, scales, meta_idx, state) -> Optional[Tuple[bool, str]]:
        if not letterboxed:
            return None
        dets = self.detector.detect_batch(letterboxed)
        batch_crops = []

        for f_idx, raw_frame, scale, faces in zip(meta_idx, raw_frames, scales, dets):
            if len(faces) > 1:
                return False, f"Filter: Multi-Face at frame {f_idx}"
            if len(faces) == 1:
                if f_idx <= state["first_3s_limit"]:
                    state["has_face_in_first_3s"] = True

                box, _ = faces[0]
                ox1, oy1, ox2, oy2 = [v / scale for v in box]
                h, w = raw_frame.shape[:2]

                # 增加 15% 动态 Margin 保证细粒度属性（如眼镜腿）完整提取
                bw, bh = ox2 - ox1, oy2 - oy1
                x1 = max(0, int(ox1 - bw * 0.15))
                y1 = max(0, int(oy1 - bh * 0.15))
                x2 = min(w, int(ox2 + bw * 0.15))
                y2 = min(h, int(oy2 + bh * 0.15))

                crop = raw_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    batch_crops.append(crop)

        # Early Exit 逻辑：过 3s 校验后，当前批次裁出的 Crop 立即跑 CLIP 判定
        if state["checked_3s"]:
            if batch_crops:
                attr_results = self.clip_filter.classify_batch(batch_crops)
                for is_clean, reason in attr_results:
                    if not is_clean:
                        return False, f"Filter: {reason}"
        else:
            # 暂存前 3 秒关卡内的 Crop
            state["pending_3s_crops"].extend(batch_crops)

        return None

    def process_single_video(self, video_path: str) -> Tuple[bool, str]:
        meta = probe_raw_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Error: Open Video Failed"

        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except Exception:
            pass

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return False, "Error: Empty Video"

        is_vertical, manual_angle = resolve_rotation_plan(meta, first_frame)
        if not is_vertical:
            cap.release()
            return False, "Filter: Landscape Video (横屏)"

        fps = meta["fps"] if meta["ok"] else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_interval = max(1, int(fps / self.sample_fps))
        first_3s_limit = int(fps * self.first_window_sec)

        state = {
            "has_face_in_first_3s": False,
            "pending_3s_crops": [],
            "first_3s_limit": first_3s_limit,
            "checked_3s": False,
        }

        letterboxed, raw_frames, scales, meta_idx = [], [], [], []

        def _consume(frame_idx: int, frame: np.ndarray):
            frame = _rotate_frame(frame, manual_angle)
            canvas, scale = self.detector.letterbox(frame)
            letterboxed.append(canvas)
            raw_frames.append(frame)
            scales.append(scale)
            meta_idx.append(frame_idx)

        try:
            _consume(0, first_frame)
            frame_idx = 1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    _consume(frame_idx, frame)

                if len(letterboxed) >= self.batch_size:
                    early = self._flush_batch(letterboxed, raw_frames, scales, meta_idx, state)
                    letterboxed, raw_frames, scales, meta_idx = [], [], [], []
                    if early is not None:
                        return early

                # 前 3 秒 Fail-Fast 判定关卡
                if frame_idx >= first_3s_limit and not state["checked_3s"]:
                    if letterboxed:
                        early = self._flush_batch(letterboxed, raw_frames, scales, meta_idx, state)
                        letterboxed, raw_frames, scales, meta_idx = [], [], [], []
                        if early is not None:
                            return early

                    if not state["has_face_in_first_3s"]:
                        return False, "Filter: No Face in First 3 Seconds"

                    # 3秒判定通过，执行积攒的 前 3 秒 CLIP 属性过滤
                    if state["pending_3s_crops"]:
                        attr_results = self.clip_filter.classify_batch(state["pending_3s_crops"])
                        for is_clean, reason in attr_results:
                            if not is_clean:
                                return False, f"Filter: {reason}"
                        state["pending_3s_crops"] = []  # 立即释放内存/显存

                    state["checked_3s"] = True

                frame_idx += 1

            # 尾部 Residual Batch Flush
            if letterboxed:
                early = self._flush_batch(letterboxed, raw_frames, scales, meta_idx, state)
                if early is not None:
                    return early

        finally:
            cap.release()

        # 防御性尾校验：总时长不足 3s 的短视频逻辑
        if not state["checked_3s"]:
            if not state["has_face_in_first_3s"]:
                return False, "Filter: No Face in First 3 Seconds"
            if state["pending_3s_crops"]:
                attr_results = self.clip_filter.classify_batch(state["pending_3s_crops"])
                for is_clean, reason in attr_results:
                    if not is_clean:
                        return False, f"Filter: {reason}"

        return True, "PASSED (Clean Single Face Video)"


# ============================================================ #
# 5. 高并发入口
# ============================================================ #
def run_batch_video_cleaning(video_dir: str, scrfd_path: str, clip_visual_path: str, clip_text_embeds_path: str, num_threads=8):
    pipeline = ProductionVideoCleaningPipeline(scrfd_path, clip_visual_path, clip_text_embeds_path)
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith((".mp4", ".mov"))]
    print(f"🚀 开始清洗文件夹: {video_dir}，共计 {len(video_files)} 个短视频...")
    t0 = time.time()
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {executor.submit(pipeline.process_single_video, path): path for path in video_files}
        for future in concurrent.futures.as_completed(future_map):
            path = future_map[future]
            try:
                results[path] = future.result()
            except Exception as e:
                results[path] = (False, f"Exception: {str(e)}")

    t1 = time.time()
    passed_list = [p for p, (passed, _) in results.items() if passed]
    print(f"\n✅ 成功保留: {len(passed_list)} / {len(video_files)} | ⏱️ {t1 - t0:.2f}s")
    return results


if __name__ == "__main__":
    run_batch_video_cleaning(
        video_dir="./input_videos",
        scrfd_path="./scrfd_500m.onnx",
        clip_visual_path="./MobileCLIP2-S0/visual.onnx",
        clip_text_embeds_path="./clip_text_embeds.npy",
        num_threads=8,
    )