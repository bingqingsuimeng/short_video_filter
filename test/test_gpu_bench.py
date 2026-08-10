"""GPU benchmark + visualization test (Fixed)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import glob
import pipeline_v6
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, 'test')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Providers ----
print(f"providers: {ort.get_available_providers()}")
print(f"device: {ort.get_device()}\n")

# ---- Load models ----
cls_name = [c for c in dir(pipeline_v6) if 'Detector' in c][0]
detector_cls = getattr(pipeline_v6, cls_name)

# 防御性 Patch：若 pipeline_v6 未修，此处在类层面强行修复 axis=2 错位 Bug
def _patch_anchor_centers(self):
    self._anchor_centers = {}
    for stride in self.strides:
        h = w = self.input_size // stride
        ys, xs = np.mgrid[:h, :w]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centers = np.stack([centers, centers], axis=2).reshape(-1, 2)  # axis=2 修复
        self._anchor_centers[stride] = centers

detector_cls._patch_anchor_centers = _patch_anchor_centers

scrfd = detector_cls(os.path.join(ROOT, 'scrfd_500m.onnx'), conf_thres=0.3)
scrfd._patch_anchor_centers()  # 应用修复

# 兼容 detect_single 接口
if not hasattr(scrfd, 'detect_single'):
    scrfd.detect_single = lambda canvas: scrfd.detect_batch([canvas])[0]

clip = pipeline_v6.CLIPAttributeFilter(
    os.path.join(ROOT, 'MobileCLIP2-S0/visual.onnx'),
    os.path.join(ROOT, 'clip_text_embeds.npy'),
    input_size=256,
    margin=0.01
)
print(f"SCRFD EP: {scrfd.sess.get_providers()}")
print(f"CLIP EP:  {clip.sess.get_providers()}\n")

# ---- Warmup ----
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
canvas, _ = scrfd.letterbox(dummy)
scrfd.detect_single(canvas)
clip.classify_batch([dummy[:256, :256]])
print("Warmup done.\n")

# ---- Run on test images ----
N = 10  # Benchmark 循环次数
scrfd_times = []
clip_times = []
total_faces = 0

img_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')) + glob.glob(os.path.join(TEST_DIR, '*.png')))

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    basename = os.path.basename(img_path)

    # 1. Letterbox 与检测
    canvas, scale = scrfd.letterbox(img)

    t0 = time.perf_counter()
    for _ in range(N):
        faces = scrfd.detect_single(canvas)
    scrfd_ms = (time.perf_counter() - t0) / N * 1000
    scrfd_times.append(scrfd_ms)

    vis = img.copy()
    print(f"{basename} ({w}x{h}): {len(faces)} face(s), scrfd={scrfd_ms:.1f}ms")

    # 2. 坐标还原与 CLIP 属性分类
    for (bbox, score) in faces:
        x1l, y1l, x2l, y2l = bbox

        # 换算回原图坐标 (canvas_coord / scale)
        ox1, oy1, ox2, oy2 = x1l / scale, y1l / scale, x2l / scale, y2l / scale

        # 加上与 v6 流水线一致的 15% 动态 Margin 扩展
        bw, bh = ox2 - ox1, oy2 - oy1
        x1o = max(0, int(ox1 - bw * 0.15))
        y1o = max(0, int(oy1 - bh * 0.15))
        x2o = min(w, int(ox2 + bw * 0.15))
        y2o = min(h, int(oy2 + bh * 0.15))

        if x2o <= x1o or y2o <= y1o:
            continue

        crop = img[y1o:y2o, x1o:x2o]
        if crop.size == 0:
            continue

        # Benchmark CLIP 推理耗时
        t1 = time.perf_counter()
        for _ in range(N):
            is_clean, label = clip.classify_batch([crop])[0]
        clip_ms = (time.perf_counter() - t1) / N * 1000
        clip_times.append(clip_ms)
        total_faces += 1

        tag = 'OK' if is_clean else 'BAD'
        print(f"   [{tag}] {label} score={score:.3f} rect=({x1o},{y1o})-({x2o},{y2o})")

        # 画框可视化
        color = (0, 255, 0) if is_clean else (0, 0, 255)
        thickness = max(1, min(x2o - x1o, y2o - y1o) // 20)
        cv2.rectangle(vis, (x1o, y1o), (x2o, y2o), color, thickness)

        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(th + 6, y1o) - th - 6
        cv2.rectangle(vis, (x1o, label_y), (x1o + tw + 4, label_y + th + 6), color, -1)
        cv2.putText(vis, text, (x1o + 2, label_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存渲染结果
    out_path = os.path.join(RESULTS_DIR, basename)
    cv2.imwrite(out_path, vis)
    print(f"   -> saved {out_path}\n")

# ---- Summary ----
print("=" * 60)
if scrfd_times:
    print(f"Average SCRFD: {np.mean(scrfd_times):.1f}ms ± {np.std(scrfd_times):.1f}ms")
if clip_times:
    print(f"Average CLIP:  {np.mean(clip_times):.1f}ms ± {np.std(clip_times):.1f}ms")
print(f"Total faces classified: {total_faces}")
print(f"Results saved to: {RESULTS_DIR}/\n")

# ---- Estimate for 2000 videos x 15s ----
if scrfd_times and clip_times:
    fps = 30
    frames_per_video = 15 * fps  # 450 frames
    total_frames = 2000 * frames_per_video

    avg_scrfd = np.mean(scrfd_times)
    avg_clip = np.mean(clip_times)

    time_scrfd = total_frames * avg_scrfd / 1000
    time_clip = total_frames * avg_clip / 1000
    time_total = time_scrfd + time_clip

    print("=" * 60)
    print("预估：2000 个视频 x 15秒 (fps=30):")
    print(f"   总帧数: {total_frames:,}")
    print(f"   SCRFD: {time_scrfd / 3600:.1f} 小时")
    print(f"   CLIP:  {time_clip / 3600:.1f} 小时")
    print(f"   合计:  {time_total / 3600:.1f} 小时 (单 GPU)")
    print(f"   4 worker: {time_total / 4 / 3600:.1f} 小时")
    print("=" * 60)
