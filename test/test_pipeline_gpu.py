"""Test pipeline_v6 GPU acceleration"""
import sys, os
# Ensure project root is on path (run from project root: python test/test_pipeline_gpu.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import glob
import pipeline_v6
import onnxruntime as ort
import time

# Resolve paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, 'test')

print(f"providers: {ort.get_available_providers()}")
print(f"device: {ort.get_device()}")
print()

# Load detectors
cls_name = [c for c in dir(pipeline_v6) if 'Detector' in c][0]
scrfd = getattr(pipeline_v6, cls_name)(os.path.join(ROOT, 'scrfd_500m.onnx'), conf_thres=0.3)
clip = pipeline_v6.CLIPAttributeFilter(
    os.path.join(ROOT, 'MobileCLIP2-S0/visual.onnx'),
    os.path.join(ROOT, 'clip_text_embeds.npy'),
    input_size=256,
    margin=0.01
)

print(f"SCRFD EP: {scrfd.sess.get_providers()}")
print(f"CLIP EP:  {clip.sess.get_providers()}")
print()

# Test on each image in test/
total_ms = 0
total_clip_ms = 0
count = 0
for img_path in sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')) + glob.glob(os.path.join(TEST_DIR, '*.png'))):
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # letterbox then detect_single (fixed batch=1 model)
    canvas, scale = scrfd.letterbox(img)
    t0 = time.time()
    faces = scrfd.detect_single(canvas)
    ms = (time.time() - t0) * 1000
    total_ms += ms
    count += 1

    print(f"{img_path} ({w}x{h}): {len(faces)} face(s), scrfd={ms:.1f}ms")

    for i, (bbox, score) in enumerate(faces[:3]):
        # bbox is (x1l, y1l, w, h) in letterbox space
        x1l, y1l, bw, bh = bbox
        x2l = x1l + bw
        y2l = y1l + bh
        # Convert to original image space, clip to bounds
        x1o = max(0, int(x1l / scale))
        y1o = max(0, int(y1l / scale))
        x2o = min(img.shape[1], int(x2l / scale))
        y2o = min(img.shape[0], int(y2l / scale))
        if x2o <= x1o or y2o <= y1o:
            continue
        crop = img[y1o:y2o, x1o:x2o]
        crop = cv2.resize(crop, (256, 256))

        t1 = time.time()
        is_bad, label = clip.classify_batch([crop])[0]
        clip_ms = (time.time() - t1) * 1000
        total_clip_ms += clip_ms

        tag = 'BAD' if is_bad else 'OK'
        print(f"  face {i+1}: [{tag}] {label} (clip={clip_ms:.1f}ms)")

if count > 0:
    print(f"\n=== Average: scrfd={total_ms/count:.1f}ms, clip={total_clip_ms/count:.1f}ms ===")
else:
    print("No test images found")
