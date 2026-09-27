#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_video.py — 从视频里挑出「人脸质量好」的原始帧，导出为 JPG（不重新编码视频）

流程（方案1）:
  1. ffmpeg 按【原始分辨率】解码；若源帧率 > --max-fps（默认 10），则用 `fps`
     滤镜【均匀】降到 10fps（30fps→10fps，丢掉重复画面，省时）。
  2. 每帧并行缩放到 --target-long 长边（只影响检测，不 upscale），送 SCRFD
     TensorRT 批量检测；框/关键点映射回原始坐标。
  3. 姿态（Stage1，68 点）：用 1k3d68（68 点 3D 关键点，TensorRT）对每张脸算
     可靠的 (yaw, pitch, roll) 角度 + 睁眼度 EAR。SCRFD 5 点 solvePnP 角度噪声大，
     已被 68 点模型取代（见 face_pose68.py）。
  4. 眼神（Stage2，级联第二级）：仅对 Stage1 全过的单脸帧跑眼神估计，
     「头正但眼睛不看镜头」的帧剔除。两种实现（--gaze-model）：
       iris     : 按 SCRFD 框裁剪+放大后跑 MediaPipe FaceMesh，虹膜中心相对
                  眼眶的偏移量 mag（face_gaze.py；未装自动跳过）
       resnet34 : yakhyo MobileGaze 全脸注视角（gaze_resnet.py，GPU onnxruntime
                  加载 resnet34_gaze.onnx），输出 (pitch, yaw) 注视角（度）。
  5. 判定（满足全部才保留，否则剔除）:
       - 恰好 1 张脸（detect 已按 conf 过滤；>1 张脸 → 剔除）
       - |yaw| <= --yaw（侧脸/转头）
       - --pitch-min <= pitch <= --pitch-max（仰头/低头不看镜头）
       - down_ratio >= --down-min（低头比复核）
       - min-EAR >= --ear-min（闭眼/眯眼）
       - gaze: |gaze_yaw|<=--gaze-yaw-max 且 |gaze_pitch|<=--gaze-pitch-max
         （resnet34 注视角，默认 20°/15°）；iris 模式为 mag<=--gaze-max
       - |gaze_dy - 基线| <= --gaze-dy-dev（纵向眼神 2-pass，默认 11°，
         抓高/低机位下绝对阈值看不见的"往上看/往下看"；0=关）
       - 人头（Stage1.5，head2）：其它闸门全过且单人脸的帧再跑人头检测，
         画面里 NMS 后 head 总数 >= 2（含主角的头）→ 剔除（过滤背景里出现
         其它人头的帧；--no-head-gate 关闭，--head-conf 调阈值，默认 0.30）
  6. 保留帧写出【原始分辨率 BGR】JPG 到 out_dir/，剔除帧直接丢弃。

命名（⚠️ 临时功能，人工确定阈值后即可用 --no-score-name 关闭或删除本段）:
  {视频名}_{帧序号:05d}_{置信度:.2f}_{侧脸角度:.0f}.jpg
  侧脸角度 = 1k3d68 的带符号 yaw，取自被判定"合格"的脸；
    判定只看 |yaw|，这里带符号便于肉眼判断左右。
  同目录另写 pose_report.csv：frame,verdict,score,yaw,pitch,roll,down_ratio,nfaces 逐帧
  （yaw/pitch/roll 均来自 68 点模型，可靠，供人工定阈值）。

用法:
  python filter_video.py E:\\change\\zi_xia\\zi_xia_001.mp4
  python filter_video.py <dir> --conf 0.8 --yaw 45 --down-min 0.46 --max-fps 10
"""
import os
import sys
import json
import time
import queue
import shutil
import tempfile
import subprocess
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.face_det import SCRFDTRTDetector, estimate_pose
from src.face_pose68 import FacePose68


# ---------------- helpers ----------------
def _parse_rate(s):
    """'30000/1001' -> 29.97 ; '30' -> 30.0 ; '' -> 0.0"""
    if not s:
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        b = float(b)
        return float(a) / b if b else 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def probe(path):
    """Return (width, height, fps, nframes) of the first video stream."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    fps = _parse_rate(st.get("avg_frame_rate") or st.get("r_frame_rate"))
    nframes = int(st.get("nb_read_frames") or 0)
    return w, h, fps, nframes


def pitch_down_proxy(kps):
    """
    轻量的『低头/仰头』指示，只用 SCRFD 的 5 个点（双眼/鼻/嘴角），
    比 solvePnP 的 pitch 稳得多（不做 PnP，只是两个距离的比值，尺度无关）。

    原理：低头时下颌方向(鼻→嘴)的纵向间距被压缩、(眼→鼻)被拉长，
    所以比值 r = (鼻→嘴) / (眼→鼻)  随低头单调【下降】。
      正脸  r ≈ 1.3 上下
      低头  r 明显变小（越低头越小）
      仰头  r 变大
    返回 float；点太退化（眼→鼻 ≈ 0）时返回 None。
    方向：r 越小 = 越低头，r 越大 = 越仰头。
    """
    E = (np.asarray(kps[0]) + np.asarray(kps[1])) / 2.0   # 眼线中点
    N = np.asarray(kps[2])                                 # 鼻尖
    M = (np.asarray(kps[3]) + np.asarray(kps[4])) / 2.0    # 嘴线中点
    d_en = float(N[1] - E[1])   # 眼→鼻 纵向（>0）
    d_nm = float(M[1] - N[1])   # 鼻→嘴 纵向（>0）
    if d_en < 1e-3:
        return None
    return d_nm / d_en


# ---------------- filter ----------------
class VideoFilter:
    def __init__(self, engine, conf, target_long, batch, detect_chunk,
                 max_fps, jpg_quality, score_name, verbose=False, jpg_workers=8,
                 down_min=0.46, pose_engine=None, pitch_max=25.0,
                 pitch_min=-25.0, ear_min=0.25, gaze_max=0.12, gaze_dy_dev=0.0,
                 gaze_model="iris", gaze_model_path=None,
                 gaze_pitch_max=15.0, gaze_yaw_max=15.0,
                 head_on=True, head_conf=0.30, head_model_path=None):
        self.det = SCRFDTRTDetector(engine, max_batch=batch, conf_thres=conf)
        # 68 点 3D 姿态模型（可靠角度）；引擎缺失则回退 SCRFD 5 点 solvePnP
        self.pose68 = None
        if pose_engine and os.path.exists(pose_engine):
            try:
                self.pose68 = FacePose68(pose_engine, max_batch=batch)
            except Exception as e:
                print(f"[filter] !! 68 点姿态模型加载失败，回退 5 点法: {e}")
        elif pose_engine:
            print(f"[filter] !! 未找到 68 点姿态引擎 {pose_engine}，回退 5 点 solvePnP")
        self.conf = conf
        self.target_long = target_long
        self.batch = batch
        self.detect_chunk = detect_chunk
        self.max_fps = max_fps
        self.jpg_q = jpg_quality
        self.score_name = score_name   # 临时命名开关
        self.verbose = verbose
        self.jpg_workers = jpg_workers
        self.down_min = down_min       # 低头比下限：down_ratio < down_min 视为低头，剔除
        self.pitch_max = pitch_max     # 抬头上限：pitch > pitch_max 视为仰头，剔除（None=关）
        self.pitch_min = pitch_min     # 低头上限：pitch < pitch_min 视为低头，剔除（None=关）
        self.ear_min = ear_min         # 睁眼下限：min-EAR < ear_min 视为闭眼，剔除（None=0=关）
        self.gaze_dy_dev = gaze_dy_dev  # 纵向眼神 2-pass 闸门：|dy-本视频dy中位数|>此值 剔除（0=关）
        # Stage2 眼神闸门（级联第二级）：仅对 Stage1 全过的帧跑眼神估计，
        # 切"头正但眼不看镜头"的帧。两种实现（--gaze-model）：
        #   iris     : MediaPipe 虹膜偏移（face_gaze.GazeGate，阈值 gaze_max，归一化单位）
        #   resnet34 : yakhyo MobileGaze 全脸注视角（gaze_resnet.GazeResNet，GPU onnxruntime，
        #              阈值 gaze_pitch_max / gaze_yaw_max，单位度；加载失败自动降级关闭）
        self.gaze_model = gaze_model
        self.gaze_max = gaze_max
        self.gaze_pitch_max = gaze_pitch_max
        self.gaze_yaw_max = gaze_yaw_max
        self.gaze = None
        if gaze_model == "resnet34":
            try:
                from src.gaze_resnet import GazeResNet
                self.gaze = GazeResNet(gaze_model_path)
                print(f"[filter] Stage2 眼神闸门 = resnet34 ({self.gaze.ep})")
            except Exception as e:
                print(f"[filter] !! 眼神闸门(resnet34)加载失败，Stage2 关闭: {e}")
        elif gaze_max is not None and gaze_max > 0:
            try:
                from src.face_gaze import GazeGate
                self.gaze = GazeGate()
            except Exception as e:
                print(f"[filter] !! 眼神闸门加载失败，Stage2 关闭: {e}")
        # Stage1.5 人头闸门：其它闸门全过且单人脸的帧，跑 head2 人头检测，
        # NMS 后 head 总数 >= 2（含主角的头）→ 判 multi_head 丢弃
        self.head = None
        if head_on:
            try:
                from src.head_gate import HeadGate
                self.head = HeadGate(head_model_path, conf_thresh=head_conf)
                print(f"[filter] 人头闸门 = head2 (TensorRT) conf={head_conf}")
            except Exception as e:
                print(f"[filter] !! 人头闸门(head2)加载失败，人头闸门关闭: {e}")

    # 判定单帧：返回 (verdict, score, (yaw,pitch,roll), down_ratio, nfaces)
    #   no_face : 0 张脸
    #   multi   : >1 张脸（都已过 conf 阈值）→ 剔除
    #   pose    : 单脸但 |yaw|>yaw_lim（侧脸/转头）
    #   up      : 单脸但 pitch>pitch_max（仰头，不看镜头）
    #   downp   : 单脸但 pitch<pitch_min（低头，不看镜头；down_ratio 之外的角度通道）
    #   blink   : 单脸但 min-EAR<ear_min（闭眼/眯眼）
    #   keep    : 单脸且 yaw/pitch/EAR 全过（conf 已达标；低头比由主循环用 down_min 复核）
    #   角度/EAR 来自 1k3d68 68 点模型；pose=None 时回退 SCRFD 5 点 solvePnP（无 EAR）。
    def _judge(self, dets, kpss, pose=None, ear=None):
        """dets: (n,5) [x1,y1,x2,y2,score]（已按 conf 过滤）; kpss: (n,5,2);
        pose: 本帧主脸（最高分）的 (yaw,pitch,roll)，来自 68 点模型；None 则回退 5 点法。
        ear: 本帧主脸 min(EAR左眼,EAR右眼)；None 则跳过闭眼判定。"""
        n = len(dets)
        if n == 0:
            return "no_face", 0.0, (0.0, 0.0, 0.0), None, 0
        top = int(np.argmax(dets[:, 4])) if n > 1 else 0
        down = pitch_down_proxy(kpss[top])
        score = float(dets[top, 4])
        if pose is None:   # 68 点模型不可用时的回退
            pose = estimate_pose(kpss[top], dets[top, :4]) or (0.0, 0.0, 0.0)
        if n > 1:          # 多人脸 → 剔除
            return "multi", score, pose, down, n
        yaw, pitch, _roll = pose
        if abs(yaw) > self.yaw_lim:
            return "pose", score, pose, down, 1
        if self.pitch_max is not None and pitch > self.pitch_max:
            return "up", score, pose, down, 1
        if self.pitch_min is not None and pitch < self.pitch_min:
            return "downp", score, pose, down, 1
        if self.ear_min is not None and ear is not None and ear < self.ear_min:
            return "blink", score, pose, down, 1
        return "keep", score, pose, down, 1

    def process(self, video_path, out_dir, yaw_lim, shared_csv=False):
        self.yaw_lim = yaw_lim
        w, h, fps, nframes = probe(video_path)
        size = w * h * 3
        stem = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(out_dir, exist_ok=True)

        # 帧率上限：源 > max_fps 才降（均匀采样）
        apply_fps = bool(self.max_fps and self.max_fps > 0 and fps > self.max_fps)
        dec_fps = self.max_fps if apply_fps else fps

        # 临时 raw 文件：Windows 下 Python 读管道极慢(~30MB/s)，读文件快~90x，
        # 且先落盘再读可彻底消除 ffmpeg stdin 管道背压死锁。
        fd_tmp, tmp_raw = tempfile.mkstemp(suffix=".raw", prefix=f"filt_{stem}_")
        os.close(fd_tmp)

        dec_cmd = ["ffmpeg", "-v", "error", "-i", video_path]
        if apply_fps:
            dec_cmd += ["-vf", f"fps={dec_fps:g}"]
        dec_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-y", tmp_raw]

        if self.verbose:
            print(f"  decode {os.path.basename(video_path)}  {w}x{h}  src_fps={fps:.3f} "
                  f"-> dec_fps={dec_fps:.3f}{' (downsampled)' if apply_fps else ''}  ~{nframes} frames")

        t0 = time.time()
        frame_no = kept = drop_noface = drop_pose = drop_down = drop_multi = 0
        drop_up = drop_blink = drop_downp = drop_gaze = drop_gazedown = 0
        drop_head = 0
        t_dec = t_read = t_det = t_judge = t_gaze = t_write = 0.0
        t_head = 0.0
        ex = ThreadPoolExecutor(max_workers=self.jpg_workers)
        futures = []
        # 临时 pose 报告（与临时命名同生命周期，定好阈值后一并删除）：
        # 每帧一行 score/yaw/pitch/roll，供人工定「不看正前方」的阈值
        report = []
        dy_all = []    # 所有测到 gaze 的帧的 dy（2-pass 基线=其中位数）
        dy_gate = []   # (帧号, 写出路径, dy)：keep 候选帧，循环后统一套用 dy 闸门
        try:
            t1 = time.time()
            subprocess.run(dec_cmd, check=True)
            t_dec = time.time() - t1

            with open(tmp_raw, "rb", buffering=0) as f:
                while True:
                    t1 = time.time()
                    chunk = []
                    for _ in range(self.detect_chunk):
                        buf = f.read(size)
                        if len(buf) < size:
                            break
                        chunk.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
                    t_read += time.time() - t1
                    if not chunk:
                        break

                    t1 = time.time()
                    dets, kpss = self.det.detect(
                        chunk, threshold=self.conf, target_long=self.target_long,
                        pipeline=True, gpu_preprocess=True)
                    t_det += time.time() - t1

                    # 68 点姿态：对本块所有 >=1 脸帧的主脸【批量】计算（整块一次
                    # TRT 调用，~ms 级）。pose_map[块内序号] = ((yaw,pitch,roll), ear)。
                    pose_map = {}
                    if self.pose68 is not None:
                        items = []
                        for i, (fr, d) in enumerate(zip(chunk, dets)):
                            if len(d) > 0:
                                top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                items.append((i, fr, d[top, :4]))
                        if items:
                            poses = self.pose68.estimate_batch(
                                [(fr, bb) for _, fr, bb in items])
                            pose_map = {i: p for (i, _, _), p in zip(items, poses)}

                    for i, (frame, d, k) in enumerate(zip(chunk, dets, kpss)):
                        frame_no += 1
                        t1 = time.time()
                        _p = pose_map.get(i)          # ((yaw,pitch,roll), ear) 或 None
                        verdict, score, pose, down, nfaces = self._judge(
                            d, k, _p[0] if _p else None,
                            _p[1] if _p else None)
                        t_judge += time.time() - t1
                        yaw, pitch, roll = pose
                        ear = _p[1] if _p else None
                        # 低头复核：单脸 pose 已合格，但 down_ratio < down_min → 按低头剔除
                        # （down_min 本身保留，即 down >= down_min 才 keep）
                        if (verdict == "keep" and self.down_min is not None
                                and down is not None and down < self.down_min):
                            verdict = "down"
                        # Stage2 眼神闸门：Stage1 全过（头基本正）的单脸帧，用
                        # MediaPipe 虹膜偏移切"头正但眼不看镜头"的帧。
                        gaze_mag = None
                        gaze_dy = None
                        if (verdict == "keep" and self.gaze is not None and nfaces == 1):
                            t1 = time.time()
                            top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                            g = self.gaze.estimate(frame, d[top, :4])
                            t_gaze += time.time() - t1
                            if g is not None:
                                if self.gaze_model == "resnet34":
                                    # g = (pitch_deg, yaw_deg)。CSV 沿用 gaze 两列：
                                    # gaze_mag=|yaw|(度, 横向偏角)  gaze_dy=pitch(度, 正=仰)
                                    gpitch, gyaw = g
                                    gaze_mag, gaze_dy = abs(gyaw), gpitch
                                    dy_all.append(gaze_dy)   # 2-pass 基线（resnet 模式单位=度）
                                    if (abs(gpitch) > self.gaze_pitch_max
                                            or abs(gyaw) > self.gaze_yaw_max):
                                        verdict = "gaze"
                                else:
                                    gaze_mag, _gdx, gaze_dy = g
                                    dy_all.append(gaze_dy)
                                    if gaze_mag > self.gaze_max:
                                        verdict = "gaze"
                        # Stage1.5 人头闸门：其它闸门全过且单人脸的帧，画面里
                        # 检出 >= 2 个人头（含主角的头）→ 判 multi_head 丢弃
                        nheads = None
                        if (verdict == "keep" and nfaces == 1
                                and self.head is not None):
                            t1 = time.time()
                            _hb = self.head.detect(frame)
                            t_head += time.time() - t1
                            nheads = len(_hb)
                            if nheads >= 2:
                                verdict = "multi_head"
                        if self.score_name:
                            report.append([frame_no, verdict, score, yaw, pitch,
                                           roll, down, nfaces, ear, gaze_mag,
                                           gaze_dy, nheads])
                        if verdict == "keep":
                            base = f"{stem}_{frame_no:05d}"
                            # 临时命名：原名_置信度_yaw_pitch_roll（68 点模型值，
                            # 供手动阈值确认，确认后改回 <stem>_kept_<frame>.jpg）
                            name = (f"{base}_{score:.2f}_{yaw:.0f}_{pitch:.0f}_{roll:.0f}.jpg"
                                    if self.score_name else f"{base}.jpg")
                            # JPEG 编码丢给线程池（libjpeg 释放 GIL），与后续
                            # 块的 read/detect 重叠，不再阻塞主线程。
                            futures.append(ex.submit(
                                cv2.imwrite, os.path.join(out_dir, name), frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpg_q)]))
                            kept += 1
                            if self.gaze_dy_dev > 0 and gaze_dy is not None:
                                dy_gate.append((frame_no, os.path.join(out_dir, name), gaze_dy))
                            if self.verbose:
                                print(f"    #{frame_no:05d} KEEP  score={score:.3f} "
                                      f"yaw={yaw:.0f} pit={pitch:.0f} rol={roll:.0f} "
                                      f"down={down:.2f}")
                        else:
                            if verdict == "no_face":
                                drop_noface += 1
                            elif verdict == "down":
                                drop_down += 1
                            elif verdict == "downp":
                                drop_downp += 1
                            elif verdict == "multi":
                                drop_multi += 1
                            elif verdict == "up":
                                drop_up += 1
                            elif verdict == "blink":
                                drop_blink += 1
                            elif verdict == "gaze":
                                drop_gaze += 1
                            elif verdict == "multi_head":
                                drop_head += 1
                            else:
                                drop_pose += 1
                            if self.verbose:
                                print(f"    #{frame_no:05d} drop  ({verdict}  down={down})")

            # 排空 JPEG 线程池（主线程提交完后大多已编码完，这里只剩尾部等待）
            t1 = time.time()
            wait(futures)
            t_write = time.time() - t1

            # Stage2 纵向眼神闸门（2-pass 后半）：基线 = 本视频所有测到 gaze 帧的
            # dy 中位数；keep 帧中 |dy-基线| > gaze_dy_dev 的按「纵向眼神偏移」剔除
            # （横向闸门 mag 抓不到往下看，往下看几乎不产生横向偏移）。
            if self.gaze_dy_dev > 0 and len(dy_all) >= 3:
                med = float(np.median(dy_all))
                hit = {}
                for no, path, dy in dy_gate:
                    if abs(dy - med) > self.gaze_dy_dev:
                        hit[no] = dy
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                if hit:
                    kept -= len(hit)
                    drop_gazedown += len(hit)
                    for r in report:
                        if r[0] in hit:
                            r[1] = "gazedown"
                    if self.verbose:
                        for no, dy in sorted(hit.items()):
                            print(f"    #{no:05d} drop (gazedown dy={dy:.3f} "
                                  f"med={med:.3f})")

            if report:
                csv_name = (f"pose_report_{stem}.csv" if shared_csv
                            else "pose_report.csv")
                csv_path = os.path.join(out_dir, csv_name)
                with open(csv_path, "w", encoding="utf-8") as cf:
                    cf.write("frame,verdict,score,yaw,pitch,roll,down_ratio,nfaces,ear,gaze_mag,gaze_dy,nheads\n")
                    for (no, vd, sc, y, p, r, dn, nf, er, gm, gd, nh) in report:
                        dtag = f"{dn:.3f}" if dn is not None else ""
                        etag = f"{er:.3f}" if er is not None else ""
                        gtag = f"{gm:.3f}" if gm is not None else ""
                        gdt = f"{gd:.3f}" if gd is not None else ""
                        nht = f"{nh}" if nh is not None else ""
                        cf.write(f"{no},{vd},{sc:.3f},{y:.1f},{p:.1f},{r:.1f},{dtag},{nf},{etag},{gtag},{gdt},{nht}\n")
        finally:
            ex.shutdown(wait=True)
            try:
                os.remove(tmp_raw)
            except OSError:
                pass

        dt = time.time() - t0
        dm = f"{self.down_min:g}" if self.down_min is not None else "?"
        print(f"  {os.path.basename(video_path)}: {frame_no} frames -> "
              f"kept {kept}  (drop {drop_noface} no-face, {drop_pose} pose, "
              f"{drop_up} up, {drop_downp} downp, {drop_gaze} gaze, "
              f"{drop_gazedown} gazedown, {drop_blink} blink, "
              f"{drop_down} down<{dm}, {drop_multi} multi, "
              f"{drop_head} multi-head)  "
              f"in {dt:.1f}s ({frame_no / dt:.0f} fps)  -> {os.path.abspath(out_dir)}")
        print(f"    [breakdown] decode={t_dec:5.2f}s  read={t_read:5.2f}s  "
              f"detect={t_det:5.2f}s  judge={t_judge:5.2f}s  gaze={t_gaze:5.2f}s  "
              f"head={t_head:5.2f}s  imwrite={t_write:5.2f}s ({kept} frames)")
        return {"frames": frame_no, "kept": kept, "no_face": drop_noface,
                "pose": drop_pose, "down": drop_down, "multi": drop_multi,
                "up": drop_up, "downp": drop_downp, "gaze": drop_gaze,
                "gazedown": drop_gazedown, "blink": drop_blink,
                "head": drop_head, "seconds": dt}

    def run(self, targets, yaw_lim, shared_csv=False):
        """targets: list of (video_path, out_dir). shared_csv: 多视频共享同一
        out_dir 时，CSV 按视频名区分（pose_report_<stem>.csv）防互相覆盖。"""
        t0 = time.time()
        failed = []
        for i, (p, od) in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] {p}", flush=True)
            try:
                self.process(p, od, yaw_lim, shared_csv=shared_csv)
            except Exception as e:   # 单视频失败不中断整批（无人值守长任务）
                failed.append((p, str(e)))
                print(f"  !! FAILED: {e}  (skip, continue)", flush=True)
        print(f"\nAll done: {len(targets)} video(s) in {time.time() - t0:.1f}s"
              + (f", {len(failed)} FAILED: {[f[0] for f in failed]}" if failed else ""))


# ---------------- 图片文件夹模式（仅人头闸门，不碰视频解码/SCRFD/pose/gaze） ----------------
_IMG_EXTS = (".jpg", ".jpeg", ".png")
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")


def process_image_dir(image_dir, out_dir, head, batch=16,
                      imread_workers=8, verbose=False):
    """纯图片文件夹过滤: 读图 → head.detect_batch → nheads>=2 判 dropped。

    这些图已全过 SCRFD/pose/gaze（kept_all 一类产物），图片模式只复用 head
    判定语义。非破坏输出: kept 图 os.link 硬链接进 out_dir（同卷；失败回退
    shutil.copy2），dropped 不复制只记录；原 image_dir 一张不动。
    流水线: ThreadPoolExecutor(imread_workers) 读图 → 有界队列 → 单线程攒满
    batch 攒一批 → detect_batch（单卡 TRT 只能串行）。
    out_dir 写 REPORT.csv (filename,nheads) 与 dropped.txt (filename nheads)。
    """
    t0 = time.time()
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith(_IMG_EXTS))
    total = len(files)
    if total == 0:
        print(f"[image-dir] {image_dir} 下没有图片(.jpg/.jpeg/.png)")
        return {"total": 0, "kept": 0, "dropped": 0, "unread": 0, "seconds": 0.0}
    os.makedirs(out_dir, exist_ok=True)

    q = queue.Queue(maxsize=batch * 8)
    SENTINEL = object()

    def _producer():
        # 滑动窗口提交 + 按文件名顺序取结果 → imread_workers 路并发读, 且出队
        # 顺序与 files 一致（REPORT/dropped 保持文件序）。窗口=队列上限+余量:
        # 窗口结束即释放该批 future, 已解码数组不再被挂起（原实现一次性提交
        # 全部任务, 完成的 future 长期持有解码数组, 1 万+ 图必然 OOM）。
        win = batch * 8 + imread_workers
        with ThreadPoolExecutor(max_workers=imread_workers) as ex:
            for i in range(0, total, win):
                futs = [(fn, ex.submit(cv2.imread, os.path.join(image_dir, fn)))
                        for fn in files[i:i + win]]
                for fn, fut in futs:
                    q.put((fn, fut.result()))
        q.put(SENTINEL)

    th = threading.Thread(target=_producer, daemon=True)
    th.start()

    kept = dropped = unread = processed = 0
    t_infer = t_link = 0.0
    last_print = [0.0]
    pending = []
    report_rows = []
    dropped_lines = []

    def _flush(items):
        nonlocal kept, dropped, unread, processed, t_infer, t_link
        if not items:
            return
        ok = [it for it in items if it[1] is not None]
        unread += len(items) - len(ok)
        for fn, fr in items:
            if fr is None:
                print(f"  !! 读图失败跳过: {fn}")
        if ok:
            t1 = time.time()
            res = head.detect_batch([fr for _, fr in ok])
            t_infer += time.time() - t1
            for (fn, _), dets in zip(ok, res):
                nheads = len(dets)
                report_rows.append((fn, nheads))
                if nheads >= 2:
                    dropped += 1
                    dropped_lines.append(f"{fn} {nheads}")
                else:
                    src = os.path.join(image_dir, fn)
                    dst = os.path.join(out_dir, fn)
                    t1 = time.time()
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                    t_link += time.time() - t1
                    kept += 1
                processed += 1
        processed += len(items) - len(ok)   # 读图失败的也计入进度
        now = time.time()
        if verbose or (now - last_print[0] >= 2.0):
            last_print[0] = now
            el = now - t0
            print(f"  [img] {processed}/{total}  kept={kept} dropped={dropped} "
                  f"elapsed={el:.0f}s  {processed / max(el, 1e-6):.1f} img/s",
                  flush=True)

    while True:
        item = q.get()
        if item is SENTINEL:
            break
        pending.append(item)
        if len(pending) >= batch:
            _flush(pending)
            pending = []
    _flush(pending)
    th.join()

    with open(os.path.join(out_dir, "REPORT.csv"), "w", encoding="utf-8") as f:
        f.write("filename,nheads\n")
        for fn, nh in report_rows:
            f.write(f"{fn},{nh}\n")
    with open(os.path.join(out_dir, "dropped.txt"), "w", encoding="utf-8") as f:
        for line in dropped_lines:
            f.write(line + "\n")

    dt = time.time() - t0
    n_unread_note = f" / unread {unread}" if unread else ""
    print(f"[image-dir] {total} images -> kept {kept} / dropped {dropped}"
          f"{n_unread_note}  in {dt:.1f}s ({total / max(dt, 1e-6):.1f} img/s)  "
          f"-> {os.path.abspath(out_dir)}")
    print(f"    [breakdown] infer={t_infer:5.1f}s  link={t_link:5.1f}s")
    return {"total": total, "kept": kept, "dropped": dropped,
            "unread": unread, "seconds": dt}


def main():
    ap = argparse.ArgumentParser(description="Filter a video to high-quality face frames (JPG), no re-encode.")
    ap.add_argument("input", nargs="?", default=None,
                    help="mp4 file or a directory of videos "
                         "(可省略, 若用 --image-dir 跑纯图片文件夹模式)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <input_dir>/<stem>_kept for a file, "
                         "or <input_dir>/kept_frames/<stem>_kept for a directory; "
                         "with a directory input an explicit --out-dir is shared by ALL videos)")
    ap.add_argument("--engine", default=None,
                    help="SCRFD TRT 引擎路径（默认 models/scrfd/scrfd_500m_bnkps_batch32_fp16.engine，"
                         "缺失回退 scrfd_500m_bnkps_batch32.engine）")
    ap.add_argument("--pose-engine", default=None,
                    help="1k3d68 68 点 3D 姿态 TRT 引擎（默认 models/pose68/1k3d68_dyn.engine；"
                         "缺失则回退 SCRFD 5 点 solvePnP）")
    ap.add_argument("--conf", type=float, default=0.8, help="face confidence threshold")
    ap.add_argument("--yaw", type=float, default=12.0,
                    help="|yaw| 上限（侧脸/转头，角度来自 1k3d68 68 点模型，可靠）；"
                         "12°=正脸，10°=严格正对，15°=放宽")
    ap.add_argument("--pitch-max", type=float, default=25.0,
                    help="抬头上限：pitch > 此值视为仰头不看镜头剔除（0=关闭；"
                         "018 好帧最大 23.5，032 在 27.7 出现抬头不看帧，取 25）")
    ap.add_argument("--pitch-min", type=float, default=-25.0,
                    help="低头上限：pitch < 此值视为低头不看镜头剔除（负值；0=关闭）")
    ap.add_argument("--ear-min", type=float, default=0.25,
                    help="睁眼下限：min-EAR < 此值视为闭眼/眯眼剔除（睁眼~0.30-0.39；"
                         "0=关闭）")
    ap.add_argument("--gaze-model", choices=["iris", "resnet34"], default="resnet34",
                    help="Stage2 眼神闸门实现（默认 resnet34，018/032/035 A/B 验证优于 iris）："
                         "resnet34=yakhyo MobileGaze 全脸注视角（TRT，本地 resnet34_gaze_fp16.engine，"
                         "FP16 缺失回退 resnet34_gaze.engine；"
                         "阈值 --gaze-pitch-max/--gaze-yaw-max，度）；iris=MediaPipe 虹膜偏移"
                         "（阈值 --gaze-max，备选）")
    ap.add_argument("--gaze-model-path", default=None,
                    help="resnet34 TRT 引擎路径（默认 models/gaze/resnet34_gaze_fp16.engine，"
                         "缺失回退 resnet34_gaze.engine）")
    ap.add_argument("--gaze-pitch-max", type=float, default=15.0,
                    help="resnet34 纵向注视角上限（度）：|pitch|>此值剔除（文档 attention 阈值 15°；"
                         "高/低机位下绝对 pitch 会漂移，主要靠 --gaze-dy-dev 基线闸门兜底）")
    ap.add_argument("--gaze-yaw-max", type=float, default=20.0,
                    help="resnet34 横向注视角上限（度）：|yaw|>此值剔除。文档 attention 阈值是 15°，"
                         "但 018/032/035 验证：15° 在头微侧/广角自拍姿态上压线误杀 10 帧（15.5~19.6° "
                         "全部确认看镜头），提到 20° 后全救回；真正横看的帧噪声到不了 20°")
    ap.add_argument("--gaze-max", type=float, default=0.12,
                    help="眼神偏移上限（Stage2 iris 模式，MediaPipe 虹膜，只过 Stage1 的帧才跑）："
                         "头正但眼不看镜头剔除；0=关闭（需 pip install mediapipe）")
    ap.add_argument("--gaze-dy-dev", type=float, default=11.0,
                    help="纵向眼神 2-pass 闸门（默认开，resnet34 模式=11°）：基线=本视频所有测到 "
                         "gaze 帧的 gaze_dy 中位数，keep 帧 |dy-基线|>此值 按纵向眼神偏移剔除"
                         "（抓绝对阈值看不见的往上看/往下看——高/低机位会整体平移注视角基准，"
                         "绝对阈值会漏）；0=关闭。resnet34 模式单位=度，11 为 018/032/035 验证值"
                         "（切 018 #7、035 #48/#108/#109，保留 032 #100 与全部确认好帧）；"
                         "iris 模式单位=归一化偏移（此值下不生效，需传 0.018 量级）")
    ap.add_argument("--no-head-gate", action="store_true",
                    help="关闭 Stage1.5 人头闸门（head2；默认开，画面里检出 >=2 个人头的帧剔除）")
    ap.add_argument("--head-conf", type=float, default=0.30,
                    help="人头检测置信度阈值（head2，默认 0.30，实测最稳误检最少）")
    ap.add_argument("--head-model-path", default=None,
                    help="head2 TRT engine 路径（默认 models/head/model_dyn_fp16.engine，"
                         "缺失逐级回退 model_dyn.engine → model.engine）")
    ap.add_argument("--image-dir", default=None,
                    help="纯图片文件夹模式（显式指定, 永远优先）: 只跑人头闸门, "
                         "nheads>=2 剔除; 输出 kept 图硬链接 + REPORT.csv + dropped.txt, "
                         "原文件夹不动。跳过视频解码/SCRFD/pose/gaze")
    ap.add_argument("--imread-workers", type=int, default=8,
                    help="图片模式: 读图线程数（默认 8）")
    ap.add_argument("--down-min", type=float, default=0.46,
                    help="低头比下限：down_ratio < 此值视为低头剔除（此值本身保留；0=关闭）")
    ap.add_argument("--target-long", type=int, default=768,
                    help="downscale long-side to this for detection (0=off)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--detect-chunk", type=int, default=64,
                    help="frames per detect() call")
    ap.add_argument("--max-fps", type=float, default=10.0,
                    help="if source fps > this, uniformly decode at this fps (0=off)")
    ap.add_argument("--quality", type=int, default=90, help="JPG quality")
    ap.add_argument("--no-score-name", action="store_true",
                    help="disable the TEMPORARY score/angle naming (just stem_index.jpg)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # ---------------- 图片文件夹模式（只加载 head 闸门, 不碰 SCRFD/pose/gaze） ----------------
    image_dir = args.image_dir
    if image_dir is None and args.input and os.path.isdir(args.input):
        _has_imgs = any(f.lower().endswith(_IMG_EXTS) for f in os.listdir(args.input))
        _has_vids = any(f.lower().endswith(_VIDEO_EXTS) for f in os.listdir(args.input))
        if _has_imgs and not _has_vids:
            image_dir = args.input
            print("[image-dir] 检测到纯图片目录（无 .mp4），进入图片过滤模式（仅人头闸门）")
    if image_dir is not None:
        if args.no_head_gate:
            sys.exit("[image-dir] 图片模式只依赖人头闸门, 与 --no-head-gate 冲突")
        from src.head_gate import HeadGate
        head = HeadGate(args.head_model_path, conf_thresh=args.head_conf)
        out_dir = args.out_dir or (os.path.normpath(image_dir) + "_head")
        process_image_dir(image_dir, out_dir, head, batch=args.batch,
                          imread_workers=args.imread_workers,
                          verbose=args.verbose)
        return
    if args.input is None:
        ap.error("需要 input（视频/视频目录）或 --image-dir（纯图片目录）")

    root = os.path.dirname(os.path.abspath(__file__))
    if args.engine:
        engine = args.engine
        if not os.path.exists(engine):
            sys.exit(f"engine not found: {engine}")
    else:
        # 默认引擎优先级: FP16 动态 batch → FP32 动态 batch，按存在性逐级回退
        _scrfd_candidates = (
            os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps_batch32_fp16.engine"),
            os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps_batch32.engine"),
        )
        engine = next((p for p in _scrfd_candidates if os.path.exists(p)), _scrfd_candidates[0])
        if not os.path.exists(engine):
            sys.exit(f"engine not found: {engine}")
    pose_engine = args.pose_engine or os.path.join(root, "models", "pose68", "1k3d68_dyn.engine")

    # build explicit (video_path, out_dir) targets — avoids double-appending the suffix
    shared_csv = False
    if os.path.isdir(args.input):
        vids = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                      if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi")))
        if args.out_dir:
            # 显式 --out-dir + 目录输入 → 所有视频共享同一个文件夹
            # （文件名已含视频名不冲突；CSV 按视频名区分防覆盖）
            targets = [(p, args.out_dir) for p in vids]
            shared_csv = True
        else:
            root = os.path.join(args.input, "kept_frames")
            targets = [(p, os.path.join(root, os.path.splitext(os.path.basename(p))[0] + "_kept"))
                       for p in vids]
    else:
        p = args.input
        final = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(p)),
            os.path.splitext(os.path.basename(p))[0] + "_kept")
        targets = [(p, final)]

    vf = VideoFilter(engine, args.conf,
                     target_long=(args.target_long or None),
                     batch=args.batch, detect_chunk=args.detect_chunk,
                     max_fps=(args.max_fps or None),
                     jpg_quality=args.quality,
                     score_name=(not args.no_score_name),
                     verbose=args.verbose,
                     down_min=(args.down_min or None),
                     pose_engine=pose_engine,
                     pitch_max=(args.pitch_max or None),
                     pitch_min=(args.pitch_min or None),
                     ear_min=(args.ear_min or None),
                     gaze_max=(args.gaze_max or None),
                     gaze_dy_dev=(args.gaze_dy_dev or 0.0),
                     gaze_model=args.gaze_model,
                     gaze_model_path=args.gaze_model_path,
                     gaze_pitch_max=args.gaze_pitch_max,
                     gaze_yaw_max=args.gaze_yaw_max,
                     head_on=(not args.no_head_gate),
                     head_conf=args.head_conf,
                     head_model_path=args.head_model_path)
    vf.run(targets, args.yaw, shared_csv=shared_csv)


if __name__ == "__main__":
    main()
