#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_video.py — 从视频里挑出「人脸质量好」的原始帧，导出为 JPG（不重新编码视频）

流程（方案1）:
  0. 解码帧源（--decode，默认 auto）: auto/gpu = ffmpeg NVDEC+GPU 转换直出
     bgr24 raw 到 stdout 管道、Python 内存流式读（与 ffmpeg 路径同一命令
     bit-exact，不落 3GB raw）；失败自动回退 ffmpeg；pynvvc = PyNvVideoCodec
     NVDEC 硬解帧源（自抽帧 + RGB→BGR，失败逐级回退 管道→落盘）；
     ffmpeg = 旧落盘路径。
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
import ctypes
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
from src.frame_dedup import FrameDedupSelector


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


def _nframes_from_meta(st, fmt_dur, fps):
    """容器元数据估帧数：nb_frames 优先，缺失时 duration×fps。都拿不到返回 0。

    允许 ±1 误差：nframes 只用于 verbose 进度显示（见 probe 文档字符串），
    解码循环由 EOF 驱动，绝不依赖这个数，所以不存在漏帧/死循环风险。
    """
    try:
        nb = int(st.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nb = 0
    if nb > 0:
        return nb
    # stream 级 duration 优先（更贴近该流），其次容器级 format.duration
    try:
        dur = float(st.get("duration") or 0) or float(fmt_dur or 0)
    except (TypeError, ValueError):
        dur = 0.0
    if fps > 0 and dur > 0:
        return max(1, int(round(dur * fps)))
    return 0


def probe(path):
    """Return (width, height, fps, nframes) of the first video stream.

    nframes 走【毫秒级】容器元数据（nb_frames，缺失时 duration×fps，允许 ±1），
    不再默认 ffprobe -count_frames——那是对整条视频逐帧软解数数，4K 视频一个
    13~23s，占 e2e ~75%（见 data/e2e_benchmark_2026-09-28.md）。nframes 仅用于
    verbose 进度显示与日志（w/h/fps 才参与解码命令与循环 shape），元数据缺失/
    解析异常时才回退 -count_frames 精确数一遍（慢但准）。
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration",
           "-of", "json", path]
    st = None
    fmt_dur = 0.0
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                             timeout=60).stdout
        j = json.loads(out)
        st = j["streams"][0]
        fmt_dur = float((j.get("format") or {}).get("duration") or 0.0)
    except Exception:
        st = None                      # 元数据路径失败 → 走下方精确兜底
    if st is not None:
        w, h = int(st["width"]), int(st["height"])
        fps = _parse_rate(st.get("avg_frame_rate") or st.get("r_frame_rate"))
        nframes = _nframes_from_meta(st, fmt_dur, fps)
        if nframes > 0:
            return w, h, fps, nframes
    # 兜底：旧方法（-count_frames 全软解逐帧计数，慢但精确）
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


# ---------------- 解码帧源 ----------------
class _PipeDecodeError(RuntimeError):
    """管道解码失败（ffmpeg 非 0 退出 / 中途断流），auto 模式据此回退落盘路径。"""


class _PipeRaw:
    """ffmpeg NVDEC(+GPU 转换) 直出 bgr24 raw 到 stdout 管道的帧源（替代 3GB 落盘）。

    与落盘路径【同一条 ffmpeg 命令】(仅输出改 pipe:1)，逐帧字节 bit-exact
    （已用 md5 对照验证）。stderr 由后台线程排空（防管道死锁，正是当年放弃
    pipe 的原因），并保留尾部错误信息用于回退诊断。

    读语义与文件一致：read(size) 返回恰好 size 字节；干净 EOF 返回 b""；
    ffmpeg 非 0 退出/中途断流抛 _PipeDecodeError。
    waited 累计等待 ffmpeg 产帧的秒数（= 解码+转换+D2H+管道传输），
    供 breakdown 把该段计入 decode。

    ⚠️ 内部必须以【小分块(32KB)】os.read：Windows 管道对大请求(如 16MB)的
    ReadFile 会阻塞到攒满/EOF，实测把吞吐从 ~1.2GB/s 压到 6~50MB/s
    （30fps 源尤其严重）。32KB 分块读全 4K 视频 ~1.2GB/s。
    """

    def __init__(self, cmd):
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        self._fd = self._proc.stdout.fileno()
        self._chunks = []
        self._avail = 0
        self._closed = False
        self.waited = 0.0
        self._err_tail = []
        self._th = threading.Thread(target=self._drain_err, daemon=True)
        self._th.start()

    def _drain_err(self):
        try:
            while True:
                line = self._proc.stderr.readline()
                if not line:
                    break
                self._err_tail.append(line.decode("utf-8", "replace").rstrip())
                if len(self._err_tail) > 30:
                    del self._err_tail[0]
        except (OSError, ValueError):
            pass

    def read(self, size):
        t1 = time.time()
        while self._avail < size:
            chunk = os.read(self._fd, 1 << 15)   # 32KB 分块(见类注释)
            if not chunk:
                break
            self._chunks.append(chunk)
            self._avail += len(chunk)
        self.waited += time.time() - t1
        if self._avail >= size:
            out = []
            need = size
            while need:
                c = self._chunks[0]
                if len(c) <= need:
                    out.append(c)
                    need -= len(c)
                    self._avail -= len(c)
                    self._chunks.pop(0)
                else:
                    out.append(c[:need])
                    self._chunks[0] = c[need:]
                    self._avail -= need
                    need = 0
            return b"".join(out)
        # 不足一帧：干净 EOF 还是错误？
        rc = self._proc.wait()
        self._th.join(timeout=1.0)
        if rc == 0 and self._avail == 0:
            return b""
        raise _PipeDecodeError(
            f"ffmpeg 管道解码中断 rc={rc} 残留{self._avail}B: "
            + " | ".join(self._err_tail[-5:]))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdout.close()
        except (OSError, AttributeError):
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except OSError:
                pass
        self._th.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------- pynvvc (PyNvVideoCodec / NVDEC) 帧源 ----------------
# 选帧标定表：GT 是 ffmpeg `-vf fps=10` 抽帧，对应【源帧号 = stride*k + delta】。
# 由内容匹配谷底法标定（4 个 4K 竖屏 HEVC 视频全部一致），见
# data/e2e_benchmark_2026-09-28.md §1/§8.5。表外帧率（VFR / 24 / 25 / 59.94…）
# 【绝不猜】→ _pynvvc_stride_delta 返回 None，回退 FFmpeg 管道路径。
_PYNVVC_ALIGN = {60.0: (6, 2), 30.0: (3, 1)}


class _PynvvcDecodeError(RuntimeError):
    """pynvvc 帧源失败（import / 建解码器 / 尺寸不符 / 解码中断 / 帧数短缺）。
    --decode pynvvc 模式据此整体回退 FFmpeg 管道路径重跑（该视频从头重跑，
    与 auto 模式管道失败回退落盘的既有语义一致）。"""


def _pynvvc_stride_delta(src_fps, max_fps):
    """按帧率返回 (stride, delta)；无法确定时返回 None。

    源 fps <= max_fps 时 ffmpeg 不加 fps 滤镜、逐帧全取，pynvvc 同样全取
    → stride=1/delta=0 是精确对应，不算猜。
    """
    if not (max_fps and max_fps > 0 and src_fps > max_fps):
        return 1, 0
    for fps_k, (s, d) in _PYNVVC_ALIGN.items():
        if abs(src_fps - fps_k) < 0.05:
            return s, d
    return None


class _PynvvcSource:
    """PyNvVideoCodec(NVDEC 硬解) 帧源，接口对齐 _PipeRaw：
      read(size) -> 恰好 size 字节的 1-D uint8 ndarray（bgr24，下一选中帧）；
                    干净 EOF -> b""；解码异常 -> _PynvvcDecodeError
      waited     -> read() 内累计耗时（解码+拉帧+RGB→BGR），_run_pass 计入 decode 段
      close()    -> 释放 NVDEC 会话（pynvvc 不支持重放/seek，每个视频新建、用完即弃）

    与 _PipeRaw 的关键差异：
    - 选帧按【源帧号 = stride*k + delta】自己抽帧（仅 60/30fps 已标定，表外回退）
    - pynvvc 宿主内存输出【真 RGB】（2.2.3 无 BGR24），必须转 BGR 才能进管线：
      pose68 / gaze / head / imwrite 全按 BGR 语义，letterbox kernel 也是 BGR 输入
    - EOF 按 get_stream_metadata().num_frames 兜底核对（pynvvc 到 EOF 会刷
      INVALID INDEX WARN，无害）；实际以解出的帧为准，元数据只用来发现中途断流
    - read() 返回 ndarray 而非 bytes：np.frombuffer 对它是零拷贝视图，
      省掉 tobytes 那次全帧拷贝（4K 帧 ~4ms，625 帧 ~2.6s）——调用方
      `np.frombuffer(buf, np.uint8).reshape(h, w, 3)` 与 len(buf) 语义不变
    """

    def __init__(self, video, w, h, fps, max_fps):
        self.waited = 0.0
        self._closed = False
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        sd = _pynvvc_stride_delta(self.fps, max_fps)
        if sd is None:
            raise _PynvvcDecodeError(
                f"src_fps={self.fps:g} 不在已标定集合 {sorted(_PYNVVC_ALIGN)}")
        self.stride, self.delta = sd
        self._next_want = self.delta
        self._given = 0
        try:
            import PyNvVideoCodec as pnv
            self.dec = pnv.CreateSimpleDecoder(
                video, outputColorType=pnv.OutputColorType.RGB)
            md = self.dec.get_stream_metadata()
            self.num_frames = int(getattr(md, "num_frames", 0) or 0)
            mw = int(getattr(md, "width", 0) or 0)
            mh = int(getattr(md, "height", 0) or 0)
        except Exception as e:
            self.dec = None
            raise _PynvvcDecodeError(f"pynvvc 建解码器失败: {e!r}") from e
        # 尺寸/旋转守卫：pynvvc 与 ffprobe 尺寸不一致（如旋转 metadata 处理不同）
        # 时像素内容必错 → 回退 FFmpeg 管道（ffmpeg 会 autorotate）
        if (mw, mh) != (self.w, self.h):
            self.dec = None
            raise _PynvvcDecodeError(
                f"pynvvc 解码尺寸 {mw}x{mh} != ffprobe {self.w}x{self.h}"
                f"（旋转/元数据差异）")
        self._it = self._iter_frames()

    def _read_frame(self, f):
        """pynvvc DecodedFrame（宿主内存, useDeviceMemory=0 默认）→ (h,w,3) RGB 视图"""
        ptr = int(f.cuda()[0].dataptr)
        buf = (ctypes.c_ubyte * (self.h * self.w * 3)).from_address(ptr)
        return np.ctypeslib.as_array(buf).reshape(self.h, self.w, 3)

    def _iter_frames(self):
        """顺序解全部源帧，只产出选中帧（源帧号 = stride*k + delta）。"""
        got = 0
        while True:
            try:
                frames = self.dec.get_batch_frames(8)
            except Exception:
                return                    # EOF（pynvvc 以异常/空列表表达）
            if not frames:
                return
            for f in frames:
                j = got
                got += 1
                if j == self._next_want:
                    yield self._read_frame(f)
            del frames

    def _expected_frames(self):
        """按标定选帧规则期望产出多少帧（元数据源帧数 → 选中帧数）。"""
        if self.num_frames <= 0:
            return 0
        if self.stride == 1 and self.delta == 0:
            return self.num_frames
        return max(0, (self.num_frames - self.delta - 1) // self.stride + 1)

    def read(self, size):
        t1 = time.time()
        try:
            rgb = next(self._it)
        except StopIteration:
            self.waited += time.time() - t1
            # 元数据说还有帧却解不出来了 → 中途断流，回退 FFmpeg 管道兜底
            exp = self._expected_frames()
            if self.num_frames > 0 and self._given < exp - 1:
                raise _PynvvcDecodeError(
                    f"pynvvc 提前断流: 解出 {self._given} 帧 < 期望 {exp}"
                    f"（源 {self.num_frames} 帧）") from None
            return b""
        except Exception as e:
            self.waited += time.time() - t1
            raise _PynvvcDecodeError(f"pynvvc 解码中断: {e!r}") from e
        self.waited += time.time() - t1        # 解码+拉帧段
        t2 = time.time()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # 唯一一次全帧拷贝(带通道交换)
        self.waited += time.time() - t2        # 色彩转换段（计入 decode）
        self._next_want += self.stride
        self._given += 1
        out = bgr.reshape(-1)                  # 1-D 视图: len == h*w*3, 零拷贝
        if out.size != size:
            raise _PynvvcDecodeError(
                f"pynvvc 帧 {self.w}x{self.h} bytes={out.size} != 期望 {size}")
        return out

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._it = None
        self.dec = None          # 释放 NVDEC 会话（不支持重放, 用完即弃）

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------- 解码/推理流水线（producer-consumer） ----------------
# 修改前的结构特点: 解码与 GPU 推理【完全串行】—— _run_pass 主线程循环
# 「read 64 帧(阻塞等解码) → 攒批推理」, 解码时 GPU 闲、推理时解码器闲。
# 这里把「从帧源 read(size) 取帧」挪进后台线程(producer), 经【有界队列】交给
# 主线程攒批推理(consumer)。帧源 read/close/waited 接口在 _PipeRaw /
# _PynvvcSource 已统一, 所以一条包装覆盖全部「边解边读」帧源。
#
# ⚠️ 实测(见 data/e2e_benchmark_2026-09-28.md §9.2): 线程级重叠受 CPython GIL
# 限制 —— pynvvc 的 get_batch_frames 是【全程持 GIL】的阻塞 C 调用(主线程
# GIL 轮询 63 it/s, 理想 ~2000; ThreadedDecoder 同样 64 it/s 且吞吐更差),
# 管道路径则被「Python 侧 32KB 分块读循环」与推理侧 numpy/cv2 的 GIL 争用
# 拉平, 两源实测重叠都≈0(pynvvc 净收益 1~3%, 管道 ≈0)。真正的解码/推理
# 重叠需要【子进程】解码(共享内存传帧), 本类保留作为其落点; 判定逐位不变。
# 内存: 4K bgr24 一帧 ~25MB → 队列深度上限 16 ≈ 400MB
# （--pipeline-depth 可调; 队列满时 producer 阻塞, 天然背压不涨内存;
#   实测稳态深度 ~2 帧 ≈ 50MB, 上限只在推理侧骤慢时才触及）。
_SRC_EOF = object()      # 生产者→消费者 的结束哨兵（不会与任何数据帧混淆）


class _PipelinedSource:
    """帧源流水线包装: producer 线程不断 make_src().read(size) 取帧入有界队列,
    主线程从队列取帧攒批推理。read/close/waited 接口与底层帧源一致, 对
    _run_pass 完全透明（调用方代码不变）。

    异常传播与回退链: producer 线程里的一切异常（含 _PipeDecodeError /
    _PynvvcDecodeError / 建解码器失败）都【原样】存下, 消费者 read() 在取到
    结束哨兵时原样重抛 —— 异常类型/信息不变, process() 的既有逐级回退
    （pynvvc→管道→落盘 / auto 管道→落盘）语义完全保留; 回退时 close() 停
    producer 并释放底层帧源（NVDEC 会话 / ffmpeg 进程不泄漏）。

    提前退出（Ctrl-C / 异常）: close() 置 stop 事件, producer 的 put/read
    循环带 stop 检查退出; 线程为 daemon, 主线程被打断时进程也能退出。

    为什么帧源在 producer 线程里构建: pynvvc 解码器与 CUDA context 的线程
    归属绑定, 创建和使用放同一线程最稳; 构建失败也走同一条异常传播通道。
    """

    def __init__(self, make_src, size, depth=16):
        self._make_src = make_src
        self._size = int(size)
        self._q = queue.Queue(maxsize=max(2, int(depth)))
        self._src = None            # 底层帧源（producer 线程构建并持有）
        self._err = None            # producer 捕获的异常（主线程 read() 重抛）
        self._waited_done = 0.0     # close() 后留存 waited（decode 段计时用）
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._produce, daemon=True,
                                    name="decode-producer")
        self._th.start()

    def _produce(self):
        try:
            src = self._make_src()
        except BaseException as e:      # 建帧源失败 → 记下, 主线程 read() 重抛
            self._err = e
            self._put(_SRC_EOF)
            return
        self._src = src
        while not self._stop.is_set():
            try:
                buf = src.read(self._size)
            except BaseException as e:  # 解码错误 → 保留原类型/信息供回退链
                self._err = e
                break
            if len(buf) < self._size:   # 干净 EOF（与主循环 len<size 判据一致）
                break
            if not self._put(buf):      # stop 已置位（主线程提前退出）
                return
        self._put(_SRC_EOF)

    def _put(self, item):
        """带 stop 检查的有界 put: 队列满且主线程已停时退出（不死锁）。"""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def read(self, size):
        """取一帧（恰好 size 字节 / size 元素）; EOF 返回 b""; 解码错误原样抛。"""
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if not self._th.is_alive():
                    if self._err is not None:
                        raise self._err
                    return b""    # producer 已退且无哨兵（仅 close 后会发生）
                continue
            if item is _SRC_EOF:
                if self._err is not None:
                    raise self._err
                return b""
            return item

    @property
    def waited(self):
        """底层帧源在 producer 线程内累计的解码等待（read 全部结束后读取）。"""
        s = self._src
        if s is not None:
            return float(getattr(s, "waited", 0.0))
        return self._waited_done

    def close(self):
        self._stop.set()
        deadline = time.time() + 5.0
        while self._th.is_alive() and time.time() < deadline:
            try:
                self._q.get_nowait()   # 排空队列, 解锁阻塞在 put 上的 producer
            except queue.Empty:
                self._th.join(timeout=0.05)
        self._th.join(timeout=2.0)
        if self._src is not None:
            # 先留存 waited（decode 段计时要落在 close 之后）, 再释放帧源
            self._waited_done = float(getattr(self._src, "waited", 0.0))
            self._src.close()          # 释放 NVDEC 会话 / 结束 ffmpeg 进程
            self._src = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------- filter ----------------
class VideoFilter:
    def __init__(self, engine, conf, target_long, batch, detect_chunk,
                 max_fps, jpg_quality, score_name, verbose=False, jpg_workers=8,
                 down_min=0.46, pose_engine=None, pitch_max=25.0,
                 pitch_min=-25.0, ear_min=0.25, min_face_h=0.0,
                 gaze_max=0.12, gaze_dy_dev=0.0,
                 gaze_model="iris", gaze_model_path=None,
                 gaze_pitch_max=15.0, gaze_yaw_max=15.0,
                 head_on=True, head_conf=0.30, head_model_path=None,
                 dedup=False, dedup_cut_lo=2.0, dedup_cut_hi=10.0,
                 dedup_min_seg=2, dedup_max_seg=10, dedup_thumb=128,
                 dedup_sharp_side=256, dedup_backups=2, dedup_max_mem=48.0,
                 hw_decode=True, decode_mode="auto",
                 pipeline=True, pipeline_depth=16, head_batch=True,
                 head_letterbox_workers=8):
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
        # 人脸最小尺寸（可调试，默认 0=关）：主脸框高 < min_face_h 像素 → 判 too_small
        # 直接剔除，跳过 pose68/gaze/head 推理，不落 JPG。用于剔除"远处小脸"（小脸上
        # 姿态估计不可靠）。阈值按目标视频实测标定；单位=人脸框高像素。
        self.min_face_h = min_face_h
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
        # 抽帧去重 + 段内选最佳帧（--dedup 开启时生效，默认关，行为与旧版一致）
        self.dedup = dedup
        self.dedup_cut_lo = dedup_cut_lo
        self.dedup_cut_hi = dedup_cut_hi
        self.dedup_min_seg = dedup_min_seg
        self.dedup_max_seg = dedup_max_seg
        self.dedup_thumb = dedup_thumb
        self.dedup_sharp_side = dedup_sharp_side
        self.dedup_backups = dedup_backups
        self.dedup_max_mem = dedup_max_mem
        # NVDEC GPU 硬解（--hw-decode，默认开）：dec_cmd 在 -i 前插 -hwaccel cuda；
        # 硬解失败（ffmpeg 非 0 退出/异常）自动回退 CPU 软解重跑一次
        self.hw_decode = hw_decode
        # 解码帧源（--decode）: auto=GPU 管道, 失败自动回退 ffmpeg 落盘路径;
        # gpu=仅 GPU 管道; ffmpeg=现有 3GB raw 落盘路径（旧行为, 一行未改）
        self.decode_mode = decode_mode
        # 解码/推理流水线重叠（--no-pipeline 可关; 仅对 pipe/pynvvc 帧源生效）
        self.pipeline = pipeline
        self.pipeline_depth = pipeline_depth
        # head 闸门批量化（A/B 用, 判定语义不变）
        self.head_batch = head_batch
        # head letterbox 线程池大小（4K 帧 letterbox 4.0ms → 8 线程 2.05ms, 实测）
        self.head_letterbox_workers = head_letterbox_workers

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

    def _gate_pre(self, frame, d, k, p68, dy_all):
        """闸门链前半：_judge → 低头复核 → Stage2 眼神（不含 head）。
        （从 _gate_chain 原样拆出，判定逻辑/顺序一行未改。）

        返回 (verdict, score, pose, down, nfaces, ear, gaze_mag, gaze_dy,
              t_judge, t_gaze)。verdict=="keep" 且 nfaces==1 且 head 闸门开启
        → 该帧还需过 Stage1.5 head（调用方决定逐帧或成批跑）。"""
        t1 = time.time()
        verdict, score, pose, down, nfaces = self._judge(
            d, k, p68[0] if p68 else None, p68[1] if p68 else None)
        t_judge = time.time() - t1
        ear = p68[1] if p68 else None
        # 低头复核：单脸 pose 已合格，但 down_ratio < down_min → 按低头剔除
        if (verdict == "keep" and self.down_min is not None
                and down is not None and down < self.down_min):
            verdict = "down"
        # Stage2 眼神闸门：Stage1 全过（头基本正）的单脸帧
        gaze_mag = None
        gaze_dy = None
        t_gaze = 0.0
        if (verdict == "keep" and self.gaze is not None and nfaces == 1):
            t1 = time.time()
            top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
            g = self.gaze.estimate(frame, d[top, :4])
            t_gaze = time.time() - t1
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
        return (verdict, score, pose, down, nfaces, ear, gaze_mag,
                gaze_dy, t_judge, t_gaze)

    def _gate_chain(self, frame, d, k, p68, dy_all):
        """单帧完整闸门链：_gate_pre → Stage1.5 人头（逐帧 batch=1）。
        （dedup 的 backup 补判复用同一实现；视频主循环默认走成批 head，
        见 _run_pass 内 head_jobs 两阶段。）

        p68: 本帧主脸的 68 点结果 ((yaw,pitch,roll), ear)，None 则 _judge 内回退
             SCRFD 5 点 solvePnP（无 EAR）。
        dy_all: 本视频所有测到 gaze 帧的 dy 收集（2-pass 基线），本方法在测到
                gaze 时 append（resnet34 模式单位=度，iris 模式=归一化偏移）。

        返回 (verdict, score, pose, down, nfaces, ear, gaze_mag, gaze_dy, nheads,
              t_judge, t_gaze, t_head)；verdict 语义与 _judge 文档一致，另加
              "down"（低头比复核）/ "gaze" / "multi_head"。
        """
        (verdict, score, pose, down, nfaces, ear, gaze_mag, gaze_dy,
         t_judge, t_gaze) = self._gate_pre(frame, d, k, p68, dy_all)
        # Stage1.5 人头闸门：其它闸门全过且单人脸，检出 >=2 个人头 → multi_head
        nheads = None
        t_head = 0.0
        if (verdict == "keep" and nfaces == 1 and self.head is not None):
            t1 = time.time()
            _hb = self.head.detect(frame)
            t_head = time.time() - t1
            nheads = len(_hb)
            if nheads >= 2:
                verdict = "multi_head"
        return (verdict, score, pose, down, nfaces, ear, gaze_mag,
                gaze_dy, nheads, t_judge, t_gaze, t_head)

    @staticmethod
    def _decode_to_raw(dec_cmd, soft_cmd, hw_decode):
        """跑解码命令把视频落 raw 盘。hw_decode=True 时 dec_cmd 带 -hwaccel cuda；
        硬解失败（ffmpeg 非 0 退出/异常）→ 打印 WARNING 并用 soft_cmd（纯软解）
        重跑一次兜底。返回实际使用的解码模式: 'hw' / 'cpu' / 'cpu-fallback'。"""
        try:
            subprocess.run(dec_cmd, check=True)
            return "hw" if hw_decode else "cpu"
        except (subprocess.SubprocessError, OSError) as e:
            if not hw_decode:
                raise
            msg = str(e).splitlines()[0] if str(e) else "无输出"
            print(f"  WARNING: NVDEC 硬解失败（{e.__class__.__name__}: {msg}），"
                  f"回退 CPU 软解重跑", flush=True)
            subprocess.run(soft_cmd, check=True)
            print(f"  [decode] 回退软解完成，本次运行继续（hw -> cpu-fallback）",
                  flush=True)
            return "cpu-fallback"

    def process(self, video_path, out_dir, yaw_lim, shared_csv=False):
        self.yaw_lim = yaw_lim
        w, h, fps, nframes = probe(video_path)
        size = w * h * 3
        stem = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(out_dir, exist_ok=True)

        # 帧率上限：源 > max_fps 才降（均匀采样）
        apply_fps = bool(self.max_fps and self.max_fps > 0 and fps > self.max_fps)
        dec_fps = self.max_fps if apply_fps else fps

        mode = self.decode_mode
        # 解码帧源尝试顺序（失败逐级整体回退重跑，最终兜底 ffmpeg 落盘路径）：
        #   auto   : FFmpeg NVDEC 管道 → ffmpeg 落盘（默认，行为与旧版一致）
        #   pynvvc : PyNvVideoCodec 硬解 → FFmpeg NVDEC 管道 → ffmpeg 落盘
        #   gpu    : 仅 FFmpeg NVDEC 管道；ffmpeg: 仅落盘路径（旧行为）
        _SRC_LABEL = {"pynvvc": "pynvvc(NVDEC) 帧源", "pipe": "FFmpeg NVDEC 管道",
                      "file": "ffmpeg 落盘"}
        chain = {"auto": ("pipe", "file"),
                 "pynvvc": ("pynvvc", "pipe", "file"),
                 "gpu": ("pipe",),
                 "ffmpeg": ("file",)}[mode]
        last_err = None
        for i, kind in enumerate(chain):
            try:
                return self._run_pass(video_path, out_dir, yaw_lim, shared_csv,
                                      w, h, size, fps, apply_fps, dec_fps,
                                      nframes, stem,
                                      use_pipe=(kind == "pipe"),
                                      use_pynvvc=(kind == "pynvvc"))
            except (_PynvvcDecodeError, _PipeDecodeError) as e:
                last_err = e
                msg = str(e).splitlines()[0] if str(e) else "无输出"
                note = ""
                if i + 1 < len(chain):
                    note = f"，回退 {_SRC_LABEL[chain[i + 1]]}路径重跑"
                print(f"  WARNING: {_SRC_LABEL[kind]}解码失败"
                      f"（{e.__class__.__name__}: {msg}）{note}", flush=True)
        raise last_err

    def _run_pass(self, video_path, out_dir, yaw_lim, shared_csv,
                  w, h, size, fps, apply_fps, dec_fps, nframes, stem, use_pipe,
                  use_pynvvc=False):
        # 帧源: use_pynvvc=True → PyNvVideoCodec(NVDEC) 硬解帧源（自抽帧 + RGB→BGR）；
        #       use_pipe=True → 与落盘路径【同一条 ffmpeg 命令】(仅输出改 pipe:1)，
        #       NVDEC+GPU 转换直出 bgr24 到内存，不落 3GB raw（输出 bit-exact）；
        #       use_pipe=False → 现有落盘路径（原代码原样保留）。
        # 这里只准备 make_src 工厂（真正构建延后）: 流水线开时帧源在 producer
        # 线程里构建（CUDA context 归属其使用线程）; 关时在主线程构建（原行为）。
        _make_src = None
        if use_pynvvc:
            # 建解码器/标定失败抛 _PynvvcDecodeError，由 process() 回退下一路帧源
            tmp_raw = None
            _make_src = lambda: _PynvvcSource(video_path, w, h, fps, self.max_fps)
        elif use_pipe:
            _tail = []
            if apply_fps:
                _tail += ["-vf", f"fps={dec_fps:g}"]
            _tail += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
            pipe_cmd = (["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", video_path]
                        if self.hw_decode
                        else ["ffmpeg", "-v", "error", "-i", video_path]) + _tail
            tmp_raw = None
            _make_src = lambda cmd=pipe_cmd: _PipeRaw(cmd)
        else:
            f_src = None
            # 临时 raw 文件：Windows 下 Python 读管道极慢(~30MB/s)，读文件快~90x，
            # 且先落盘再读可彻底消除 ffmpeg stdin 管道背压死锁。
            fd_tmp, tmp_raw = tempfile.mkstemp(suffix=".raw", prefix=f"filt_{stem}_")
            os.close(fd_tmp)
            # 解码命令：hw_decode=True 时 -i 前插 -hwaccel cuda（NVDEC）；
            # 另备一条无 -hwaccel 的软解命令，硬解失败时兜底重跑
            _tail = []
            if apply_fps:
                _tail += ["-vf", f"fps={dec_fps:g}"]
            _tail += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-y", tmp_raw]
            dec_cmd = (["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", video_path]
                       if self.hw_decode else ["ffmpeg", "-v", "error", "-i", video_path]) + _tail
            soft_cmd = ["ffmpeg", "-v", "error", "-i", video_path] + _tail

        # 解码/推理流水线重叠（--no-pipeline 可关）: producer 线程解码 → 有界队列
        # → 主线程攒批推理。仅对「边解边读」的帧源（pynvvc/管道）生效; ffmpeg 落盘
        # 路径的解码在读取前已整体完成, 无可重叠, 保持原样（旧行为一行未改）。
        # 帧源构建失败/解码失败经 _PipelinedSource.read() 原样抛回主线程 →
        # process() 的逐级回退链（pynvvc→管道→落盘 / auto 管道→落盘）语义不变。
        pipelined = bool(_make_src is not None and self.pipeline)
        if _make_src is not None:
            if pipelined:
                f_src = _PipelinedSource(_make_src, size,
                                         depth=self.pipeline_depth)
            else:
                f_src = _make_src()
        else:
            f_src = None

        if self.verbose:
            _src = "pynvvc" if use_pynvvc else ("pipe" if use_pipe else "file")
            print(f"  decode {os.path.basename(video_path)}  {w}x{h}  src_fps={fps:.3f} "
                  f"-> dec_fps={dec_fps:.3f}{' (downsampled)' if apply_fps else ''}  "
                  f"~{nframes} frames  hwdec={'cuda' if self.hw_decode else 'cpu'}  "
                  f"src={_src}  pipeline={'on' if pipelined else 'off'}"
                  + (f"(depth={self.pipeline_depth})" if pipelined else ""))

        t0 = time.time()
        frame_no = kept = drop_noface = drop_pose = drop_down = drop_multi = 0
        drop_up = drop_blink = drop_downp = drop_gaze = drop_gazedown = 0
        drop_head = 0
        drop_toosmall = 0
        t_dec = t_read = t_det = t_judge = t_gaze = t_write = 0.0
        t_head = 0.0
        ex = ThreadPoolExecutor(max_workers=self.jpg_workers)
        futures = []
        # 临时 pose 报告（与临时命名同生命周期，定好阈值后一并删除）：
        # 每帧一行 score/yaw/pitch/roll，供人工定「不看正前方」的阈值
        report = []
        dy_all = []    # 所有测到 gaze 的帧的 dy（2-pass 基线=其中位数）
        dy_gate = []   # (帧号, 写出路径, dy)：keep 候选帧，循环后统一套用 dy 闸门
        # dedup 统计（未开 --dedup 时全为 0，不参与汇总）
        decoded_frames = 0
        dedup_rows = []   # (frame, diff, seg_id, seg_start, seg_end, sharp, is_rep)
        seg_stats = {"segs": 0, "sent": 0, "reps": 0, "bk_judged": 0,
                     "sec_judged": 0, "sec_hits": 0,
                     "drop_seg": 0, "backup_hits": 0}
        dedup_dy_hits = 0
        try:
            if use_pynvvc:
                dec_mode = "pynvvc"
                t_dec = 0.0
            elif use_pipe:
                dec_mode = "pipe"
                t_dec = 0.0
            else:
                t1 = time.time()
                dec_mode = self._decode_to_raw(dec_cmd, soft_cmd, self.hw_decode)
                t_dec = time.time() - t1

            if self.dedup:
                # ---------- dedup 模式 ----------
                # 连续相似帧分段、段内选最清晰 rep 送检；快速动作逐帧独立成段
                # → 不漏帧。段关闭才产出；rep 被闸门 drop 时按清晰度次序对
                # backups 补跑闸门取第一个 keep，全 fail 计 drop_seg。
                # 输出文件名仍用【原始解码帧号】。
                deduper = FrameDedupSelector(
                    cut_lo=self.dedup_cut_lo, cut_hi=self.dedup_cut_hi,
                    min_seg=self.dedup_min_seg, max_seg=self.dedup_max_seg,
                    thumb_side=self.dedup_thumb,
                    sharp_side=self.dedup_sharp_side,
                    n_backups=self.dedup_backups,
                    max_mem_mb=self.dedup_max_mem)
                segs = []   # 已关闭待送检段: (rep_frame, rep_idx, backups, meta)
                SEG_Q_CAP = max(8, self.detect_chunk // 4)
                SEG_Q_BYTES = 160 * 1024 * 1024  # 挂起段(全段帧)内存上限
                # 解码帧号→秒 的换算(段内跨秒补覆盖用); fps 探测失败时按 10 兜底
                sec_fps = dec_fps if (dec_fps and dec_fps > 0) else 10.0

                def _count_drop(vd):
                    nonlocal drop_noface, drop_pose, drop_down, drop_multi
                    nonlocal drop_up, drop_blink, drop_downp, drop_gaze, drop_head
                    nonlocal drop_toosmall
                    if vd == "no_face":
                        drop_noface += 1
                    elif vd == "down":
                        drop_down += 1
                    elif vd == "downp":
                        drop_downp += 1
                    elif vd == "multi":
                        drop_multi += 1
                    elif vd == "up":
                        drop_up += 1
                    elif vd == "blink":
                        drop_blink += 1
                    elif vd == "gaze":
                        drop_gaze += 1
                    elif vd == "multi_head":
                        drop_head += 1
                    elif vd == "too_small":
                        drop_toosmall += 1
                    else:
                        drop_pose += 1

                def _detect_batch(frames):
                    """一列帧批量 SCRFD 检测 + 68 点姿态。
                    返回 (dets, kpss, pose_map)，pose_map[批内序号]=((yaw,pitch,roll),ear)。"""
                    nonlocal t_det
                    t1 = time.time()
                    dets, kpss = self.det.detect(
                        frames, threshold=self.conf,
                        target_long=self.target_long,
                        pipeline=True, gpu_preprocess=True)
                    t_det += time.time() - t1
                    # too_small 帧跳过 pose68 推理（min_face_h 开启时）
                    ts_mask = set()
                    if self.min_face_h and self.min_face_h > 0:
                        for i, d in enumerate(dets):
                            if len(d) > 0:
                                top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                if (d[top, 3] - d[top, 1]) < self.min_face_h:
                                    ts_mask.add(i)
                    pose_map = {}
                    if self.pose68 is not None:
                        items = []
                        for i, (fr, d) in enumerate(zip(frames, dets)):
                            if len(d) > 0 and i not in ts_mask:
                                top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                items.append((i, fr, d[top, :4]))
                        if items:
                            poses = self.pose68.estimate_batch(
                                [(fr, bb) for _, fr, bb in items])
                            pose_map = {i: p for (i, _, _), p in zip(items, poses)}
                    return dets, kpss, pose_map

                def _judge_and_write(frame, idx, d, k, p68):
                    """单帧跑闸门链；keep 则写出（文件名用原始解码帧号 idx）
                    返回 True；否则计 drop 返回 False。"""
                    nonlocal frame_no, kept, t_judge, t_gaze, t_head, drop_toosmall
                    # too_small 直接判，跳过 pose/gaze/head（_gate_chain 内含昂贵闸门）
                    if (self.min_face_h and self.min_face_h > 0
                            and len(d) > 0):
                        top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                        if (d[top, 3] - d[top, 1]) < self.min_face_h:
                            frame_no += 1
                            score = float(d[top, 4])
                            if self.score_name:
                                report.append([idx, "too_small", score, 0.0, 0.0,
                                               0.0, None, len(d), None, None,
                                               None, None])
                            drop_toosmall += 1
                            if self.verbose:
                                print(f"    #{idx:05d} drop  (too_small  "
                                      f"fh={d[top, 3]-d[top, 1]:.0f})")
                            return False
                    (verdict, score, pose, down, nfaces, ear, gaze_mag,
                     gaze_dy, nheads, _tj, _tg, _th) = self._gate_chain(
                         frame, d, k, p68, dy_all)
                    t_judge += _tj
                    t_gaze += _tg
                    t_head += _th
                    frame_no += 1   # dedup 模式下 = 送检帧数(reps + backup 补判)
                    yaw, pitch, roll = pose
                    if self.score_name:
                        report.append([idx, verdict, score, yaw, pitch,
                                       roll, down, nfaces, ear, gaze_mag,
                                       gaze_dy, nheads])
                    if verdict == "keep":
                        base = f"{stem}_{idx:05d}"
                        name = (f"{base}_{score:.2f}_{yaw:.0f}_{pitch:.0f}_{roll:.0f}.jpg"
                                if self.score_name else f"{base}.jpg")
                        futures.append(ex.submit(
                            cv2.imwrite, os.path.join(out_dir, name), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpg_q)]))
                        kept += 1
                        if self.gaze_dy_dev > 0 and gaze_dy is not None:
                            dy_gate.append((idx, os.path.join(out_dir, name), gaze_dy))
                        if self.verbose:
                            print(f"    #{idx:05d} KEEP  score={score:.3f} "
                                  f"yaw={yaw:.0f} pit={pitch:.0f} rol={roll:.0f} "
                                  f"down={down:.2f}")
                        return True
                    _count_drop(verdict)
                    if self.verbose:
                        print(f"    #{idx:05d} drop  ({verdict}  down={down})")
                    return False

                def _process_seg_batch(sb):
                    """一批已关闭段: reps 批量送检→闸门。
                    - rep keep: 段内每个"与 rep 不同秒"的秒, 按清晰度降序逐一补判
                      该秒的帧, 命中 keep 即停(early stop)——眼神/头部在秒内变化快,
                      只补最清晰一帧不够(实证: 154 第15秒);
                    - rep drop: 段内剩余帧按清晰度降序逐一补判, 命中第一个 keep
                      即停(early stop), 全 fail 计 drop_seg。
                    两者合起来: 每个解码秒的候选帧要么有 keep 的 rep, 要么该秒在
                    段内的帧被全扫过 → 覆盖与 A 同帧集等价(仅剩 dy 2-pass 基线差异
                    这一 v1 已知项)。
                    补判帧合并成一次批量 detect + pose68（补判只在 rep 结果出来
                    后发生; gaze/head 昂贵闸门只对过 Stage1 的帧跑）。"""
                    seg_stats["segs"] += len(sb)
                    rep_frames = [s[0] for s in sb]
                    dets, kpss, pose_map = _detect_batch(rep_frames)
                    jobs = []         # job_pos → (kind, satisfied_set)
                    extra_items = []  # (job_pos, kind, sec_or_None, frame, idx)
                    for si, seg in enumerate(sb):
                        _, rep_idx, _bk, meta = seg
                        seg_stats["sent"] += 1
                        seg_stats["reps"] += 1
                        ok = _judge_and_write(rep_frames[si], rep_idx,
                                               dets[si], kpss[si],
                                               pose_map.get(si))
                        job_pos = len(jobs)
                        if ok:
                            # 秒号约定与逐秒直方图一致: sec = frame // dec_fps
                            rep_sec = rep_idx // sec_fps
                            by_sec = {}   # sec → [(frame, idx), ...] 清晰度降序
                            for (idx, _sh, fr) in meta["frames"]:
                                s2 = idx // sec_fps
                                if s2 != rep_sec:
                                    by_sec.setdefault(s2, []).append((fr, idx))
                            if by_sec:
                                jobs.append(("sec", set()))
                                for s2 in sorted(by_sec):
                                    for (fr, idx) in by_sec[s2]:
                                        extra_items.append(
                                            (job_pos, "sec", s2, fr, idx))
                        else:
                            jobs.append(("scan", set()))
                            for (idx, _sh, fr) in meta["frames"]:
                                if idx != rep_idx:
                                    extra_items.append(
                                        (job_pos, "scan", None, fr, idx))
                    solved_scan = set()
                    if extra_items:
                        dets_e, kpss_e, pose_map_e = _detect_batch(
                            [it[3] for it in extra_items])
                        for ei, (job_pos, kind, s2, fr, idx) in enumerate(extra_items):
                            if kind == "scan" and job_pos in solved_scan:
                                continue
                            if kind == "sec" and s2 in jobs[job_pos][1]:
                                continue   # 该秒已有 keep, 剩余候选跳过
                            seg_stats["sent"] += 1
                            if kind == "scan":
                                seg_stats["bk_judged"] += 1
                            else:
                                seg_stats["sec_judged"] += 1
                            if _judge_and_write(fr, idx, dets_e[ei], kpss_e[ei],
                                                pose_map_e.get(ei)):
                                if kind == "scan":
                                    seg_stats["backup_hits"] += 1
                                    solved_scan.add(job_pos)
                                else:
                                    seg_stats["sec_hits"] += 1
                                    jobs[job_pos][1].add(s2)
                    for job_pos, (kind, _sat) in enumerate(jobs):
                        if kind == "scan" and job_pos not in solved_scan:
                            seg_stats["drop_seg"] += 1

                def _drain_segs():
                    while (len(segs) >= SEG_Q_CAP
                           or sum(s[3]["nbytes_total"] for s in segs) > SEG_Q_BYTES):
                        n = min(len(segs), self.detect_chunk)
                        _process_seg_batch(segs[:n])
                        del segs[:n]

                with (f_src if f_src is not None
                      else open(tmp_raw, "rb", buffering=0)) as f:
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
                        for fr in chunk:
                            decoded_frames += 1
                            res = deduper.push(fr, decoded_frames)
                            if res is not None:
                                rep_fr, rep_idx, backups, meta = res
                                segs.append((rep_fr, rep_idx, backups, meta))
                                for (fi, dif, sh, isrep) in meta["rows"]:
                                    dedup_rows.append(
                                        (fi, dif, meta["seg_id"],
                                         meta["seg_start"], meta["seg_end"],
                                         sh, isrep))
                        _drain_segs()
                    res = deduper.flush()   # EOF 强关最后一段
                    if res is not None:
                        rep_fr, rep_idx, backups, meta = res
                        segs.append((rep_fr, rep_idx, backups, meta))
                        for (fi, dif, sh, isrep) in meta["rows"]:
                            dedup_rows.append(
                                (fi, dif, meta["seg_id"],
                                 meta["seg_start"], meta["seg_end"],
                                 sh, isrep))
                    # 排空剩余段：EOF 后不再依赖触发条件，否则尾部不满阈值
                    # 的段会永远留在队列里不被送检(丢帧/丢秒)
                    while segs:
                        n = min(len(segs), self.detect_chunk)
                        _process_seg_batch(segs[:n])
                        del segs[:n]
            else:
                with (f_src if f_src is not None
                      else open(tmp_raw, "rb", buffering=0)) as f:
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

                        # 人脸最小尺寸闸门（可调试，默认关）：主脸框高 < min_face_h
                        # → 本块内标 too_small，跳过 pose68/gaze/head 推理。
                        ts_mask = set()
                        if self.min_face_h and self.min_face_h > 0:
                            for i, d in enumerate(dets):
                                if len(d) > 0:
                                    top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                    if (d[top, 3] - d[top, 1]) < self.min_face_h:
                                        ts_mask.add(i)

                        # 68 点姿态：对本块所有 >=1 脸帧的主脸【批量】计算（整块一次
                        # TRT 调用，~ms 级）。pose_map[块内序号] = ((yaw,pitch,roll), ear)。
                        # too_small 帧不送 pose68（省推理）。
                        pose_map = {}
                        if self.pose68 is not None:
                            items = []
                            for i, (fr, d) in enumerate(zip(chunk, dets)):
                                if len(d) > 0 and i not in ts_mask:
                                    top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                    items.append((i, fr, d[top, :4]))
                            if items:
                                poses = self.pose68.estimate_batch(
                                    [(fr, bb) for _, fr, bb in items])
                                pose_map = {i: p for (i, _, _), p in zip(items, poses)}

                        # ---- 单帧收尾: 填 report 行 + keep 落 JPG / drop 计数 ----
                        def _count_drop(vd):
                            nonlocal drop_noface, drop_pose, drop_down
                            nonlocal drop_multi, drop_up, drop_blink, drop_downp
                            nonlocal drop_gaze, drop_head
                            if vd == "no_face":
                                drop_noface += 1
                            elif vd == "down":
                                drop_down += 1
                            elif vd == "downp":
                                drop_downp += 1
                            elif vd == "multi":
                                drop_multi += 1
                            elif vd == "up":
                                drop_up += 1
                            elif vd == "blink":
                                drop_blink += 1
                            elif vd == "gaze":
                                drop_gaze += 1
                            elif vd == "multi_head":
                                drop_head += 1
                            else:
                                drop_pose += 1

                        def _emit(frame, no, verdict, score, pose, down, nfaces,
                                  ear, gaze_mag, gaze_dy, nheads):
                            """一行 report + keep 落 JPG / drop 计数, 返回该行。
                            （调用方按块内帧序插入 rows[], CSV 帧序与旧版一致。）"""
                            nonlocal kept
                            yaw, pitch, roll = pose
                            row = [no, verdict, score, yaw, pitch, roll, down,
                                   nfaces, ear, gaze_mag, gaze_dy, nheads]
                            if verdict == "keep":
                                base = f"{stem}_{no:05d}"
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
                                    dy_gate.append((no, os.path.join(out_dir, name), gaze_dy))
                                if self.verbose:
                                    print(f"    #{no:05d} KEEP  score={score:.3f} "
                                          f"yaw={yaw:.0f} pit={pitch:.0f} rol={roll:.0f} "
                                          f"down={down:.2f}")
                            else:
                                _count_drop(verdict)
                                if self.verbose:
                                    print(f"    #{no:05d} drop  ({verdict}  down={down})")
                            return row

                        # ---- 两阶段闸门: 阶段1 逐帧 judge/低头复核/gaze(顺序、
                        #      判定逻辑与旧版一致); 阶段2 把本块所有送入 head 的
                        #      帧合成一次 detect_batch ----
                        # head2 引擎 profile max_batch=32; 旧路径逐帧 batch=1、
                        # 每帧一次 execute+stream.synchronize(实测 15.8ms/帧),
                        # 成批后 9.2ms/帧(--head-batch 关可回旧路径 A/B)。
                        head_jobs = []    # (块内序号 i, frame)
                        head_meta = {}    # i → (frame_no, score, pose, down, nfaces, ear, gaze_mag, gaze_dy)
                        rows = [None] * len(chunk)   # report 行按帧序暂存
                        for i, (frame, d, k) in enumerate(zip(chunk, dets, kpss)):
                            frame_no += 1
                            # too_small 直接判：跳过 pose/gaze/head 推理，不落 JPG
                            if (self.min_face_h and self.min_face_h > 0
                                    and i in ts_mask):
                                top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                                score = float(d[top, 4])
                                if self.score_name:
                                    rows[i] = [frame_no, "too_small", score,
                                               0.0, 0.0, 0.0, None, len(d),
                                               None, None, None, None]
                                drop_toosmall += 1
                                if self.verbose:
                                    print(f"    #{frame_no:05d} drop  (too_small  "
                                          f"fh={d[top, 3]-d[top, 1]:.0f})")
                                continue
                            _p = pose_map.get(i)          # ((yaw,pitch,roll), ear) 或 None
                            (verdict, score, pose, down, nfaces, ear, gaze_mag,
                             gaze_dy, _tj, _tg) = self._gate_pre(frame, d, k, _p, dy_all)
                            t_judge += _tj
                            t_gaze += _tg
                            if (verdict == "keep" and nfaces == 1
                                    and self.head is not None):
                                # Stage1.5 head 延后, 与块内其它过闸帧合成一批
                                head_jobs.append((i, frame))
                                head_meta[i] = (frame_no, score, pose, down,
                                                nfaces, ear, gaze_mag, gaze_dy)
                                continue
                            rows[i] = _emit(frame, frame_no, verdict, score, pose,
                                            down, nfaces, ear, gaze_mag, gaze_dy,
                                            None)
                        if head_jobs:
                            t1 = time.time()
                            if self.head_batch:
                                _hbs = self.head.detect_batch(
                                    [fr for _, fr in head_jobs],
                                    workers=self.head_letterbox_workers)
                            else:
                                # A/B: 原逐帧路径（batch=1, 每帧一次 execute+sync）
                                _hbs = [self.head.detect(fr)
                                        for _, fr in head_jobs]
                            t_head += time.time() - t1
                            for (i, frame), _hb in zip(head_jobs, _hbs):
                                (no, score, pose, down, nfaces, ear,
                                 gaze_mag, gaze_dy) = head_meta[i]
                                nheads = len(_hb)
                                verdict = "multi_head" if nheads >= 2 else "keep"
                                rows[i] = _emit(frame, no, verdict, score, pose,
                                                down, nfaces, ear, gaze_mag,
                                                gaze_dy, nheads)
                        if self.score_name:
                            report.extend(r for r in rows if r is not None)

            if use_pipe or use_pynvvc:
                # 帧源内部等待时间（管道: ffmpeg 解码+GPU 转换+D2H+管道传输；
                # pynvvc: NVDEC 解码+拉帧+RGB→BGR）计入 decode 段。
                # 流水线关: read 段计时包含帧源内部等待, 减掉避免双计;
                # 流水线开: 该等待发生在 producer 线程, 主线程的 read 段只剩
                # 「等队列」的时间, 不再减 —— 两段各自独立计时, 此时 breakdown
                # 各段之和 < 墙钟(GIL 限制下两段实际仍近似串行, 见 §9.2)。
                t_dec += f_src.waited
                if not pipelined:
                    t_read = max(0.0, t_read - f_src.waited)

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
                    if self.dedup:
                        dedup_dy_hits += len(hit)   # dedup 开启且 dy 闸门命中(汇总统计)
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
            if dedup_rows:
                # 仅 --dedup 时产出（与 pose_report.csv 并存，不动后者格式）：
                # 逐解码帧的段切分/清晰度明细，供调 cut_lo/cut_hi
                with open(os.path.join(out_dir, "dedup_report.csv"), "w",
                          encoding="utf-8") as dfh:
                    dfh.write("frame,diff,seg_id,seg_start,seg_end,sharp,is_rep\n")
                    for (fi, dif, sid, s0, s1, sh, isrep) in dedup_rows:
                        dfh.write(f"{fi},{dif:.3f},{sid},{s0},{s1},{sh:.1f},{isrep}\n")
        finally:
            if f_src is not None:
                f_src.close()
            ex.shutdown(wait=True)
            if tmp_raw is not None:
                try:
                    os.remove(tmp_raw)
                except OSError:
                    pass

        dt = time.time() - t0
        dm = f"{self.down_min:g}" if self.down_min is not None else "?"
        if self.dedup:
            head_label = f"decoded {decoded_frames} -> sent {frame_no}"
            rate = decoded_frames / dt
        else:
            head_label = f"{frame_no} frames"
            rate = frame_no / dt
        print(f"  {os.path.basename(video_path)}: {head_label} -> "
              f"kept {kept}  (drop {drop_noface} no-face, {drop_pose} pose, "
              f"{drop_up} up, {drop_downp} downp, {drop_gaze} gaze, "
              f"{drop_gazedown} gazedown, {drop_blink} blink, "
              f"{drop_down} down<{dm}, {drop_multi} multi, "
              f"{drop_head} multi-head, {drop_toosmall} too-small)  "
              f"in {dt:.1f}s ({rate:.0f} fps)  -> {os.path.abspath(out_dir)}")
        print(f"    [breakdown] decode={t_dec:5.2f}s  read={t_read:5.2f}s  "
              f"detect={t_det:5.2f}s  judge={t_judge:5.2f}s  gaze={t_gaze:5.2f}s  "
              f"head={t_head:5.2f}s  imwrite={t_write:5.2f}s ({kept} frames)")
        if self.dedup:
            print(f"    [dedup] segments={seg_stats['segs']}  "
                  f"reps={seg_stats['reps']}  "
                  f"backup-judged={seg_stats['bk_judged']}  "
                  f"backup-hit={seg_stats['backup_hits']}  "
                  f"sec-judged={seg_stats['sec_judged']}  "
                  f"sec-hit={seg_stats['sec_hits']}  "
                  f"drop_seg={seg_stats['drop_seg']}  "
                  f"compression={decoded_frames / max(frame_no, 1):.2f}x  "
                  f"dy-gate-hits(dedup kept)={dedup_dy_hits}")
        out = {"frames": frame_no, "kept": kept, "no_face": drop_noface,
               "decode_mode": dec_mode,
               "pose": drop_pose, "down": drop_down, "multi": drop_multi,
               "up": drop_up, "downp": drop_downp, "gaze": drop_gaze,
               "gazedown": drop_gazedown, "blink": drop_blink,
               "head": drop_head, "toosmall": drop_toosmall,
               "pipeline": pipelined, "pipe_depth": self.pipeline_depth,
               "seconds": dt}
        if self.dedup:
            out.update({"decoded": decoded_frames, "segments": seg_stats["segs"],
                        "drop_seg": seg_stats["drop_seg"],
                        "backup_hits": seg_stats["backup_hits"],
                        "dedup_dy_hits": dedup_dy_hits})
        return out

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
    ap.add_argument("--min-face-h", type=float, default=0.0,
                    help="人脸最小尺寸（可调试，默认 0=关闭；单位=人脸框高像素）："
                         "主脸框高 < 此值的帧判 too_small 直接剔除，跳过姿态/眼神/人头"
                         "闸门推理、不落 JPG。用于剔除'远处小脸'（小脸上 1k3d68 姿态估计"
                         "不可靠）。阈值按目标视频实测标定；0=关闭，代码路径与旧版一致")
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
    ap.add_argument("--hw-decode", action=argparse.BooleanOptionalAction, default=True,
                    help="NVDEC GPU 硬解（默认开：ffmpeg 在 -i 前加 -hwaccel cuda；"
                         "--no-hw-decode 回 CPU 软解，即旧行为）。硬解失败"
                         "（非 0 退出/异常）自动 WARNING 并回退软解重跑")
    ap.add_argument("--decode", choices=["auto", "gpu", "ffmpeg", "pynvvc"], default="auto",
                    help="解码帧源（默认 auto）: gpu=ffmpeg NVDEC+GPU 转换直出 bgr24 "
                         "rawvideo 到 stdout 管道、Python 内存流式读（与 ffmpeg 路径同一"
                         "命令 bit-exact，不落 3GB raw，解码与下游检测重叠）；"
                         "auto=gpu，失败（含中途断流）自动 WARNING 回退 ffmpeg 落盘路径"
                         "重跑；pynvvc=PyNvVideoCodec(NVDEC) 硬解帧源（按源帧号 "
                         "stride*k+delta 抽帧，仅 60/30fps 已标定、表外帧率自动回退管道；"
                         "失败同样逐级回退 管道→落盘）；ffmpeg=现有 3GB raw 落盘路径"
                         "（旧行为，一行未改）。--hw-decode 对 auto/gpu/ffmpeg 路径均生效")
    ap.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=True,
                    help="解码/推理流水线重叠（默认开, --no-pipeline 关, 便于 A/B）: "
                         "后台线程从帧源取帧入有界队列, 主线程攒批推理。"
                         "仅对 pipe/pynvvc 帧源生效; 判定结果逐位不变。"
                         "⚠️ 受 CPython GIL 限制, 线程级重叠实测仅 1~3 个百分点"
                         "（pynvvc 的解码调用全程持 GIL, 见基准文档 §9.2）,"
                         "真正的解码/推理重叠需子进程解码")
    ap.add_argument("--pipeline-depth", type=int, default=16,
                    help="流水线队列深度（默认 16 帧; 4K bgr24 一帧 ~25MB → "
                         "16 帧 ≈ 400MB 宿主内存上限, 队列满时解码线程阻塞背压）")
    ap.add_argument("--head-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="head 闸门按块批量化（默认开, --no-head-batch 关）: 把一块内"
                         "所有送入 head 的帧合成一次 detect_batch（原来逐帧 batch=1、"
                         "每帧一次 execute+sync）。判定语义不变（逐位等价已验证）")
    ap.add_argument("--dedup", action="store_true",
                    help="抽帧去重+段内选最佳帧(默认关): 连续相似帧分段, 段内选最清晰一张"
                         "(避运动模糊)送检; 快速动作/scene cut 逐帧独立成段→不漏帧; "
                         "rep 被闸门 drop 时按清晰度次序对 backup 帧补跑闸门取第一个 keep")
    ap.add_argument("--dedup-cut-lo", type=float, default=2.0,
                    help="dedup 低阈值(128px 灰度缩略图相邻帧平均绝对差): diff>=cut_lo 且"
                         "段长>=min_seg 关段(滞回防闪烁)")
    ap.add_argument("--dedup-cut-hi", type=float, default=10.0,
                    help="dedup 高阈值: diff>=cut_hi 立即关段(快速动作/scene cut, "
                         "逐帧独立成段不漏帧)")
    ap.add_argument("--dedup-min-seg", type=int, default=2,
                    help="dedup 最小段长(帧), 低于此长度 diff>=cut_lo 不关段(防闪烁)")
    ap.add_argument("--dedup-max-seg", type=int, default=10,
                    help="dedup 段最大帧数(10fps 下 10=1s 兜底: 全程无动作也每秒出 1 帧)")
    ap.add_argument("--dedup-thumb", type=int, default=128,
                    help="dedup diff 用的灰度缩略图长边(px)")
    ap.add_argument("--dedup-sharp-side", type=int, default=256,
                    help="dedup 清晰度(Laplacian 方差)用的灰度图长边(px)")
    ap.add_argument("--dedup-backup", type=int, default=2,
                    help="dedup 每段保留的次清晰 backup 帧数(rep 被 drop 时按清晰度次序补跑闸门)")
    ap.add_argument("--dedup-max-mem", type=float, default=48.0,
                    help="dedup 段内全分辨率帧内存上限(MB, 高分辨率按帧大小自适应压段长; "
                         "1080p 默认 48 可容 ~8 帧, 4K 帧~24MB 只容 2 帧, 4K 想要真去重可调到 160)")
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
                     min_face_h=args.min_face_h,
                     gaze_max=(args.gaze_max or None),
                     gaze_dy_dev=(args.gaze_dy_dev or 0.0),
                     gaze_model=args.gaze_model,
                     gaze_model_path=args.gaze_model_path,
                     gaze_pitch_max=args.gaze_pitch_max,
                     gaze_yaw_max=args.gaze_yaw_max,
                     head_on=(not args.no_head_gate),
                     head_conf=args.head_conf,
                     head_model_path=args.head_model_path,
                     dedup=args.dedup,
                     dedup_cut_lo=args.dedup_cut_lo,
                     dedup_cut_hi=args.dedup_cut_hi,
                     dedup_min_seg=args.dedup_min_seg,
                     dedup_max_seg=args.dedup_max_seg,
                     dedup_thumb=args.dedup_thumb,
                     dedup_sharp_side=args.dedup_sharp_side,
                     dedup_backups=args.dedup_backup,
                     dedup_max_mem=args.dedup_max_mem,
                     hw_decode=args.hw_decode,
                     decode_mode=args.decode,
                     pipeline=args.pipeline,
                     pipeline_depth=args.pipeline_depth,
                     head_batch=args.head_batch)
    vf.run(targets, args.yaw, shared_csv=shared_csv)


if __name__ == "__main__":
    main()
