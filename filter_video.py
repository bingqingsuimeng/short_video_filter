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
from src.frame_dedup import FrameDedupSelector, ShotTracker


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
    # post6 nsys profiling 兼容: nsys 注入会使子进程(ffprobe)的 stdout 静默变空
    # (实测 exit 0 且文件重定向同样为空), 令元数据探测失败。设 SVF_PROBE_CACHE
    # 时改读预生成的缓存 JSON(键=abs path, 值=[w,h,fps,nframes]; 缓存由同机
    # 无 profiling 的真实 probe() 生成, 数值与实跑一致)。不设该变量时此分支
    # 完全不存在, 默认行为逐字节不变。
    _pc = os.environ.get("SVF_PROBE_CACHE")
    if _pc:
        try:
            with open(_pc, "r", encoding="utf-8") as _pf:
                _hit = json.load(_pf).get(os.path.abspath(path))
            if _hit:
                return int(_hit[0]), int(_hit[1]), float(_hit[2]), int(_hit[3])
        except Exception:
            pass                      # 缓存缺失/损坏 → 落回真实 ffprobe
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
                 dedup_min_seg=2, dedup_max_seg=None, dedup_thumb=128,
                 dedup_sharp_side=256, dedup_backups=2, dedup_max_mem=48.0,
                 dedup_per_shot_max=None, dedup_per_shot_min=1,
                 dedup_max_gap_sec=1.0,
                 hw_decode=True, decode_mode="auto",
                 pipeline=True, pipeline_depth=16, head_batch=True,
                 head_letterbox_workers=8, overlap="auto"):
        # D 轮懒加载(§14): 宿主 SCRFD 实例不再在 __init__ 急建 —— pynvvc-gpu 臂
        # 下 _ensure_gpu_arm 会对同一引擎文件另建 device 实例, 宿主实例是纯死重
        # (§13.7-B: 日志两次 "engine loaded")。__init__ 只存配置, 首次访问时
        # 构建(det property); 非 GPU 臂在 __init__ 末尾立即物化(见下), 计时口径
        # 与旧行为一致(引擎加载在逐视频墙钟之外, r1 口径照旧单列)。
        self._det_engine = engine
        self._det_batch = batch
        self._det_conf = conf
        self._det_obj = None
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
        # D 轮懒加载(§14): 同 SCRFD, 宿主实例改 head property 懒建。加载失败
        # 的语义与旧版一致: 打印 WARNING 后闸门全局关闭(head 恒为 None, 不回退)。
        self._head_on = head_on
        self._head_conf = head_conf
        self._head_model_path = head_model_path
        self._head_obj = None
        self._head_state = "pending"   # pending → ok / failed / off
        # 抽帧去重 + 段内选最佳帧（--dedup 开启时生效，默认关，行为与旧版一致）
        self.dedup = dedup
        self.dedup_cut_lo = dedup_cut_lo
        self.dedup_cut_hi = dedup_cut_hi
        self.dedup_min_seg = dedup_min_seg
        # 段长时间兜底: dedup_max_seg=None(默认)→按 --dedup-max-gap-sec 换算
        # (_dedup_eff_max_seg); 显式给定帧数则优先旧口径
        self.dedup_max_seg = (int(dedup_max_seg)
                              if dedup_max_seg and int(dedup_max_seg) > 0
                              else None)
        if self.dedup and self.dedup_max_seg is None and not (
                dedup_max_gap_sec and float(dedup_max_gap_sec) > 0):
            raise ValueError("--dedup-max-gap-sec 必须 > 0"
                             "(或显式给 --dedup-max-seg)")
        self.dedup_thumb = dedup_thumb
        self.dedup_sharp_side = dedup_sharp_side
        self.dedup_backups = dedup_backups
        self.dedup_max_mem = dedup_max_mem
        # 镜头感知采样(opt-in, 默认全关=旧行为): 硬切(diff>=cut_hi)为镜头边界;
        # per_shot_max=镜头内送检配额上限(None=关, 逐段送检旧行为);
        # per_shot_min=每镜头至少 M 帧送检(1=旧逐段 rep 机制天然满足);
        # max_gap_sec=段长时间兜底间隔(秒), 按 dec_fps 换算 max_seg
        _psm = (dedup_per_shot_max if dedup_per_shot_max is None
                else int(dedup_per_shot_max))
        self.dedup_per_shot_max = _psm if (_psm is not None and _psm > 0) else None
        self.dedup_per_shot_min = max(1, int(dedup_per_shot_min))
        self.dedup_max_gap_sec = float(dedup_max_gap_sec)
        if (self.dedup_per_shot_max is not None
                and self.dedup_per_shot_min > self.dedup_per_shot_max):
            print(f"[filter] WARNING: --dedup-per-shot-min({self.dedup_per_shot_min}) "
                  f"> --dedup-per-shot-max({self.dedup_per_shot_max}), "
                  f"配额镜头按 q=max(N,M)={self.dedup_per_shot_min} 帧送检")
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
        # 解码∥推理重叠（仅 pynvvc-gpu 臂; --overlap auto/on/off）:
        # auto=ThreadedDecoder 可用则开(初始化失败自动回退 A 轮串行臂),
        # off=保留 A 轮串行行为。详见 src/gpu_decode._OverlappedGpuSource。
        self.overlap = overlap
        # --decode pynvvc-gpu 显存直通臂：懒建设备侧资源（首次使用时创建,
        # 见 _ensure_gpu_arm）；其余帧源路径完全不触碰这些资源, 旧行为不变
        self.engine_path = engine                  # SCRFD 原引擎文件（device 臂同引擎另建实例）
        self._head_engine_path = head_model_path   # head 引擎路径（None → 默认解析）
        self._gpu = None
        # D 轮懒加载(§14)收尾: 非 pynvvc-gpu 臂在 __init__ 内立即物化宿主
        # SCRFD/head 实例, 与旧行为逐字节同(引擎加载在 run() 墙钟之外, r1
        # 口径照旧单列); pynvvc-gpu 臂保持懒建(--dedup 也走零拷贝臂, 同样
        # 懒建) —— GPU 臂初始化失败回退宿主臂时首次访问 det/head 再建
        # (回退链无损)。
        if decode_mode != "pynvvc-gpu":
            _ = self.det
            if head_on:
                _ = self.head
        # D 轮 C 项: 落地 pinned slab 一次性预分配(4K UHD 容量 ×OVL_SETS,
        # 仅 pynvvc-gpu 臂 + 重叠开启时)。放在 __init__(r1 口径的"预热单列"
        # 段, 与引擎加载同级, run() 墙钟之外): cuMemHostAlloc 走驱动全局锁,
        # 与后续一切 CUDA 宿主 API 串行(探针 _test/post6_pinned_stall_probe.py:
        # 后台线程分配 1.7GB 期间, 主线程 4MB 小分配最大停顿 245ms) ——
        # 后台线程方案无法与解码会话建立真并行, 一次付清最干净。首视频起
        # 零分配; 超 4K 容量的视频仍走 _run_pass_gpu 内按需扩容路径(逐字节
        # 同旧行为); 分配失败(内存不足) → None → 可分页逐帧回退路径。
        self._land_slabs_pre = None
        if decode_mode == "pynvvc-gpu" and not dedup and overlap != "off":
            try:
                from src.gpu_decode import OVL_SETS as _ov
                import pycuda.driver as _d
                _cap4k = (self.detect_chunk + 8) * 3840 * 2160 * 3
                self._land_slabs_pre = [
                    _d.pagelocked_empty((_cap4k,), np.uint8)
                    for _ in range(_ov)]
            except Exception as _e:
                print(f"[filter] pinned slab 预分配失败, 回退按需分配: {_e}")
                self._land_slabs_pre = None

    # ---- 宿主臂引擎懒加载 property(D 轮, §14) ----
    # 语义保持: 加载失败 → 打印一次 WARNING 后闸门关闭(head 恒 None, 不抛出);
    # SCRFD 失败仍向上抛(与旧 __init__ 一致 —— 旧版在构造处直接失败)。
    @property
    def det(self) -> SCRFDTRTDetector:
        if self._det_obj is None:
            self._det_obj = SCRFDTRTDetector(
                self._det_engine, max_batch=self._det_batch,
                conf_thres=self._det_conf)
        return self._det_obj

    @det.setter
    def det(self, v) -> None:
        self._det_obj = v          # 允许外部注入/覆盖(测试兼容)

    @property
    def head(self):
        if self._head_state == "pending":
            if self._head_on:
                try:
                    from src.head_gate import HeadGate
                    self._head_obj = HeadGate(self._head_model_path,
                                              conf_thresh=self._head_conf)
                    print(f"[filter] 人头闸门 = head2 (TensorRT) "
                          f"conf={self._head_conf}")
                    self._head_state = "ok"
                except Exception as e:
                    print(f"[filter] !! 人头闸门(head2)加载失败，人头闸门关闭: {e}")
                    self._head_state = "failed"
            else:
                self._head_state = "off"
        return self._head_obj

    @head.setter
    def head(self, v) -> None:
        self._head_obj = v         # 允许外部注入/覆盖(测试兼容)
        if v is None and self._head_state == "pending":
            self._head_state = "failed"

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

    def _gate_judge(self, frame, d, k, p68):
        """闸门链前半·阶段A（不含 gaze）：_judge → 低头复核。
        （C 轮从 _gate_pre 拆出，判定逻辑/顺序一行未改；_gate_pre 仍为
        阶段A+阶段B 的组合，host 臂/dedup 路径行为不变。）

        返回 (verdict, score, pose, down, nfaces, ear, t_judge)。"""
        t1 = time.time()
        verdict, score, pose, down, nfaces = self._judge(
            d, k, p68[0] if p68 else None, p68[1] if p68 else None)
        t_judge = time.time() - t1
        ear = p68[1] if p68 else None
        # 低头复核：单脸 pose 已合格，但 down_ratio < down_min → 按低头剔除
        if (verdict == "keep" and self.down_min is not None
                and down is not None and down < self.down_min):
            verdict = "down"
        return (verdict, score, pose, down, nfaces, ear, t_judge)

    def _gaze_apply(self, g, verdict, dy_all):
        """Stage2 眼神闸门的判定应用（单帧与批量共用；与 _gate_pre 原实现
        逐行同逻辑）。g: resnet34 → (pitch_deg, yaw_deg) / iris →
        (mag, dx, dy)，None=未测到（裁剪无效）。
        返回 (verdict, gaze_mag, gaze_dy)。"""
        gaze_mag = None
        gaze_dy = None
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
        return verdict, gaze_mag, gaze_dy

    def _gate_gaze(self, frame, d, verdict, nfaces, dy_all):
        """闸门链前半·阶段B：Stage2 眼神（逐帧版，_gate_pre/_gate_chain 用）。
        Stage1 全过（头基本正）的单脸帧才跑。
        返回 (verdict, gaze_mag, gaze_dy, t_gaze)。"""
        gaze_mag = None
        gaze_dy = None
        t_gaze = 0.0
        if (verdict == "keep" and self.gaze is not None and nfaces == 1):
            t1 = time.time()
            top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
            g = self.gaze.estimate(frame, d[top, :4])
            t_gaze = time.time() - t1
            verdict, gaze_mag, gaze_dy = self._gaze_apply(g, verdict, dy_all)
        return verdict, gaze_mag, gaze_dy, t_gaze

    def _gate_pre(self, frame, d, k, p68, dy_all):
        """闸门链前半：_judge → 低头复核 → Stage2 眼神（不含 head）。
        （= _gate_judge + _gate_gaze 组合，判定逻辑/顺序与原单函数实现
        完全一致；_run_pass_gpu 的批量版分两阶段调用同一套逻辑。）

        返回 (verdict, score, pose, down, nfaces, ear, gaze_mag, gaze_dy,
              t_judge, t_gaze)。verdict=="keep" 且 nfaces==1 且 head 闸门开启
        → 该帧还需过 Stage1.5 head（调用方决定逐帧或成批跑）。"""
        (verdict, score, pose, down, nfaces, ear,
         t_judge) = self._gate_judge(frame, d, k, p68)
        verdict, gaze_mag, gaze_dy, t_gaze = self._gate_gaze(
            frame, d, verdict, nfaces, dy_all)
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

    def _dedup_eff_max_seg(self, dec_fps):
        """段长时间兜底上限(帧)。显式 --dedup-max-seg 优先(旧裸帧数口径);
        否则按保底间隔换算 max(min_seg, round(max_gap_sec * dec_fps)) ——
        dec_fps=实际送检帧率(源>max-fps 时=max-fps, 否则=源 fps; pynvvc 表外
        帧率回退管道臂同样先按 max-fps 降采样后再判)。默认 1.0s@10fps=10 帧,
        与旧硬编码 dedup_max_seg=10@10fps 逐位一致。"""
        if self.dedup_max_seg:
            return self.dedup_max_seg
        sf = dec_fps if (dec_fps and dec_fps > 0) else 10.0
        return max(self.dedup_min_seg, int(round(self.dedup_max_gap_sec * sf)))

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
        # --decode pynvvc-gpu + --dedup 现已原生支持（GPU INTER_AREA 核顺带产出
        # dedup 两张小图，段内帧驻留显存池，见 _run_pass_gpu_dedup），不再回退。
        # 解码帧源尝试顺序（失败逐级整体回退重跑，最终兜底 ffmpeg 落盘路径）：
        #   auto   : FFmpeg NVDEC 管道 → ffmpeg 落盘（默认，行为与旧版一致）
        #   pynvvc : PyNvVideoCodec 硬解 → FFmpeg NVDEC 管道 → ffmpeg 落盘
        #   pynvvc-gpu : 显存直通零拷贝臂（device 帧不落宿主）→ 宿主 pynvvc → 管道 → 落盘
        #   gpu    : 仅 FFmpeg NVDEC 管道；ffmpeg: 仅落盘路径（旧行为）
        _SRC_LABEL = {"pynvvc": "pynvvc(NVDEC) 帧源", "pipe": "FFmpeg NVDEC 管道",
                      "file": "ffmpeg 落盘",
                      "pynvvc-gpu": "pynvvc-gpu(显存直通) 帧源"}
        chain = {"auto": ("pipe", "file"),
                 "pynvvc": ("pynvvc", "pipe", "file"),
                 "pynvvc-gpu": ("pynvvc-gpu", "pynvvc", "pipe", "file"),
                 "gpu": ("pipe",),
                 "ffmpeg": ("file",)}[mode]
        last_err = None
        for i, kind in enumerate(chain):
            try:
                if kind == "pynvvc-gpu":
                    return self._run_pass_gpu(video_path, out_dir, yaw_lim,
                                              shared_csv, w, h, size, fps,
                                              apply_fps, dec_fps, nframes,
                                              stem)
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
                     "drop_seg": 0, "backup_hits": 0,
                     "quota_shots": 0, "quota_sent": 0, "min_judged": 0}
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
                    min_seg=self.dedup_min_seg,
                    max_seg=self._dedup_eff_max_seg(dec_fps),
                    thumb_side=self.dedup_thumb,
                    sharp_side=self.dedup_sharp_side,
                    n_backups=self.dedup_backups,
                    max_mem_mb=self.dedup_max_mem)
                # ---- 镜头感知采样(opt-in, 见 src/frame_dedup.py ShotTracker) ----
                # shot_mode=True(给了 --dedup-per-shot-max 或 --dedup-per-shot-min>1)
                # 才改变送检流(段→镜头工作项); 默认只记镜头表+sent 记账, 逐字节零影响。
                shot_mode = (self.dedup_per_shot_max is not None
                             or self.dedup_per_shot_min > 1)
                _fb = size                      # 全分辨率帧字节(护栏估算用)
                _eff = self._dedup_eff_max_seg(dec_fps)
                if _fb * _eff > self.dedup_max_mem * 1024 * 1024:
                    _eff = max(self.dedup_min_seg,
                               int(self.dedup_max_mem * 1024 * 1024) // _fb)
                _q = (max(self.dedup_per_shot_max, self.dedup_per_shot_min)
                      if self.dedup_per_shot_max is not None else 0)
                # 镜头缓冲护栏: 至少 512MB; 配额模式下保证配额激活前峰值
                # ((N+1) 段)放得下, 上限 1GB
                _need = ((self.dedup_per_shot_max + 1) * _eff + _q) * _fb \
                    if self.dedup_per_shot_max is not None else 0
                _guard = max(512 * 1024 * 1024, min(_need + 64 * 1024 * 1024,
                                                    1024 * 1024 * 1024))
                tracker = ShotTracker(
                    self.dedup_cut_hi,
                    per_shot_max=self.dedup_per_shot_max,
                    per_shot_min=self.dedup_per_shot_min,
                    mem_guard_bytes=_guard)
                _cur_judged = [None]   # 当前镜头工作项的已判帧号集(top-up 去重用)
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
                    # 镜头感知: 按帧号给镜头表记账 sent(纯记账, 不影响判定)
                    if tracker is not None:
                        tracker.note_judged(idx)
                        if _cur_judged[0] is not None:
                            _cur_judged[0].add(idx)
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

                def _process_shot_item(item):
                    """镜头工作项处理(仅 shot_mode; 复用同一套闸门链):
                    - 配额镜头(quota=True): 镜头内 top-q 帧(时间序)批量送检,
                      逐帧过完整闸门链(闸门零改动)。全 drop 不再补判 —— 淘汰帧
                      已按配额即时释放(内存有界), 送检 q>=M 已满足送检保底。
                    - 非配额镜头: 原逐段流(_process_seg_batch 原样), 之后若
                      per-shot-min=M>1 且该镜头送检数<M, 按「段内回退序」
                      (逐段时间序、段内清晰度降序, 即 rep 被 drop 时的既有
                      回退序)对未判过的帧逐帧补判至 M 帧送检。"""
                    _cur_judged[0] = set()
                    try:
                        seg_stats["segs"] += item["nsegs"]
                        if item["quota"]:
                            seg_stats["quota_shots"] += 1
                            sel = item["sel"]
                            dets, kpss, pose_map = _detect_batch(
                                [fr for (_idx, fr) in sel])
                            for si, (idx, fr) in enumerate(sel):
                                seg_stats["sent"] += 1
                                seg_stats["quota_sent"] += 1
                                _judge_and_write(fr, idx, dets[si], kpss[si],
                                                 pose_map.get(si))
                        else:
                            _process_seg_batch(item["segs"])
                            if item["min"] > 1:
                                row = item["row"]
                                pool = item["topup"]
                                ji = 0
                                while (row["sent"] < item["min"]
                                       and ji < len(pool)):
                                    idx, fr = pool[ji]
                                    ji += 1
                                    if idx in _cur_judged[0]:
                                        continue   # 已判过(rep/sec/scan 补判)
                                    d1, k1, p1 = _detect_batch([fr])
                                    seg_stats["sent"] += 1
                                    seg_stats["min_judged"] += 1
                                    _judge_and_write(fr, idx, d1[0], k1[0],
                                                     p1.get(0))
                    finally:
                        _cur_judged[0] = None

                def _pending_bytes():
                    # 挂起队列字节: shot_mode 用工作项自带字节, 原路径按段 meta
                    if shot_mode:
                        return sum(s["nbytes"] for s in segs)
                    return sum(s[3]["nbytes_total"] for s in segs)

                def _process_items(batch):
                    if shot_mode:
                        for it in batch:
                            _process_shot_item(it)
                    else:
                        _process_seg_batch(batch)   # 原路径原样(逐段批送检)

                def _drain_segs():
                    while (len(segs) >= SEG_Q_CAP
                           or _pending_bytes() > SEG_Q_BYTES):
                        n = min(len(segs), self.detect_chunk)
                        _process_items(segs[:n])
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
                                _seg = (rep_fr, rep_idx, backups, meta)
                                if shot_mode:
                                    # 镜头工作项替代裸段入队(配额/保底 opt-in)
                                    for it in tracker.add_segment(
                                            _seg, meta["frames"]):
                                        segs.append(it)
                                else:
                                    tracker.add_segment(_seg)   # 纯记录镜头表
                                    segs.append(_seg)
                                for (fi, dif, sh, isrep) in meta["rows"]:
                                    dedup_rows.append(
                                        (fi, dif, meta["seg_id"],
                                         meta["seg_start"], meta["seg_end"],
                                         sh, isrep))
                        _drain_segs()
                    res = deduper.flush()   # EOF 强关最后一段
                    if res is not None:
                        rep_fr, rep_idx, backups, meta = res
                        _seg = (rep_fr, rep_idx, backups, meta)
                        if shot_mode:
                            for it in tracker.add_segment(_seg, meta["frames"]):
                                segs.append(it)
                        else:
                            tracker.add_segment(_seg)
                            segs.append(_seg)
                        for (fi, dif, sh, isrep) in meta["rows"]:
                            dedup_rows.append(
                                (fi, dif, meta["seg_id"],
                                 meta["seg_start"], meta["seg_end"],
                                 sh, isrep))
                    items = tracker.close_eof()   # EOF 关最后一镜头(被动模式仅记表)
                    if shot_mode:
                        segs.extend(items)
                    # 排空剩余段：EOF 后不再依赖触发条件，否则尾部不满阈值
                    # 的段会永远留在队列里不被送检(丢帧/丢秒)
                    while segs:
                        n = min(len(segs), self.detect_chunk)
                        _process_items(segs[:n])
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
                # 逐解码帧的段切分/清晰度明细，供调 cut_lo/cut_hi。
                # 共享 out_dir（目录批处理/显式 --out-dir）时按视频命名
                # <stem>_dedup_report.csv 防互相覆盖（与 pose_report_<stem>.csv
                # 同规则），单视频仍为 dedup_report.csv（向后兼容）。
                _dname = (f"{stem}_dedup_report.csv" if shared_csv
                          else "dedup_report.csv")
                with open(os.path.join(out_dir, _dname), "w",
                          encoding="utf-8") as dfh:
                    dfh.write("frame,diff,seg_id,seg_start,seg_end,sharp,is_rep\n")
                    for (fi, dif, sid, s0, s1, sh, isrep) in dedup_rows:
                        dfh.write(f"{fi},{dif:.3f},{sid},{s0},{s1},{sh:.1f},{isrep}\n")
            if tracker is not None and tracker.table:
                # 镜头表(镜头感知采样, --dedup 即产出): 镜头号/段数/起止帧/帧数/
                # 送检帧数/是否配额镜头/关闭原因(cut=硬切, eof, memcap=护栏强关)。
                # 命名规则与 dedup_report 相同: 批处理 <stem>_shot_table.csv,
                # 单视频 shot_table.csv。
                _sname = (f"{stem}_shot_table.csv" if shared_csv
                          else "shot_table.csv")
                with open(os.path.join(out_dir, _sname), "w",
                          encoding="utf-8") as sfh:
                    sfh.write("shot_id,seg_count,frame_start,frame_end,frames,"
                              "sent,quota,close_reason\n")
                    for r in tracker.table:
                        sfh.write(f"{r['shot_id']},{r['seg_count']},"
                                  f"{r['frame_start']},{r['frame_end']},"
                                  f"{r['frames']},{r['sent']},{r['quota']},"
                                  f"{r['close_reason']}\n")
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
            _qtail = (f"  quota-shots={seg_stats['quota_shots']}  "
                      f"quota-sent={seg_stats['quota_sent']}  "
                      f"min-judged={seg_stats['min_judged']}"
                      if seg_stats["quota_shots"] or seg_stats["min_judged"]
                      else "")
            print(f"    [dedup] segments={seg_stats['segs']}  "
                  f"reps={seg_stats['reps']}  "
                  f"backup-judged={seg_stats['bk_judged']}  "
                  f"backup-hit={seg_stats['backup_hits']}  "
                  f"sec-judged={seg_stats['sec_judged']}  "
                  f"sec-hit={seg_stats['sec_hits']}  "
                  f"drop_seg={seg_stats['drop_seg']}  "
                  f"compression={decoded_frames / max(frame_no, 1):.2f}x  "
                  f"dy-gate-hits(dedup kept)={dedup_dy_hits}"
                  f"{_qtail}")
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

    # ---------- --decode pynvvc-gpu: NVDEC 显存直通零拷贝臂（非 dedup 专用） ----------
    def _ensure_gpu_arm(self):
        """懒建 pynvvc-gpu 臂的设备侧资源（首次创建, 跨视频复用）：
        自研 pycuda 流 + 自研 kernel cubin + 显存帧池 + 与宿主臂同引擎文件的
        独立 SCRFD/head device 实例（绑定自研流, 与 NVDEC 解码器同流保序;
        宿主臂实例 self.det / self.head 不受影响, 继续供回退路径使用）。
        D 轮(§14): 宿主实例已懒加载, 此处只看 self._head_on / self._head_conf
        配置, 不触碰 self.head（避免触发宿主实例无谓构建）。device 版 head
        构建失败时与旧版宿主构建失败同语义: 打印 WARNING 后闸门全局关闭。"""
        if self._gpu is None:
            import pycuda.driver as _drv
            from src.gpu_decode import GpuKernels, DeviceFramePool
            from src.head_gate import HeadGate, _default_engine_path
            stream = _drv.Stream()
            # D 轮(A 项): 解码器独立流 —— NVDEC+YUV→RGB 转换 kernel+D2D 走
            # dstream, 消费 kernel 留主流; read_block 出口事件护栏保序
            # (src/gpu_decode.py §14 保序证明)。
            dstream = _drv.Stream()
            g = {"stream": stream,
                 "dstream": dstream,
                 "kernels": GpuKernels(),
                 "pool": DeviceFramePool(),
                 "det": SCRFDTRTDetector(self.engine_path, max_batch=self.batch,
                                         conf_thres=self.conf)}
            g["det"].stream = stream
            if self._head_on:
                try:
                    g["head"] = HeadGate(
                        self._head_engine_path or _default_engine_path(),
                        conf_thresh=self._head_conf)
                    g["head"].stream = stream
                except Exception as e:
                    print(f"[filter] !! 人头闸门(head2)加载失败，人头闸门关闭: {e}")
                    g["head"] = None
                    self._head_obj = None
                    self._head_state = "failed"
            else:
                g["head"] = None
            self._gpu = g
        return self._gpu

    def _run_pass_gpu(self, video_path, out_dir, yaw_lim, shared_csv,
                      w, h, size, fps, apply_fps, dec_fps, nframes, stem):
        """--decode pynvvc-gpu: NVDEC device 帧显存直通零拷贝臂。
        --dedup 时改走 _run_pass_gpu_dedup（缩略图 GPU 化 + 段帧显存驻留），
        非 dedup 路径保持原实现一行未改。

        判定流水与 _run_pass 非 dedup 分支逐行同逻辑（同计数器/同 CSV/同
        Stage2 dy 闸门/同 breakdown 行），区别只在帧的来源与去向：
          块     = read_block(detect_chunk): NVDEC device 帧按与 _PynvvcSource
                   相同的 stride/delta 规则选中后 D2D 拷入显存池（不落宿主）
          detect = 池槽 RGB --自研 INTER_AREA 核(输出 BGR, 与 cv2 逐字节一致)-->
                   768 长边小图 --> 现役 preproc_kernel letterbox --> 原 SCRFD 引擎
          人脸帧 = 池内 RGB→BGR 原地换通道 → 同步 D2H 落宿主（pose68/gaze/JPG 用;
                   no_face / too_small 帧不落地 → 免全帧 D2H）
          head   = 池槽(已 BGR) --自研 640 定点 letterbox 核--> fp32 blob -->
                   原宿主 head 引擎（免宿主 letterbox 与 H2D）
        送入各引擎的 blob 与宿主臂逐字节相同（配方验证 _test/post3_e2/e3），
        判定结果结构性一致。初始化/解码失败抛 _PynvvcDecodeError →
        process() 逐级回退 宿主 pynvvc → 管道 → 落盘 重跑。"""
        if self.dedup:
            return self._run_pass_gpu_dedup(
                video_path, out_dir, yaw_lim, shared_csv, w, h, size, fps,
                apply_fps, dec_fps, nframes, stem)
        import pycuda.driver as _drv
        from pycuda.gpuarray import GPUArray
        from src.gpu_decode import (AreaTables, OVL_SETS, _PynvvcGpuSource,
                                    _OverlappedGpuSource, scrfd_block_detect,
                                    head_slots_detect)
        try:
            g = self._ensure_gpu_arm()
        except Exception as e:
            raise _PynvvcDecodeError(f"pynvvc-gpu 臂初始化失败: {e!r}") from e
        stream, kernels, pool = g["stream"], g["kernels"], g["pool"]

        # SCRFD 768 长边几何（与 face_det._prepare_chunk 同式: 只缩不放）
        if self.target_long and max(h, w) > self.target_long:
            sf = self.target_long / max(h, w)
            nw7, nh7 = int(round(w * sf)), int(round(h * sf))
            rescale = 1.0 / sf
        else:
            nw7, nh7 = w, h
            rescale = 1.0

        try:
            # 解码∥推理重叠(--overlap): ThreadedDecoder 内部 C++ 线程解码不占
            # GIL, producer 线程拉帧+D2D, 主线程满速推理(探针1 P3b)。池容量
            # OVL_SETS×(chunk+8): OVL_SETS 块一轮回, producer 领先
            # ≤ OVL_SETS-1 块(信号量 OVL_SETS token)
            use_overlap = self.overlap != "off"
            # C 轮: 落地 pinned slab —— 首选 __init__ 预分配的 4K 容量 slab
            # (见 __init__ 注释: cuMemHostAlloc 走驱动全局锁, 后台线程无法与
            # CUDA 初始化真并行, 故一次付清在 r1 口径预热段); 预分配缺失/
            # 容量不足(>4K 视频)/分配失败 → 原地按需分配; 再失败 → None →
            # 可分页逐帧回退路径(逐字节同)。--overlap off 臂不开 pinned
            # slab, 行为不变。
            fb = h * w * 3
            land_cap = (self.detect_chunk + 8) * fb
            land_slabs = g.setdefault("_land_slabs", [None] * OVL_SETS)
            while len(land_slabs) < OVL_SETS:      # 槽组 2→3 扩容(跨视频复用 g)
                land_slabs.append(None)
            land_sets = [None] * OVL_SETS
            if use_overlap:
                _pre = self._land_slabs_pre
                for _si in range(OVL_SETS):
                    if (land_slabs[_si] is not None
                            and len(land_slabs[_si]) >= land_cap):
                        continue                   # 已有足够大的 slab(跨视频复用)
                    _p = _pre[_si] if _pre is not None else None
                    if _p is not None and len(_p) >= land_cap:
                        land_slabs[_si] = _p       # 吃预分配
                    else:
                        try:                       # 按需分配(与原实现同式)
                            land_slabs[_si] = _drv.pagelocked_empty(
                                (land_cap,), np.uint8)
                        except Exception:
                            land_slabs[_si] = None
                land_sets = list(land_slabs)
            # 显存帧池: 末批溢出帧 ≤8 不覆盖本块未消费槽; 重叠臂 producer
            # 领先 ≤ OVL_SETS-1 块 → OVL_SETS 组槽轮换(token OVL_SETS 个:
            # D2D(j) 覆写的第 j%OVL_SETS 组上一用户是块 j-OVL_SETS(≤j-2),
            # 其判定链在 consumer pop(j-1) 前已同步完成); 串行臂连续游标
            # 1 组驻留。分辨率变化时池自动重建。(C 轮实测 2→3 无收益, 维持 2)
            pool.ensure((self.detect_chunk + 8)
                        * (OVL_SETS if use_overlap else 1), h, w)
            # B 轮: 先建帧源再建 AreaTables/768 暂存 —— 4K 表构建 ~0.3-0.5s
            # 纯 CPU, 与解码器会话建立+首帧解码重叠(时间线探针: 首块 pop
            # 等待 0.5-0.9s, 其中表构建占大头)。串行臂同样受益(解码器预热
            # 提前), 判定不变(同一批对象, 只是创建顺序)。
            src = None
            if use_overlap:
                try:
                    src = _OverlappedGpuSource(video_path, w, h, fps,
                                               self.max_fps, stream, pool,
                                               self.detect_chunk,
                                               dstream=g["dstream"])
                except _PynvvcDecodeError as e:
                    msg = str(e).splitlines()[0] if str(e) else "无输出"
                    print(f"  WARNING: 重叠解码(ThreadedDecoder)初始化失败"
                          f"（{msg}），回退 A 轮串行臂", flush=True)
            if src is None:
                # A 轮串行臂（--overlap off 或重叠初始化失败的回退, 逐位同基线）
                src = _PynvvcGpuSource(video_path, w, h, fps, self.max_fps,
                                       stream, pool, dstream=g["dstream"])
            tables = AreaTables(w, h, nw7, nh7)   # INTER_AREA 路径判定+表(整视频复用)
            # 768 小图暂存（detect_chunk 个槽, 块间复用; 方法结束即释放）
            small_ga = GPUArray((self.detect_chunk * nh7 * nw7 * 3,), np.uint8)
        except _PynvvcDecodeError:
            raise
        except Exception as e:
            raise _PynvvcDecodeError(f"pynvvc-gpu 臂初始化失败: {e!r}") from e

        if self.verbose:
            print(f"  decode {os.path.basename(video_path)}  {w}x{h}  src_fps={fps:.3f} "
                  f"-> dec_fps={dec_fps:.3f}{' (downsampled)' if apply_fps else ''}  "
                  f"~{nframes} frames  hwdec=nvdec(dev)  src=pynvvc-gpu  "
                  f"768={w}x{h}->{nw7}x{nh7}({tables.kind})  pipeline=off(dev)"
                  f"  overlap={'on' if isinstance(src, _OverlappedGpuSource) else 'off'}")

        t0 = time.time()
        frame_no = kept = drop_noface = drop_pose = drop_down = drop_multi = 0
        drop_up = drop_blink = drop_downp = drop_gaze = drop_gazedown = 0
        drop_head = 0
        drop_toosmall = 0
        # t_dec= NVDEC 拉帧+D2D(帧源内部累计) + 人脸帧落地(swap+D2H);
        # t_read 恒 0(无宿主全帧读取) —— breakdown 行保持同一格式
        t_dec = t_read = t_det = t_judge = t_gaze = t_write = 0.0
        t_head = 0.0
        ex = ThreadPoolExecutor(max_workers=self.jpg_workers)
        futures = []
        # B 轮: 落地 D2H 的常驻 pinned 缓冲(两套轮换)。可分页同步 memcpy_dtoh
        # 的问题(时间线探针实测): (a) ~6.5ms/帧@4K(3.8GB/s), (b) 每帧
        # np.empty 首触缺页, (c) 走 legacy 默认流与所有阻塞流互斥 —— 三者都
        # 落在消费关键路径上。pinned slab 一次分配(g 内跨视频复用), 帧视图
        # 零分配; async D2H 走 io 流不锁其它流。两套轮换 + 复用前等上一块
        # JPG futures(imwrite 线程池持有帧引用, 视频结束才统一 wait —— 套
        # 复用必须先确认引用已释放)。分配失败回退可分页逐帧路径(逐字节同)。
        # C 轮: 分配本体已提到帧源创建之前由后台线程执行(遮盖于解码器预热/
        # AreaTables 构建), 见上方 slab_th; 首块 read_block 后 join。
        land_turn = 0
        land_futs = [None] * OVL_SETS
        # 临时 pose 报告（与 _run_pass 同格式同生命周期）
        report = []
        dy_all = []    # 所有测到 gaze 的帧的 dy（2-pass 基线=其中位数）
        dy_gate = []   # (帧号, 写出路径, dy)：keep 候选帧，循环后统一套用 dy 闸门
        # dedup 统计（本臂不支持 dedup, 全为 0, 不参与汇总）
        decoded_frames = 0
        dedup_rows = []
        seg_stats = {"segs": 0, "sent": 0, "reps": 0, "bk_judged": 0,
                     "sec_judged": 0, "sec_hits": 0,
                     "drop_seg": 0, "backup_hits": 0}
        dedup_dy_hits = 0
        try:
            dec_mode = "pynvvc-gpu"
            t_dec = 0.0

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
                """一行 report + keep 落 JPG / drop 计数（与 _run_pass 逐行同）。"""
                nonlocal kept
                yaw, pitch, roll = pose
                row = [no, verdict, score, yaw, pitch, roll, down,
                       nfaces, ear, gaze_mag, gaze_dy, nheads]
                if verdict == "keep":
                    base = f"{stem}_{no:05d}"
                    name = (f"{base}_{score:.2f}_{yaw:.0f}_{pitch:.0f}_{roll:.0f}.jpg"
                            if self.score_name else f"{base}.jpg")
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

            while True:
                s0, m = src.read_block(self.detect_chunk)  # 等待已累计进 src.waited
                if m == 0:
                    break

                # SCRFD: 池槽 RGB → 自研 INTER_AREA 核(出 BGR) → 现役 letterbox
                # kernel → 原 SCRFD 引擎（16 一批, 与宿主臂 det.detect 分批一致）
                t1 = time.time()
                dets, kpss = scrfd_block_detect(
                    kernels, g["det"], stream, pool, s0, m, h, w,
                    small_ga, nh7, nw7, rescale, tables, self.conf)
                t_det += time.time() - t1

                # 人脸最小尺寸闸门（与 _run_pass 同逻辑）
                ts_mask = set()
                if self.min_face_h and self.min_face_h > 0:
                    for i, d in enumerate(dets):
                        if len(d) > 0:
                            top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                            if (d[top, 3] - d[top, 1]) < self.min_face_h:
                                ts_mask.add(i)

                # 人脸帧落地: 池内 RGB→BGR 原地换通道 → async D2H(pinned
                # slab, 不逐帧 np.empty 首触缺页) → 一次 stream.synchronize()。
                # 仅人脸帧; no_face / too_small 帧不落地。
                t1 = time.time()
                face_idx = [i for i in range(m)
                            if len(dets[i]) > 0 and i not in ts_mask]
                for i in face_idx:
                    kernels.swap_inplace(stream, pool.slot(s0 + i), h * w)
                set_i = land_turn % OVL_SETS
                if land_futs[set_i]:
                    # 该套 pinned slab 上一块(块 k-OVL_SETS)的 JPG 已落盘才能
                    # 覆写(_emit 把帧引用交给 imwrite 线程池, 视频结束才统一
                    # wait)
                    wait(land_futs[set_i])
                    land_futs[set_i] = None
                slab = land_sets[set_i]
                _fmark = len(futures)   # 本块新提交的 JPG futures(登记 slab 复用)
                chunk = {}
                if slab is not None:
                    # async D2H → pinned slab 帧视图, 入队后统一同步
                    for j, i in enumerate(face_idx):
                        fr = chunk[i] = slab[j * fb:(j + 1) * fb].reshape(
                            h, w, 3)
                        _drv.memcpy_dtoh_async(fr.reshape(-1),
                                               pool.slot(s0 + i), stream)
                    stream.synchronize()   # 落地 D2H 完成后再进宿主判定链
                else:
                    # pinned 分配失败 → 旧可分页同步路径(逐字节同)
                    for i in face_idx:
                        fr = np.empty((h, w, 3), np.uint8)
                        _drv.memcpy_dtoh(fr.reshape(-1), pool.slot(s0 + i))
                        chunk[i] = fr
                land_turn += 1
                t_dec += time.time() - t1   # 落地(swap+D2H 等待)计入 decode 段

                # 68 点姿态：本块 >=1 脸帧主脸批量计算（不计时, 与 _run_pass 一致）
                pose_map = {}
                if self.pose68 is not None:
                    items = []
                    for i in face_idx:
                        d = dets[i]
                        top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                        items.append((i, chunk[i], d[top, :4]))
                    if items:
                        poses = self.pose68.estimate_batch(
                            [(fr, bb) for _, fr, bb in items])
                        pose_map = {i: p for (i, _, _), p in zip(items, poses)}

                # ---- 两阶段闸门（与 _run_pass 逐行同逻辑）----
                # C 轮 gaze 批量化（判定结构仍全部在块内完成，无跨块延后）:
                #   阶段A 逐帧 _gate_judge（_judge+低头复核, 与原循环同输出, 不含
                #         gaze）; keep 单脸候选收集进 gaze_jobs/pre_map
                #   阶段B 块内候选一次批量眼神推理（gaze.estimate_batch: 固定
                #         batch=1 引擎流水化 enqueue + 线程池预处理, 与逐帧
                #         estimate 逐位一致, 等价表 _test/post5_gaze_probe.py;
                #         dy_all 收集顺序=帧序, 与原实现一致）
                #   阶段C 按帧序分发 head/_emit（与原实现同判定; 仅 emit 调用
                #         时序变化 —— 计数/JPG 文件名/report 行序均按键控,
                #         输出字节不变）
                head_jobs = []    # (块内序号 i, frame)
                head_meta = {}    # i → (frame_no, score, pose, down, nfaces, ear, gaze_mag, gaze_dy)
                rows = [None] * m
                gaze_jobs = []    # (块内序号 i, frame, bbox)
                pre_map = {}      # i → (verdict, score, pose, down, nfaces, ear, frame_no)
                for i in range(m):
                    frame_no += 1
                    d = dets[i]
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
                    frame = chunk.get(i)   # 仅人脸帧非 None（其余闸门不触碰 frame）
                    k = kpss[i]
                    _p = pose_map.get(i)
                    (verdict, score, pose, down, nfaces, ear,
                     _tj) = self._gate_judge(frame, d, k, _p)
                    t_judge += _tj
                    if verdict == "keep" and nfaces == 1:
                        top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                        if self.gaze is not None:
                            gaze_jobs.append((i, frame, d[top, :4]))
                        pre_map[i] = (verdict, score, pose, down, nfaces,
                                      ear, frame_no)
                        continue
                    # 不进 Stage2/Stage3 的帧在此判定终结（与原实现同输出）
                    rows[i] = _emit(frame, frame_no, verdict, score, pose,
                                    down, nfaces, ear, None, None, None)
                # 阶段B: 批量 gaze（无 keep 候选则零开销; workers=线程池预处理,
                # cv2 释放 GIL, blob 与逐帧一致）
                if gaze_jobs:
                    t1 = time.time()
                    gouts = self.gaze.estimate_batch(
                        [(fr, bb) for _, fr, bb in gaze_jobs])
                    t_gaze += time.time() - t1
                    for (i, frame, _bb), gz in zip(gaze_jobs, gouts):
                        (verdict, score, pose, down, nfaces, ear,
                         no) = pre_map.pop(i)
                        verdict, gaze_mag, gaze_dy = self._gaze_apply(
                            gz, verdict, dy_all)
                        if (verdict == "keep" and nfaces == 1
                                and g["head"] is not None):
                            head_jobs.append((i, frame))
                            head_meta[i] = (no, score, pose, down, nfaces,
                                            ear, gaze_mag, gaze_dy)
                            continue
                        rows[i] = _emit(frame, no, verdict, score, pose,
                                        down, nfaces, ear, gaze_mag,
                                        gaze_dy, None)
                # 阶段C: 剩余 keep 候选（gaze 关闭, 或 gaze 后仍 keep 且 head
                # 关闭）按帧序分发 —— 与原实现 verdict 流完全一致
                for i, (verdict, score, pose, down, nfaces, ear,
                        no) in list(pre_map.items()):
                    frame = chunk.get(i)
                    if (verdict == "keep" and nfaces == 1
                            and g["head"] is not None):
                        head_jobs.append((i, frame))
                        head_meta[i] = (no, score, pose, down, nfaces, ear,
                                        None, None)
                        continue
                    rows[i] = _emit(frame, no, verdict, score, pose, down,
                                    nfaces, ear, None, None, None)
                if head_jobs:
                    t1 = time.time()
                    # device 直通: 池槽(已 BGR) → 自研 640 定点 letterbox 核 →
                    # 原宿主 head 引擎（按 head.max_batch 分批, 与 detect_batch 一致）
                    _hbs = head_slots_detect(
                        kernels, g["head"], stream, pool,
                        [s0 + i for i, _fr in head_jobs], h, w)
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
                if slab is not None:
                    # 本块 JPG futures 视频结束才统一 wait, 登记到 land_futs:
                    # 2 块之后覆写该套 slab 前先 wait(见上)
                    land_futs[set_i] = futures[_fmark:]

            # NVDEC 拉帧+D2D 等待（read_block 内部累计）计入 decode 段
            t_dec += src.waited

            # 排空 JPEG 线程池
            t1 = time.time()
            wait(futures)
            t_write = time.time() - t1

            # Stage2 纵向眼神闸门（2-pass 后半，与 _run_pass 逐行同逻辑）
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
                        dedup_dy_hits += len(hit)
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
            if dedup_rows:   # 本臂恒空, 保留同款守卫以防复用
                with open(os.path.join(out_dir, "dedup_report.csv"), "w",
                          encoding="utf-8") as dfh:
                    dfh.write("frame,diff,seg_id,seg_start,seg_end,sharp,is_rep\n")
                    for (fi, dif, sid, s0, s1, sh, isrep) in dedup_rows:
                        dfh.write(f"{fi},{dif:.3f},{sid},{s0},{s1},{sh:.1f},{isrep}\n")
        finally:
            src.close()
            ex.shutdown(wait=True)

        dt = time.time() - t0
        dm = f"{self.down_min:g}" if self.down_min is not None else "?"
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
        out = {"frames": frame_no, "kept": kept, "no_face": drop_noface,
               "decode_mode": dec_mode,
               "pose": drop_pose, "down": drop_down, "multi": drop_multi,
               "up": drop_up, "downp": drop_downp, "gaze": drop_gaze,
               "gazedown": drop_gazedown, "blink": drop_blink,
               "head": drop_head, "toosmall": drop_toosmall,
               "pipeline": False, "pipe_depth": self.pipeline_depth,
               "overlap": isinstance(src, _OverlappedGpuSource),
               "seconds": dt}
        return out

    def _run_pass_gpu_dedup(self, video_path, out_dir, yaw_lim, shared_csv,
                            w, h, size, fps, apply_fps, dec_fps, nframes, stem):
        """--decode pynvvc-gpu + --dedup: 零拷贝臂的抽帧去重模式。

        与宿主 dedup 臂(_run_pass dedup 分支)逐位等价, 判定链复用同一批
        宿主函数(_gate_judge/_gate_gaze/head.detect/pose68.estimate_batch),
        区别只在数据搬运:
          解码   = read_block(同 _PynvvcSource stride/delta 选帧), device 帧
                   不落宿主全帧
          小图   = 复用已验证的 INTER_AREA 核(area_fast/generic_u8, oswap=1)
                   在 GPU 上从 RGB device 帧生成 dedup 两张 BGR 小图
                   (thumb 128 / sharp 256 长边), 经 pinned 缓冲 D2H —— 与
                   宿主 cv2.resize(BGR) 逐字节一致(_test/dedup_gpu_thumb_probe);
                   灰度化留在宿主 cv2.cvtColor(小图输入字节与宿主臂相同,
                   逐位等价由构造保证, 避免 cvtColor SIMD 路径复刻风险)
          段状态 = SlotSegDeduper(与 FrameDedupSelector 逐行同构, 帧以驻留池
                   槽号表示; 等价性 _test/dedup_slot_mirror_test.py)
          驻留   = 段内帧 D2D 进独立驻留池(段关闭送检完成才释放槽位),
                   解码池照常轮换 → 与 --overlap 无冲突(解码线程只覆写
                   解码池; 驻留池由消费侧分配器管理)
          送检   = 段批 rep/extras 槽位连续化 D2D 进 scratch → device SCRFD
                   (按 det.max_batch 分批, 与宿主 det.detect 内部分批一致)
                   → 人脸帧原地换序 BGR → 同步 D2H → 宿主 pose68 批量 →
                   逐帧闸门链(与宿主 _judge_and_write 逐行同逻辑)
        初始化/解码失败抛 _PynvvcDecodeError → process() 逐级回退
        宿主 pynvvc → 管道 → 落盘 重跑。"""
        import heapq
        import pycuda.driver as _drv
        from pycuda.gpuarray import GPUArray
        from src.frame_dedup import SlotSegDeduper, ShotTracker
        from src.gpu_decode import (AreaTables, DeviceFramePool, OVL_SETS,
                                    _OverlappedGpuSource, _PynvvcGpuSource,
                                    scrfd_block_detect)
        try:
            g = self._ensure_gpu_arm()
        except Exception as e:
            raise _PynvvcDecodeError(f"pynvvc-gpu 臂初始化失败: {e!r}") from e
        stream, kernels, pool = g["stream"], g["kernels"], g["pool"]

        # SCRFD 768 长边几何（与非 dedup 臂同式: 只缩不放）
        if self.target_long and max(h, w) > self.target_long:
            sf = self.target_long / max(h, w)
            nw7, nh7 = int(round(w * sf)), int(round(h * sf))
            rescale = 1.0 / sf
        else:
            nw7, nh7 = w, h
            rescale = 1.0

        # dedup 两张小图几何（与宿主 _gray_thumb 逐行同式: s=side/long_side,
        # int(w*s) 截断; long_side <= side 时不缩 → area fast 1x 路径覆盖）
        def _small_geo(side):
            if max(h, w) > side:
                s = side / max(h, w)
                return max(1, int(w * s)), max(1, int(h * s))
            return w, h

        tnw, tnh = _small_geo(self.dedup_thumb)
        snw, snh = _small_geo(self.dedup_sharp_side)

        try:
            # 解码∥推理重叠(与非 dedup 臂同款; 驻留池设计解除了段帧与
            # 解码池槽位复用的冲突, dedup 模式无需降级 overlap)
            use_overlap = self.overlap != "off"
            src = None
            if use_overlap:
                try:
                    src = _OverlappedGpuSource(video_path, w, h, fps,
                                               self.max_fps, stream, pool,
                                               self.detect_chunk,
                                               dstream=g["dstream"])
                except _PynvvcDecodeError as e:
                    msg = str(e).splitlines()[0] if str(e) else "无输出"
                    print(f"  WARNING: 重叠解码(ThreadedDecoder)初始化失败"
                          f"（{msg}），回退 A 轮串行臂", flush=True)
            if src is None:
                src = _PynvvcGpuSource(video_path, w, h, fps, self.max_fps,
                                       stream, pool, dstream=g["dstream"])
            tables = AreaTables(w, h, nw7, nh7)     # SCRFD 768 表
            ttab = AreaTables(w, h, tnw, tnh)       # thumb 128 表
            stab = AreaTables(w, h, snw, snh)       # sharp 256 表
            small_ga = GPUArray((self.detect_chunk * nh7 * nw7 * 3,), np.uint8)
            thumb_ga = GPUArray((self.detect_chunk * tnh * tnw * 3,), np.uint8)
            sharp_ga = GPUArray((self.detect_chunk * snh * snw * 3,), np.uint8)
            try:
                thumb_host = _drv.pagelocked_empty(
                    (self.detect_chunk * tnh * tnw * 3,), np.uint8)
                sharp_host = _drv.pagelocked_empty(
                    (self.detect_chunk * snh * snw * 3,), np.uint8)
            except Exception:
                thumb_host = np.empty((self.detect_chunk * tnh * tnw * 3,),
                                      np.uint8)
                sharp_host = np.empty((self.detect_chunk * snh * snw * 3,),
                                      np.uint8)
            # ---- dedup 显存驻留池(段内帧) + 检测连续化 scratch 池 ----
            # 驻留池容量 = 挂起段帧数上限(与宿主 SEG_Q_BYTES/SEG_Q_CAP 同口径,
            # 挂起段可跨块存活, 故用分配器复用而非游标回绕 —— 回绕会覆写
            # 尚未送检的挂起段帧) + 本块帧数 + 段内上限 + 余量。
            fb = h * w * 3
            eff_max_est = self._dedup_eff_max_seg(dec_fps)
            if fb * eff_max_est > self.dedup_max_mem * 1024 * 1024:
                eff_max_est = max(self.dedup_min_seg,
                                  int(self.dedup_max_mem * 1024 * 1024) // fb)
            SEG_Q_CAP = max(8, self.detect_chunk // 4)
            SEG_Q_BYTES = 160 * 1024 * 1024
            left_frames = min((SEG_Q_CAP - 1) * eff_max_est,
                              SEG_Q_BYTES // fb)
            # 镜头感知采样(opt-in): shot_mode=配额/保底开启才改变送检流;
            # 驻留池需额外容纳镜头缓冲(配额激活前峰值 (N+1) 段 + q 帧, 与宿主臂
            # 同一护栏公式; 纯记录模式不持帧不加分)
            shot_mode = (self.dedup_per_shot_max is not None
                         or self.dedup_per_shot_min > 1)
            _q = (max(self.dedup_per_shot_max, self.dedup_per_shot_min)
                  if self.dedup_per_shot_max is not None else 0)
            _need = ((self.dedup_per_shot_max + 1) * eff_max_est + _q) * fb \
                if self.dedup_per_shot_max is not None else 0
            _guard = max(512 * 1024 * 1024,
                         min(_need + 64 * 1024 * 1024, 1024 * 1024 * 1024))
            shot_cap = (min(_guard // fb,
                            (self.dedup_per_shot_max + 2) * eff_max_est + _q + 8)
                        if shot_mode else 0)
            stage_cap = left_frames + self.detect_chunk + eff_max_est + 8 \
                + shot_cap
            if stage_cap * fb > 3 * 1024 * 1024 * 1024:
                raise _PynvvcDecodeError(
                    f"dedup 显存驻留池超预算"
                    f"({stage_cap * fb / 1024 ** 3:.1f}GB), 回退宿主臂")
            if "dedup_stage" not in g:
                g["dedup_stage"] = DeviceFramePool()
            if "dedup_scratch" not in g:
                g["dedup_scratch"] = DeviceFramePool()
            stage = g["dedup_stage"]
            scratch = g["dedup_scratch"]
            # 主解码池(与非 dedup 臂同款; dedup 模式额外有驻留池)
            pool.ensure((self.detect_chunk + 8)
                        * (OVL_SETS if use_overlap else 1), h, w)
            stage.ensure(stage_cap, h, w)
            scratch.ensure(g["det"].max_batch + 8, h, w)
            # 段状态机(与宿主 FrameDedupSelector 逐行同构, 槽号版)
            deduper = SlotSegDeduper(
                cut_lo=self.dedup_cut_lo, cut_hi=self.dedup_cut_hi,
                min_seg=self.dedup_min_seg,
                max_seg=self._dedup_eff_max_seg(dec_fps),
                thumb_side=self.dedup_thumb,
                sharp_side=self.dedup_sharp_side,
                n_backups=self.dedup_backups,
                max_mem_mb=self.dedup_max_mem, frame_bytes=fb)
        except _PynvvcDecodeError:
            raise
        except Exception as e:
            raise _PynvvcDecodeError(f"pynvvc-gpu 臂初始化失败: {e!r}") from e

        if self.verbose:
            print(f"  decode {os.path.basename(video_path)}  {w}x{h}  "
                  f"src_fps={fps:.3f} -> dec_fps={dec_fps:.3f}"
                  f"{' (downsampled)' if apply_fps else ''}  ~{nframes} frames"
                  f"  hwdec=nvdec(dev)  src=pynvvc-gpu+dedup"
                  f"  thumb={tnw}x{tnh}({ttab.kind})  sharp={snw}x{snh}"
                  f"({stab.kind})  768={w}x{h}->{nw7}x{nh7}({tables.kind})"
                  f"  overlap={'on' if isinstance(src, _OverlappedGpuSource) else 'off'}")

        t0 = time.time()
        frame_no = kept = drop_noface = drop_pose = drop_down = drop_multi = 0
        drop_up = drop_blink = drop_downp = drop_gaze = drop_gazedown = 0
        drop_head = 0
        drop_toosmall = 0
        t_dec = t_read = t_det = t_judge = t_gaze = t_write = 0.0
        t_head = 0.0
        ex = ThreadPoolExecutor(max_workers=self.jpg_workers)
        futures = []
        report = []
        dy_all = []
        dy_gate = []
        decoded_frames = 0
        dedup_rows = []   # (frame, diff, seg_id, seg_start, seg_end, sharp, is_rep)
        seg_stats = {"segs": 0, "sent": 0, "reps": 0, "bk_judged": 0,
                     "sec_judged": 0, "sec_hits": 0,
                     "drop_seg": 0, "backup_hits": 0,
                     "quota_shots": 0, "quota_sent": 0, "min_judged": 0}
        dedup_dy_hits = 0
        try:
            dec_mode = "pynvvc-gpu"
            t_dec = 0.0
            sec_fps = dec_fps if (dec_fps and dec_fps > 0) else 10.0
            segs = []   # 已关闭待送检段: (rep_slot, rep_idx, backups, meta)
            _cur_judged = [None]   # 当前镜头工作项已判帧号集(top-up 去重用)

            # ---- 驻留池槽分配器(最低自由槽优先, 确定性复用) ----
            free_slots = []
            stage_water = 0      # 从未分配过的最高水位

            def _stage_alloc():
                nonlocal stage_water
                if free_slots:
                    return heapq.heappop(free_slots)
                if stage_water < stage_cap:
                    s = stage_water
                    stage_water += 1
                    return s
                return None

            def _stage_release(slots):
                for s in slots:
                    heapq.heappush(free_slots, s)

            # 镜头感知采样记账器(opt-in): 配额激活淘汰帧 → 槽位即时归还驻留池;
            # 纯记录模式(默认参数)只记镜头表, 不改变送检流
            tracker = ShotTracker(
                self.dedup_cut_hi,
                per_shot_max=self.dedup_per_shot_max,
                per_shot_min=self.dedup_per_shot_min,
                mem_guard_bytes=_guard, release=_stage_release)

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

            def _to_host_bgr(slot):
                """驻留池槽 RGB → 原地换序 BGR → 同步 D2H 落宿主(与非 dedup 臂
                pinned 失败回退路径同款 memcpy_dtoh 同步写法)。调用方保证同一
                槽至多换序一次, 且换序后不再走 RGB 通道序的检测核(rep 不进
                extras 补判名单, 与宿主臂排除规则一致)。"""
                kernels.swap_inplace(stream, stage.slot(slot), h * w)
                fr = np.empty((h, w, 3), np.uint8)
                _drv.memcpy_dtoh(fr.reshape(-1), stage.slot(slot))
                return fr

            def _detect_slots(slot_list):
                """驻留槽位列表 → 按 det.max_batch 分批: 连续化 D2D 进 scratch
                → device SCRFD。分批边界与宿主 det.detect 内部分批一致
                (同帧同批 → dets/kpss 逐位一致)。"""
                dets, kpss = [], []
                mb = g["det"].max_batch
                for c0 in range(0, len(slot_list), mb):
                    grp = slot_list[c0:c0 + mb]
                    for i, sl in enumerate(grp):
                        _drv.memcpy_dtod_async(scratch.slot(i),
                                               stage.slot(sl), fb, stream)
                    d_i, k_i = scrfd_block_detect(
                        kernels, g["det"], stream, scratch, 0, len(grp),
                        h, w, small_ga, nh7, nw7, rescale, tables, self.conf)
                    dets.extend(d_i)
                    kpss.extend(k_i)
                return dets, kpss

            def _pose_batch(items):
                """(批内序号, 宿主 BGR, bbox) → 批量 pose68(与宿主 _detect_batch
                同一次 estimate_batch 调用式, 批成员一致 → pose 逐位一致)。"""
                if self.pose68 is not None and items:
                    poses = self.pose68.estimate_batch(
                        [(fr, bb) for _, fr, bb in items])
                    return {i: p for (i, _, _), p in zip(items, poses)}
                return {}

            def _face_land(slot_list, dets):
                """人脸帧(非 too_small)D2H 落宿主, 返回 (items, frames)。
                items/frames 键=批内序号; 规则与宿主 _detect_batch 一致。"""
                items = []
                for i, d in enumerate(dets):
                    if len(d) > 0:
                        top = int(np.argmax(d[:, 4])) if len(d) > 1 else 0
                        if (self.min_face_h and self.min_face_h > 0
                                and (d[top, 3] - d[top, 1]) < self.min_face_h):
                            continue
                        items.append((i, _to_host_bgr(slot_list[i]),
                                      d[top, :4]))
                frames = {i: fr for i, fr, _ in items}
                return items, frames

            def _judge_and_write(frame, idx, d, k, p68):
                """与 _run_pass dedup 分支 _judge_and_write 逐行同逻辑; 闸门链
                拆两段: Stage1(_gate_judge, 不触像素)在 D2H 之后, keep 才进
                gaze/head(与 _gate_chain 的级联次序逐行一致) —— no_face /
                multi 帧不触碰像素(frame=None 亦安全)。"""
                nonlocal frame_no, kept, t_judge, t_gaze, t_head, drop_toosmall
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
                (verdict, score, pose, down, nfaces, ear,
                 _tj) = self._gate_judge(frame, d, k, p68)
                t_judge += _tj
                gaze_mag = gaze_dy = None
                nheads = None
                if verdict == "keep":
                    verdict, gaze_mag, gaze_dy, _tg = self._gate_gaze(
                        frame, d, verdict, nfaces, dy_all)
                    t_gaze += _tg
                    if (verdict == "keep" and nfaces == 1
                            and self.head is not None):
                        t1 = time.time()
                        _hb = self.head.detect(frame)
                        t_head += time.time() - t1
                        nheads = len(_hb)
                        if nheads >= 2:
                            verdict = "multi_head"
                frame_no += 1   # = 送检帧数(reps + backup 补判)
                # 镜头感知: 按帧号给镜头表记账 sent(纯记账, 不影响判定)
                if tracker is not None:
                    tracker.note_judged(idx)
                    if _cur_judged[0] is not None:
                        _cur_judged[0].add(idx)
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
                        dy_gate.append((idx, os.path.join(out_dir, name),
                                        gaze_dy))
                    if self.verbose:
                        print(f"    #{idx:05d} KEEP  score={score:.3f} "
                              f"yaw={yaw:.0f} pit={pitch:.0f} rol={roll:.0f} "
                              f"down={down:.2f}")
                    return True
                _count_drop(verdict)
                if self.verbose:
                    print(f"    #{idx:05d} drop  ({verdict}  down={down})")
                return False

            def _process_seg_batch(sb, do_release=True):
                """与 _run_pass dedup 分支 _process_seg_batch 逐行同逻辑(段批
                rep 送检→闸门→sec/scan 补判 early-stop), 帧来源改驻留池槽。
                do_release=False: 槽位归还交由调用方(镜头工作项统一归还一次,
                防同槽双重归还)。"""
                seg_stats["segs"] += len(sb)
                rep_slots = [s[0] for s in sb]
                dets, kpss = _detect_slots(rep_slots)
                items, rep_frames = _face_land(rep_slots, dets)
                pose_map = _pose_batch(items)
                jobs = []         # job_pos → (kind, satisfied_set)
                extra_items = []  # (job_pos, kind, sec_or_None, slot, idx)
                for si, seg in enumerate(sb):
                    _, rep_idx, _bk, meta = seg
                    seg_stats["sent"] += 1
                    seg_stats["reps"] += 1
                    ok = _judge_and_write(rep_frames.get(si), rep_idx,
                                          dets[si], kpss[si],
                                          pose_map.get(si))
                    job_pos = len(jobs)
                    if ok:
                        rep_sec = rep_idx // sec_fps
                        by_sec = {}
                        for (idx, _sh, sl) in meta["frames"]:
                            s2 = idx // sec_fps
                            if s2 != rep_sec:
                                by_sec.setdefault(s2, []).append((sl, idx))
                        if by_sec:
                            jobs.append(("sec", set()))
                            for s2 in sorted(by_sec):
                                for (sl, idx) in by_sec[s2]:
                                    extra_items.append(
                                        (job_pos, "sec", s2, sl, idx))
                    else:
                        jobs.append(("scan", set()))
                        for (idx, _sh, sl) in meta["frames"]:
                            if idx != rep_idx:
                                extra_items.append(
                                    (job_pos, "scan", None, sl, idx))
                solved_scan = set()
                if extra_items:
                    ex_slots = [it[3] for it in extra_items]
                    dets_e, kpss_e = _detect_slots(ex_slots)
                    items_e, ex_frames = _face_land(ex_slots, dets_e)
                    pose_map_e = _pose_batch(items_e)
                    for ei, (job_pos, kind, s2, sl, idx) in enumerate(extra_items):
                        if kind == "scan" and job_pos in solved_scan:
                            continue
                        if kind == "sec" and s2 in jobs[job_pos][1]:
                            continue   # 该秒已有 keep, 剩余候选跳过
                        seg_stats["sent"] += 1
                        if kind == "scan":
                            seg_stats["bk_judged"] += 1
                        else:
                            seg_stats["sec_judged"] += 1
                        if _judge_and_write(ex_frames.get(ei), idx,
                                            dets_e[ei], kpss_e[ei],
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
                # 本批全部送检完成 → 段内全部驻留槽归还(meta['frames'] 覆盖
                # 全段帧, 含 rep/backups)
                if do_release:
                    for seg in sb:
                        _stage_release([sl for (_idx, _sh, sl) in seg[3]["frames"]])

            def _process_shot_item(item):
                """镜头工作项处理(仅 shot_mode; 与宿主臂 _process_shot_item
                逐行同逻辑, 帧来源改驻留池槽):
                - 配额镜头: 镜头内 top-q 槽(时间序)批量送检, 逐帧过完整闸门链;
                  全 drop 不再补判(淘汰槽已按配额即时归还, 显存有界)。
                - 非配额镜头: 原逐段流, 之后 per-shot-min=M>1 时按段内回退序
                  对未判过的槽补判至 M 帧送检。
                处理完统一归还 item['release'](各槽恰归还一次)。"""
                _cur_judged[0] = set()
                try:
                    seg_stats["segs"] += item["nsegs"]
                    if item["quota"]:
                        seg_stats["quota_shots"] += 1
                        sel = item["sel"]
                        slots = [sl for (_idx, sl) in sel]
                        dets, kpss = _detect_slots(slots)
                        items_f, sel_frames = _face_land(slots, dets)
                        pose_map = _pose_batch(items_f)
                        for si, (idx, _sl) in enumerate(sel):
                            seg_stats["sent"] += 1
                            seg_stats["quota_sent"] += 1
                            _judge_and_write(sel_frames.get(si), idx,
                                             dets[si], kpss[si],
                                             pose_map.get(si))
                    else:
                        _process_seg_batch(item["segs"], do_release=False)
                        if item["min"] > 1:
                            row = item["row"]
                            pool = item["topup"]
                            ji = 0
                            while (row["sent"] < item["min"]
                                   and ji < len(pool)):
                                idx, sl = pool[ji]
                                ji += 1
                                if idx in _cur_judged[0]:
                                    continue   # 已判过(rep/sec/scan 补判)
                                d1, k1 = _detect_slots([sl])
                                items_f, fr_map = _face_land([sl], d1)
                                pose_map = _pose_batch(items_f)
                                seg_stats["sent"] += 1
                                seg_stats["min_judged"] += 1
                                _judge_and_write(fr_map.get(0), idx,
                                                 d1[0], k1[0], pose_map.get(0))
                    # 本镜头全部送检完成 → 统一归还槽位(淘汰槽已在配额激活/
                    # 重选时提前归还, 不在本清单内)
                    _stage_release(item["release"])
                finally:
                    _cur_judged[0] = None

            def _pending_bytes():
                # 挂起队列字节: shot_mode 用工作项自带字节, 原路径按段 meta
                if shot_mode:
                    return sum(s["nbytes"] for s in segs)
                return sum(s[3]["nbytes_total"] for s in segs)

            def _process_items(batch):
                if shot_mode:
                    for it in batch:
                        _process_shot_item(it)
                else:
                    _process_seg_batch(batch)   # 原路径原样(逐段批送检)

            def _drain_segs():
                while (len(segs) >= SEG_Q_CAP
                       or _pending_bytes() > SEG_Q_BYTES):
                    n = min(len(segs), self.detect_chunk)
                    _process_items(segs[:n])
                    del segs[:n]

            while True:
                s0, m = src.read_block(self.detect_chunk)  # 等待已累计进 waited
                if m == 0:
                    break
                # GPU 顺带产出两张小图(复用已验证 INTER_AREA 核, RGB→BGR)
                t1 = time.time()
                kernels.area_batch(stream, pool.slot(s0),
                                   thumb_ga.__cuda_array_interface__["data"][0],
                                   m, h, w, tnh, tnw, ttab, oswap=1)
                kernels.area_batch(stream, pool.slot(s0),
                                   sharp_ga.__cuda_array_interface__["data"][0],
                                   m, h, w, snh, snw, stab, oswap=1)
                _drv.memcpy_dtoh_async(thumb_host,
                                       thumb_ga.__cuda_array_interface__["data"][0],
                                       stream)
                _drv.memcpy_dtoh_async(sharp_host,
                                       sharp_ga.__cuda_array_interface__["data"][0],
                                       stream)
                stream.synchronize()
                t_dec += time.time() - t1   # 小图生成+D2H 计入 decode 段
                # 逐帧: D2D 驻留 + 宿主段状态机(与 _run_pass dedup 分支同构,
                # 帧序/帧号口径一致)。排水按【累计解码帧数每 detect_chunk 帧】
                # 触发 —— 宿主臂按 64 帧读块推帧后排水, 本臂 overlap 首块只
                # 有 16 帧, 按块排水会使段批分组与宿主错位(report 行序/送检
                # 批次漂移), 故用计数器对齐累计帧数 64k 的排水点。
                for j in range(m):
                    sl = _stage_alloc()
                    if sl is None:
                        raise _PynvvcDecodeError(
                            "dedup 显存驻留池耗尽(段挂起超预算), 回退宿主臂")
                    _drv.memcpy_dtod_async(stage.slot(sl), pool.slot(s0 + j),
                                           fb, stream)
                    tb = tnh * tnw * 3
                    sb = snh * snw * 3
                    tv = thumb_host[j * tb:(j + 1) * tb].reshape(tnh, tnw, 3)
                    sv = sharp_host[j * sb:(j + 1) * sb].reshape(snh, snw, 3)
                    decoded_frames += 1
                    res = deduper.push_slot(tv, sv, fb, sl, decoded_frames)
                    if res is not None:
                        rep_sl, rep_idx, backups, meta = res
                        _seg = (rep_sl, rep_idx, backups, meta)
                        if shot_mode:
                            # 镜头工作项替代裸段入队(配额/保底 opt-in)
                            for it in tracker.add_segment(_seg, meta["frames"]):
                                segs.append(it)
                        else:
                            tracker.add_segment(_seg)   # 纯记录镜头表
                            segs.append(_seg)
                        for (fi, dif, sh, isrep) in meta["rows"]:
                            dedup_rows.append(
                                (fi, dif, meta["seg_id"],
                                 meta["seg_start"], meta["seg_end"],
                                 sh, isrep))
                    if decoded_frames % self.detect_chunk == 0:
                        _drain_segs()
            # EOF: 宿主臂在最后一个不满 detect_chunk 的读块末尾也会排水一次
            # (_run_pass 行业 1284), 这里补同一次排水, 保证尾部段的送检批次
            # 与宿主臂逐位同序
            _drain_segs()
            res = deduper.flush()   # EOF 强关最后一段
            if res is not None:
                rep_sl, rep_idx, backups, meta = res
                _seg = (rep_sl, rep_idx, backups, meta)
                if shot_mode:
                    for it in tracker.add_segment(_seg, meta["frames"]):
                        segs.append(it)
                else:
                    tracker.add_segment(_seg)
                    segs.append(_seg)
                for (fi, dif, sh, isrep) in meta["rows"]:
                    dedup_rows.append(
                        (fi, dif, meta["seg_id"],
                         meta["seg_start"], meta["seg_end"], sh, isrep))
            items = tracker.close_eof()   # EOF 关最后一镜头(被动模式仅记表)
            if shot_mode:
                segs.extend(items)
            # 排空剩余段(EOF 后不再依赖触发条件; 与宿主臂同口径)
            while segs:
                n = min(len(segs), self.detect_chunk)
                _process_items(segs[:n])
                del segs[:n]

            # NVDEC 拉帧+D2D 等待(read_block 内部累计)计入 decode 段
            t_dec += src.waited

            # 排空 JPEG 线程池
            t1 = time.time()
            wait(futures)
            t_write = time.time() - t1

            # Stage2 纵向眼神闸门(2-pass 后半, 与 _run_pass 逐行同逻辑)
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
                    dedup_dy_hits += len(hit)
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
                # 共享 out_dir(目录批处理/显式 --out-dir)按视频命名防互相覆盖;
                # 单视频保持 dedup_report.csv(向后兼容)
                _dname = (f"{stem}_dedup_report.csv" if shared_csv
                          else "dedup_report.csv")
                with open(os.path.join(out_dir, _dname), "w",
                          encoding="utf-8") as dfh:
                    dfh.write("frame,diff,seg_id,seg_start,seg_end,sharp,is_rep\n")
                    for (fi, dif, sid, s0, s1, sh, isrep) in dedup_rows:
                        dfh.write(f"{fi},{dif:.3f},{sid},{s0},{s1},{sh:.1f},{isrep}\n")
            if tracker is not None and tracker.table:
                # 镜头表(--dedup 即产出; 与宿主臂同款): 批处理 <stem>_shot_table.csv,
                # 单视频 shot_table.csv
                _sname = (f"{stem}_shot_table.csv" if shared_csv
                          else "shot_table.csv")
                with open(os.path.join(out_dir, _sname), "w",
                          encoding="utf-8") as sfh:
                    sfh.write("shot_id,seg_count,frame_start,frame_end,frames,"
                              "sent,quota,close_reason\n")
                    for r in tracker.table:
                        sfh.write(f"{r['shot_id']},{r['seg_count']},"
                                  f"{r['frame_start']},{r['frame_end']},"
                                  f"{r['frames']},{r['sent']},{r['quota']},"
                                  f"{r['close_reason']}\n")
        finally:
            src.close()
            ex.shutdown(wait=True)

        dt = time.time() - t0
        dm = f"{self.down_min:g}" if self.down_min is not None else "?"
        head_label = f"decoded {decoded_frames} -> sent {frame_no}"
        rate = decoded_frames / dt
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
        print(f"    [dedup] segments={seg_stats['segs']}  "
              f"reps={seg_stats['reps']}  "
              f"backup-judged={seg_stats['bk_judged']}  "
              f"backup-hit={seg_stats['backup_hits']}  "
              f"sec-judged={seg_stats['sec_judged']}  "
              f"sec-hit={seg_stats['sec_hits']}  "
              f"drop_seg={seg_stats['drop_seg']}  "
              f"compression={decoded_frames / max(frame_no, 1):.2f}x  "
              f"dy-gate-hits(dedup kept)={dedup_dy_hits}"
              + (f"  quota-shots={seg_stats['quota_shots']}  "
                 f"quota-sent={seg_stats['quota_sent']}  "
                 f"min-judged={seg_stats['min_judged']}"
                 if seg_stats["quota_shots"] or seg_stats["min_judged"] else ""))
        out = {"frames": frame_no, "kept": kept, "no_face": drop_noface,
               "decode_mode": dec_mode,
               "pose": drop_pose, "down": drop_down, "multi": drop_multi,
               "up": drop_up, "downp": drop_downp, "gaze": drop_gaze,
               "gazedown": drop_gazedown, "blink": drop_blink,
               "head": drop_head, "toosmall": drop_toosmall,
               "pipeline": False, "pipe_depth": self.pipeline_depth,
               "overlap": isinstance(src, _OverlappedGpuSource),
               "decoded": decoded_frames, "segments": seg_stats["segs"],
               "drop_seg": seg_stats["drop_seg"],
               "backup_hits": seg_stats["backup_hits"],
               "dedup_dy_hits": dedup_dy_hits,
               "seconds": dt}
        return out

    def run(self, targets, yaw_lim, shared_csv=False, on_video=None):
        """targets: list of (video_path, out_dir). shared_csv: 多视频共享同一
        out_dir 时，CSV 按视频名区分（pose_report_<stem>.csv）防互相覆盖。
        on_video: 可选回调 (i, n, video_path, out_stats, seconds)，每个视频
        process() 成功返回后触发（失败不触发）；None=原行为。
        返回 (results, failed, dt): results=[(video_path, out_stats), ...]。"""
        t0 = time.time()
        failed = []
        results = []
        for i, (p, od) in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] {p}", flush=True)
            t1 = time.time()
            try:
                out = self.process(p, od, yaw_lim, shared_csv=shared_csv)
                results.append((p, out))
                if on_video is not None:
                    on_video(i, len(targets), p, out, time.time() - t1)
            except Exception as e:   # 单视频失败不中断整批（无人值守长任务）
                failed.append((p, str(e)))
                print(f"  !! FAILED: {e}  (skip, continue)", flush=True)
        dt = time.time() - t0
        print(f"\nAll done: {len(targets)} video(s) in {dt:.1f}s"
              + (f", {len(failed)} FAILED: {[f[0] for f in failed]}" if failed else ""))
        return results, failed, dt


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


def merge_pose_reports(out_dir, video_files):
    """目录批处理模式收尾: 把 out_dir 内逐视频 pose_report_<stem>.csv 合并为
    pose_report_all.csv —— 首列 video(视频文件名, 含扩展名), 其余列与单视频
    pose_report.csv 逐字节一致(行原样前缀 video, 不经 csv 库重序列化)。
    合并后删除源 CSV(语义同 _test/batch225_merge.py)。
    video_files: 按处理顺序的视频文件名字典序列表; 失败视频无源 CSV, 跳过并 WARN。
    返回 (数据行数, 成功合并的视频数)。"""
    header = ["video", "frame", "verdict", "score", "yaw", "pitch", "roll",
              "down_ratio", "nfaces", "ear", "gaze_mag", "gaze_dy", "nheads"]
    src_header = ",".join(header[1:])
    total = merged = 0
    all_path = os.path.join(out_dir, "pose_report_all.csv")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for fn in video_files:
            stem = os.path.splitext(fn)[0]
            src = os.path.join(out_dir, f"pose_report_{stem}.csv")
            if not os.path.isfile(src):
                print(f"  WARNING: 缺少 {src}(该视频可能处理失败), "
                      f"合并 CSV 不含其行")
                continue
            with open(src, encoding="utf-8") as sf:
                lines = [ln for ln in sf.read().splitlines() if ln]
            if not lines or lines[0] != src_header:
                print(f"  WARNING: {src} 表头异常, 跳过")
                continue
            f.write("\n".join(f"{fn},{ln}" for ln in lines[1:]))
            f.write("\n")
            total += len(lines) - 1
            merged += 1
            os.remove(src)
    print(f"  [merge] {merged}/{len(video_files)} 视频 -> {all_path} "
          f"({total} 数据行), 源 pose_report_<stem>.csv 已并入删除")
    return total, merged


def merge_dedup_reports(out_dir, video_files):
    """目录批处理模式收尾: 把 out_dir 内逐视频 <stem>_dedup_report.csv 合并为
    dedup_report_all.csv —— 风格与 merge_pose_reports 一致(首列 video,
    行原样前缀, 合并后删除源 CSV)。--dedup 未产出文件的视频跳过并 WARN
    (dedup 关闭或该视频无行)。返回 (数据行数, 成功合并的视频数)。"""
    header = ["video", "frame", "diff", "seg_id", "seg_start", "seg_end",
              "sharp", "is_rep"]
    src_header = ",".join(header[1:])
    total = merged = 0
    all_path = os.path.join(out_dir, "dedup_report_all.csv")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for fn in video_files:
            stem = os.path.splitext(fn)[0]
            src = os.path.join(out_dir, f"{stem}_dedup_report.csv")
            if not os.path.isfile(src):
                continue   # 无 --dedup 时整批都没有, 静默; 单视频缺失见下
            with open(src, encoding="utf-8") as sf:
                lines = [ln for ln in sf.read().splitlines() if ln]
            if not lines or lines[0] != src_header:
                print(f"  WARNING: {src} 表头异常, 跳过")
                continue
            f.write("\n".join(f"{fn},{ln}" for ln in lines[1:]))
            f.write("\n")
            total += len(lines) - 1
            merged += 1
            os.remove(src)
    if merged or os.path.isfile(all_path):
        print(f"  [merge] dedup 明细 {merged}/{len(video_files)} 视频 -> "
              f"{all_path} ({total} 数据行)"
              + (", 源 <stem>_dedup_report.csv 已并入删除" if merged else ""))
    return total, merged


def merge_shot_tables(out_dir, video_files):
    """目录批处理模式收尾: 把 out_dir 内逐视频 <stem>_shot_table.csv 合并为
    shot_table_all.csv —— 风格与 merge_dedup_reports 一致(首列 video,
    行原样前缀, 合并后删除源 CSV)。--dedup 未产出镜头表的视频跳过(静默)。
    返回 (镜头行数, 成功合并的视频数)。"""
    header = ["video", "shot_id", "seg_count", "frame_start", "frame_end",
              "frames", "sent", "quota", "close_reason"]
    src_header = ",".join(header[1:])
    total = merged = 0
    all_path = os.path.join(out_dir, "shot_table_all.csv")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for fn in video_files:
            stem = os.path.splitext(fn)[0]
            src = os.path.join(out_dir, f"{stem}_shot_table.csv")
            if not os.path.isfile(src):
                continue
            with open(src, encoding="utf-8") as sf:
                lines = [ln for ln in sf.read().splitlines() if ln]
            if not lines or lines[0] != src_header:
                print(f"  WARNING: {src} 表头异常, 跳过")
                continue
            f.write("\n".join(f"{fn},{ln}" for ln in lines[1:]))
            f.write("\n")
            total += len(lines) - 1
            merged += 1
            os.remove(src)
    if merged:
        print(f"  [merge] 镜头表 {merged}/{len(video_files)} 视频 -> "
              f"{all_path} ({total} 镜头), 源 <stem>_shot_table.csv 已并入删除")
    return total, merged


def main():
    ap = argparse.ArgumentParser(
        description="Filter a video to high-quality face frames (JPG), no re-encode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""用法示例:
  python filter_video.py video.mp4                    单视频 -> <视频所在目录>/<stem>_kept + pose_report.csv
  python filter_video.py video_dir                     视频目录 -> <video_dir>/kept_frames/<stem>_kept 每视频一目录(旧行为)
  python filter_video.py video_dir out_dir             目录批处理: 目录下全部视频(.mp4/.mkv/.mov/.avi,单层,按名排序),
                                                       kept JPG 平铺进 out_dir(带视频 stem 前缀),
                                                       CSV 合并为 out_dir/pose_report_all.csv(首列 video)
  python filter_video.py video_dir out_dir --gaze-dy-dev 11
                                                       所有 flag 与单视频模式完全相同(默认 --decode auto;
                                                       显存直通臂显式加 --decode pynvvc-gpu, overlap 默认 auto)""")
    ap.add_argument("input", nargs="?", default=None,
                    help="mp4 file or a directory of videos "
                         "(可省略, 若用 --image-dir 跑纯图片文件夹模式)")
    ap.add_argument("output", nargs="?", default=None,
                    help="输出目录(第二位置参数, 仅当第一个位置参数是视频目录时有效): "
                         "目录批处理模式——所有视频 kept JPG 平铺进该目录, "
                         "pose_report 合并为 pose_report_all.csv(首列 video)。"
                         "与 --out-dir 同时给时以本参数为准(WARN)")
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
    ap.add_argument("--decode", choices=["auto", "gpu", "ffmpeg", "pynvvc",
                                         "pynvvc-gpu"], default="auto",
                    help="解码帧源（默认 auto）: gpu=ffmpeg NVDEC+GPU 转换直出 bgr24 "
                         "rawvideo 到 stdout 管道、Python 内存流式读（与 ffmpeg 路径同一"
                         "命令 bit-exact，不落 3GB raw，解码与下游检测重叠）；"
                         "auto=gpu，失败（含中途断流）自动 WARNING 回退 ffmpeg 落盘路径"
                         "重跑；pynvvc=PyNvVideoCodec(NVDEC) 硬解帧源（按源帧号 "
                         "stride*k+delta 抽帧，仅 60/30fps 已标定、表外帧率自动回退管道；"
                         "失败同样逐级回退 管道→落盘）；pynvvc-gpu=显存直通零拷贝臂"
                         "（NVDEC device 帧不落宿主: 自研 INTER_AREA/letterbox 核在显存"
                         "完成预处理 → 原引擎零拷贝推理, blob 与宿主臂逐字节相同, 判定"
                         "结构性一致; 仅人脸帧 D2H 落地; 支持 --dedup（缩略图 GPU 生成"
                         " + 段内帧显存驻留, 判定与宿主 dedup 臂逐位一致）; "
                         "表外帧率/失败逐级回退 pynvvc→管道→落盘）；"
                         "ffmpeg=现有 3GB raw 落盘路径（旧行为，一行未改）。"
                         "--hw-decode 对 auto/gpu/ffmpeg 路径均生效")
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
    ap.add_argument("--overlap", choices=["auto", "on", "off"], default="auto",
                    help="解码∥推理重叠（仅 --decode pynvvc-gpu 臂）: auto=默认,"
                         "ThreadedDecoder 内部 C++ 线程解码不占 GIL, 后台 producer "
                         "拉帧+D2D, 主线程满速推理（初始化失败自动 WARNING 回退 A 轮"
                         "串行臂）; off=A 轮串行行为（A/B 用）; on=强制开（失败同样"
                         "回退串行）。判定逐位不变（同一共享流保序, 帧即拉即拷）")
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
    ap.add_argument("--dedup-max-seg", type=int, default=None,
                    help="dedup 段最大帧数(旧裸帧数口径; 默认 None=按 "
                         "--dedup-max-gap-sec 换算, 显式给定时优先)")
    ap.add_argument("--dedup-max-gap-sec", type=float, default=1.0,
                    help="镜头感知采样: 段长时间兜底间隔(秒, 默认 1.0=旧口径 "
                         "max_seg=10@10fps)。按实际送检帧率换算 "
                         "max_seg=max(min_seg, round(此值*dec_fps)), off-table "
                         "fps 回退臂同样正确")
    ap.add_argument("--dedup-per-shot-max", type=int, default=4,
                    help="镜头感知采样: 每镜头送检配额上限 N(默认 4; 传 0 关闭"
                         "=逐段送检旧行为)。硬切(diff>=cut_hi)视为镜头边界, "
                         "软切(diff>=cut_lo)为镜头内次级边界; 镜头内段数>N 时"
                         "不再逐段送检, 镜头内全部候选帧按清晰度排序只送最清晰 "
                         "q=max(N,--dedup-per-shot-min) 帧(仍逐帧过闸门)")
    ap.add_argument("--dedup-per-shot-min", type=int, default=1,
                    help="镜头感知采样: 每镜头至少 M 帧送检(默认 1=旧逐段 rep "
                         "机制天然满足)。M>1 且镜头段数<M 时按段内回退序"
                         "(逐段时间序、段内清晰度降序)补判至 M 帧")
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
    if (args.dedup and args.dedup_max_seg is None
            and args.dedup_max_gap_sec <= 0):
        ap.error("--dedup-max-gap-sec 必须 > 0(或显式给 --dedup-max-seg)")

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
        out_dir = args.out_dir or args.output or (os.path.normpath(image_dir) + "_head")
        process_image_dir(image_dir, out_dir, head, batch=args.batch,
                          imread_workers=args.imread_workers,
                          verbose=args.verbose)
        return
    if args.input is None:
        ap.error("需要 input（视频/视频目录）或 --image-dir（纯图片目录）")

    # ---------------- 目录批处理模式: python filter_video.py <input目录> <output目录> ----------------
    # 两个位置参数: 第一=视频目录(单层), 第二=统一输出目录。复用现有
    # 「目录输入 + 共享 out_dir」分支(逐视频 process()), 收尾合并 CSV。
    dir_batch = False
    if args.output is not None:
        if args.out_dir is not None:
            print(f"  WARNING: 同时给了第二位置参数(输出目录 {args.output}) 与 "
                  f"--out-dir, 以第二位置参数为准(--out-dir 忽略)")
        if not os.path.isdir(args.input):
            ap.error("第二位置参数(输出目录)仅目录批处理模式有效: "
                     "第一个位置参数必须是视频目录")
        dir_batch = True

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
    batch_out = None   # 目录批处理模式: 统一输出目录（收尾合并 CSV 用）
    if os.path.isdir(args.input):
        vids = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                      if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi")))
        if dir_batch:
            # 目录批处理: 全部视频 kept JPG 平铺进 batch_out(带 stem 前缀不冲突),
            # CSV 按视频名区分(pose_report_<stem>.csv)收尾合并为 pose_report_all.csv
            batch_out = args.output
            os.makedirs(batch_out, exist_ok=True)
            _pre = [f for f in os.listdir(batch_out)
                    if os.path.isfile(os.path.join(batch_out, f))]
            if _pre:
                print(f"  WARNING: 输出目录 {batch_out} 已有 {len(_pre)} 个文件, "
                      f"同名 JPG/CSV 将被覆盖(重跑语义=更新)")
            # 同 stem 不同扩展名(如 a.mp4 + a.mkv)理论重名风险预检:
            _seen_stems = {}
            for _p in vids:
                _s = os.path.splitext(os.path.basename(_p))[0].lower()
                if _s in _seen_stems:
                    print(f"  WARNING: 输入内 {_seen_stems[_s]} 与 "
                          f"{os.path.basename(_p)} 同 stem(忽略扩展名/大小写), "
                          f"输出 JPG 可能互相覆盖, 建议先重命名输入")
                _seen_stems[_s] = os.path.basename(_p)
            if not vids:
                sys.exit(f"{args.input} 下没有视频(.mp4/.mkv/.mov/.avi)")
            targets = [(p, batch_out) for p in vids]
            shared_csv = True
        elif args.out_dir:
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
                     dedup_per_shot_max=args.dedup_per_shot_max,
                     dedup_per_shot_min=args.dedup_per_shot_min,
                     dedup_max_gap_sec=args.dedup_max_gap_sec,
                     dedup_thumb=args.dedup_thumb,
                     dedup_sharp_side=args.dedup_sharp_side,
                     dedup_backups=args.dedup_backup,
                     dedup_max_mem=args.dedup_max_mem,
                     hw_decode=args.hw_decode,
                     decode_mode=args.decode,
                     pipeline=args.pipeline,
                     pipeline_depth=args.pipeline_depth,
                     head_batch=args.head_batch,
                     overlap=args.overlap)
    if dir_batch:
        # 目录批处理模式: 逐视频进度行 + 重名防护 + 收尾合并 CSV + 汇总
        stems_done = []   # [(视频文件名, stem)] 已处理
        out_listing = [set(os.listdir(batch_out))]   # 处理前目录快照(闭包共享)

        def _on_video(i, n, p, out, dt):
            stem = os.path.splitext(os.path.basename(p))[0]
            # 重名防护: 本视频新增文件若撞上前面已处理视频的 stem 前缀
            # (仅同目录不同扩展名同 stem 时可能), 加视频序号后缀防覆盖
            cur = set(os.listdir(batch_out))
            new_files = sorted(cur - out_listing[0])
            out_listing[0] = cur
            renamed = 0
            for prev_fn, prev_stem in stems_done:
                if prev_stem.lower() == stem.lower() and \
                        prev_fn != os.path.basename(p):
                    for fn in new_files:
                        if fn.lower().startswith(stem.lower() + "_"):
                            new = (fn[:-4] + f"_v{i}.jpg"
                                   if fn.lower().endswith(".jpg")
                                   else f"{fn}_v{i}")
                            print(f"  WARNING: 输出重名 {fn} "
                                  f"(与 {prev_fn} 同 stem), 改名为 {new}")
                            os.replace(os.path.join(batch_out, fn),
                                       os.path.join(batch_out, new))
                            renamed += 1
                    break
            stems_done.append((os.path.basename(p), stem))
            print(f"[{i}/{n}] {stem}: kept={out['kept']} 判定帧={out['frames']} "
                  f"耗时{out['seconds']:.1f}s 臂={out['decode_mode']}"
                  + (f" 重名改名{renamed}个" if renamed else ""), flush=True)

        results, failed, dt = vf.run(targets, args.yaw, shared_csv=True,
                                     on_video=_on_video)
        merge_pose_reports(batch_out, [os.path.basename(p) for p, _ in targets])
        if args.dedup:
            merge_dedup_reports(batch_out,
                                [os.path.basename(p) for p, _ in targets])
            merge_shot_tables(batch_out,
                              [os.path.basename(p) for p, _ in targets])
        total_kept = sum(o["kept"] for _, o in results)
        print(f"\n[目录批处理] done={len(results)} failed={len(failed)} "
              f"kept 合计={total_kept}  总墙钟 {dt:.1f}s  -> "
              f"{os.path.abspath(batch_out)}")
        if failed:
            print(f"  failed 清单: {[os.path.basename(f[0]) for f in failed]}")
    else:
        vf.run(targets, args.yaw, shared_csv=shared_csv)


if __name__ == "__main__":
    main()
