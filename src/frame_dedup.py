# -*- coding: utf-8 -*-
"""
frame_dedup.py — 抽帧去重 + 段内选最佳帧(零新增依赖, 只用 cv2/numpy)

思想(借鉴 PySceneDetect 的 content-score(MAD) + 滞回, 但做的是段内选帧而非场景切分):
  * 每帧算长边 thumb_side 的灰度缩略图, diff = mean|thumb - prev_thumb|
    (相邻帧变化量, 即逐帧 MAD);
  * 同时算长边 sharp_side 的灰度图, sharp = Laplacian(gray).var()
    (清晰度, 越大 = 运动模糊越少);
  * 段关闭条件(满足任一, 当前帧开新段并作为新段第一帧):
      1) diff >= cut_hi                快速动作/scene cut, 逐帧独立成段 → 不漏帧
      2) diff >= cut_lo 且 段长>=min_seg  滞回防闪烁
      3) 段长 >= max_seg               时间上限兜底(10fps 下 max_seg=10 → 全程无动作
                                       也每秒出 1 帧)
  * 段关闭时 rep = 段内 argmax(sharp)(最清晰一张, 避运动模糊), 另取次清晰
    n_backups 张为 backup。meta['frames'] 保留【全段帧】按清晰度降序
    [(idx, sharp, frame), ...], 供下游闸门回退:
      - rep 被闸门 drop → 按 meta['frames'] 清晰度次序对剩余帧补跑闸门,
        命中第一个 keep 即停(early stop);
      - rep keep → 段内其它"解码秒"各取最清晰一帧补判(防跨秒段丢失另一秒)。
  * 内存上界: deduper 自身最多持有段内 max_seg 个全分辨率帧; 段关闭后全段帧
    由调用方(seg 队列)持有直到送检, 调用方需控制队列总字节(见 filter_video.py
    的 SEG_Q_BYTES); 高分辨率(如 4K)按帧大小把 eff_max_seg 自适应压到
    max_mem_mb 内。

用法:
    sel = FrameDedupSelector()
    for frame, idx in decoded_frames:
        res = sel.push(frame, idx)      # 段未关闭返回 None
        if res is not None:
            rep_frame, rep_idx, backups, meta = res
            # rep/backups 是全分辨率 BGR; 送下游管线
    res = sel.flush()                   # EOF 强关最后一段(空段返回 None)
"""
import cv2
import numpy as np


class FrameDedupSelector:
    def __init__(self, cut_lo=2.0, cut_hi=10.0, min_seg=2, max_seg=10,
                 thumb_side=128, sharp_side=256, n_backups=2,
                 max_mem_mb=48.0):
        if not (0 <= cut_lo < cut_hi):
            raise ValueError("需要 0 <= cut_lo < cut_hi")
        self.cut_lo = float(cut_lo)
        self.cut_hi = float(cut_hi)
        self.min_seg = max(1, int(min_seg))
        self.max_seg = max(self.min_seg, int(max_seg))
        self.thumb_side = max(8, int(thumb_side))
        self.sharp_side = max(8, int(sharp_side))
        self.n_backups = max(0, int(n_backups))
        self.max_mem_bytes = int(max_mem_mb) * 1024 * 1024
        self._prev_thumb = None
        self._seg = []          # [(frame_bgr, idx, diff, sharp), ...] 当前段
        self._seg_id = 0

    # ---- 低成本度量 ----
    def _gray_thumb(self, frame, side):
        """长边缩到 side(INTER_AREA)的灰度图; 已经 <= side 则不放大。"""
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side > side:
            s = side / long_side
            frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- 段关闭 ----
    def _close_seg(self):
        """关闭当前段: rep=argmax(sharp), backups=次清晰 n_backups 张(全分辨率)。
        返回 (rep_frame, rep_idx, [(bk_frame, bk_idx), ...], meta)。
        meta['rows']   = [(frame_idx, diff, sharp, is_rep), ...] 全段逐帧, 供
                          dedup_report.csv;
        meta['frames'] = [(idx, sharp, frame), ...] 全段帧按清晰度降序(含 rep 在
                          首位), 供闸门回退补判(全段候选 + early stop);
        meta['nbytes_total'] = 全段帧字节数(调用方控队列内存用)。"""
        seg = self._seg
        order = sorted(range(len(seg)), key=lambda i: seg[i][3], reverse=True)
        rep_i = order[0]
        rep_frame, rep_idx, _, _ = seg[rep_i]
        backups = [(seg[i][0], seg[i][1]) for i in order[1:1 + self.n_backups]]
        frames_sorted = [(seg[i][1], seg[i][3], seg[i][0]) for i in order]
        meta = {
            "seg_id": self._seg_id,
            "seg_start": seg[0][1],
            "seg_end": seg[-1][1],
            "rows": [(seg[i][1], seg[i][2], seg[i][3], 1 if i == rep_i else 0)
                     for i in range(len(seg))],
            "frames": frames_sorted,
            "nbytes_total": sum(seg[i][0].nbytes for i in range(len(seg))),
        }
        self._seg_id += 1
        return rep_frame, rep_idx, backups, meta

    def push(self, frame_bgr, frame_idx):
        """推一帧(BGR 全分辨率, frame_idx=原始解码帧号)。
        段未关闭返回 None; 段关闭返回 (rep_frame, rep_idx,
        [(bk_frame, bk_idx), ...], meta), 且 frame_bgr 成为新段第一帧。"""
        thumb = self._gray_thumb(frame_bgr, self.thumb_side)
        if self._prev_thumb is None:
            diff = 0.0
        else:
            diff = float(np.abs(thumb.astype(np.int16)
                                - self._prev_thumb.astype(np.int16)).mean())
        self._prev_thumb = thumb
        sharp = float(cv2.Laplacian(
            self._gray_thumb(frame_bgr, self.sharp_side), cv2.CV_64F).var())

        close = False
        if self._seg:
            # 高分辨率自适应内存上限: 段内全分辨率帧总字节 <= max_mem_bytes
            eff_max = self.max_seg
            nbytes = frame_bgr.nbytes
            if nbytes * eff_max > self.max_mem_bytes:
                eff_max = max(self.min_seg, self.max_mem_bytes // nbytes)
            n = len(self._seg)
            if diff >= self.cut_hi:
                close = True          # 1) 快速动作 / scene cut → 不漏帧
            elif diff >= self.cut_lo and n >= self.min_seg:
                close = True          # 2) 滞回防闪烁
            elif n >= eff_max:
                close = True          # 3) 时间上限兜底
        if close:
            rep = self._close_seg()
            self._seg = []            # 旧段帧引用移交 meta['frames'], 由调用方
            self._seg.append((frame_bgr, frame_idx, diff, sharp))  # 在送检后释放
            return rep
        self._seg.append((frame_bgr, frame_idx, diff, sharp))
        return None

    def flush(self):
        """EOF 强关最后一段; 无剩余帧返回 None。"""
        if self._seg:
            rep = self._close_seg()
            self._seg = []
            return rep
        return None


class ShotTracker:
    """镜头感知采样(镜头切分 + 镜头内配额 + 保底)的镜头侧记账器。

    与 FrameDedupSelector/SlotSegDeduper(段状态机)配合: 段每关闭一段, 调用方把
    段元组 (rep, rep_idx, backups, meta) 连同 meta['frames'](帧引用列表, 宿主臂
    =ndarray, GPU 臂=驻留池槽号)交给本类。镜头判定完全复用段状态机的既有决策,
    本类【不改变】任何 diff/关段逻辑:

      * 镜头边界 = 段首帧 diff >= cut_hi(硬切)。段状态机里 diff>=cut_hi 恒为
        第 1 优先关段条件, 且该触发帧成为下一段首帧 → 「下一段 rows[0][1]
        (diff) >= cut_hi」⟺ 上一次关段是硬切。软切(cut_lo)/兜底切(max_seg)
        均为镜头内次级边界, 镜头延续(长静止镜头被 max_seg 切出的众多段同属
        一个镜头 —— 这正是配额要压的对象)。

      * 送检配额(per_shot_max=N, 默认 None=关): 镜头内关闭段数 > N 时激活,
        该镜头不再逐段送检; 镜头内【全部候选帧】按清晰度降序(平局取帧号小者,
        与段内 argmax 稳定排序同约定)只保留最清晰 q=max(N, per_shot_min) 帧
        送检, 逐帧仍过完整闸门链(闸门逻辑零改动)。未入选帧的引用立即经
        release 回调归还(宿主臂=解引用交给 GC, GPU 臂=归还驻留池槽位),
        缓冲有界(<= q 帧 + 当前段)。

      * 镜头保底(per_shot_min=M, 默认 1): 每镜头至少 M 帧送检。旧逐段 rep
        机制天然满足 M=1(每镜头至少 1 段、每段至少送 1 个 rep); M>1 且镜头
        段数 < M 时, 在该镜头逐段流处理完后, 按「段内回退序」(逐段时间序、
        段内清晰度降序 —— 即非配额镜头 rep 被 drop 时的既有回退序)对未判过
        的帧补判至 M 帧送检(由调用方执行, 本类只出 topup 名单)。

      * 内存护栏 mem_guard_bytes: 镜头缓冲字节超护栏时 —— 配额可用则立即激活
        (缩到 q 帧); 仅 min 模式(per_shot_max=None)无配额可用则强制关镜头
        (close_reason=memcap), 防超长镜头缓冲爆内存。纯记录模式不持帧, 不触发。

    两种用法:
      * 纯记录模式 add_segment(seg, frames=None): 只记镜头表(shot_id/起止帧/
        帧数/段数)与 idx→镜头 映射(note_judged 记账 sent), 不持有任何
        seg/meta/帧引用, 不改变调用方送检流 —— 默认参数下逐字节零影响。
      * 配额模式 add_segment(seg, meta['frames']): 返回因本段而关闭的镜头工作
        项(0/1 个), 调用方把工作项替代裸段元组入送检队列; 工作项分两种:
          quota=True : {"sel": [(idx, ref), ...] 时间序送检名单, 无逐段流}
          quota=False: {"segs": [段元组, ...], "topup": [(idx, ref), ...]}
        item["release"] = 处理完后需归还的引用清单(各引用恰归还一次)。
    """

    def __init__(self, cut_hi, per_shot_max=None, per_shot_min=1,
                 mem_guard_bytes=512 * 1024 * 1024, release=None):
        self.cut_hi = float(cut_hi)
        pm = per_shot_max if per_shot_max is None else int(per_shot_max)
        self.per_shot_max = pm if (pm is not None and pm > 0) else None
        self.per_shot_min = max(1, int(per_shot_min))
        self.mem_guard = int(mem_guard_bytes)
        self._release = release
        self._shot_id = 0
        self._cur = None          # 当前(未关闭)镜头积累器
        self.table = []           # 镜头表行 dict 列表(调用方落 CSV)
        self.shot_map = {}        # 解码帧号 idx → shot_id(note_judged 记账)
        self._row_by_id = {}      # shot_id → 已关闭镜头表行
        self._sent_open = {}      # shot_id → 未关闭镜头的已送检帧数

    def _q(self):
        """配额镜头实际送检帧数: 上限 N 与保底 M 取大(M>N 时配额按 M 执行)。"""
        return max(self.per_shot_max, self.per_shot_min)

    def _new_shot(self):
        self._shot_id += 1
        self._cur = {"id": self._shot_id, "segs": [], "frames": [],
                     "nsegs": 0, "nframes": 0, "bytes": 0, "fbytes": 0,
                     "quota": False, "start": None, "end": None}

    def _reselect(self, cur, q):
        """镜头内重选 top-q(清晰度降序, 平局帧号小者优先), 淘汰帧立即归还。"""
        frames = cur["frames"]
        if len(frames) <= q:
            return
        order = sorted(range(len(frames)),
                       key=lambda i: (-frames[i][1], frames[i][0]))
        cur["frames"] = [frames[i] for i in order[:q]]
        drop = [frames[i] for i in order[q:]]
        if cur["fbytes"]:
            cur["bytes"] = len(cur["frames"]) * cur["fbytes"]
        if drop and self._release is not None:
            self._release([f[2] for f in drop])

    def _activate(self, cur):
        cur["quota"] = True
        cur["segs"] = []          # 配额镜头不再逐段送检, 释放段/rep/backups 引用
        self._reselect(cur, self._q())

    def _close(self, reason):
        cur = self._cur
        self._cur = None
        row = {"shot_id": cur["id"], "seg_count": cur["nsegs"],
               "frame_start": cur["start"], "frame_end": cur["end"],
               "frames": cur["nframes"],
               "sent": self._sent_open.pop(cur["id"], 0),
               "quota": 1 if cur["quota"] else 0, "close_reason": reason}
        self.table.append(row)
        self._row_by_id[cur["id"]] = row
        if cur["quota"]:
            sel = sorted(cur["frames"], key=lambda f: f[0])   # 送检按时间序
            return {"kind": "shot", "quota": True, "shot_id": cur["id"],
                    "nsegs": cur["nsegs"], "sel": [(f[0], f[2]) for f in sel],
                    "nbytes": len(sel) * cur["fbytes"], "row": row,
                    "release": [f[2] for f in sel]}
        return {"kind": "shot", "quota": False, "shot_id": cur["id"],
                "nsegs": cur["nsegs"], "segs": cur["segs"],
                "topup": [(f[0], f[2]) for f in cur["frames"]],
                "min": self.per_shot_min, "nbytes": cur["bytes"], "row": row,
                "release": [f[2] for f in cur["frames"]]}

    def add_segment(self, seg, frames=None):
        """推入一个已关闭段。seg=(rep, rep_idx, backups, meta)(两臂同构);
        frames=meta['frames']([(idx, sharp, ref), ...] 清晰度降序)或 None
        (纯记录模式)。返回因本段而关闭的镜头工作项列表(0/1 个)。"""
        meta = seg[3]
        items = []
        if self._cur is None:
            self._new_shot()
        elif meta["rows"][0][1] >= self.cut_hi:
            # 本段首帧 diff>=cut_hi ⟺ 上一次关段是硬切 → 镜头边界
            items.append(self._close("cut"))
            self._new_shot()
        cur = self._cur
        cur["nsegs"] += 1
        cur["nframes"] += len(meta["rows"])
        if cur["start"] is None:
            cur["start"] = meta["seg_start"]
        cur["end"] = meta["seg_end"]
        sid = cur["id"]
        for r in meta["rows"]:
            self.shot_map[r[0]] = sid
        if frames is None:
            return items          # 纯记录模式: 不持任何 seg/meta/帧引用
        cur["segs"].append(seg)   # 配额激活前需保留段元组(非配额镜头逐段送检)
        cur["frames"].extend(frames)
        n = max(1, len(meta["rows"]))
        cur["fbytes"] = meta["nbytes_total"] // n   # 帧全同尺寸, 取均值即精确值
        cur["bytes"] += meta["nbytes_total"]
        if self.per_shot_max is not None:
            if not cur["quota"]:
                if cur["nsegs"] > self.per_shot_max:
                    self._activate(cur)          # 段数超上限 → 配额生效
            else:
                self._reselect(cur, self._q())   # 运行中维持 top-q
        if cur["bytes"] > self.mem_guard:
            if self.per_shot_max is not None:
                if not cur["quota"]:
                    self._activate(cur)          # 护栏: 立即缩到 q 帧
            elif cur["segs"]:
                # 仅 min 模式无配额可用 → 强制关镜头防爆内存
                items.append(self._close("memcap"))
        return items

    def close_eof(self):
        """EOF: 强制关闭当前镜头(close_reason=eof), 返回工作项列表(0/1 个)。"""
        if self._cur is not None:
            return [self._close("eof")]
        return []

    def note_judged(self, idx):
        """闸门每判一帧记账一次(送检口径, 含补判); 由调用方 _judge_and_write
        调用。镜头未关闭时先挂账, 关闭时并入表行。"""
        sid = self.shot_map.get(idx)
        if sid is None:
            return
        row = self._row_by_id.get(sid)
        if row is not None:
            row["sent"] += 1
        else:
            self._sent_open[sid] = self._sent_open.get(sid, 0) + 1


class SlotSegDeduper(FrameDedupSelector):
    """显存槽号版段状态机 —— 与 FrameDedupSelector 的分段/选帧决策【逐行同构】,
    供 --decode pynvvc-gpu --dedup 零拷贝臂使用。

    差异只有一个: 帧不再以全分辨率 BGR ndarray 持有, 而是【显存池槽号】
    (调用方保证槽内帧在段关闭+送检完成前不被覆写, 见 filter_video.py
    _run_pass_gpu_dedup 的驻留池设计)。push_slot 的输入是两张小图(调用方
    从 GPU INTER_AREA 核输出 D2H 而来, 与宿主 _gray_thumb 的 resize 输出
    逐位一致) + 本帧字节数 + 槽号; diff/sharp 计算与段关闭判定三条件、
    高分辨率 eff_max 自适应、_close_seg 的 rep/backups/meta 结构全部照抄
    FrameDedupSelector.push/_close_seg, 一行未改语义 —— 宿主臂与 GPU 臂
    因此对同一视频产出相同的段切分与 rep 选择。

    push_slot(thumb_small, sharp_small, nbytes, slot, idx) 参数:
      thumb_small : 长边 thumb_side 的 BGR 小图(未转灰度; 灰度化在此处用
                    cv2 完成, 与宿主 _gray_thumb 的 cvtColor 同输入同结果)
      sharp_small : 长边 sharp_side 的 BGR 小图
      nbytes      : 全分辨率帧字节数(= h*w*3, eff_max 自适应用, 与宿主
                    frame_bgr.nbytes 同值)
      slot        : 本帧在显存驻留池中的槽号(meta['frames'] 以槽号代替帧)
      idx         : 原始解码帧号(与宿主 push 的 frame_idx 同语义)
    返回/flush 语义与 FrameDedupSelector 一致, 但 meta['frames'] 元组为
    (idx, sharp, slot)。nbytes_total 基类实现取 seg[i][0].nbytes, 槽号版
    改用 __init__ 传入的固定帧字节数(GPU 臂全部同尺寸)。"""

    def __init__(self, *a, frame_bytes=0, **kw):
        super().__init__(*a, **kw)
        self._frame_bytes = int(frame_bytes)

    def _close_seg(self):
        """与 FrameDedupSelector._close_seg 逐行同构, 仅 nbytes_total 改用
        固定帧字节数(段内帧全同尺寸, sum(frame.nbytes) 等价替换)。"""
        seg = self._seg
        order = sorted(range(len(seg)), key=lambda i: seg[i][3], reverse=True)
        rep_i = order[0]
        rep_frame, rep_idx, _, _ = seg[rep_i]
        backups = [(seg[i][0], seg[i][1]) for i in order[1:1 + self.n_backups]]
        frames_sorted = [(seg[i][1], seg[i][3], seg[i][0]) for i in order]
        meta = {
            "seg_id": self._seg_id,
            "seg_start": seg[0][1],
            "seg_end": seg[-1][1],
            "rows": [(seg[i][1], seg[i][2], seg[i][3], 1 if i == rep_i else 0)
                     for i in range(len(seg))],
            "frames": frames_sorted,
            "nbytes_total": self._frame_bytes * len(seg),
        }
        self._seg_id += 1
        return rep_frame, rep_idx, backups, meta

    def push_slot(self, thumb_small, sharp_small, nbytes, slot, idx):
        thumb = cv2.cvtColor(thumb_small, cv2.COLOR_BGR2GRAY)
        if self._prev_thumb is None:
            diff = 0.0
        else:
            diff = float(np.abs(thumb.astype(np.int16)
                                - self._prev_thumb.astype(np.int16)).mean())
        self._prev_thumb = thumb
        sharp = float(cv2.Laplacian(
            cv2.cvtColor(sharp_small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        close = False
        if self._seg:
            # 高分辨率自适应内存上限: 段内全分辨率帧总字节 <= max_mem_bytes
            # (与 FrameDedupSelector.push 同式; nbytes=全分辨率帧字节)
            eff_max = self.max_seg
            if nbytes * eff_max > self.max_mem_bytes:
                eff_max = max(self.min_seg, self.max_mem_bytes // nbytes)
            n = len(self._seg)
            if diff >= self.cut_hi:
                close = True          # 1) 快速动作 / scene cut → 不漏帧
            elif diff >= self.cut_lo and n >= self.min_seg:
                close = True          # 2) 滞回防闪烁
            elif n >= eff_max:
                close = True          # 3) 时间上限兜底
        if close:
            rep = self._close_seg()
            self._seg = []            # 旧段帧引用移交 meta['frames'], 由调用方
            self._seg.append((slot, idx, diff, sharp))  # 在送检后释放
            return rep
        self._seg.append((slot, idx, diff, sharp))
        return None
