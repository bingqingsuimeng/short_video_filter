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
