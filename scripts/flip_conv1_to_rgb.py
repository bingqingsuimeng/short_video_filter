# -*- coding: utf-8 -*-
"""
flip_conv1_to_rgb.py — 把第一个卷积层的输入通道维权重翻转 (w[:, ::-1, :, :])，
使 BGR↔RGB 通道约定被吸收进权重（数学上精确等价：第一层的加权和只是交换
求和顺序，BN/bias 在 conv 之后不受影响）。

等价性质（对任意输入 x，swap 指沿通道维反转）：
    flipped_model(swap(x)) == original_model(x)

即：若原模型由"管线预处理输出 RGB"驱动（本仓库三个模型现状：face_det 的
letterbox kernel / gaze_resnet / head_gate 的预处理都把 BGR 帧转成 RGB 再喂
引擎），则翻转后的模型必须喂"同一预处理的**未交换**版本（BGR blob）"，
输出逐位一致。这正对应"显存直通"路线：NVDEC 直出 RGB → 不改动
letterbox kernel（其内部做一次通道交换）→ 交换后的 BGR blob → 翻转引擎。

用法（每个模型一条，输出新文件，绝不覆盖原 ONNX）：
    python scripts/flip_conv1_to_rgb.py <in.onnx> <out.onnx>

校验（不通过则拒绝写出）：
  1. 图输入只被一个节点消费，且该节点是 Conv（无多 stem 分支 / 非 Conv 输入）；
  2. Conv 的权重是 initializer 且为 4 维 (out_ch, 3, kh, kw)；
  3. 翻转后重载回读，逐元素等于原权重[:, ::-1, :, :]。
"""
import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper as onh


def find_first_conv(model, onnx_path):
    """定位消费图输入的唯一节点，要求其为 Conv 且权重 (out,3,kh,kw)。
    返回 (node, weight_initializer_name)。"""
    g = model.graph
    in_names = [i.name for i in g.input]
    if len(in_names) != 1:
        raise RuntimeError(f"{onnx_path}: 图输入 {len(in_names)} 个 {in_names}，"
                           f"非单输入结构，不自动翻转")
    gin = in_names[0]
    consumers = [n for n in g.node if gin in n.input]
    if len(consumers) != 1:
        raise RuntimeError(f"{onnx_path}: 图输入 '{gin}' 被 {len(consumers)} 个节点"
                           f"消费 {[ (n.op_type, n.name) for n in consumers ]}，"
                           f"非单 stem 结构，不自动翻转")
    node = consumers[0]
    if node.op_type != "Conv":
        raise RuntimeError(f"{onnx_path}: 输入首个消费者是 {node.op_type}"
                           f"({node.name}) 而非 Conv，不自动翻转")
    w_name = node.input[1]
    inits = {i.name: i for i in g.initializer}
    if w_name not in inits:
        raise RuntimeError(f"{onnx_path}: Conv '{node.name}' 权重 '{w_name}' "
                           f"不是 initializer，不自动翻转")
    dims = inits[w_name].dims
    if len(dims) != 4 or dims[1] != 3:
        raise RuntimeError(f"{onnx_path}: Conv '{node.name}' 权重 '{w_name}' "
                           f"dims={list(dims)}，输入通道数 != 3，不自动翻转")
    print(f"[flip] {os.path.basename(onnx_path)}: 输入 '{gin}' -> "
          f"{node.op_type} '{node.name}'，权重 '{w_name}' {list(dims)}")
    return node, w_name


def flip(in_path, out_path):
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise RuntimeError("拒绝覆盖原 ONNX：输出路径不能与输入相同")
    model = onnx.load(in_path)  # 默认加载外部数据（若权重外置）
    _, w_name = find_first_conv(model, in_path)
    g = model.graph
    init = next(i for i in g.initializer if i.name == w_name)
    w = onh.to_array(init)
    w_flipped = np.ascontiguousarray(w[:, ::-1, :, :])
    init.CopyFrom(onh.from_array(w_flipped, name=init.name))

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out_path)

    # 回读自检：新文件权重 == 原权重[:, ::-1]
    w2 = onh.to_array(next(i for i in onnx.load(out_path).graph.initializer
                           if i.name == w_name))
    assert w2.shape == w.shape and np.array_equal(w2, w[:, ::-1, :, :]), \
        "回读校验失败"
    assert np.array_equal(w2, w) is False or w.shape[1] == 1, "翻转未生效"
    nz = float(np.abs(w2 - w).max())
    print(f"[flip] saved -> {out_path}  (权重通道维已反转, "
          f"|w_flipped - w|max={nz:.4g}, 预期=|c2-c0|量级)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx_in", help="原 ONNX（不修改）")
    ap.add_argument("onnx_out", help="翻转后的新 ONNX（*_rgb.onnx）")
    args = ap.parse_args()
    flip(args.onnx_in, args.onnx_out)
