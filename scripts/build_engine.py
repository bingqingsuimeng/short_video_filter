# -*- coding: utf-8 -*-
"""
Build a TensorRT engine for SCRFD-500m with 5-keypoint output.

- Reshapes the ONNX input to dynamic batch [-1, 3, SIZE, SIZE]
- Builds an FP16 engine with an optimization profile (min/opt/max batch)
- Saves the serialized engine to disk

Usage:
    python build_engine.py [onnx_path] [engine_path] [--fp16]

默认(无 --fp16): 行为与旧版逐字节一致 —— reshape + 建 FP32 引擎到
    models/scrfd/scrfd_500m_bnkps_batch32.engine
--fp16: reshape 后先复用 build_head_engine.convert_onnx_to_fp16() 把 ONNX 转成
    FP16 混合精度(图 IO 保持 fp32), 再按同一 profile 建引擎到
    models/scrfd/scrfd_500m_bnkps_batch32_fp16.engine
    (TRT 11 强类型网络: 精度由 ONNX 图张量 dtype 决定, 无全局 FP16 flag;
    中间产物 models/scrfd/scrfd_500m_bnkps_batch32_fp16.onnx 保留以便复现)
"""
import argparse
import os
import sys

import numpy as np
import onnx
import tensorrt as trt

INPUT_SIZE = 640
MIN_BATCH = 1
OPT_BATCH = 16
MAX_BATCH = 32


def reshape_onnx(src, dst, size=INPUT_SIZE):
    """Rewrite input dim to [-1, 3, size, size] (dynamic batch, fixed spatial)."""
    model = onnx.load(src)
    d = model.graph.input[0].type.tensor_type.shape.dim
    # dim 0 = batch -> dynamic
    d[0].dim_param = "batch"
    d[0].UnsetField("dim_value") if d[0].HasField("dim_value") else None
    # dims 2,3 = H,W -> fixed
    d[2].dim_value = size
    d[2].UnsetField("dim_param") if d[2].HasField("dim_param") else None
    d[3].dim_value = size
    d[3].UnsetField("dim_param") if d[3].HasField("dim_param") else None

    # BUGFIX: the head Transposes were exported with perm=[2,3,0,1]
    # (NCHW -> HWNC). Harmless at batch=1 (identical flatten order), but at
    # batch>1 it interleaves the batch dimension INSIDE the anchor axis
    # (flat order becomes [h,w,b,c] instead of [b,h,w,c]), so every
    # batch slot's scores mix multiple images. Fix perm to [0,2,3,1].
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name == "perm" and list(attr.ints) == [2, 3, 0, 1]:
                del attr.ints[:]
                attr.ints.extend([0, 2, 3, 1])
                fixed += 1
    print(f"[reshape] fixed {fixed} head Transpose perm [2,3,0,1] -> [0,2,3,1]")
    onnx.save(model, dst)
    print(f"[reshape] {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"input=[-1,3,{size},{size}]")
    return dst


def build(onnx_path, engine_path, fp16=True, fp16_converted=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())
    for err in range(parser.num_errors):
        print("  ONNX parse error:", parser.get_error(err))

    in_tensor = network.get_input(0)
    in_name = in_tensor.name
    print(f"[build] input '{in_name}' dims={list(in_tensor.shape)}")

    config = builder.create_builder_config()
    # 1 GiB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16_converted:
        # ONNX 图已是 FP16 混合精度(IO fp32): TRT 11 强类型网络精度取自图
        # dtype, 无需也无法再设 flag
        print("[build] ONNX already FP16 mixed precision -> precision from graph dtype")
    elif fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")
    else:
        # TRT 11 removed the global FP16 builder flag (precision is now
        # auto/per-layer). Build in FP32 — SCRFD-500m@640 is still fast.
        print("[build] FP16 flag unavailable -> building FP32 engine")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        in_name,
        (MIN_BATCH, 3, INPUT_SIZE, INPUT_SIZE),
        (OPT_BATCH, 3, INPUT_SIZE, INPUT_SIZE),
        (MAX_BATCH, 3, INPUT_SIZE, INPUT_SIZE),
    )
    config.add_optimization_profile(profile)

    print("[build] building engine (this can take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] saved engine -> {engine_path} "
          f"({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx_path", nargs="?", default=None)
    ap.add_argument("engine_path", nargs="?", default=None)
    ap.add_argument("--fp16", action="store_true",
                    help="先转 ONNX 为 FP16 混合精度(IO 保持 fp32, 复用 "
                         "build_head_engine.convert_onnx_to_fp16)再建引擎到 "
                         "scrfd_500m_bnkps_batch32_fp16.engine")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    onnx_path = args.onnx_path or (
        os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps.onnx"))
    if args.engine_path:
        engine_path = args.engine_path
    elif args.fp16:
        engine_path = os.path.join(root, "models", "scrfd",
                                   "scrfd_500m_bnkps_batch32_fp16.engine")
    else:
        engine_path = os.path.join(root, "models", "scrfd",
                                   "scrfd_500m_bnkps_batch32.engine")
    tmp_reshaped = engine_path + ".reshaped.onnx"
    reshape_onnx(onnx_path, tmp_reshaped)
    if args.fp16:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_head_engine import convert_onnx_to_fp16
        fp16_onnx = os.path.splitext(engine_path)[0] + ".onnx"
        convert_onnx_to_fp16(tmp_reshaped, fp16_onnx)
        build(fp16_onnx, engine_path, fp16=True, fp16_converted=True)
    else:
        build(tmp_reshaped, engine_path, fp16=True)
