# -*- coding: utf-8 -*-
"""
Build a TensorRT engine for SCRFD-500m with 5-keypoint output.

- Reshapes the ONNX input to dynamic batch [-1, 3, SIZE, SIZE]
- Builds an FP16 engine with an optimization profile (min/opt/max batch)
- Saves the serialized engine to disk

Usage:
    python build_engine.py [onnx_path] [engine_path]
"""
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


def build(onnx_path, engine_path, fp16=True):
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
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else (
        os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps.onnx"))
    engine_path = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.join(root, "models", "scrfd", "scrfd_500m_bnkps_batch32.engine"))
    tmp_reshaped = engine_path + ".reshaped.onnx"
    reshape_onnx(onnx_path, tmp_reshaped)
    build(tmp_reshaped, engine_path, fp16=True)
