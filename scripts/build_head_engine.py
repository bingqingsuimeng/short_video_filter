# -*- coding: utf-8 -*-
"""
Build a dynamic-batch TensorRT engine for the head detector (models/head/model.onnx).

- head2 是单类 head 的 YOLO11l ONNX, 输入 640x640, 全部 Transpose batch 安全,
  无需 build_engine.py 里 SCRFD 专用的 Transpose perm 修复 / onnx reshape
- 动态 batch profile: min=1 / opt=16 / max=32, 空间维度固定 640x640
- TRT 11 无全局 FP16 flag(自动/逐层精度), 与 build_engine.py 写法一致

Usage:
    python build_head_engine.py [onnx_path] [engine_path]

默认: models/head/model.onnx -> models/head/model_dyn.engine
"""
import os
import sys

import tensorrt as trt

INPUT_SIZE = 640
MIN_BATCH = 1
OPT_BATCH = 16
MAX_BATCH = 32


def build(onnx_path, engine_path):
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
    out_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print(f"[build] outputs={out_names}")

    config = builder.create_builder_config()
    # 1 GiB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

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
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "models", "head", "model.onnx")
    engine_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "models", "head", "model_dyn.engine")
    build(onnx_path, engine_path)
