# -*- coding: utf-8 -*-
"""
Build a TensorRT engine for resnet34_gaze.onnx (yakhyo MobileGaze / uniface).

- 输入固定 [1, 3, 448, 448]（batch=1，onnx 本身即固定 batch，无需 reshape）
- 输出：yaw [1, 90]、pitch [1, 90]（90 个角度 bin 的 logits）
- TRT 11 无全局 FP16 flag（自动/逐层精度），与 build_engine.py 的写法一致

Usage:
    python build_gaze_engine.py [onnx_path] [engine_path]
"""
import os
import sys

import tensorrt as trt


def build(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())
    for err in range(parser.num_errors):
        print("  ONNX parse error:", parser.get_error(err))

    in_names = [network.get_input(i).name for i in range(network.num_inputs)]
    out_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print(f"[build] inputs={in_names} outputs={out_names}")
    for i in range(network.num_outputs):
        print(f"[build]   out[{i}] '{out_names[i]}' dims={list(network.get_output(i).shape)}")

    config = builder.create_builder_config()
    # 1 GiB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")
    else:
        print("[build] FP16 flag unavailable (TRT11) -> 自动精度")

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
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "models", "gaze", "resnet34_gaze.onnx"),
          sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "models", "gaze", "resnet34_gaze.engine"))
