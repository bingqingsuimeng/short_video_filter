# -*- coding: utf-8 -*-
"""
Build a TensorRT engine for resnet34_gaze.onnx (yakhyo MobileGaze / uniface).

- 输入固定 [1, 3, 448, 448]（batch=1，onnx 本身即固定 batch，无需 reshape）
- 输出：yaw [1, 90]、pitch [1, 90]（90 个角度 bin 的 logits）
- TRT 11 无全局 FP16 flag（自动/逐层精度），与 build_engine.py 的写法一致

Usage:
    python build_gaze_engine.py [onnx_path] [engine_path]
    python build_gaze_engine.py --fp16
    python build_gaze_engine.py --dyn        # 动态 batch(批量化 gaze 闸门用)

默认: models/gaze/resnet34_gaze.onnx -> models/gaze/resnet34_gaze.engine(行为不变)
--fp16: 先把 ONNX 转 FP16 混合精度(图 IO 保持 fp32, 复用
        build_head_engine.convert_onnx_to_fp16)再建
        models/gaze/resnet34_gaze_fp16.engine
        (中间产物 models/gaze/resnet34_gaze_fp16.onnx 保留以便复现)
--dyn: 先 reshape 输入 batch 维为动态(图结构 batch 安全: GAP->Flatten->Gemm,
        无 Reshape/Transpose), 复用已转好的 resnet34_gaze_fp16.onnx 按同一
        profile(min=1/opt=16/max=32)建
        models/gaze/resnet34_gaze_dyn_fp16.engine
        (中间产物 models/gaze/resnet34_gaze_dyn_fp16.onnx 保留以便复现;
         与 build_engine.py 的 SCRFD 动态 batch 同法)
"""
import argparse
import os
import sys

import tensorrt as trt

MIN_BATCH = 1
OPT_BATCH = 16
MAX_BATCH = 32


def reshape_dynamic(src, dst):
    """输入 batch 维 1 -> 动态 'batch'(空间维保持 448)。本图 GAP->Flatten->Gemm
    对 batch 天然安全(无硬编码 batch 的 Reshape/Transpose, 见 build_engine.py
    对 SCRFD 的 Transpose perm 修复 —— 本图 Transpose 数=0)。"""
    import onnx
    model = onnx.load(src)
    d = model.graph.input[0].type.tensor_type.shape.dim
    assert d[0].dim_value == 1, f"期望固定 batch=1, 实际 {d[0].dim_value}"
    d[0].dim_param = "batch"
    d[0].ClearField("dim_value")
    onnx.save(model, dst)
    print(f"[reshape] {src} -> {dst}  input=[batch,3,448,448]")
    return dst


def build(onnx_path, engine_path, fp16=True, profile=False):
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

    if profile:
        in_tensor = network.get_input(0)
        p = builder.create_optimization_profile()
        p.set_shape(in_tensor.name,
                    (MIN_BATCH, 3, 448, 448),
                    (OPT_BATCH, 3, 448, 448),
                    (MAX_BATCH, 3, 448, 448))
        config.add_optimization_profile(p)
        print(f"[build] profile batch min={MIN_BATCH} opt={OPT_BATCH} "
              f"max={MAX_BATCH}")

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
                    help="先把 ONNX 转 FP16 混合精度(IO 保持 fp32, 复用 "
                         "build_head_engine.convert_onnx_to_fp16)再建引擎; "
                         "默认引擎名改为 resnet34_gaze_fp16.engine")
    ap.add_argument("--dyn", action="store_true",
                    help="动态 batch 引擎: reshape 输入 batch 维 + 优化 profile "
                         "min=1/opt=16/max=32, 默认引擎名 "
                         "resnet34_gaze_dyn_fp16.engine(复用已转好的 fp16 onnx)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    onnx_path = args.onnx_path or os.path.join(root, "models", "gaze", "resnet34_gaze.onnx")
    if args.engine_path:
        engine_path = args.engine_path
    elif args.dyn:
        engine_path = os.path.join(root, "models", "gaze",
                                   "resnet34_gaze_dyn_fp16.engine")
    elif args.fp16:
        engine_path = os.path.join(root, "models", "gaze", "resnet34_gaze_fp16.engine")
    else:
        engine_path = os.path.join(root, "models", "gaze", "resnet34_gaze.engine")

    if args.dyn:
        # 复用 fp16 混合精度图(不存在则先转), reshape batch 维 -> profile build
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_head_engine import convert_onnx_to_fp16
        fp16_onnx = os.path.splitext(onnx_path)[0] + "_fp16.onnx"
        if not os.path.exists(fp16_onnx):
            convert_onnx_to_fp16(onnx_path, fp16_onnx)
        dyn_onnx = os.path.splitext(engine_path)[0] + ".onnx"
        reshape_dynamic(fp16_onnx, dyn_onnx)
        build(dyn_onnx, engine_path, fp16=False, profile=True)
    elif args.fp16:
        # 与 build_pose_engine.py --fp16 同路径: 转图 -> 普通 build,
        # 精度由 ONNX 图 dtype 决定(TRT 11 无全局 flag)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_head_engine import convert_onnx_to_fp16
        fp16_onnx = os.path.splitext(onnx_path)[0] + "_fp16.onnx"
        convert_onnx_to_fp16(onnx_path, fp16_onnx)
        build(fp16_onnx, engine_path)
    else:
        # 无 flag: 与原脚本逐字节一致的 FP32 构建
        build(onnx_path, engine_path)
