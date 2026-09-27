# -*- coding: utf-8 -*-
"""
Build a dynamic-batch TensorRT engine for the 1k3d68 68-point 3D landmark
model (models/pose68/1k3d68.onnx, input 192x192, output 3309).

- 该 onnx 输入 batch 维本来就是动态(dim_param, mxnet 导出), 输出 fc1 声明
  batch=1 —— TRT parser 会按输入 batch 自动提升输出 batch(现有
  1k3d68_dyn.engine 即由此 onnx + min=1/opt=16/max=32 profile 建成, 输出
  [-1,3309]), 因此无需 SCRFD 式的 reshape_onnx。
- 图算子集: Conv x54 / BatchNormalization x52 / Relu x51 / Add x16 /
  Flatten x2 / Identity x1 / MaxPool x1 / Gemm x1, 无 Range/Resize/常量 shape
  运算。BatchNormalization 与 Gemm 的输出 dtype 均为第一个输入的 dtype,
  已加入 build_head_engine._infer_types 的 passthrough 列表。
- profile 与现有 1k3d68_dyn.engine 完全一致: min=1 / opt=16 / max=32,
  空间维固定 192x192。
- 默认 FP32 -> models/pose68/1k3d68_dyn.engine(同 profile; 仓库现有默认
  引擎已存在, 勿用无 --fp16 模式覆盖它, 本脚本默认路径仅作等价重建用)。
- --fp16: 先复用 build_head_engine.convert_onnx_to_fp16() 把 ONNX 转成
  FP16 混合精度(图 IO 保持 fp32, 中间产物 models/pose68/1k3d68_dyn_fp16.onnx
  保留以便复现), 再按同一 profile 建引擎到
  models/pose68/1k3d68_dyn_fp16.engine(TRT 11 强类型网络: 精度由 ONNX 图
  张量 dtype 决定, 无全局 FP16 flag)。
- 混合精度细化保护: 全图 fp16 时末尾 fc1 Gemm 的 fp16 舍入会让 68 点
  landmark 产生 ~0.1px 级系统性偏差, EAR(小距离之比)被放大成
  ~+0.005-0.011 的偏高, 翻转 blink 边界帧。--fp16 默认把保护方案
  DEFAULT_FP32_OPS/DEFAULT_FP32_NODES 带上(重跑 --fp16 直接得到修复版
  引擎); --no-fp32-ops 关掉保护退化为全 fp16; --fp32-ops/--fp32-nodes
  显式覆盖(算子类型如 Gemm,逗号分隔; 或节点名如 fc1,逗号分隔)。
  --fp16-tag 给中间 onnx/引擎加命名后缀(如 pGemm -> 1k3d68_dyn_fp16_pGemm.*)。

Usage:
    python build_pose_engine.py [onnx_path] [engine_path] [--fp16]
        [--fp32-ops Gemm] [--fp32-nodes fc1] [--fp16-tag pGemm]
        [--no-fp32-ops]
"""
import argparse
import os

import tensorrt as trt

INPUT_SIZE = 192
MIN_BATCH = 1
OPT_BATCH = 16
MAX_BATCH = 32

# --fp16 的默认保护方案(混合精度细化, 由 100 张 EAR 对比迭代确定;
# 置空列表即默认全 fp16)。
# 根因(2026-09-27, 100 张 + 42 帧实测): TRT 里 fp16 Conv 的权重/激活舍入
# 经 54 层累积成 EAR 系统性偏高 ~+0.0038(max 0.011), 把 EAR 0.245-0.25
# 眯眼边界帧从 blink 翻成 keep; 只保护 fc1 Gemm 无效(+0.0039), 只保护
# BN 也无效(TRT +0.0027 且仍 1 翻转), 保护 stage4+head Conv 无效——
# 偏差来自前/中期大 FLOP Conv, 无法与速度兼得。最终: 保护全部主要算子
# (Conv/BN/Relu/Add/Gemm), 仅 MaxPool/Flatten/Identity/Cast 留 fp16
# (MaxPool/Flatten 无数值舍入)——精度回到 FP32 级(EAR mean +0.00002,
# 0 边界翻转), 速度 4.12ms/帧 ≈ FP32 4.09ms(本模型 fp16 无速度收益)。
DEFAULT_FP32_OPS = ["Conv", "BatchNormalization", "Relu", "Add", "Gemm"]
DEFAULT_FP32_NODES = None


def build(onnx_path, engine_path, fp16=False,
          fp32_ops=None, fp32_nodes=None, fp16_onnx=None):
    if fp16:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_head_engine import convert_onnx_to_fp16
        fp16_onnx = fp16_onnx or os.path.splitext(onnx_path)[0] + "_dyn_fp16.onnx"
        convert_onnx_to_fp16(onnx_path, fp16_onnx,
                             fp32_ops=fp32_ops, fp32_nodes=fp32_nodes)
        onnx_path = fp16_onnx

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

    print(f"[build] building {'FP16' if fp16 else 'FP32'} engine "
          f"(this can take a few minutes)...")
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
                         "build_head_engine.convert_onnx_to_fp16, 默认带 "
                         "DEFAULT_FP32_OPS/NODES 保护)再建引擎; "
                         "默认引擎名改为 1k3d68_dyn_fp16.engine")
    ap.add_argument("--fp32-ops", default=None,
                    help="逗号分隔的算子类型(如 Gemm 或 Gemm,BatchNormalization),"
                         "这些算子保留 fp32 计算; 覆盖默认保护方案")
    ap.add_argument("--fp32-nodes", default=None,
                    help="逗号分隔的节点名(如 fc1), 这些节点保留 fp32 计算; "
                         "与 --fp32-ops 取并集")
    ap.add_argument("--no-fp32-ops", action="store_true",
                    help="关闭默认保护方案, 建纯全 fp16 引擎")
    ap.add_argument("--fp16-tag", default=None,
                    help="给 FP16 中间 onnx/引擎名加后缀(如 pGemm -> "
                         "1k3d68_dyn_fp16_pGemm.onnx / .engine)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    onnx_path = args.onnx_path or os.path.join(root, "models", "pose68", "1k3d68.onnx")
    base_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    if args.fp16_tag:
        base_stem = base_stem + "_" + args.fp16_tag
    if args.engine_path:
        engine_path = args.engine_path
    elif args.fp16:
        engine_path = os.path.join(root, "models", "pose68",
                                   base_stem + "_dyn_fp16.engine")
    else:
        engine_path = os.path.join(root, "models", "pose68",
                                   base_stem + "_dyn.engine")

    fp32_ops = fp32_nodes = None
    if args.fp16:
        if args.no_fp32_ops:
            fp32_ops = fp32_nodes = None
        else:
            fp32_ops = list(DEFAULT_FP32_OPS or [])
            fp32_nodes = list(DEFAULT_FP32_NODES or [])
        if args.fp32_ops:
            fp32_ops = [s for s in args.fp32_ops.split(",") if s]
        if args.fp32_nodes:
            fp32_nodes = (fp32_nodes or []) + \
                [s for s in args.fp32_nodes.split(",") if s]

    fp16_onnx = (os.path.join(root, "models", "pose68",
                              base_stem + "_dyn_fp16.onnx") if args.fp16
                 else None)
    build(onnx_path, engine_path, fp16=args.fp16,
          fp32_ops=fp32_ops, fp32_nodes=fp32_nodes, fp16_onnx=fp16_onnx)
