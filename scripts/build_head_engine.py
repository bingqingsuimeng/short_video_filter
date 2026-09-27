# -*- coding: utf-8 -*-
"""
Build a dynamic-batch TensorRT engine for the head detector (models/head/model.onnx).

- head2 是单类 head 的 YOLO11l ONNX, 输入 640x640, 全部 Transpose batch 安全,
  无需 build_engine.py 里 SCRFD 专用的 Transpose perm 修复 / onnx reshape
- 动态 batch profile: min=1 / opt=16 / max=32, 空间维度固定 640x640
- 默认 FP32(onnx 全 float32)。TRT 11 移除了全局 FP16/INT8 builder flag
  (强类型网络, 精度取自 ONNX 图 dtype; 官方迁移指南推荐先用 ModelOpt AutoCast
  把 ONNX 转混合精度再 build, 见 docs.nvidia.com .../tensorrt-10x-to-11x-python-api-patterns.html)
  --fp16: 先把 ONNX 转成 FP16 混合精度(图 IO 保持 fp32, 与 HeadGate 现有接口一致),
  再按同一 profile 建引擎。优先用 modelopt.onnx.autocast(若已安装),
  否则用内置 onnx cast(等价语义: 首尾插 Cast, 全部 float 权重/常量转 float16)。

Usage:
    python build_head_engine.py [onnx_path] [engine_path] [--fp16]

默认: models/head/model.onnx -> models/head/model_dyn.engine
--fp16: models/head/model.onnx -> models/head/model_dyn_fp16.engine
        (转换中间产物 models/head/model_fp16.onnx, 保留以便复现)
"""
import argparse
import os

import numpy as np
import onnx
import onnx.checker
import onnx.helper as oh
import onnx.numpy_helper as onh
import tensorrt as trt

INPUT_SIZE = 640
MIN_BATCH = 1
OPT_BATCH = 16
MAX_BATCH = 32


_T = onnx.TensorProto
_INT = {_T.INT32, _T.INT64}


def _target_input_type(op, k):
    """节点第 k 个浮点输入的目标精度:
    Range(opset<=16 无 fp16 实现)全部输入、Resize 的 scales(2)/sizes(3)
    按 ONNX 规格必须 fp32, 其余一律 fp16。"""
    if op == "Range":
        return _T.FLOAT
    if op == "Resize" and k in (2, 3):
        return _T.FLOAT
    return _T.FLOAT16


def _fp32_slot_tensors(g):
    """必须保持 fp32 的输入槽对应的张量集合(权重转换步骤据此跳过)。"""
    tensors = set()
    for n in g.node:
        if n.op_type == "Range":
            tensors.update(n.input)
        elif n.op_type == "Resize":
            for k in (2, 3):
                if k < len(n.input) and n.input[k]:
                    tensors.add(n.input[k])
    return tensors


def _infer_types(g):
    """本模型算子集上的前向类型推断, 返回 {tensor_name: elem_type}。"""
    types = {}
    for t in g.input:
        types[t.name] = t.type.tensor_type.elem_type
    for init in g.initializer:
        types[init.name] = init.data_type
    for n in g.node:
        op = n.op_type
        if op == "Shape":
            for o in n.output:
                types[o] = _T.INT64
        elif op == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    for o in n.output:
                        types[o] = a.t.data_type
                    break
        elif op == "ConstantOfShape":
            v = {a.name: a for a in n.attribute}.get("value")
            dt = v.t.data_type if (v is not None and v.HasField("t")) else _T.FLOAT
            for o in n.output:
                types[o] = dt
        elif op == "Cast":
            to = {a.name: a.i for a in n.attribute}.get("to", _T.FLOAT)
            for o in n.output:
                types[o] = to
        elif op == "Range":
            if n.input:
                types[n.output[0]] = types.get(n.input[0], _T.FLOAT)
        elif op in ("Conv", "MatMul", "Split", "Gather", "Resize", "Concat",
                    "Expand", "Slice", "MaxPool", "Reshape", "Unsqueeze",
                    "Transpose", "Sigmoid", "Softmax", "Add", "Mul", "Sub",
                    "Div", "Flatten", "Identity"):
            # 输出类型 = 主输入(第一个)类型; 全 int 输入(如 int64 Gather/
            # Reshape shape 运算)不得回退成 float
            base = types.get(n.input[0], _T.FLOAT) if n.input else _T.FLOAT
            for o in n.output:
                types[o] = base
        else:
            print(f"[fp16] 警告: 未知算子 {op}({n.name}) 类型未推断")
    return types


def convert_onnx_to_fp16(onnx_path, out_path):
    """FP32 ONNX -> FP16 混合精度 ONNX。

    语义等价于 modelopt.onnx.autocast.convert_to_mixed_precision(
        onnx_path, low_precision_type="fp16", keep_io_types=True):
    图输入/输出保持 float32; 全部 float 权重/常量转 float16, 网络内部按
    类型驱动统一为 fp16 —— 对每个节点, 浮点输入类型与节点目标精度
    (Range 必须 fp32, 其余 fp16)不一致的边自动插 Cast。
    返回输出路径。
    """
    # 首选官方 ModelOpt AutoCast(若环境装了 nvidia-modelopt)
    try:
        import modelopt.onnx.autocast as autocast
        converted = autocast.convert_to_mixed_precision(
            onnx_path, low_precision_type="fp16", keep_io_types=True)
        onnx.save(converted, out_path)
        print(f"[fp16] 转换完成(modelopt.onnx.autocast) -> {out_path}")
        return out_path
    except ImportError:
        pass

    # 内置 onnx cast: 与 autocast(keep_io_types=True) 同语义
    model = onnx.load(onnx_path)
    g = model.graph
    keep_f32 = _fp32_slot_tensors(g)

    # 1) 全部 float 权重 / Constant 常量 -> fp16(规格要求 fp32 的槽位
    #    对应张量跳过, 如 Resize scales)
    n16 = 0
    for init in g.initializer:
        if init.data_type == _T.FLOAT and init.name not in keep_f32:
            arr = onh.to_array(init).astype(np.float16)
            init.CopyFrom(onh.from_array(arr, name=init.name))
            n16 += 1
    for node in g.node:
        if node.op_type == "Constant" and node.output[0] not in keep_f32:
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == _T.FLOAT:
                    arr = onh.to_array(attr.t).astype(np.float16)
                    attr.t.CopyFrom(onh.from_array(arr, name=attr.t.name))

    # 2) 图输入后插 fp32->fp16 Cast 并重接下游(图 IO 保持 fp32)
    for inp in list(g.input):
        in16 = inp.name + "_fp16"
        cast_name = f"cast_{inp.name}_to_fp16"
        g.node.insert(0, oh.make_node("Cast", [inp.name], [in16],
                                      to=_T.FLOAT16, name=cast_name))
        for node in g.node:
            if node.name == cast_name:
                continue  # 新 Cast 本身输入就是原图输入, 不参与重接
            for k, s in enumerate(node.input):
                if s == inp.name:
                    node.input[k] = in16

    # 3) 类型驱动定点迭代: 浮点输入与节点目标精度不符的边插 Cast。
    #    多轮是因为 Cast 会改变上游输出类型(如 Range 输入被拉成 fp32 后
    #    其输出才成为 fp32), 一轮推断不够
    n_edge = 0
    for _ in range(8):
        types = _infer_types(g)
        inserted = 0
        for n in list(g.node):
            if n.op_type == "Cast":
                continue
            for k, s in enumerate(n.input):
                target = _target_input_type(n.op_type, k)
                t = types.get(s)
                if t in (_T.FLOAT, _T.FLOAT16) and t != target:
                    c = f"__cast{n_edge}_{s}"
                    idx = next(i for i, nn in enumerate(g.node)
                               if nn.name == n.name)
                    g.node.insert(idx, oh.make_node(
                        "Cast", [s], [c], to=target, name=c))
                    n.input[k] = c
                    n_edge += 1
                    inserted += 1
        if inserted == 0:
            break

    # 4) 图输出为 fp16 时末尾插 fp16->fp32 Cast
    types = _infer_types(g)
    producers = {}
    for n in g.node:
        for o in n.output:
            producers[o] = n.name
    for out in list(g.output):
        if types.get(out.name) == _T.FLOAT16:
            out16 = out.name + "_fp16"
            for node in g.node:
                for k, s in enumerate(node.output):
                    if s == out.name:
                        node.output[k] = out16
            g.node.append(oh.make_node("Cast", [out16], [out.name],
                                       to=_T.FLOAT,
                                       name=f"cast_{out.name}_to_fp32"))

    # 5) 清掉原图 value_info(旧 fp32 类型标注, 会与新连线冲突), 终检
    del g.value_info[:]
    types2 = _infer_types(g)
    bad = []
    for n in g.node:
        if n.op_type == "Range" and types2.get(n.output[0]) != _T.FLOAT:
            bad.append(f"Range {n.name} 非 fp32")
        if n.op_type == "Cast":
            continue
        fl = [types2.get(s) for s in n.input
              if types2.get(s) in (_T.FLOAT, _T.FLOAT16)]
        # shape/index 辅助输入不计入(Reshape/Expand/Slice/Gather/Resize
        # 的第二类输入, 本模型里甚至是 fp32 常量拼接)
        if n.op_type in ("Reshape", "Expand", "Slice", "Gather", "Resize"):
            fl = fl[:1]
        if len(set(fl)) > 1:
            bad.append(f"{n.name}({n.op_type}) 混合精度输入 {set(fl)}")
    if bad:
        raise RuntimeError("FP16 转换自检失败:\n" + "\n".join(bad[:20]))
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out_path)

    n_t16 = sum(1 for v in types2.values() if v == _T.FLOAT16)
    n_t32 = sum(1 for v in types2.values() if v == _T.FLOAT)
    n_io_f32 = sum(1 for t in list(g.input) + list(g.output)
                   if t.type.tensor_type.elem_type == _T.FLOAT)
    print(f"[fp16] 转换完成(内置 onnx cast) -> {out_path}")
    print(f"[fp16]   float16 权重={n16}, 边界/精度 Cast={n_edge} 个")
    print(f"[fp16]   张量类型: fp16={n_t16}, fp32={n_t32}, "
          f"图 IO 保持 float32={n_io_f32}/{len(g.input)+len(g.output)}")
    return out_path


def build(onnx_path, engine_path, fp16=False):
    if fp16:
        fp16_onnx = os.path.splitext(onnx_path)[0] + "_fp16.onnx"
        convert_onnx_to_fp16(onnx_path, fp16_onnx)
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
                    help="先把 ONNX 转 FP16 混合精度(IO 保持 fp32)再建引擎; "
                         "默认引擎名改为 model_dyn_fp16.engine")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
    onnx_path = args.onnx_path or os.path.join(root, "models", "head", "model.onnx")
    if args.engine_path:
        engine_path = args.engine_path
    elif args.fp16:
        engine_path = os.path.join(root, "models", "head", "model_dyn_fp16.engine")
    else:
        engine_path = os.path.join(root, "models", "head", "model_dyn.engine")
    build(onnx_path, engine_path, fp16=args.fp16)
