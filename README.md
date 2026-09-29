# 短视频人脸过滤器(short_video_filter)

从视频里挑出 **「单人正面、睁眼看镜头、无人头干扰」** 的帧,输出**原分辨率 JPG**(不重编码视频),并附逐帧姿态报告 `pose_report.csv`。

> **本仓库只含代码与配置,不含任何模型文件(ONNX / TensorRT 引擎全部留本地,git 已忽略)。** 模型获取方式见下文「模型获取与引擎构建」。

## 管线

```
ffmpeg 解码(原分辨率)
  → SCRFD-500m 批量人脸检测(TensorRT, 动态 batch 1/16/32, 输入 640×640, conf 默认 0.8)
  → 68 点姿态(1k3d68 TRT 引擎: yaw / pitch / roll / EAR;缺引擎时回退 SCRFD 5 点 solvePnP)
  → 视线闸门(resnet34 Gaze TRT, 448×448 batch1; --gaze-dy-dev 纵向 2-pass 基线闸门)
  → Stage1.5 人头闸门(head2 人头检测 TRT, NMS 后 head 数 ≥2 剔除)
  → 输出原分辨率 JPG(quality 默认 90) + pose_report.csv
```

各闸门默认阈值(均可命令行覆盖):

| 闸门 | 参数 | 默认 | 含义 |
|------|------|------|------|
| 检测 | `--conf` | 0.8 | SCRFD 人脸置信度下限 |
| 姿态 | `--yaw` | 12.0 | \|yaw\| 上限(度),超=侧脸剔除 |
| 姿态 | `--pitch-max` | 25.0 | pitch 上限(度),超=仰头不看剔除(0=关) |
| 姿态 | `--pitch-min` | -25.0 | pitch 下限(度),低于=低头不看剔除(0=关) |
| 姿态 | `--down-min` | 0.46 | 低头比下限,低于=低头剔除(0=关) |
| 姿态 | `--ear-min` | 0.25 | min-EAR 下限,低于=闭眼/眯眼剔除(0=关) |
| 视线 | `--gaze-pitch-max` | 15.0 | resnet34 纵向注视角上限(度) |
| 视线 | `--gaze-yaw-max` | 20.0 | resnet34 横向注视角上限(度) |
| 视线 | `--gaze-dy-dev` | 11.0 | 纵向眼神 2-pass:基线=本视频所有测到 gaze 帧的 gaze_dy 中位数,keep 帧 \|dy-基线\|>此值剔除(高/低机位整体漂移注视角,靠它兜底;0=关) |
| 人头 | `--head-conf` | 0.30 | 人头检测置信度阈值 |

`--gaze-model` 默认 `resnet34`(A/B 验证优于 `iris`);iris 模式用 MediaPipe 虹膜偏移,阈值 `--gaze-max`(默认 0.12,0=关)。

## 环境

- **Python 3.10**(conda env `face_cleaner`)
- **CUDA 13.1 + TensorRT 11.3**(本机路径已持久化到 Windows 用户环境,见 `setup_cuda_trt.ps1` / `verify_cuda_trt.ps1`;任何新终端/IDE 启动后自动继承,无需再导出)
- 依赖安装(清华镜像):

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意:`requirements.txt` 只含运行时 pip 依赖;`tensorrt` / `pycuda` / `onnx` 由本机 CUDA+TRT 环境提供(conda env 内),不走 pip。

## 模型获取与引擎构建

**本仓库不含任何模型文件。** 以下 4 个模型需自备 ONNX 并放在指定路径,再用根目录的构建脚本生成 TensorRT 引擎:

| 模型 | ONNX 文件(放 `models/` 下) | 大小 | 来源 | 对应 engine | 重建命令 |
|------|------|------|------|------|------|
| SCRFD-500m(人脸检测 + 5 关键点) | `models/scrfd/scrfd_500m_bnkps.onnx` | ~2.4 MB | **公开**,insightface SCRFD | 默认 `models/scrfd/scrfd_500m_bnkps_batch32_fp16.engine`(FP16,动态 batch 1/16/32,640×640,IO fp32);缺失时自动回退 FP32 `models/scrfd/scrfd_500m_bnkps_batch32.engine` | `python scripts/build_engine.py --fp16`(建 FP16 默认引擎);`python scripts/build_engine.py`(建 FP32,也可显式传 `onnx engine`) |
| 1k3d68(68 点 3D 姿态) | `models/pose68/1k3d68.onnx` | ~137 MB | **公开**(1k3d68 项目) | 默认 `models/pose68/1k3d68_dyn.engine`(FP32,动态 batch 1/16/32,192×192);另有 `models/pose68/1k3d68_dyn_fp16.engine`(修复版 FP16:Conv/BN/Relu/Add/Gemm 保 FP32 的混合精度,EAR 精度回到 FP32 级但**无速度收益**,见「FP16 混合精度构建」;`--pose-engine` 显式启用,默认仍 FP32)。旧全 fp16 版(有 EAR 偏移)留档 `1k3d68_dyn_fp16_v1.engine` | `python scripts/build_pose_engine.py --fp16`(带默认保护方案,建修复版 FP16 引擎 + 中间 onnx);`--no-fp32-ops` 建旧式全 fp16;`--fp32-ops/--fp32-nodes/--fp16-tag` 自定义保护与命名;`python scripts/build_pose_engine.py`(同 profile 重建 FP32) |
| Gaze ResNet34(视线 pitch/yaw,448×448) | `models/gaze/resnet34_gaze.onnx` | ~81 MB | **自训**(yakhyo MobileGaze / uniface 类训练) | 默认 `models/gaze/resnet34_gaze_fp16.engine`(FP16,batch1,448×448,IO fp32);缺失时自动回退 FP32 `models/gaze/resnet34_gaze.engine` | `python scripts/build_gaze_engine.py --fp16`(建 FP16 默认引擎 + 中间 onnx);`python scripts/build_gaze_engine.py`(建 FP32) |
| Head YOLO11l(人头检测,单类 head,640×640) | `models/head/model.onnx` | ~101 MB | **自训**(real_head_yolo) | 默认 `models/head/model_dyn_fp16.engine`(FP16,动态 batch 1/16/32,IO fp32);缺失时自动回退 FP32 `models/head/model_dyn.engine`,再回退 batch1 固定版 `models/head/model.engine` | `python scripts/build_head_engine.py --fp16`(建 FP16 默认引擎);`python scripts/build_head_engine.py`(默认建 FP32 `model_dyn.engine`) |

> **重要:gaze(`models/gaze/resnet34_gaze.onnx`)与 head(`models/head/model.onnx`)两个 ONNX 是自训模型,不在任何公开源,克隆本仓库后需自行放置到上述路径。**

构建脚本说明(在 `scripts/` 下,均 `python scripts/<脚本> [onnx] [engine]`,不带参数时用仓库根 `models/` 下的默认路径):

- `scripts/build_engine.py` — SCRFD 专用:先把 ONNX 输入改写成动态 batch `[-1,3,640,640]`,并修复导出时的 head Transpose perm 问题(SCRFD ONNX 专有的 bug,其它模型不需要),再按 min=1/opt=16/max=32 profile 建引擎。加 `--fp16` 则在 reshape 后先转 FP16 混合精度(见下文「FP16 混合精度构建」)再建 `scrfd_500m_bnkps_batch32_fp16.engine`。
- `scripts/build_gaze_engine.py` — resnet34 onnx 本身固定 batch=1,直接建即可。默认建 FP32 `resnet34_gaze.engine`;加 `--fp16` 先把 ONNX 转 FP16 混合精度(图 IO 保持 fp32,复用 `build_head_engine.convert_onnx_to_fp16`,中间产物 `models/gaze/resnet34_gaze_fp16.onnx` 保留)再建 `models/gaze/resnet34_gaze_fp16.engine`。
- `scripts/build_head_engine.py` — head2 onnx 的 Transpose 全部 batch 安全,无需 reshape,直接按 min=1/opt=16/max=32 profile 建动态 batch 引擎。默认建 FP32 `model_dyn.engine`;加 `--fp16` 先把 ONNX 转 FP16 混合精度(图 IO 保持 fp32,中间产物 `models/head/model_fp16.onnx` 保留)再建 `models/head/model_dyn_fp16.engine`。
- `scripts/build_pose_engine.py` — 1k3d68 onnx 输入 batch 维本身动态(输出 batch 由 TRT parser 按输入提升),无需 reshape;按 min=1/opt=16/max=32 profile(与现有 `1k3d68_dyn.engine` 一致)建引擎。默认建 FP32 `1k3d68_dyn.engine`;加 `--fp16` 先把 ONNX 转 FP16 混合精度(图 IO 保持 fp32,中间产物 `models/pose68/1k3d68_dyn_fp16.onnx` 保留)再建 `models/pose68/1k3d68_dyn_fp16.engine`。**`--fp16` 默认带算子级 FP32 保护方案**(`DEFAULT_FP32_OPS = Conv/BN/Relu/Add/Gemm`,修复 fp16 累积舍入导致的 EAR 系统性偏移,见下);`--no-fp32-ops` 关掉保护(建旧式全 fp16),`--fp32-ops Gemm` / `--fp32-nodes fc1` 显式覆盖(算子类型或节点名,逗号分隔),`--fp16-tag pGemm` 给中间 onnx/引擎加命名后缀。保护转换实现在 `build_head_engine.convert_onnx_to_fp16(fp32_ops=..., fp32_nodes=...)`——被保护节点的输入 Cast→fp32 进节点、权重/常量保持 fp32、输出 fp32(下游 fp16 节点自动插 Cast 回 fp16),全图 fp16 骨架与 IO fp32 不变;其它模型的 `--fp16` 不受影响(保护参数默认空)。
- 这些脚本都是 TRT 11 风格:`Logger(WARNING)` + `set_memory_pool_limit(WORKSPACE, 1<<30)`,无全局 FP16 flag(自动/逐层精度)。

可选:`preproc_kernel.cubin` 是 `preproc_kernel.cu`(SCRFD letterbox 预处理 CUDA 核)的编译产物,用 `nvcc -cubin preproc_kernel.cu` 可重编;**缺失时 `face_det` 自动回退 CPU 预处理**,不影响正确性。

## FP16 混合精度构建(TRT 11)

head2 与 SCRFD 已实测验证的 FP16 流程与踩坑(均为 RTX 3060)。

### 为什么不能像 TRT 10 那样加 flag

- TRT 11 **移除了全局 `BuilderFlag.FP16` / `INT8`**。旧代码里的
  `if fp16 and hasattr(trt.BuilderFlag, "FP16")` 检查在 TRT 11 下恒为 False,
  **静默降级成 FP32** —— 本项目旧 `build_engine.py` 曾因此建出过「伪 FP16」
  引擎(流程/命名看着是 FP16,实际全 FP32)。
- 逐层 `layer.precision = trt.float16` 在 11.x 也不可用(`ILayer` 没有
  `precision` 属性,精度由网络结构/图 dtype 决定)。

### 正确方法:把 ONNX 图本身转成混合精度

TRT 11 默认**强类型网络**,每层精度**由 ONNX 图张量 dtype 决定**。因此:

1. 先把 FP32 ONNX 转成 FP16 混合精度 ONNX(图输入/输出保持 fp32,内部全 fp16);
2. 再走普通 build 流程建引擎即可(无需任何 flag)。

转换实现(按优先级):

- 官方推荐 **ModelOpt AutoCast**(若装了 `nvidia-modelopt`):
  `modelopt.onnx.autocast.convert_to_mixed_precision(onnx, low_precision_type="fp16", keep_io_types=True)`;
- 无 ModelOpt 时用 `scripts/build_head_engine.py` 内置的 `convert_onnx_to_fp16()`
  (本机环境走此路径)—— 与 autocast(`keep_io_types=True`)等价:全部浮点
  权重/常量转 fp16、图输入后插 `Cast→fp16`、图输出前插 `Cast→fp32`
  (下游接口零改动)、Range / Resize 辅助槽保 fp32、按算子目标精度自动补边界
  Cast,末尾跑 `onnx.checker` + 混合精度自检。

各构建脚本的 `--fp16` 用法:

| 模型 | 命令 | FP16 引擎(中间 ONNX) |
|------|------|------|
| SCRFD-500m | `python scripts/build_engine.py --fp16` | `models/scrfd/scrfd_500m_bnkps_batch32_fp16.engine`(`scrfd_500m_bnkps_batch32_fp16.onnx`) |
| Head YOLO11l | `python scripts/build_head_engine.py --fp16` | `models/head/model_dyn_fp16.engine`(`models/head/model_fp16.onnx`) |
| 1k3d68(68 点姿态) | `python scripts/build_pose_engine.py --fp16`(默认带 Conv/BN/Relu/Add/Gemm 的 FP32 保护,修复 EAR 偏移;`--no-fp32-ops` 建旧式全 fp16) | `models/pose68/1k3d68_dyn_fp16.engine`(`models/pose68/1k3d68_dyn_fp16.onnx`;全 fp16 版留档 `*_v1.engine`) |
| Gaze ResNet34 | `python scripts/build_gaze_engine.py --fp16` | `models/gaze/resnet34_gaze_fp16.engine`(`models/gaze/resnet34_gaze_fp16.onnx`) |

不带 `--fp16` 时行为与 FP32 构建完全一致,默认引擎不变。

### 转换坑清单

- **图 IO 必须保 fp32**:输入后插 `Cast→fp16`、输出前插 `Cast→fp32`,
  推理缓冲/下游代码接口零改动。
- **Range 的全部输入、Resize 的 `scales`(输入 2)/ `sizes`(输入 3)必须 fp32**
  (ONNX 规格要求,`onnx.checker` 会拒)。
- **全 int 输入的 Gather 别误转 float**:类型推断按「输出类型 = 第一个输入
  类型」passthrough,shape/index 运算不能回退成 float。
- **原图旧 `value_info` 要清掉**:残留的 fp32 类型标注会与新连线冲突,
  ORT/TRT 按旧类型校验会报错。
- **动态 batch 模型先修结构再转 fp16**:SCRFD 必须先把输入 reshape 成
  `[-1,3,640,640]` 并修 head Transpose perm(`[2,3,0,1]→[0,2,3,1]`,
  batch>1 时分数串扰的 bug),再转 fp16。
- **类型推断的算子 passthrough 列表要覆盖目标模型的算子集**:head 图没有
  Relu,SCRFD 有 41 个 —— 缺漏会让 dtype 统计/自检失真(已在
  `build_head_engine.py` 的 `_infer_types` 补上 `Relu`)。

### 验证方法学(TRT 11.3 Python 读不出逐层精度!)

Python API 无法从序列化引擎读回逐层精度,靠**三重佐证**:

1. **ONNX 图 dtype 统计**:fp16 张量占绝对多数(head 98.2%、SCRFD 95.8%;
   剩余 fp32 均为图 IO / Range / Resize 辅助槽);
2. **引擎体积约减半**:head 110→55MB,SCRFD 3.13→1.84MB;
3. **输出对比**(真正判据)。

**正确性对比协议**:从 `E:\output\buding\kept_all` 固定 seed 抽 100 张
(覆盖判定边界档),逐张对比 FP16 vs FP32 的 **判定一致率**(最重要:
检出数 / verdict)、框数 / 坐标差(px) / conf 差,不一致张逐张列出并肉眼核
(到底几张脸、谁对)。测试脚本在 `_test/`(如 `scrfd_fp16_compare100.py`),
产物在 `E:\output\_fp16_svf\`。

### 实测收益(RTX 3060)

| 模型 | batch1 | batch16 | 引擎体积 | 正确性 |
|------|--------|---------|----------|--------|
| Head YOLO11l | 2.17×(15.65→7.21ms) | 2.79×(10.70→3.83ms) | 110→55MB | 100 张分层抽样判定 100% 一致 |
| SCRFD-500m | 1.26×(0.99→0.78ms) | 1.91×(2.38→1.25ms) | 3.13→1.84MB | 100 张检出数 100% 一致(框差 p50/p95/max = 0.15/0.34/0.84px,score 最大差 0.0013);端到端 42 帧短视频 verdict / nfaces 100% 一致 |
| 1k3d68(68 点,**修复版**) | —(无收益,不采用) | 1.00×(4.09→4.11ms) | 155→155MB | 100 张 pose 相关 verdict 100/100 一致(68 点差 p50/max = 0.006/0.30px,角度差 max 0.034°,EAR 带符号差 mean +0.00002 / max 0.0005,0 边界翻转);端到端 42 帧 verdict 42/42 一致,kept=9 与 FP32 完全相同 |
| Gaze ResNet34(视线) | 3.51×(3.97→1.13ms) | —(batch1 模型) | 147→45MB | 100 张 Stage2 gaze 判定 100/100 一致(pitch 差 mean -0.01°/p95 0.09°/max 0.17°,yaw 差 mean +0.003°/max 0.28°,**无系统性偏移**;7 张 |pitch|14~16° 边界帧无翻转);端到端 59 帧 verdict 59/59 一致(含 gazedown/gaze 帧) |

SCRFD 是小模型,纯推理偏内存带宽瓶颈,故加速比低于 head(计算更重)。

**1k3d68 的 FP16 陷阱(EAR 系统性偏移的根因与修复,2026-09-27 排查)**:
初版全 fp16 引擎(3.00× 提速,已留档 `1k3d68_dyn_fp16_v1.engine`)实测
EAR 系统性偏高 mean +0.0038 / max 0.011——EAR 是 68 点中眼睛区几个
landmark 的**小像素距离之比**,landmark 的亚像素方向性偏差会被放大,
把 EAR 0.245~0.25 的眯眼边界帧从 blink 翻成 keep(100 张 1 张、42 帧
视频 4 张)。算子级归因(TRT 逐方案实测):根因是 **54 个 fp16 Conv 的
权重/激活舍误差沿骨干累积**(同一素材输入相近 → 偏差方向一致);
只保护末尾 fc1 Gemm 无效(+0.0039)、只保护 BN 无效(TRT +0.0027 且仍
1 翻转)、保护 stage4+head Conv 无效(偏差来自前/中期大 FLOP 层,与
速度直接冲突)。最终修复:混合精度 ONNX 里**保护全部主要算子
Conv/BN/Relu/Add/Gemm 为 fp32**(仅 MaxPool/Flatten/Identity/Cast 留
fp16,MaxPool/Flatten 无数值舍入),`--fp16` 默认即此方案
(`--no-fp32-ops` 退化为全 fp16)。代价:速度回到 FP32 水平(4.11 vs
4.09ms/帧,本模型 fp16 无收益),体积 155MB——**精度换不来回速,
pose68 默认引擎保持 FP32**,`--pose-engine` 显式启用修复版 FP16 仅在
需要统一走 fp16 流程时用。归因数据:
`E:\output\_fp16_svf\pose68\`(diag_decomp.txt / summary_variants.txt /
compare100_*.csv)。

### 默认引擎优先级

- **SCRFD / gaze / head 默认 FP16,缺失自动回退 FP32**(按存在性逐级回退,
  无需任何 flag):
  - SCRFD:`scrfd_500m_bnkps_batch32_fp16.engine` → `scrfd_500m_bnkps_batch32.engine`
  - gaze:`resnet34_gaze_fp16.engine` → `resnet34_gaze.engine`
  - head:`model_dyn_fp16.engine` → `model_dyn.engine` → `model.engine`(batch1 固定版)
  - 均可用 `--engine` / `--gaze-model-path` / `--head-model-path` 显式指定。
- **pose68 例外,默认维持 FP32** `1k3d68_dyn.engine`:EAR 是眼睛区几个
  landmark 的小像素距离之比,FP16 舍入会被放大(眯眼边界帧翻转);修复版
  FP16(`1k3d68_dyn_fp16.engine`)保精度后速度收益归零(见上文归因),
  故默认仍 FP32,`--pose-engine` 显式启用。

## 用法

### 视频模式

```bash
python filter_video.py <video.mp4> --gaze-dy-dev 11
```

常用参数(默认值均已在 `filter_video.py` argparse 中,上面表格为主):

- `--out-dir <dir>`:输出目录。默认:单个文件 → `<input所在目录>/<stem>_kept`;目录输入(多视频)→ `<input目录>/kept_frames/<stem>_kept`,多视频共享同一 `--out-dir` 时 CSV 按视频名区分(`pose_report_<stem>.csv`)防覆盖
- `--engine` / `--pose-engine` / `--gaze-model-path` / `--head-model-path`:各模型 TRT 引擎路径。默认:仓库根 `models/scrfd/scrfd_500m_bnkps_batch32_fp16.engine`(缺 FP16 回退 `scrfd_500m_bnkps_batch32.engine`)/ `models/pose68/1k3d68_dyn.engine`(FP32,EAR 精度考虑,见「默认引擎优先级」)/ `models/gaze/resnet34_gaze_fp16.engine`(缺 FP16 回退 `resnet34_gaze.engine`)/ `models/head/model_dyn_fp16.engine`(缺 FP16 按 `model_dyn.engine` → `model.engine` 逐级回退)
- `--batch 16`:SCRFD 检测 batch(≤32);`--detect-chunk 64`:每次 detect() 的帧数
- `--max-fps 10.0`:源视频 fps 高于此值时按此 fps 均匀抽帧解码(0=关);`--target-long 768`:检测前 downscale 长边(0=关)
- `--no-head-gate`:关闭 Stage1.5 人头闸门(默认开)
- `--no-score-name`:禁用临时 score/angle 命名,输出统一为 `<stem>_<index>.jpg`
- `--quality 90`:JPG 质量;`--verbose`:逐帧打印

### 图片文件夹模式(对已保留的 jpg 再跑人头过滤,非破坏)

```bash
python filter_video.py --image-dir <图片目录> --out-dir <输出目录>
```

只跑人头闸门(不碰视频解码/SCRFD/pose/gaze):`nheads>=2` 的帧剔除;kept 图以**硬链接**放进 `--out-dir`,dropped 记入 `dropped.txt` + `REPORT.csv`(列 `filename,nheads`),**源目录完全不动**。也可直接给一个纯图片目录作为位置参数(目录内无 .mp4 时自动进入该模式)。注意此模式与 `--no-head-gate` 冲突(该模式只依赖人头闸门)。

### 人头闸门(Stage1.5)说明

- 用 head2(单类 head YOLO11l)检测整帧人头,`conf>=0.30`,NMS IoU 0.6;
- **只数 NMS 后的 head 总数:`≥2 头 = 画面里含他人 → 整帧丢弃**;
- 只在「其它闸门全过 + 恰好检出 1 张人脸」时触发,避免空转;
- **强后果:人流 / 景点密集视频可能整段 kept=0** —— 这是项目已接受的决策(要「单人」帧,宁缺毋滥)。
- 默认 FP16 引擎实测:纯推理 batch16 较 FP32 提速 ~2.8×,100 张分层抽样对比 kept/dropped 判定 100% 一致。

## 输出

- kept 帧:**原分辨率 JPG**(quality 默认 90),默认命名带 score/angle 信息(`--no-score-name` 关闭)
- `pose_report.csv` 逐帧(或 `pose_report_<stem>.csv`):
  `frame,verdict,score,yaw,pitch,roll,down_ratio,nfaces,ear,gaze_mag,gaze_dy,nheads`
- 图片文件夹模式额外:`REPORT.csv`(`filename,nheads`)+ `dropped.txt`

## 性能剖析(Nsight Systems)

本机装有 Nsight Systems 2026.5.1(`nsys` 已在 PATH),管线级的系统级剖析配方与全部踩坑记录在 `.claude/skills/nsight-systems/SKILL.md`(本机实战手册:实测采集命令、recipe 分析 SOP、kernel 名免 NVTX 归因法、中文 locale GBK bug 等避坑清单),nsys 相关任务直接按手册执行,无需重新调研。官方离线文档存档在 `docs/nsight-systems/`(内容大,grep 按问题检索)。剖析产物放 `E:\output\nsys\`;基准与各轮优化数据见 `data/e2e_benchmark_2026-09-28.md`。

实测过的采集命令(完整配方与约束见上述手册):

```bash
nsys profile -t cuda,nvtx,nvvideo,python-gil --gpu-video-devices=0 \
  --sample none --cpuctxsw none -o E:/output/nsys/<报告名> -f true \
  <ASCII-only包装.bat> <app参数...>
```

## 项目结构

```
short_video_filter/
├── filter_video.py          # 主入口: 视频模式 + 图片文件夹模式, 全管线调度
├── src/                     # 管线模块 + 运行时数据
│   ├── __init__.py
│   ├── face_det.py          # SCRFD-500m TRT 批量检测 + letterbox 预处理(含 cubin 加速/回退)
│   ├── face_pose68.py       # 1k3d68 68 点 → yaw/pitch/roll/EAR/down_ratio
│   ├── face_gaze.py         # 视线闸门封装(resnet34 / iris 两种实现)
│   ├── gaze_resnet.py       # resnet34 Gaze TRT 推理(448×448, 输出 pitch/yaw 度)
│   ├── head_gate.py         # Stage1.5 人头闸门(head2 TRT, NMS IoU 0.6, ≥2 头剔除)
│   ├── meanshape_68.pkl     # 68 点平均脸(face_pose68 用, 运行时数据, 在 git 内)
│   ├── preproc_kernel.cu    # SCRFD 预处理 CUDA 核源码(.cubin 缺失时自动回退 CPU)
│   └── preproc_kernel.cubin # CUDA 核编译产物(留本地, git 忽略, nvcc 可重编)
├── models/                  # 模型文件(全部留本地, git 忽略;获取/重建见上表)
│   ├── scrfd/               # scrfd_500m_bnkps.onnx / scrfd_500m_bnkps_batch32_fp16.engine(默认,FP16, + .reshaped.onnx)
│   │                        #   / scrfd_500m_bnkps_batch32.engine(FP32 回退, + .reshaped.onnx)
│   │                        #   / scrfd_500m_bnkps_batch32_fp16.onnx(--fp16 中间产物, 见「FP16 混合精度构建」)
│   ├── pose68/              # 1k3d68.onnx / 1k3d68_dyn.engine(FP32 默认) /
│                            #   1k3d68_dyn_fp16.engine(修复版 FP16, --fp16 产物,
│                            #   Conv/BN/Relu/Add/Gemm 保 FP32) /
│                            #   1k3d68_dyn_fp16_v1.engine(旧全 fp16, EAR 偏移, 留档)
│   ├── gaze/                # resnet34_gaze.onnx / resnet34_gaze_fp16.engine(默认,FP16, + .onnx)
│                            #   / resnet34_gaze.engine(FP32 回退)
│   └── head/                # model.onnx / model_dyn_fp16.engine(默认,FP16) /
│                            #   model_dyn.engine(FP32 回退) / model.engine(batch1 回退) /
│                            #   model_fp16.onnx(--fp16 中间产物) /
│                            #   threshold.json(head 训练 F1 最优阈值参考, 运行时用 0.30)
├── scripts/                 # 引擎构建 + 环境脚本
│   ├── build_engine.py      # SCRFD 动态 batch 引擎构建(含 Transpose 修复, --fp16)
│   ├── build_gaze_engine.py # resnet34 引擎构建
│   ├── build_head_engine.py # head2 动态 batch 引擎构建(min1/opt16/max32, --fp16 混合精度)
│   ├── setup_cuda_trt.ps1   # CUDA 13.1 + TRT 11.3 路径写入 Windows 用户环境(一次性)
│   └── verify_cuda_trt.ps1  # 验证 import tensorrt / 驱动是否就绪
├── data/                    # 旧 CLIP 项目逐人结果记录(*_checkpoint.json, 留本地, git 忽略)
├── requirements.txt
├── CLAUDE.md
├── .gitignore               # 忽略所有 *.onnx / *.engine / *.cubin / *_checkpoint.json / data/
└── README.md
```
