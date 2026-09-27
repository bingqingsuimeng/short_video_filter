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
| SCRFD-500m(人脸检测 + 5 关键点) | `models/scrfd/scrfd_500m_bnkps.onnx` | ~2.4 MB | **公开**,insightface SCRFD | `models/scrfd/scrfd_500m_bnkps_batch32.engine`(动态 batch 1/16/32,640×640) | `python scripts/build_engine.py`(默认即上述路径,也可显式传 `onnx engine`) |
| 1k3d68(68 点 3D 姿态) | `models/pose68/1k3d68.onnx` | ~137 MB | **公开**(1k3d68 项目) | `models/pose68/1k3d68_dyn.engine`(动态 batch) | 由 `models/pose68/1k3d68.onnx` 按 `scripts/build_engine.py` 中 `build()` 的动态 batch profile 模式构建(解析网络 + `set_shape` min/opt/max + `build_serialized_network`) |
| Gaze ResNet34(视线 pitch/yaw,448×448) | `models/gaze/resnet34_gaze.onnx` | ~81 MB | **自训**(yakhyo MobileGaze / uniface 类训练) | `models/gaze/resnet34_gaze.engine`(batch1,448×448) | `python scripts/build_gaze_engine.py`(默认即上述路径) |
| Head YOLO11l(人头检测,单类 head,640×640) | `models/head/model.onnx` | ~101 MB | **自训**(real_head_yolo) | 默认 `models/head/model_dyn_fp16.engine`(FP16,动态 batch 1/16/32,IO fp32);缺失时自动回退 FP32 `models/head/model_dyn.engine`,再回退 batch1 固定版 `models/head/model.engine` | `python scripts/build_head_engine.py --fp16`(建 FP16 默认引擎);`python scripts/build_head_engine.py`(默认建 FP32 `model_dyn.engine`) |

> **重要:gaze(`models/gaze/resnet34_gaze.onnx`)与 head(`models/head/model.onnx`)两个 ONNX 是自训模型,不在任何公开源,克隆本仓库后需自行放置到上述路径。**

构建脚本说明(在 `scripts/` 下,均 `python scripts/<脚本> [onnx] [engine]`,不带参数时用仓库根 `models/` 下的默认路径):

- `scripts/build_engine.py` — SCRFD 专用:先把 ONNX 输入改写成动态 batch `[-1,3,640,640]`,并修复导出时的 head Transpose perm 问题(SCRFD ONNX 专有的 bug,其它模型不需要),再按 min=1/opt=16/max=32 profile 建引擎。
- `scripts/build_gaze_engine.py` — resnet34 onnx 本身固定 batch=1,直接建即可。
- `scripts/build_head_engine.py` — head2 onnx 的 Transpose 全部 batch 安全,无需 reshape,直接按 min=1/opt=16/max=32 profile 建动态 batch 引擎。默认建 FP32 `model_dyn.engine`;加 `--fp16` 先把 ONNX 转 FP16 混合精度(图 IO 保持 fp32,中间产物 `models/head/model_fp16.onnx` 保留)再建 `models/head/model_dyn_fp16.engine`。
- 三个脚本都是 TRT 11 风格:`Logger(WARNING)` + `set_memory_pool_limit(WORKSPACE, 1<<30)`,无全局 FP16 flag(自动/逐层精度)。

可选:`preproc_kernel.cubin` 是 `preproc_kernel.cu`(SCRFD letterbox 预处理 CUDA 核)的编译产物,用 `nvcc -cubin preproc_kernel.cu` 可重编;**缺失时 `face_det` 自动回退 CPU 预处理**,不影响正确性。

## 用法

### 视频模式

```bash
python filter_video.py <video.mp4> --gaze-dy-dev 11
```

常用参数(默认值均已在 `filter_video.py` argparse 中,上面表格为主):

- `--out-dir <dir>`:输出目录。默认:单个文件 → `<input所在目录>/<stem>_kept`;目录输入(多视频)→ `<input目录>/kept_frames/<stem>_kept`,多视频共享同一 `--out-dir` 时 CSV 按视频名区分(`pose_report_<stem>.csv`)防覆盖
- `--engine` / `--pose-engine` / `--gaze-model-path` / `--head-model-path`:各模型 TRT 引擎路径。默认:仓库根 `models/scrfd/scrfd_500m_bnkps_batch32.engine` / `models/pose68/1k3d68_dyn.engine` / `models/gaze/resnet34_gaze.engine` / `models/head/model_dyn_fp16.engine`(缺 FP16 按 `model_dyn.engine` → `model.engine` 逐级回退)
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
│   ├── scrfd/               # scrfd_500m_bnkps.onnx / scrfd_500m_bnkps_batch32.engine(+ .reshaped.onnx)
│   ├── pose68/              # 1k3d68.onnx / 1k3d68_dyn.engine
│   ├── gaze/                # resnet34_gaze.onnx / resnet34_gaze.engine
│   └── head/                # model.onnx / model_dyn_fp16.engine(默认,FP16) /
│                            #   model_dyn.engine(FP32 回退) / model.engine(batch1 回退) /
│                            #   model_fp16.onnx(--fp16 中间产物) /
│                            #   threshold.json(head 训练 F1 最优阈值参考, 运行时用 0.30)
├── scripts/                 # 引擎构建 + 环境脚本
│   ├── build_engine.py      # SCRFD 动态 batch 引擎构建(含 Transpose 修复)
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
