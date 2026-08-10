# 短视频人脸清洗流水线 v6

基于 SCRFD 人脸检测 + MobileCLIP2-S0 零分类的短视频人脸属性清洗管线，支持 GPU 加速。

## 系统要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Linux x86_64 |
| GPU | NVIDIA（支持 CUDA 12 计算），显存 >= 4GB |
| 驱动 | CUDA >= 12.0（CUDA 13 驱动可向后兼容运行 CUDA 12 程序） |
| Python | 3.10 |
| 磁盘 | 约 2GB |

## 安装步骤

### Step 1. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 2. 创建 Python 3.10 虚拟环境

```bash
conda create -n face_cleaner python=3.10 -y
conda activate face_cleaner
```

### Step 3. 安装 pip 依赖

```bash
cd /path/to/pipeline
pip install -r requirements.txt
```

> 如果网络较慢，可使用清华源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### Step 4. 配置 CUDA 12 库路径（关键！）

`onnxruntime-gpu` 依赖 CUDA 12 运行时库，但 pip 安装的 `nvidia-*` 包将 `.so` 放在 site-packages 内，
系统动态链接器无法自动发现。需要将它们加入 `LD_LIBRARY_PATH`。

**方法 A：conda activate 自动加载（推荐）**

```bash
mkdir -p ~/miniconda3/envs/face_cleaner/etc/conda/activate.d

cat > ~/miniconda3/envs/face_cleaner/etc/conda/activate.d/activate.sh << 'EOF'
#!/bin/bash
# 自动将 pip 安装的 nvidia CUDA 12 库路径加入 LD_LIBRARY_PATH
CUDA_LIBS=$(python3 -c "
import site, os
cuda_home = os.path.join(site.getsitepackages()[0], 'nvidia')
libs = []
for root, dirs, files in os.walk(cuda_home):
    if os.path.basename(root) == 'lib':
        libs.append(root)
print(':'.join(sorted(libs)))
" 2>/dev/null)
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIBS}:${LD_LIBRARY_PATH}"
fi
EOF

chmod +x ~/miniconda3/envs/face_cleaner/etc/conda/activate.d/activate.sh
```

重新激活环境使生效：

```bash
conda deactivate
conda activate face_cleaner
```

### Step 5. 验证 GPU 加速

```bash
cd /path/to/pipeline
python test/test_pipeline_gpu.py
```

**预期输出：**

```
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
device: GPU

SCRFD EP: ['CUDAExecutionProvider', 'CPUExecutionProvider']
CLIP EP:  ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

> 如果显示只有 `CPUExecutionProvider`，说明 `LD_LIBRARY_PATH` 未正确配置，请重新执行 Step 4。

## 项目结构

```
.
├── pipeline_v6.py              # 主流水线（入口）
├── requirements.txt            # pip 依赖
├── README.md                   # 本文件
├── scrfd_500m.onnx            # SCRFD 人脸检测模型（约 2.3MB）
├── clip_text_embeds.npy       # 预计算的文本嵌入（5 类 x 512 维）
├── MobileCLIP2-S0/
│   ├── visual.onnx            # MobileCLIP2-S0 视觉编码器（256x256 输入）
│   └── text.onnx              # 文本编码器（当前不使用，预计算嵌入替代）
└── test/
    ├── test_pipeline_gpu.py   # GPU 加速验证脚本
    ├── erci.png               # 动漫脸测试图
    ├── glass.png              # 戴眼镜测试图
    ├── glass_portrait.png     # 戴眼镜竖屏测试图
    ├── mask.png               # 戴口罩测试图
    ├── multi.png              # 多人脸测试图
    └── tietu.png              # 贴纸脸测试图
```

## 模型说明

| 模型 | 来源 | 用途 |
|------|------|------|
| SCRFD 500M | [insightface](https://github.com/deepinsight/insightface) | 人脸检测 |
| MobileCLIP2-S0 | [MobileCLIP](https://huggingface.co/RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX/tree/main) | 人脸属性零分类 |
| clip_text_embeds.npy | 本地预计算 | 5 类文本嵌入 |

## 使用方法

```bash
conda activate face_cleaner
python pipeline_v6.py --input <视频目录> --output <输出目录>
```

## 性能参考

NVIDIA GPU 环境下：

| 模型 | 推理耗时（每帧） |
|------|-----------------|
| SCRFD 500M | ~4ms |
| MobileCLIP2-S0 | ~4ms |

> 首次推理因 CUDA kernel 编译会有 300-600ms warmup 开销，后续推理稳定在 ~4ms。

## 常见问题

**Q: onnxruntime-gpu 找不到 `libcublasLt.so.12`**

确保 `LD_LIBRARY_PATH` 包含 nvidia 库路径：

```bash
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia
```

如果没有输出，按 Step 4 配置后重新 `conda activate face_cleaner`。

**Q: CUDA 13 驱动能用吗？**

可以。CUDA 驱动向后兼容，CUDA 13 驱动可正常运行 CUDA 12 程序。
