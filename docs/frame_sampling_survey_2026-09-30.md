# 视频内容感知抽帧 / 帧选择:工业界与开源界方案调研(重点:光流路线)

调研日期:2026-09-30。调研范围:镜头/场景切分、光流用于帧选择(含 RTX 3060 OFA 硬件光流)、轻量去重、解码端运动矢量提取、学术关键帧选择、云厂商工业实践。
结论服务对象:本机短视频人脸过滤管线(NVDEC 解码 → GPU 预处理 → 3×TensorRT 闸门 → 保留合规帧),当前抽帧 = 固定 10fps 均匀采样 + 可选灰度缩略图 MAD diff 去重(`--dedup`,基准压缩比 ~1.36x,与 GPU 零拷贝解码臂不兼容)。

> 注:本报告所有数字均标注来源;社区实测数字(标注"社区实测")仅供参考量级。

---

## 1. 方案分类总表

| 方案 | 机制一句话 | 计算成本量级(720p/1080p) | 成熟度 | 关键来源 |
|---|---|---|---|---|
| 固定间隔采样(现状默认) | 每隔 N 帧取 1,不看内容 | ≈0 | 工业标配 | — |
| 灰度缩略图 MAD diff + 滞回(现状 `--dedup`) | 缩略图逐帧平均绝对差,超阈值切段,段内取最清晰 1 帧 | CPU 侧每帧 <1ms(64×36 级) | 自研,同类机制工业通用 | PySceneDetect ContentDetector 同族(见下) |
| ffmpeg `select='gt(scene,T)'` | 逐帧直方图/SAD 场景分,超阈值保留 | 全软解 + 逐帧打分,受解码限速(720p ~百 fps) | 成熟 | [ffmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)、[wiki: scene detection](https://trac.ffmpeg.org/wiki/scene%20detection) |
| ffmpeg `scdet` 滤镜 | 专用场景切分滤镜,输出 mafd/score 元数据 | 同上 | 成熟 | [ffmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) |
| ffmpeg `thumbnail=N` | 窗口内 8×8 块差异选"代表帧" | 最便宜的一档(块计数) | 成熟 | [ffmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) |
| ffmpeg `mpdecimate` | hi/lo/frac 帧差判定丢"非有意变化"帧 | 便宜(像素差计数) | 成熟(去隔行重复帧起家) | [ffmpeg filters](https://ffmpeg.org/ffmpeg-filters.html#mpdecimate)、[SO 讨论](https://stackoverflow.com/questions/62754951) |
| PySceneDetect ContentDetector | HSV 逐帧差滑动平均 | 720p ~188fps(CPU,解码为瓶颈;社区实测) | 成熟,事实标准 | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)、[benchmarks](https://github.com/Breakthrough/PySceneDetect/blob/feature-adaptive-detector/docs/benchmarks/adaptive.md) |
| PySceneDetect AdaptiveDetector | 滚动窗口比值,抗"快摇镜误切" | 720p ~122fps(比 ContentDetector 慢 ~1.5x) | 成熟 | 同上 |
| TransNetV2 | 3D CNN 神经网络镜头边界检测(输入 48×27 RGB) | GPU 原始推理 ~1500fps、含解码 ~300fps(RTX 1080/3080 级,社区实测) | 成熟开源(F1: ClipShots 77.9 / PlanetEarth 96.2 / RAI 93.9) | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)、[issue #7](https://github.com/soCzech/TransNetV2/issues/7) |
| **OFA 硬件光流(NVOF2)** | 独立硬件单元算稠密光流,NV12/ABGR 输入 | Turing 级 1–10ms/帧;1080p 常见 1–3ms(preset/grid 相关) | 成熟(SDK 5.5;OpenCV/CV-CUDA 有封装) | [Optical Flow SDK](https://developer.nvidia.com/opticalflow-sdk)、[docs](https://docs.nvidia.com/video-technologies/optical-flow-sdk/index.html)、[OpenCV cudaoptflow](https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html)、[CV-CUDA optical_flow](https://cv-cuda.readthedocs.io/en/latest/optical_flow.html) |
| OpenCV CUDA Farneback | GPU 金字塔稠密光流 | 1080p ~4–15ms/帧(GTX1080Ti ~6.2ms,社区实测;CPU ~450ms) | 成熟 | [docs](https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html) |
| OpenCV/optflow TVL1 (CUDA) | 高质量稠密光流,迭代 TV 正则 | 慢(Farneback 的数倍) | 成熟但性能垫档 | [docs](https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html) |
| RAFT 等深度光流 | 全对相关体 + GRU 迭代精化 | 3060 @1080p 估 ~0.5–1s/帧(RAFT-large),RAFT-small 也要百 ms 级 | 研究级,对本场景过重 | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) |
| 解码端运动矢量提取 | `export_mvs` 从软解码流直接拿编码器 MV | 软解限速(720p h264 数百 fps),MV 本身零额外计算 | 成熟 | [extract_mvs.c 示例](https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/extract_mvs.c)、[mvextractor](https://github.com/LukasBommes/mvextractor) |
| 感知哈希(pHash/PDQ) | 32×32 DCT→64bit 哈希,汉明距离判重复 | 极便宜 | 成熟(videohash/PDQ 生态) | [videohash](https://github.com/akamhy/videohash)、[ThreatExchange(PDQ)](https://github.com/facebook/ThreatExchange) |
| CLIP 特征选帧 | 每帧 CLIP 图像编码,按文本相似度/多样性选帧 | 每帧一次 CLIP 前向(中) | 工具多,选帧场景成熟 | [frame-selection](https://github.com/LucasVentura/frame-selection)、[topic: frame-selection](https://github.com/topics/frame-selection) |
| 学习型自适应帧选择(AdaFrame/SCSampler) | RL/LSTM 或显著性打分,按"信息量"选帧 | 模型前向(中–高) | 学术为主 | [AdaFrame (CVPR2019, arXiv 1811.12432)](https://arxiv.org/abs/1811.12432)、[SCSampler](https://openaccess.thecvf.com/content_CVPR_2019/html/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_CVPR_2019_paper.html) |
| 云厂商"智能抽帧/智能封面" | 场景切分 + 画面质量/美学打分 + 间隔保底 | 服务端 | 商用 | [阿里云 ICE](https://help.aliyun.com/zh/ims/user-guide/smart-crop-and-smart-screenshot)、[腾讯云 CI](https://cloud.tencent.com/document/product/460/89518)、[AWS MediaConvert](https://aws.amazon.com/mediaconvert/features/) |

---

## 2. 镜头/场景切分:工业界的"哪些帧属于同一镜头"

- **ContentDetector**(PySceneDetect 默认):HSV 逐像素绝对差累加成滑动平均,超过阈值(~27)判切。快、对运动敏感(摇镜会误切)。AdaptiveDetector 用双窗口滚动平均的比值判切,显著减少运动误切,代价是慢 ~1.5x 且可能漏闪切。官方 720p/168k 帧基准:Content ~188fps、Adaptive ~122fps,瓶颈在解码不在打分(来源:[adaptive benchmark](https://github.com/Breakthrough/PySceneDetect/blob/feature-adaptive-detector/docs/benchmarks/adaptive.md)、[项目 benchmarks.md](https://github.com/Breakthrough/PySceneDetect/blob/main/docs/benchmarks.md))。
- **ffmpeg**:`select='gt(scene,T)'` 直方图分数;`scdet` 输出原始 MAFD;`thumbnail=N` 在 N 帧窗口内用 8×8 块差异挑代表帧(与本项目 `--dedup` "段内取最清晰/代表帧"同构,是最便宜的工程做法)(来源:[ffmpeg-filters](https://ffmpeg.org/ffmpeg-filters.html)、[wiki](https://trac.ffmpeg.org/wiki/scene%20detection))。
- **TransNetV2**:48×27 小图输入的 3D CNN,精度远超传统方法(ClipShots F1 77.9),GPU 上推理极快(社区实测原始推理 ~1500fps、接 PySceneDetect 后 ~300fps),但要多一个模型依赖与解码通道。社区共识:PySceneDetect ContentDetector 是速度/精度最佳折中,TransNetV2 更准但更重(来源:[TransNetV2](https://github.com/soCzech/TransNetV2)、[PySceneDetect docs](https://www.scenedetect.com/))。

**对本管线**:固定 `--max-fps 10` 的问题不是"采样率",而是"不区分镜头"。镜头切分信息(MAD 尖峰即可粗略替代)是把"每镜头配额 + 镜头内去重"组合起来的前提。

## 3. 光流路线专项评估(用户点名路线)

### 3.1 RTX 3060 的 OFA 可用性:确认支持

- OFA(Optical Flow Accelerator)自 Turing 起为**独立硬件单元**;RTX 3060 属 Ampere GA10x,**带 OFA,每卡 1 个 OFA 会话**(GA10x 会话数:3 NVENC / 3 NVDEC / 1 OFA),最大 8192×8192、最小 58×48(来源:[Optical Flow SDK](https://developer.nvidia.com/opticalflow-sdk)、[NVOF docs](https://docs.nvidia.com/video-technologies/optical-flow-sdk/index.html))。
- **并发性(关键好消息)**:Video Codec SDK 文档明确"OFA 是独立硬件单元,NVDEC/NVENC 与 OFA 的会话可以并发互不影响(受各自性能上限约束)"——即**光流计算不与 NVDEC 解码争抢**,也不占 CUDA/SM(来源:同上 docs;社区在 [VideoCodecSDK issue #96](https://github.com/NVIDIA/video-codec-sdk) 亦确认 OFA≠NVDEC,nvidia-smi 不显示 OFA 占用)。
- **与 NVDEC 同流复用(格式匹配)**:NVOF2 输入支持 **NV12**(2 字节 pitch 对齐)与 ABGR/10-bit 格式;NVDEC 零拷贝臂输出的正是 NV12 device memory → 理论上可直接把解码面喂给 NVOF,不落 host、不过 CUDA kernel(来源:[NVOF2 应用说明/GitHub README](https://github.com/NVIDIA/OpticalFlowSDK);需注意 2 字节 pitch 对齐)。
- 运行时依赖:NVOF2 需 r511+ 驱动(nvofapi 运行库随驱动),SDK(头文件/样例)从 NVIDIA 开发者站下载(来源:[Optical Flow SDK](https://developer.nvidia.com/opticalflow-sdk))。DLSS Frame Generation / Lossless Scaling 等"驱动级补帧"即用 OFA,工业可用性已被大规模验证(来源:[NVIDIA DLSS 3](https://www.nvidia.com/en-us/geforce/news/dlss3-ai-powered-neural-graphics-innovations/))。

### 3.2 吞吐量级

- NVIDIA 应用说明:硬件引擎单帧 1–10ms(Turing,分辨率相关),CPU/GPU 侧开销很小;"硬件引擎占大头"(来源:[NVOF2 应用说明,经 OpticalFlowSDK README](https://github.com/NVIDIA/OpticalFlowSDK))。
- 社区实测区间很宽:有 RTX 3090 上集成不当只跑出 ~25fps(1080p)的个例,也有 OpenCV NVOF2/Farneback 对比中 OFA ~1ms/帧 的说法;OpenCV 官方对比把 NVOF2 列为最快档、TVL1 最慢档(来源:[OpenCV forum NVOF1 vs NVOF2](https://forum.opencv.org/t/nvof1-vs-nvof2/56528)、[OpenCV cudaoptflow](https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html))。
- **对本管线**:60fps 源、10fps 采样,即使逐解码帧做光流也只是 60 次/秒 × 1–3ms ≈ 6–18% 的一个专用单元占用,且不占 SM——相对 3 个 TensorRT 引擎可忽略。

### 3.3 Python 侧 API 选择

| 封装 | 状态 | 备注 |
|---|---|---|
| CV-CUDA `cvcuda.optical_flow` | 有算子与 Python API,基于 NVOF;8×8 网格,输出为像素分数单位(不同封装 1/16~1/64 不一,用前按文档换算) | 需 CUDA 11.8+/驱动 525+;**Windows wheel 可用性需在本机验证**(来源:[CV-CUDA optical_flow](https://cv-cuda.readthedocs.io/en/latest/optical_flow.html)、[CV-CUDA GitHub](https://github.com/CUDA/cv-cuda)) |
| OpenCV `cuda::NvidiaOpticalFlow_2_0` | 官方封装,CC≥7.5,preset/grid/temporal-hints 可配;Python 绑定较新 | 构建时需要 Optical Flow SDK(来源:[OpenCV docs](https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html)、[forum](https://forum.opencv.org/t/nvof1-vs-nvof2/56528)) |
| NVIDIA VPF (VideoProcessingFramework) | 自带"解码→光流"示例管线(C++) | Python 友好度一般(来源:[VPF](https://github.com/NVIDIA/VideoProcessingFramework)) |
| ffmpeg | **无内置 nvof 滤镜**(仅邮件列表补丁如 vf_nlmeans_nvof) | 想走 ffmpeg 需自己写(来源:[FFmpeg 邮件列表补丁,经检索](https://ffmpeg.org/pipermail/ffmpeg-devel/)) |

### 3.4 光流决定抽帧密度的真实案例

- **学术**:AdaFrame(CVPR2019,RL 自适应选帧,FCVID 平均每视频仅 5.4 帧)、SCSampler(CVPR2019,用 RGB 差/运动显著性对片段打分采样)是"按信息量选帧"的代表作;光流幅度作为"帧重要性/显著性"代理在视频摘要文献中是常见特征(FlowRank 等)(来源:[AdaFrame](https://arxiv.org/abs/1811.12432)、[SCSampler](https://openaccess.thecvf.com/content_CVPR_2019/html/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_CVPR_2019_paper.html))。
- **工程/产品**:工业界用 OFA 光流的主要是**补帧/插帧**(DLSS FG、Lossless Scaling、NVOF 自带的 NvOFFRUC 帧率上变换样例)与目标跟踪/标注(CVAT 类工具的稠密光流插值),**"用光流幅度决定抽帧密度"没有看到大规模公开的工业标准实现**;云厂商转码的抽帧(见 §6)用的是场景检测+质量分,不是光流。
- 监控摘要类系统多用背景差分/运动检测而非稠密光流(成本考虑)。

### 3.5 光流 vs 朴素 MAD/直方图 diff:收益到底在哪

| 维度 | MAD/直方图 diff | 光流 |
|---|---|---|
| 输出 | 单个全局标量(变化量) | 每块 (dx,dy) 向量场 |
| 全局运动(摇镜/变焦) | 全帧差都大 → 误判"每帧都不同",去重失效 | 方向一致+幅度均匀 → 可识别为全局运动,不误杀 |
| 局部运动(静止镜头里人走动) | 差值小 → 易被当"重复" | 局部块幅度大 → 可判"有内容变化" |
| 镜头切换 | 尖峰可判切 | 同样可判切(幅度尖峰+相关性崩) |
| 成本 | ≈0(缩略图) | OFA 1–3ms/帧(3060,不占 SM);CUDA Farneback 4–15ms/帧(占 SM,会与 TensorRT 抢算力);RAFT 不可行 |

**结论**:光流的增量收益集中在两点——(a) 摇镜/高动态镜头的鲁棒去重与密度自适应,(b) 运动方向信息(可选做运动补偿帧差)。对"静止/低动态镜头去重"这一主诉,MAD+滞回已能拿到大头(实测 1.36x)。**OFA 路线在 3060 上计算成本可忽略、格式与 NVDEC 输出匹配、可与解码并发,技术可行性好;短板全在工程集成(Python 绑定/Windows wheel)与调参。建议先做独立原型对基准视频对比两信号的分段差异,再决定并入。**

## 4. 专项回答:H.264/HEVC 解码端能否"免费"拿运动矢量

- **能,但只在软解路径**:ffmpeg `-flags2 +export_mvs` 使解码器把 MV 作为 `AV_FRAME_DATA_MOTION_VECTORS` side data 输出(`AVMotionVector`:source/src_x/src_y/dst_x/dst_y,1/4 像素单位);C 参考实现 `doc/examples/extract_mvs.c`,可视化 `codecview=mv=pf+bf+bb`(来源:[ffmpeg-filters#codecview](https://ffmpeg.org/ffmpeg-filters.html#codecview)、[extract_mvs.c](https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/extract_mvs.c))。
- **H.264**:2015 年起支持([提交 def9785, lavc/hevc: export motion vectors](https://github.com/FFmpeg/FFmpeg/commit/def978538c5e7613e2b11bdcb01c40a7bb3efc47))。**HEVC**:同样走该 side data,但对部分编码参数(B 帧结构等)导出不全,需按源验证(来源:同上提交及其说明、[mvextractor](https://github.com/LukasBommes/mvextractor))。
- **NVDEC 不输出 MV**:官方 SDK 讨论确认 NVDEC 不暴露块级 MV;商业方案(如 VIDEA SDK)通过码流解析拿到 MV/块信息(来源:[NVIDIA VideoCodecSDK](https://github.com/NVIDIA/video-codec-sdk) 相关 issue/论坛讨论、[检索到的 SO 讨论](https://stackoverflow.com/questions/tagged/nvdec))。
- **现成工具**:[LukasBommes/mvextractor](https://github.com/LukasBommes/mvextractor)(C++,PyPI 可装,边解码边出 MV+帧+时间戳)、[amberwangyili/mv-tracker](https://github.com/amberwangyili/mv-tracker)(MPEG4/HEVC/H264,附 MAD)、PyAV `frame.side_data`。
- **怎么用**:P/B 帧的 MV 是编码器运动搜索的产物,信息"免费";常用特征 = 平均 MV 幅度、零 MV 块占比(静止镜头占比高)、MV 尖峰(切镜/I 帧)。I 帧无 MV。
- **对本管线的矛盾点**:MV 提取必须软解,与 NVDEC 零拷贝臂互斥 → 只适合**离线预扫/单独轻量 pass**(720p h264 软解数百 fps,60s 视频几十秒内扫完),不适合在线并入 GPU 臂。

## 5. 轻量去重的其他工业做法

- **mpdecimate**:hi=768/lo=320/frac=0.33(8bit 默认)判定"非有意变化"丢帧;`max` 正值=最多连丢 N 帧,负值=保底间隔(注意文档默认语义有过争议,用前显式设 `max`)(来源:[ffmpeg-filters#mpdecimate](https://ffmpeg.org/ffmpeg-filters.html#mpdecimate)、[SO](https://stackoverflow.com/questions/62754951))。**思路可借鉴到 `--dedup` 的"段长上限/保底帧"参数化**。
- **pHash 生态**:videohash(FFmpeg 场景选帧→144×144→旋转/拼贴→64bit 视频级哈希)、Facebook PDQ(大规模去重)、VideoDuplicateFinder(逐帧哈希阈值)(来源:[videohash](https://github.com/akamhy/videohash)、[ThreatExchange](https://github.com/facebook/ThreatExchange)、[VideoDuplicateFinder](https://github.com/0x90d/videoduplicatefinder))。视频级哈希是"整段重复视频"层面的,帧级去重还是帧差/直方图便宜。
- **auto-editor**:开源自动剪辑工具,`--motion-threshold 2%` 用帧运动量剪掉静止段——"运动量驱动剪辑"在创作工具里已是标配功能(帧差实现,非光流)(来源:[auto-editor](https://github.com/wyattblue/auto-editor))。

## 6. 工业界实践(转码服务/短视频平台方向)

- **AWS MediaConvert**:抽帧=帧捕获(FrameCapture)+ 场景检测(SceneChangeDetect,可调严格度),典型组合是"切镜帧或最大间隔,谁先到谁触发",正是"内容感知 + 间隔保底"的混合策略;场景检测同时用于 HLS 分段(来源:[MediaConvert features](https://aws.amazon.com/mediaconvert/features/)、[缩略图教程章节 PDF](https://docs.aws.amazon.com/mediaconvert/latest/ug/mediaconvert-guides-tutorials-thumbnails-chapter.pdf))。
- **阿里云 ICE(智能媒体服务)**:"智能抽帧截图"= 智能判断画面切换与截图时机,产出"重复率低、质量高、内容丰富"的截图;另有智能封面(美学/构图/清晰度打分选帧)(来源:[阿里云智能抽帧截图文档](https://help.aliyun.com/zh/ims/user-guide/smart-crop-and-smart-screenshot));MPS 场景检测做分镜拆分(来源:[shot-detection](https://help.aliyun.com/zh/mps/user-guide/shot-detection))。
- **腾讯云**:数据万象 CI"视频智能截帧"(标签+置信度+时间点);MPS"场景智能检测"(转场/黑边/黑白屏/马赛克)(来源:[CI 智能截帧](https://cloud.tencent.com/document/product/460/89518)、[MPS 场景智能检测](https://cloud.tencent.com/document/product/861/91626))。
- **共性总结**:云厂商与创作工具(auto-editor)的公开做法 = **场景/转场检测(帧差族) + 每镜头/每窗口质量或美学打分选代表帧 + 最大间隔保底**。**没有一家公开方案把稠密光流作为抽帧主信号**;光流在工业界的公开用武之地是插帧(补帧率)、稳像、跟踪。短视频平台(TikTok/快手)无公开技术细节,不可考。

## 7. 对本管线的落地建议(按改动成本从小到大)

1. **【小改】把 `--dedup` 的缩略图生成搬进 GPU 臂,消除宿主解码回退。**
   做法:GPU 预处理阶段顺带 resize 出 ~64×36 灰度缩略图,写入 pinned 环形缓冲(~2.3KB/帧,D2H 可忽略);CPU 端沿用现有 MAD+滞回逻辑不动。
   预期收益:两条解码臂都能吃到 ~1.36x 压缩,且 GPU 臂不再回退 ~1.7x 慢路径——等于净收益最大的零风险项。
   风险:GPU resize 插值与 CPU 灰度路径有差异,阈值需重标定一次;改动局限在预处理内核+一个拷贝。

2. **【中改】从"段内 1 帧"升级为"镜头感知采样":MAD 尖峰判切 + 每镜头配额 + 最大间隔保底。**
   做法:用现有 MAD 信号在段切分之外增加"镜头切分"(MAD 尖峰阈值,或离线跑 PySceneDetect/ffmpeg `thumbnail`/`scdet` 生成边界);每个镜头内保底 1–2 帧(按清晰度),低动态段稀疏、高动态段加密;段长上限从 10 帧(1s)放宽到 1–3s 并配 `mpdecimate` 式"max 连丢上限"保底。
   预期收益:消除跨镜头重复,压缩比应显著高于 1.36x;每镜头覆盖率可控(可回答"这段视频拍到了什么")。这也是 AWS/阿里云公开采用的"切镜触发+最大间隔保底"同款策略。
   风险:阈值/配额需要用基准视频重标定;离线预扫多一次解码(CPU,百 fps 级,离线可接受)。

3. **【大改/试点】OFA 硬件光流自适应采样(NVOF2 路线,先原型后并入)。**
   做法:CV-CUDA `cvcuda.optical_flow`(优先,纯 Python)或 OpenCV `cuda::NvidiaOpticalFlow_2_0`;把 NVDEC 输出的 NV12 面直接喂 OFA(独立硬件单元,与解码并发,不占 SM);用平均光流幅度、零矢量占比、主导方向一致性三个统计量驱动:(a) 摇镜/全局运动镜头识别(全局一致方向→不因"每帧都不同"而放弃去重),(b) 高局部运动段加密采样、静止段稀疏,(c) 镜头切换辅助判据。
   预期收益:对高动态/手持摇镜密集的源改善最明显(固定 10fps 在这些镜头要么全重复要么漏关键动作);3060 上计算开销可忽略(1–3ms/帧,独立单元)。
   风险:工程集成量最大(Python 绑定可用性需验证:CV-CUDA Windows wheel、OpenCV 需带 NVOF2 构建;备选 C++/pybind);流程向量单位 1/16~1/64 像素需按文档换算;先写独立脚本对基准视频对比"OF 信号 vs MAD 信号"的分段差异,有增量再并入主管线。

---

## 附:主要来源清单

- NVIDIA Optical Flow SDK:https://developer.nvidia.com/opticalflow-sdk ; https://docs.nvidia.com/video-technologies/optical-flow-sdk/index.html ; https://github.com/NVIDIA/OpticalFlowSDK
- NVIDIA Video Codec SDK(会话数/并发):https://github.com/NVIDIA/video-codec-sdk
- DLSS 3(OFA 工业用例):https://www.nvidia.com/en-us/geforce/news/dlss3-ai-powered-neural-graphics-innovations/
- OpenCV CUDA 光流:https://docs.opencv.org/4.x/dc/d6b/group__cudaoptflow.html ; https://forum.opencv.org/t/nvof1-vs-nvof2/56528
- CV-CUDA 光流算子:https://cv-cuda.readthedocs.io/en/latest/optical_flow.html ; https://github.com/CUDA/cv-cuda
- NVIDIA VPF:https://github.com/NVIDIA/VideoProcessingFramework
- PySceneDetect:https://github.com/Breakthrough/PySceneDetect ; https://www.scenedetect.com/ ; benchmarks:https://github.com/Breakthrough/PySceneDetect/blob/feature-adaptive-detector/docs/benchmarks/adaptive.md
- TransNetV2:https://github.com/soCzech/TransNetV2 ; https://github.com/soCzech/TransNetV2/issues/7
- ffmpeg:filters doc https://ffmpeg.org/ffmpeg-filters.html ; scene detection wiki https://trac.ffmpeg.org/wiki/scene%20detection ; mpdecimate 讨论 https://stackoverflow.com/questions/62754951
- MV 提取:https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/extract_mvs.c ; https://github.com/FFmpeg/FFmpeg/commit/def978538c5e7613e2b11bdcb01c40a7bb3efc47 ; https://github.com/LukasBommes/mvextractor ; https://github.com/amberwangyili/mv-tracker
- 去重:https://github.com/akamhy/videohash ; https://github.com/facebook/ThreatExchange ; https://github.com/0x90d/videoduplicatefinder
- 创作工具:https://github.com/wyattblue/auto-editor
- 学术:https://arxiv.org/abs/1811.12432 (AdaFrame) ; https://openaccess.thecvf.com/content_CVPR_2019/html/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_CVPR_2019_paper.html (SCSampler) ; https://github.com/princeton-vl/RAFT
- CLIP 选帧:https://github.com/LucasVentura/frame-selection ; https://github.com/topics/frame-selection
- 云厂商:https://aws.amazon.com/mediaconvert/features/ ; https://docs.aws.amazon.com/mediaconvert/latest/ug/mediaconvert-guides-tutorials-thumbnails-chapter.pdf ; https://help.aliyun.com/zh/ims/user-guide/smart-crop-and-smart-screenshot ; https://help.aliyun.com/zh/mps/user-guide/shot-detection ; https://cloud.tencent.com/document/product/460/89518 ; https://cloud.tencent.com/document/product/861/91626

> 置信度备注:标注"社区实测"的数字来自 issue/论坛,仅供量级参考;OFA 单帧毫秒数区间来自 NVIDIA 应用说明;阿里云/腾讯云文档 URL 来自搜索结果快照,如 404 请在厂商帮助中心按标题检索"智能抽帧截图 / 视频智能截帧 / 场景智能检测"。
