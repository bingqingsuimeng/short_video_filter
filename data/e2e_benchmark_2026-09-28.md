# E2E 基准与 pynvvc 硬解方案验证记录(2026-09-28)

> 实验执行:子代理串行完成,GPU RTX 3060 12GB,全程单任务。
> 产物:`E:\output\_gpudecode\_p1ab\`(p1_report.json / p1_e2e_report.json / p1_e2e_report_r2.json / 三联对比图);
> 脚本:`_test/p1_pynvvc_ab.py`、`_test/p1_e2e_ab.py`。

## 0. 结论摘要

> **2026-09-28 第二轮(流水线重叠 + head 体检)见 §9**:head 闸门批量化 1.76x、e2e 17.69→16.30s;
> 线程级解码/推理重叠受 GIL 限制(实测重叠≈0),真正重叠需子进程解码。

- **pynvvc(PyNvVideoCodec 2.2.3)帧源可直接替换现有 FFmpeg 管道解码段**:全链路 625 帧 kept/drop 判定 98.6% 一致,9 帧翻转(1.4%)全部为阈值压线帧(conf≈0.800 / |yaw|≈12.0 / gaze_dy 压 15°/11°),两轮复测确定性复现,非帧源缺陷。
- **解码段提速 1.26~1.30x**,不含探测段管线提速 1.18~1.22x;但 **e2e 全链路仅 1.03~1.04x**——瓶颈在 `probe()` 的 ffprobe `-count_frames`(纯 CPU 全软解数帧,占 e2e ~75%,两臂同价)。
- 检测级零影响:同一 SCRFD FP16 引擎下 148/148 框完全一致,score |Δ|≤0.0044,IoU≥0.961。
- **三项优化已落地(见第 7 节)**:probe 元数据化 + pynvvc 帧源内置为 `--decode pynvvc` + 免 tobytes 二次拷贝;4 视频 e2e 96.29s→**17.69s(5.44x)**,不含探测段 1.40x,纯解码段 1.62x;625 帧 kept/drop 与基线 A 仅 9 帧已归因压线翻转(与基线 B 逐位一致,零新增)。

## 1. 测试对象与参数

| key | 视频 | 分辨率 | 源 fps | 源帧数 | GT 帧(fps=10 抽帧) |
|---|---|---|---|---|---|
| 006 | buding_tuji_006_2026_09_11_20_59_35.mp4 | 2160x3840 | 60 | 631 | 105 |
| 099 | buding_tuji_099_2026_09_12_12_14_06.mp4 | 2160x3840 | 60 | 1010 | 168 |
| 154 | buding_tuji_154_2026_09_12_15_19_20.mp4 | 2160x3840 | 30 | 583 | 194 |
| 214 | buding_tuji_214_2026_09_12_18_36_52.mp4 | 1900x3378 | 60 | 947 | 158 |

参数(两臂完全一致):conf 0.8 / yaw 12 / pitch ±25 / ear 0.25 / down 0.46 / gaze resnet34(15°/20°)+ dy_dev 11 / head2 conf 0.30 / max-fps 10 / target-long 768 / batch 16 / chunk 64 / score_name 开。
A 臂 = `--decode auto`(FFmpeg 管道);B 臂 = `--decode gpu` + monkeypatch 帧源为 pynvvc(源帧号 = `stride*k+delta`,60fps→(6,2)、30fps→(3,1),内容匹配谷底法标定)。

## 2. 像素层差异(pynvvc RGB→BGR vs FFmpeg bgr24 GT,逐帧对齐后)

| 视频 | MAD 均值 | MAD p95 | 单像素最大差 | 有符号通道差 B/G/R | md5 相同帧 |
|---|---|---|---|---|---|
| 006 | 2.07 | 2.22 | 21 | +2.15 / −0.82 / −1.69 | 0/105 |
| 099 | 1.76 | 3.20 | 32 | +1.13 / −0.00 / +0.22 | 0/168 |
| 154 | 3.01 | 3.24 | 43 | +1.40 / −3.24 / −2.35 | 0/194 |
| 214 | 2.14 | 2.76 | 41 | +2.14 / +0.46 / −0.78 | 0/158 |

归因:非 range 错(线性拟合斜率 0.96~1.01)、非解码内容错;是**色彩矩阵解释差(601 vs 709 型)+ 10→8bit 舍入路径 + 色度上采样滤波**的组合。MAD ≈ 0.7%~1.2% 满量程,人眼不可分。

## 3. 检测级 A/B(SCRFD scrfd_500m_bnkps_batch32_fp16,conf 0.8,target_long 768)

| 视频 | 抽样帧 | 框数 gt/pnv | 漏检/多检 | score |Δ| mean/max | IoU min |
|---|---|---|---|---|---|
| 006 | 61 | 53/53 | 0/0 | 0.0005 / 0.0039 | 0.983 |
| 099 | 31 | 11/11 | 0/0 | 0.0009 / 0.0029 | 0.995 |
| 154 | 35 | 35/35 | 0/0 | 0.0014 / 0.0044 | 0.961 |
| 214 | 52 | 49/49 | 0/0 | 0.0005 / 0.0015 | 0.994 |

## 4. 全链路 kept/drop A/B(两轮复测,确定性一致)

| 视频 | 帧数 | kept A / B | 翻转帧数 | JPG 集一致 |
|---|---|---|---|---|
| 006 | 105 | 0 / 0 | 0 | ✓ |
| 099 | 168 | 59 / 57 | 3 | ✗ |
| 154 | 194 | 159 / 160 | 1 | ✗ |
| 214 | 158 | 42 / 40 | 5 | ✗ |
| **合计** | 625 | 260 / 257 | **9(1.4%)** | |

翻转明细与归因(全部压线,双帧源重跑验证):

| 帧 | A → B | 归因(双源实测) |
|---|---|---|
| 099 #38 | downp → no_face | conf 0.8018 vs 0.7988(压 0.8) |
| 154 #10 | no_face → keep | conf 0.7998 vs 0.8003(压 0.8) |
| 099 #105 | keep → pose | \|yaw\|≈12.0 压线 |
| 214 #144 | keep → pose | \|yaw\|≈12.0 压线 |
| 214 #48 | keep → gaze | gaze_dy 压 15° 绝对阈值(13.56 vs 15.48) |
| 214 #121 | keep → gazedown | 压 ±11° 2-pass 基线(-4.23 vs -4.64) |
| 214 #97 | gaze → keep | gaze_dy 15.14 vs 13.68,压 15° |
| 214 #65 | gaze → multi_head | 级联顺序效应:B 的 gaze 过了才轮到 head 闸门 |

判定一致帧上的数值漂移:score |Δ|max 0.003~0.005;pose68 角度 |Δ|max 0.2~3.4°;gaze 角 |Δ|max 1.7~4.4°。

## 5. E2E 耗时基线(修改前,两轮;r2 为冷缓存/负载波动参考)

### 5.1 汇总(单进程跑一臂 4 视频,引擎加载单列)

| 口径 | r1 A | r1 B | 加速 | r2 A | r2 B | 加速 |
|---|---|---|---|---|---|---|
| **e2e 全链路**(视频打开→CSV 落盘) | 96.29s | 92.85s | **1.037x** | 139.30s | 134.69s | **1.034x** |
| └ 其中 ffprobe `-count_frames` 探测段 | 71.79s | 72.81s | 1:1 | 106.62s | 106.95s | 1:1 |
| **不含探测段**(管线 internal) | 24.50s | 20.04s | **1.223x** | 32.68s | 27.74s | **1.178x** |
| **纯解码段**(decode+read) | 16.50s | 12.72s | **1.297x** | 20.66s | 16.46s | **1.255x** |
| 引擎加载(import+4 引擎,单次) | 0.65s | 0.47s | — | 0.47s | 0.48s | — |

### 5.2 逐视频 e2e(r1)

| 视频 | A 臂 | B 臂 | 加速 |
|---|---|---|---|
| 006 | 17.78s | 16.45s | 1.08x |
| 099 | 29.49s | 29.02s | 1.02x |
| 154 | 24.15s | 23.04s | 1.05x |
| 214 | 24.87s | 24.34s | 1.02x |

### 5.3 管线 internal 分段(r1 合计,A / B)

| 段 | A 臂 | B 臂 | 说明 |
|---|---|---|---|
| decode | 11.20s | 12.13s | B 含 ~4.2s CPU RGB→BGR+bytes 拷贝(4K 帧 6.8ms × 625) |
| read | 5.30s | 0.59s | 管道读 vs 宿主内存直读 |
| detect | 1.22s | 1.05s | SCRFD |
| gaze | 1.76s | 1.62s | |
| head | 3.69s | 3.42s | |
| imwrite | 0.03s | 0.04s | |

## 6. 优化计划(按收益排序)

1. **干掉 probe() 的 ffprobe `-count_frames`**:占 e2e ~75%,改读容器元数据(nb_frames/duration),预计 e2e 砍掉 3/4。
2. **pynvvc 帧源集成**为 `--decode` 显式新选项(默认行为不变,保留 fallback);选帧按帧率自动算 stride,60fps→(6,2)、30fps→(3,1) 已标定,其它帧率需兜底回退。
3. **RGB→BGR 拷贝优化**:B 臂解码段 1/3 耗在这(CPU 6.8ms/帧);方案:设备端转换,或改 letterbox 预处理 kernel 支持 RGB 直喂(免交换)。

## 7. 修改后复测(2026-09-28)

> 复测口径与第 5 节完全一致(同 4 视频、单进程一臂 4 视频、引擎加载单列)。
> 脚本 `_test/post_opt_verify.py`(两臂各一进程串行跑 4 视频),产物
> `E:\output\_gpudecode\_p1ab\post_优化\`(JSON 报告)与 `post_opt\`(跑批 JPG/CSV,
> out_dir 必须纯 ASCII,原因见下⚠️)。

### 7.1 修改内容(全部落在 `filter_video.py`)

| 任务 | 落地方式 | 位置 |
|---|---|---|
| 1. 干掉 probe 的 `-count_frames` | `probe()` 改读 ffprobe 容器元数据(`nb_frames`,缺失时 `duration×fps`,毫秒级);元数据缺失/解析异常才回退旧 `-count_frames` 精确数帧。nframes 本就只用于 verbose 进度显示(解码循环由 EOF 驱动),±1 误差无风险 | `filter_video.py:85-104`(`_nframes_from_meta`)、`:107-146`(`probe`) |
| 2. pynvvc 帧源集成 | 新增 `--decode pynvvc`(默认 auto 行为不变):`_PynvvcSource` 接口对齐 `_PipeRaw`(read/close/waited),按源帧号 `stride*k+delta` 自抽帧;标定表 `{60fps:(6,2), 30fps:(3,1)}`,表外帧率**绝不猜**→回退管道;守卫:pynvvc 与 ffprobe 尺寸不一致(旋转 metadata)回退、解出帧数 < 元数据期望-1(中途断流)回退;失败链 `pynvvc→管道→落盘`(process() 重构为统一回退链,auto 语义不变) | `:284-421`(帧源)、`:645-672`(回退链)、`:676-682/740/1110`(_run_pass)、`:1441`(CLI) |
| 3. RGB→BGR 拷贝优化 | **方案 c**:`read()` 不再做 `ascontiguousarray().tobytes()` 二次全帧拷贝,`cvtColor` 结果直接 `reshape(-1)` 返回 1-D ndarray——调用方 `np.frombuffer(buf,...)` 对它是零拷贝视图,`len(buf)==size` 语义不变,`_PipeRaw`(bytes)与 pynvvc(ndarray)同接口。**选 c 的依据**:实测 4K 帧现状 6.64ms → 纯 cvtColor 2.56ms;而方案 a 路线的纯 memcpy(RGB 直通)反而 3.98ms > cvtColor(cv2 SIMD 交换比 numpy 拷贝快),且 a 需改 letterbox kernel + pose68/gaze/head/imwrite 四个 BGR 消费端 + kept 帧 JPG 色序处理,收益(仅省 no_face 帧的交换,~58%×3.5ms)远小于风险。方案 b(RGBP planar/device 输出)受 pynvvc 2.2.3 跨 context 限制(§8.3)未采 | `:387-413`(`_PynvvcSource.read`) |

微基准(`_test/post_opt_probe_bench.py`,4K 帧 ×30 中位):cvtColor+tobytes(现状) 6.644ms/帧、cvtColor 2.561ms、cvtColor+预分配 dst 2.466ms、numpy 反转 22.347ms、纯 memcpy 3.979ms、零拷贝视图 0ms。

### 7.2 耗时对比(r1 口径;修改后 = `--decode pynvvc` 全栈)

| 口径 | 基线 A r1 | 基线 B r1 | **修改后(pynvvc)** | vs 基线 A 加速 | vs 基线 B 加速 |
|---|---|---|---|---|---|
| **e2e 全链路**(视频打开→CSV 落盘) | 96.29s | 92.85s | **17.69s** | **5.44x** | **5.25x** |
| └ 探测段(ffprobe) | 71.79s | 72.81s | **0.09s**(4 视频 22~24ms/个) | 800x | 810x |
| **不含探测段**(管线 internal) | 24.50s | 20.04s | **17.46s** | **1.40x** | **1.15x** |
| **纯解码段**(decode+read) | 16.50s | 12.72s | **10.19s** | **1.62x** | **1.25x** |
| 引擎加载(import+4 引擎,单次) | 0.65s | 0.47s | 0.55s | — | — |

逐视频 e2e(基线 A / 基线 B / 修改后):006 17.78/16.45/**2.61**s;099 29.49/29.02/**5.07**s;154 24.15/23.04/**5.96**s;214 24.87/24.34/**4.06**s。

修改后管线 internal 分段合计(decode/read/detect/gaze/head/imwrite)= 10.08/0.11/1.04/1.61/3.39/0.04s(基线 B:12.13/0.59/1.05/1.62/3.42/0.04s)——解码段省的 ~2.5s 即任务 3 砍掉的 tobytes 拷贝(4K 帧 6.6→2.6ms × 625)。

任务 1 单独收益(隔离验证,`--decode auto` 只换新 probe,记 postAUTO):e2e 24.11s(vs 基线 A **4.0x**)、探测段 0.09s、不含探测段 23.93s(1.02x,管线不动)、纯解码段 16.03s(1.03x)。即 e2e 从 96.29→24.11s 的 4x 全部来自 probe;pynvvc 再把不含探测段 23.93→17.46s(1.37x)。

### 7.3 kept/drop 等价性(625 帧,逐帧 CSV 比对)

| 对比 | 结果 |
|---|---|
| 修改后 auto vs 基线 A | **625/625 帧判定全一致、score/yaw/pitch/roll/down_ratio/ear/gaze_mag/gaze_dy 全部 \|Δ\|=0.0、JPG 文件名集一致**(probe 改动零影响) |
| 修改后 pynvvc vs 基线 B(monkeypatch 版) | **625/625 帧判定全一致、8 项指标 \|Δ\|=0.0、JPG 文件名集一致**(内置帧源 + 免 tobytes 与已验证实现逐位等价) |
| 修改后 pynvvc vs 基线 A | 翻转 **9 帧(1.4%),与第 4 节基线 B vs A 的集合完全相同,零新增**:099 #38(downp→no_face,conf 0.802 压 0.8)、099 #68(keep→gazedown,dy −6.91 vs −7.80 压中位数±11°)、099 #105(keep→pose,\|yaw\|=12.0 压线)、154 #10(no_face→keep,conf 0.800 压 0.8)、214 #48(gaze→keep,dy 13.56 vs 15.48 压 15°)、214 #65(keep→multi_head,级联顺序)、214 #97(gaze→keep,dy 15.14 vs 13.68)、214 #121(keep→gazedown,dy −4.23 vs −4.64)、214 #144(keep→pose,\|yaw\|=12.0) |

> 注:第 4 节翻转明细表只列了 8 帧,漏了 099 #68(dy 2-pass 基线压线),本轮补齐为 9 帧全集。

### 7.4 口径变化与工程注意点

1. **探测段已从"17~27s/视频"变为"~23ms/视频"**,e2e 全链路里不再有 ffprobe 软解数帧;`nb_frames` 在 4 个测试视频上与 `-count_frames` 精确一致,但其它容器(MKV/TS/网络流)可能缺失或偏差 ±1 → 自动回退/估算,不影响解码(循环由 EOF 驱动)。
2. ⚠️ **`cv2.imwrite` 在 Windows 对非 ASCII 路径(目录或文件名)静默失败**(rc=False、无异常、无日志):out_dir 含中文时 kept JPG 一张都不落盘,CSV 照常产出,极易误判成"全被剔除"。本批复测跑批目录因此用 `post_opt`(ASCII),报告 JSON/CSV 放 `post_优化`(Python io 不受限)。生产 out_dir 命名需保持 ASCII(可在 `_run_pass` 开头加一次 `cv2.imwrite` 探针做启动自检,未实施)。
3. pynvvc 路径解码段仍有 ~2.5ms/帧的一次全帧拷贝(RGB→BGR 交换),这是"4K 帧必须以 BGR 交给 pose68/gaze/head/imwrite"架构下的下限;进一步优化需把交换挪进检测预处理或改 device 输出(§8.3 风险)。
4. `--decode pynvvc` 与 `--dedup` 可叠加(同一 f_src 接口),已冒烟验证(154 + --dedup:128 段/compression 1.36x,无异常);dedup 下的等价性未做逐帧复测(本轮 dedup 关,与第 5 节口径一致)。

## 8. 已知风险与工程注意点

1. `import PyNvVideoCodec`(小写 import 报错);2.2.3 无 BGR24 直出,输出真 RGB。
2. pynvvc 不支持重放/seek,重复 get_batch_frames 继续往后读;EOF 时刷 `INVALID INDEX` WARN(无害);pts 有 −3 帧偏移,对齐靠内容匹配而非时间戳。
3. 帧缓冲在宿主内存依赖 `useDeviceMemory=0` 默认值,升级版本若改 device 输出需重新处理跨 context。
4. 压线帧翻转不可避免(score≈0.800±0.003 / |yaw|≈12.0);若要求同视频两遍 100% 一致,需给闸门加边际带/滞回。
5. stride/delta 标定表只对 60/30fps 有效;VFR/其它帧率需重新标定或回退 FFmpeg 路径。
6. 本轮仅验证 4 个 4K 竖屏 HEVC Main10 视频;H.264/AV1 与其它分辨率未验证。
7. **任务 1/2/3 落地后新增**:probe 元数据路径对非常规容器(MKV/TS/无 nb_frames)自动回退 `-count_frames` 或 duration×fps 估算(±1,只影响 verbose 显示);`--decode pynvvc` 表外帧率/尺寸不符/中途断流均自动回退 FFmpeg 管道;`cv2.imwrite` 对非 ASCII 路径静默失败(§7.4.2)——out_dir 必须纯 ASCII。

## 9. 流水线重叠与 head 体检复测(2026-09-28 第二轮)

> 实验执行:子代理串行完成,单卡 RTX 3060,全程单 GPU 任务。
> 产物:`E:\output\_gpudecode\_p1ab\post2\`(run_*.json / run_<tag>/<视频名>/ 的 JPG+CSV);
> 脚本:`_test/post2_harness.py`(run/cmp/merge)、`_test/post2_head_bench.py`(head 微基准)、
> `_test/post2_gil_probe.py`(GIL 探针)、`_test/post2_threaded_probe.py`(ThreadedDecoder 探测)、
> `_test/post2_overlap_probe.py`(重叠度时间线)、`_test/post2_fallback_smoke.py`(回退链冒烟)、
> `_test/post2_abort_reuse.py`(提前退出后 GPU 复用)、`_test/post2_clean_smoke.py`(落盘/dedup/image-dir 冒烟)。

### 9.0 结论摘要

1. **head 闸门体检+批量化是本轮唯一实打实的收益**:head 段 3.39s→**1.93s(1.76x)**,e2e 17.69s→**16.30s(1.09x)**;625 帧 kept/drop、8 项指标(含 nheads)、JPG 文件名集与**文件字节**全部逐位一致。
2. **解码/推理流水线(producer-consumer 线程)已落地且逐位等价,但没换来预期收益**:CPython GIL 把解码与推理锁回串行——pynvvc 净收益 1~3%(154 视频三轮交替 5.207s→5.148s),FFmpeg 管道路径 ≈0%。**真正要拿到 max(解码, 推理) 必须用子进程解码**(见 §9.6),本轮线程版保留作为其落点(结构/异常传播/回退链都已验证)。
3. 新瓶颈=解码段 10.2~11.6s(16.3ms/4K 帧),推理侧合计已压到 ~5.1s(detect 1.3 + gaze 1.8 + head 1.9);decode:inference ≈ 2:1。

### 9.1 改动内容(全部已提交)

| # | 改动 | 位置 | 效果/口径 |
|---|---|---|---|
| 1 | **解码/推理流水线**:`_PipelinedSource` 包装帧源——后台 daemon 线程(producer)不断 `make_src().read(size)` 取帧入**有界队列**(默认 16 帧≈400MB 上限,实测稳态 ~2 帧≈50MB),主线程(consumer)攒批推理;帧源在 producer 线程内构建(CUDA context 归属其使用线程)。异常(建源失败/`_PipeDecodeError`/`_PynvvcDecodeError`)在 producer 线程原样存下、consumer `read()` 原样重抛 → `process()` 既有逐级回退链语义不变;`close()` 置 stop 事件+排空队列+join+关底层帧源(不泄漏 NVDEC 会话/ffmpeg 进程);CLI `--pipeline/--no-pipeline`(默认开)、`--pipeline-depth`(默认 16) | `filter_video.py:446-551`(类)、`:856-880`(帧源工厂化+接线)、`:1673-1680`(CLI) | 逐位等价;重叠收益 1~3%(GIL,见 §9.2);流水线开时 breakdown 口径变化:decode=producer 线程内解码耗时,read=主线程等队列时间,**两段不再与墙钟相加** |
| 2 | **head 闸门两阶段批量化**:`_gate_chain` 拆成 `_gate_pre`(_judge→低头复核→gaze,逐帧、顺序与原实现一致)+ head 阶段;主循环每块先逐帧判定,把所有"其它闸门全过且单人脸"的帧收集成 `head_jobs`,块末一次 `head.detect_batch()`(引擎 profile max_batch=32,自动切片),`--head-batch/--no-head-batch` 可回旧逐帧路径 A/B。`detect_batch` 新增 `workers`(线程池并行 letterbox,4K 帧 4.0ms→2.05ms;每帧独立 canvas/blob,无共享状态)。dedup 路径仍走逐帧 `_gate_chain`(行为不变) | `filter_video.py:692-760`(_gate_pre/_gate_chain 拆分)、`:1240-1330`(两阶段主循环)、`src/head_gate.py:142,150-195` | head 15.8ms/帧→**6.8ms/帧**(1.9x);判定逐位一致(§9.5) |

### 9.2 流水线为什么没换来 max(解码, 推理) —— GIL 实测(不猜)

| 实验(`_test/post2_gil_probe.py`,154 视频,4K) | 结果 | 结论 |
|---|---|---|
| 单线程 pynvvc 连解 120 帧 | 12.4 ms/帧 | 基准 |
| 后台线程解码 + 主线程 GIL 轮询(`time.sleep(0.5ms)` 循环) | 轮询 **63 it/s**(纯 sleep 基准 ~2000) | **`get_batch_frames` 是全程持 GIL 的阻塞 C 调用**,解码期间其它线程拿不到 GIL |
| 后台线程解码 + 主线程 cv2.resize(4K→768, 释放 GIL) | 墙钟 1.54s vs 串行 1.83s(重叠 85%) | 消费侧 GIL 越轻重叠越好——但真实推理侧做不到 |
| `PyNvVideoCodec.ThreadedDecoder`(C++ 内部线程+环形缓冲) | **64 it/s**(同 SimpleDecoder)且吞吐 25.9ms/帧(更慢) | 官方"后台线程解码"同样在 Python 侧持 GIL,救不了 |
| 真实管线时间线(`post2_overlap_probe.py`,154) | producer 产出间隔 p50=2.0ms/mean=24.7ms,队列深度 mean 1.4/p50 0(cap 16) | 无生产者领先:两线程互相等 GIL,完全乒乓串行 |
| 管道路径(auto)实测 | producer busy 11.2s(ffmpeg 产量下限),但 Python 侧 32KB 分块读循环(25MB 帧 ~780 次 os.read+join)与推理侧 numpy/cv2 争 GIL → e2e 与串行持平(23.90 vs 23.93s) | 管道解码虽在独立进程,Python 读帧仍吃 GIL |

净效果(154 视频,三轮交替):串行 5.207s → 流水线 5.148s(**+1.1%**);4 视频合计 +0.2~0.5s(1~3%),噪声带 ±5%。

### 9.3 head 闸门体检(`_test/post2_head_bench.py`,4K kept 帧实测)

先弄清 3.39s/268 帧(625 帧中真正送入 head 的帧数:006=0、099=63、154=160、214=45)= 12.65ms/帧花在哪:

| 项 | 实测 | 说明 |
|---|---|---|
| head2 引擎输入 | **640×640(已是训练尺寸)**,输出 (batch,5,8400),profile max_batch=32,IO fp32 | 无"letterbox 到过大尺寸"问题,**缩输入会伤小人头检出=改语义,不做** |
| 逐帧 `detect()`(旧路径,batch=1) | **15.80 ms/帧** | 每帧一次 `set_input_shape`+execute+`stream.synchronize()`(串行化 CPU/GPU) |
| `detect_batch(16)` / `(32)` | **9.17 / 9.23 ms/帧** | 批量摊薄 execute+sync;32 与 16 持平(引擎 FP16 640×640 单帧 GPU ~3.8ms 是下限) |
| bs=16 段拆分 | letterbox+pack 4.24 ms/帧;H2D 0.4;execute+sync 3.8;D2H 0.26 | letterbox 是 CPU 大头,GPU 是次大头 |
| letterbox(4K→640) 单帧 | **4.00 ms**(np.full canvas 0.30 + 纯 resize 仅 0.18 + fp32 链路 ~3.5) | 8 线程 → **2.05 ms**(cv2/numpy 释放 GIL,可并行) |
| batch=1 vs batch=16 头数/框/分 | 16 张对比 **逐位一致** | 批量不改判定(全量验证见 §9.5) |

结论:优化点=①head 按块批量(去 per-frame sync)②letterbox 搬线程池;输入尺寸不动(已是训练尺寸,动=改语义)。

### 9.4 耗时对比(r1 口径:单进程一臂 4 视频、引擎加载单列)

| 口径 | §7 修改后(基线) | 旧语义复测 ab_a(2 轮) | **head 批量 only**(t2_pipe) | **最终:流水线+head 批量**(final_pnv) | final auto(管道) |
|---|---|---|---|---|---|
| **e2e 全链路** | 17.69s | 18.45 / 21.45s | 16.52s | **16.30s** | 23.90s |
| └ 探测段 | 0.09s | 0.09s | 0.09s | 0.10s | 0.09s |
| **不含探测段** | 17.46s | 18.21 / 21.21s | 16.28s | **16.12s** | 23.74s |
| decode 段 | 10.17s | 10.64~12.32s | 10.29s | 11.56s¹ | 14.80s¹ |
| read 段 | 0.10s | 0.10s | 0.11s | 9.76s¹ | 16.97s¹ |
| detect | 1.17s | 1.11~1.42s | 1.14s | 1.37s | 1.45s |
| gaze | 1.61s | 1.55~1.92s | 1.55s | 1.78s | 2.04s |
| **head** | **3.39s** | 3.42~3.90s | **1.82s** | **1.93s** | 1.97s |
| imwrite | 0.04s | 0.13~0.15s | 0.11s | 0.00s | 0.00s |
| 引擎加载 | 0.55s | ~0.6s | 0.58s | 0.56s | 0.59s |

¹ 流水线开时 decode=后台线程解码耗时、read=主线程等队列时间,两者**时间上重叠**,不能再相加(相加=21.3s 是假象);墙钟看"不含探测段"。

- 同会话交替 A/B(旧语义 vs 最终,各 2 轮):e2e **1.13~1.28x**(a 臂受系统负载噪声影响大,b 臂稳定在 16.3~16.7s);分段看 head 3.42~3.90→1.93(**1.76~2.0x**)是负载无关的硬收益。
- 对 §7 基线 17.69s:最终 16.30s = **1.09x**;若只开 head 批量(流水线关)16.52s = 1.07x。
- 逐视频 e2e(final_pnv):006 2.79s / 099 4.60s / 154 5.16s / 214 3.74s(§7 基线:2.61/5.07/5.96/4.06s)。
- auto(管道)路径 23.90s:比 pynvvc 慢 7.6s,差距=ffmpeg 产量下限(11.2s) vs NVDEC(10.2s)+Python 读帧 GIL 争用,流水线帮不了它。

### 9.5 等价性(625 帧,全部逐位一致)

| 对比 | 结果 |
|---|---|
| 新代码 `--no-pipeline --no-head-batch` vs 28b926c 基线 | 625/625 帧判定一致、9 项指标(8 项+nheads)|Δ|=0.0、JPG 名单一致(重构零影响) |
| 最终(流水线+head 批量) vs 28b926c 基线(pynvvc 臂) | **625/625 帧判定一致、9 项指标 |Δ|=0.0、JPG 名单一致、JPG 文件 md5 逐字节一致**(268 帧 head 批量判定无一带偏,无需压线归因) |
| 最终 auto(管道) vs §5 基线 A(auto 臂) | 625/625 帧判定一致、9 项指标 |Δ|=0.0、JPG 名单一致 |
| pynvvc vs 管道的已知 9 帧压线翻转 | 与 §7.3 完全相同,本轮零新增 |

回退链与生命周期冒烟(`post2_fallback_smoke.py`/`post2_abort_reuse.py`):
- 建帧源即失败 / 解码中途断流(producer 线程内抛 `_PynvvcDecodeError`)→ 异常经 `read()` 原样抛回主线程 → WARNING + 回退 FFmpeg 管道重跑,结果与直接 `--decode auto` **逐行一致**;
- 消费者中途异常(模拟 Ctrl-C)→ producer 线程 ≤0.25s 退出、底层帧源 `close()` 被调用(无 NVDEC 会话泄漏),异常退出后同进程立即重建引擎(0.31s)/重新解码(15.7ms/帧)/完整重跑均正常;
- `--decode ffmpeg`(落盘旧路径)、`--dedup`(128 段/压缩 1.36x,与 §7.4.4 一致)、`--image-dir` 模式均正常。

### 9.6 新瓶颈分析与下一步建议

解码/推理流水线化后,瓶颈**没有**如预期移到 max(解码, 推理)——线程级重叠被 GIL 锁死,实际仍是"解码 10.2s → 推理 5.1s"串行;但两段占比已从 58%/33% 变为 63%/32%(总盘子变小)。按收益排序:

1. **子进程解码器(预期最大,~1.4x)**:把 pynvvc 解码挪进子进程,`cvtColor(dst=共享内存槽)` 直写共享内存环(4 槽×25MB),父进程零拷贝取帧。子进程无 GIL 之争,墙钟≈max(子进程 ~15ms/帧, 推理 ~8.3ms/帧)≈9.4s+开销 → 不含探测段 ~16.1s→**~11s**。本轮已把帧源工厂化/异常传播/回退链铺好,`_PipelinedSource` 换个 make_src 即可接上。注意子进程别 import `face_det`(pycuda.autoinit 会占 GPU 显存),解码核心宜放独立小模块。
2. **gaze 批量化(~0.7s)**:gaze 也是逐帧 batch=1+每帧 sync(1.78s/308 帧=5.8ms),但引擎是**固定 batch=1**,需重建动态 batch 引擎并做与 head 同款的判定等价验证。
3. **RGB→BGR 消掉(~1.6s,风险高)**:2.5ms/帧的宿主色序交换,要么 device 输出+GPU 转换(§8.3 跨 context 风险),要么让 letterbox kernel 直吃 RGB(要动 pose68/gaze/head/imwrite 四个 BGR 消费端)。
4. **不再建议**:head 输入尺寸缩到 640 以下(已是训练尺寸,伤小人头=改语义);线程级流水线继续调参(实测上限就是 1~3%)。

### 9.7 本轮新增工程注意点

1. **breakdown 口径**:流水线开时 `decode`=后台线程解码耗时、`read`=主线程等队列时间,两者重叠,段和<墙钟;对账请用"不含探测段"的墙钟。
2. **`--pipeline-depth`(默认 16)只是背压上限**,实测稳态队列 ~2 帧(≈50MB);4K 视频内存宽裕,无需调小。
3. 流水线只对「边解边读」的帧源(pynvvc/管道)生效;`--decode ffmpeg` 落盘路径解码在读取前已整体完成,保持旧行为(一行未改)。
4. dedup 与流水线可叠加(已冒烟),但 dedup 路径的 head 仍走逐帧 `_gate_chain`(batch=1),dedup 场景的 head 收益要等 dedup 路径也接两阶段批量化。
5. 本轮基准 e2e 噪声带 ±5~15%(受系统负载影响),分段耗时(尤其 head/decode)比墙钟稳;A/B 结论一律以同会话交替跑为准。
