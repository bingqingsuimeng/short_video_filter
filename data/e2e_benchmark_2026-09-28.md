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

## 10. 显存直通零拷贝集成(2026-09-29 第三轮,--decode pynvvc-gpu)

### 10.0 结论摘要

新增 `--decode pynvvc-gpu` 显存直通零拷贝臂:NVDEC device 帧(RGB)不落宿主,
自研 CUDA kernel 在显存内逐位复刻宿主预处理(INTER_AREA 768 下采样、head 640
定点 letterbox),再进**原版** preproc_kernel + **原版**引擎零拷贝推理。判定流水
与宿主 pynvvc 臂逐行同逻辑。

- **等价性:625/625 帧逐位一致**(9 项指标 |Δ|=0.0、JPG 名单一致、257 张 JPG
  md5 逐字节一致)——连 §7.3 的 9 帧压线翻转都没有(结构性一致:送入引擎的
  blob 与宿主臂逐字节相同,无阈值可漂)。
- **e2e 16.30s → 13.42s(1.21x)**,逐视频 1.14~1.28x;解码+读帧段 21.3s(重叠假象)
  → **8.60s(read=0)**;detect 1.37→0.39s(INTER_AREA 上 GPU+免 H2D)、
  head 1.93→1.06s(letterbox 上 GPU+免 H2D)。
- 主路径(SCRFD/head 的 blob)**无任何全帧 H2D**;宿主拷贝只剩「人脸帧落地 D2H」
  (pose68/gaze/JPG 三个宿主消费者共同需要,JPG 落盘必须有宿主帧,不可避免)。
- 不引入任何翻转引擎;旧三臂(管道/pynvvc 宿主/落盘)一行未改,新代码复跑宿主
  pynvvc 臂与 §9 基线 625 帧逐位一致。

### 10.1 改动内容

| 文件 | 内容 |
|---|---|
| `src/gpu_decode_kernels.cu`(新)+ `.cubin` | 5 个 kernel:`area_fast_u8`(整数比 INTER_AREA,如 2160→432=5x,精确整数累加+RNE)、`area_generic_u8`(表驱动 INTER_AREA,如 1900x3378,复刻 computeResizeAreaTab double 建表+浮点累加次序,显式 `__fmul_rn/__fadd_rn` 禁 FMA)、`head_letterbox_u8`(640 居中 pad=114 定点 INTER_LINEAR,直写 fp32 blob)、`rgb2bgr_inplace_u8`、`copy_u8`。`nvcc -cubin -arch=sm_86 -fmad=false`;注释必须纯 ASCII(MSVC cp936 吞 UTF-8 注释换行)。**preproc_kernel.cu 未动** |
| `src/gpu_decode.py`(新) | `_area_tab/area_path/AreaTables`(宿主 double 建表上传)、`GpuKernels`(ctypes cubin 加载/发射,与 face_det 同款)、`DeviceFramePool`(连续槽显存池,容量 chunk+8,块放不下整块回槽 0,保证块内物理连续)、`_PynvvcGpuSource`(useDeviceMemory=True 解码器与推理**共享同一条 pycuda 流**保序;选帧规则与 `_PynvvcSource` 完全一致;D2D=`memcpy_dtod_async`;EOF/断流守卫同款)、`scrfd_block_detect`/`head_slots_detect`(块级 device 直通) |
| `filter_video.py` | `--decode` choices 加 `pynvvc-gpu`;`process()` 链 `("pynvvc-gpu","pynvvc","pipe","file")` + `--dedup` 时显式 WARNING 回退宿主 pynvvc;`__init__` 记录 `engine_path/_head_engine_path/_gpu`(3 行,无行为变化);`_ensure_gpu_arm`(懒建:自研流+cubin+显存池+**同引擎文件另建** device 版 SCRFD/head 实例,绑自研流)+ `_run_pass_gpu`(与 `_run_pass` 非 dedup 分支逐行同逻辑)。旧臂路径零修改 |

实验铺路(`_test/post3_e1_resample_ab.py`、`post3_e2_resample_ref.py`、
`post3_e3_kernel_ab.py`):E1 证明直改 letterbox 输入(链 B/C)blob 有
~3.5 LSB 等效偏差、score 会漂 → 否决;定案「option B」= GPU 上逐位复刻
cv2 INTER_AREA + 原引擎。E2 在 32 张真实 GT 帧上建立三条 Python 参考配方
(与 cv2 逐字节一致)作 kernel 规格;E3 验证 kernel/共享流 D2D 保序/SCRFD
端到端与宿主臂逐位一致(4 视频全 PASS)。

### 10.2 数据流与拷贝点清单(零拷贝验证)

```
NVDEC RGB device 帧(解码器写帧与后续 kernel 同一条 CUDA 流,自动保序)
  → [拷贝1] D2D 整帧入显存池(设备侧↔设备侧,不过 PCIe,不过宿主)
  → [0拷贝] area_fast/generic kernel 池内直读 → 768 BGR 小图(显存)
  → [0拷贝] 现役 preproc_kernel letterbox 768→640 blob(显存,原核未改)
  → [0拷贝] SCRFD 引擎 set_tensor_address 就地推理 → 输出 D2H(9 小张量)
  → 人脸帧: rgb2bgr_inplace kernel(池内) → [拷贝2] 同步 D2H 整帧落地(宿主)
  → pose68 / gaze(宿主裁脸) / cv2.imwrite 用宿主 BGR 帧(与宿主臂同输入)
  → head 帧: head_letterbox kernel 池内直读(已 BGR)→ [0拷贝] 写 head.in_dev
    fp32 blob → 原宿主 head 引擎就地执行 → 输出 D2H(5x8400 小张量)
```

| # | 拷贝 | 方向 | 范围 | 说明 |
|---|---|---|---|---|
| 1 | D2D 池拷贝 | 设备→设备 | 全部选中帧(625) | 唯一整帧设备侧拷贝;显存带宽 ~330GB/s,4K 帧仅 ~0.15ms |
| 2 | 人脸帧落地 D2H | 设备→宿主 | 仅人脸帧(508/625:006 89/099 96/154 189/214 134) | 由 pose68+gaze+imwrite 三个宿主消费者共同需要;JPG 落盘必须有宿主帧,不可避免;非 gaze 单独引入 |
| 3 | TRT 输出 D2H | 设备→宿主 | 每批 9 小张量 / 5x8400 | 与宿主臂同 |
| 4 | pose68/gaze 脸部裁剪 H2D | 宿主→设备 | 仅人脸帧 ROI | 与宿主臂完全相同(宿主消费者,不在本臂改动范围) |
| — | SCRFD/head 图像数据 H2D | — | **无** | 宿主臂此处有 768 小图/640 blob 的 H2D,本臂全部消除 |

显存占用(4K):池 72 槽 x 24.9MB ≈ 1.75GB + 768 小图 64MB + 引擎缓冲,
合计 < 2.2GB(RTX 3060 12GB 内充裕)。

### 10.3 等价性(625 帧,全部逐位一致;`equiv_zerocopy_vs_final_pnv.json`)

| 对比 | 结果 |
|---|---|
| device 臂(zerocopy) vs 宿主 pynvvc 臂(§9 final_pnv) | **625/625 帧判定一致、9 项指标 |Δ|=0.0、JPG 名单一致、257 张 JPG md5 逐字节一致;§7.3 的 9 帧压线帧全部无翻转(零漂移,结构性等价)** |
| 旧臂未受影响验证:新代码宿主 pynvvc 臂复跑(hostcheck) vs §9 基线 | 625/625 帧判定一致、9 项指标 |Δ|=0.0、JPG 名单一致( additions 零影响) |
| 回退链 F1:device 臂初始化失败(monkeypatch) | WARNING + 回退宿主 pynvvc 重跑,006 结果 105/0 与直接 pynvvc 一致(`_test/post3_fallback_check.py`) |
| 回退链 F2:表外帧率假 fps=24 | pynvvc-gpu → pynvvc → 管道 逐级 WARNING,最终 pipe 结果 105/0 一致 |
| F3:`--decode pynvvc-gpu --dedup`(CLI) | 显式 WARNING「不支持 --dedup,回退 pynvvc 宿主帧源」,dedup 正常产出 |

### 10.4 耗时对比(r1 口径:单进程一臂 4 视频、引擎加载单列)

| 口径 | §9 最终(宿主 pynvvc,final_pnv) | **本轮:显存直通(zerocopy)** | 变化 |
|---|---|---|---|
| **e2e 全链路** | **16.30s** | **13.42s** | **1.21x** |
| └ 探测段 | 0.10s | 0.10s | — |
| **不含探测段** | 16.12s | **12.96s** | 1.24x |
| decode 段 | 11.56s(与 read 重叠) | **8.60s**(拉帧+D2D+人脸帧落地;read 恒 0) | 见下 |
| read 段 | 9.76s(重叠) | **0.00s**(无宿主全帧读取) | 消除 |
| detect | 1.37s | **0.39s** | 3.5x(INTER_AREA 上 GPU+免 H2D) |
| gaze | 1.78s | 1.54s | 1.16x(同宿主实现,帧来源变化带来的波动) |
| **head** | **1.93s** | **1.06s** | 1.82x(letterbox 上 GPU+免 H2D) |
| imwrite | 0.00s | 0.16s | 异步排空抖动,噪声级 |
| 引擎加载 | 0.56s | 0.57s(device 臂 SCRFD/head 实例在首视频内懒建,含在 006 墙钟内) | — |

- 逐视频 e2e:006 2.79→2.44s(1.14x)/ 099 4.60→3.58s(1.28x)/ 154 5.16→4.33s(1.19x)/ 214 3.74→3.07s(1.22x)。
- 解码+读帧:宿主臂 decode/read 两段重叠后实际 ~11.6s(§9.2 GIL 串行)→ 本臂 **8.60s**(且 read=0):NVDEC 拉帧不变,省掉每帧 D2H+宿主 cvtColor;新增 D2D(0.15ms/帧)与人脸帧落地 D2H(508 帧)远小于省下部分。
- 残余瓶颈:NVDEC 拉帧本身(~6-7s)+ 人脸帧落地 D2H;再往上要动 pose68/gaze/imwrite 的宿主消费语义(§9.6 建议的子进程解码对本臂无收益——宿主已无全帧读取)。

### 10.5 本轮新增工程注意点

1. **共享流保序**:解码器第 3/4 位置参数传 cudaContext/cudaStream(`CreateSimpleDecoder(video,0,ctx.handle,stream.handle,True,...,RGB)`),解码写帧与后续 kernel/D2D 在同一流上按程序序执行,无需显式同步(E3 T0 验证)。
2. **池容量 chunk+8**:一次 `get_batch_frames(8)` 内的批边界溢出选中帧(≤8)不会覆盖本块未消费槽;块尾部放不下整块则整块回槽 0,保证块内**物理连续**(area/head kernel 按连续地址消费)且不跨池尾回绕;单线程消费下旧块先读完再写新块。
3. **不引入翻转引擎**:device INTER_AREA 核直接输出 BGR(oswap=1)与宿主 cv2 结果逐字节一致,后续原核原引擎零改动;head 核 bswap=1 读已换序的池帧。任何 blob 逐字节差异都会表现为判定漂移,E1 的 0.00049 score 漂移路线已被否决。
4. **pycuda 2026.1 精简版**:D2D 用 `drv.memcpy_dtod_async(dst_int, src_int, nbytes, stream)`;无 `drv.zeros/empty`,设备缓冲用 `GPUArray`,指针取 `__cuda_array_interface__["data"][0]`。
5. **两条流水线的互斥**:`--pipeline` 线程重叠对本臂无意义(主循环自带块级批量,decode/detect 天然 GPU 串行保序),`_run_pass_gpu` 忽略该开关(breakdown 里 pipeline=off(dev));`--dedup` 需宿主全帧算清晰度,显式回退宿主臂。
6. **device 版 SCRFD/head 实例与宿主实例并存**:同引擎文件各建一份 execution context,绑到自研流;回退到宿主臂时宿主实例原样可用。同一进程两份引擎+显存池 ~2.2GB,12GB 卡安全。
7. 表外帧率(非 60/30fps)、元数据尺寸不符、解码中途断流等异常全部抛 `_PynvvcDecodeError`,复用 process() 现有逐级回退链(pynvvc→管道→落盘),语义与宿主 pynvvc 臂一致。

## 11. 解码∥推理重叠(2026-09-29 第四轮,--overlap)

### 11.0 结论摘要

新增 `--overlap auto|on|off`(auto=on):重叠臂用 ThreadedDecoder 的 C++ 内部
解码线程(NVDEC ASIC)与主线程推理在**同一条 pycuda 流**上块级重叠
(producer-consumer:信号量 2 token + 有界队列 4,pop 时释放 token)。解码在
NVDEC、推理在 SM,设备侧天然是两块硬件;探针证明共享一条流不互相限速。

- **等价性:625/625 帧逐位一致**(9 项指标 |Δ|=0.0、JPG 名单一致、257 张 JPG
  md5 逐字节一致:099 57/154 160/214 40/006 0)。
- **e2e 13.42s → 10.30~10.44s(1.28~1.30x)**,逐视频 1.04~1.46x;
  **不含探测段 9.92~9.94s**(首破 10s);`--overlap off` 臂 13.31/13.30s,
  与 A 轮一致(开关语义正确,判定逐位同基线)。
- 采纳的三项改动:①pinned slab 落地(async D2H,2 套轮换);②先建帧源再建
  AreaTables(4K 表构建 ~0.3-0.5s 纯 CPU 与解码器预热重叠);③首块 16 削 ramp。
- **2-stage 消费(判定链延后一块)两次实现均否决回退**:落地 D2H train +
  producer D2D + 解码器写帧三者并发造成 CE 带宽拥塞,pose/gates 膨胀 30~60%,
  超过隐藏收益(154 内部 2.874 → 4.196/3.819s)。
- **子进程 CUDA IPC 路线无需启动**:E1/E1b 证明单流共享已不受限, ladder 第 2 级跳过。

### 11.1 改动内容

| 文件 | 内容 |
|---|---|
| `src/gpu_decode.py` | `_create_threaded_decoder` 工厂(独立函数便于测试注入失败;`CreateThreadedDecoder(video,16,0,ctx.handle,stream.handle,True,0,0,0,0,RGB)`,与 SimpleDecoder 同风格位置参数);`_OverlappedGpuSource`(producer 线程:内层 `_PynvvcGpuSource(decoder="threaded")` 按块 `read_block` → 有界队列;主线程构建内层源——CUDA context 归属正确,初始化失败同步抛出 → 串行臂回退;`first_chunk=16` 首块小批量削 ramp;块大小由 producer 决定,消费端 `read_block(n)` 忽略 n) |
| `filter_video.py` | `--overlap` 参数(auto/on/off);`_run_pass_gpu`:`use_overlap` 分支(2 组槽池/pinned slab 落地/提前建源),回退 WARNING 文案;串行臂代码路径不变 |

### 11.2 探针数据(全部 `_test/post4_*`,不猜)

| 探针 | 结论 |
|---|---|
| E1 `post4_probe_streams.py`(154,4K):solo 2.28 / 同流压 16MB D2D 载荷 2.57 / 异流 2.57 ms/源帧 | 同流/异流 CE 负载都不限速解码 → 「写帧被同流推理拖慢」假设否定 |
| E1b `post4_probe_kernelload.py`(099,4K 全尺寸 D2D):none 2.13 / 同流 SM kernel 2.17 / 异流 2.17 ms/源帧 | SM kernel 负载同样不限速 → 解码速率 = NVDEC 主导;**双流+event 修复无必要,子进程 IPC 路线无必要** |
| 时间线 `post4_timeline.py`(154 逐块):可分页落地(sync `memcpy_dtoh` + 逐帧 `np.empty` 首触缺页)缺口 ~390-430ms/块 | 4K 帧 24.9MB 落地是隐藏大头;154 是 4 视频中唯一 consumer-bound(006/099/214 已在 NVDEC 地板 2.13-2.28ms/源帧) |
| 2-stage 回归时间线(oovl_on5/6):154 内部 2.874 → 4.196(v1 显式延后释放)/ 3.819(v2 3 组槽+pop 释放);pose 94→153-221ms、gates 316→372-560ms、producer 解码 434→569ms | CE 拥塞证据链完整 → 判定链内联(ovl_on4)为最终形态 |

### 11.3 等价性(625 帧,全部逐位一致)

| 对比 | 结果 |
|---|---|
| 重叠臂 on(`run_ovl_on_final`)vs A 轮 zerocopy 基线 | **625/625 帧判定一致、9 项指标 |Δ|=0.0、JPG md5 逐字节一致(57+160+40+0)** |
| `--overlap off`(`run_ovl_off`)vs 基线 | 同上逐位一致;13.44s ≈ A 轮 13.42(关闭即旧行为) |
| 同会话 on/off 交替 ×2(`post4_ab.json`) | 4 轮判定全部逐位一致;on 10.36/10.30s、off 13.31/13.30s |
| 回退冒烟(`post4_fallback_check.py`) | monkeypatch ThreadedDecoder 初始化失败 → WARNING「回退 A 轮串行臂」→ 串行臂跑 154,判定/JPG 逐位同基线;假 fps=24 标定外 → 链式回退(pynvvc→管道)正常完成 |
| 中断/泄漏(`post4_abort_leak.py`) | 第 3 块注入 KeyboardInterrupt → 异常向上传播、`finally: src.close()+ex.shutdown` 不死锁、进程 3.3s 自行退出(exit 130);taskkill /F 强杀后 nvidia-smi 计算进程表无 python 残留、显存回落桌面基线(557MiB/12288MiB) |
| 旧臂未受影响(`post4_oldarms_check.py`) | 宿主 pynvvc 臂 4 视频复跑 vs A 轮 `run_hostcheck` 基线:625/625 逐位一致 |

### 11.4 耗时对比(r1 口径:单进程一臂 4 视频、引擎加载单列)

| 口径 | §10 zerocopy(A 轮) | **本轮 overlap on** | 变化 |
|---|---|---|---|
| **e2e 全链路** | **13.42s** | **10.30~10.44s** | **1.28~1.30x** |
| └ 探测段 | 0.10s | 0.10s | — |
| **不含探测段** | 12.96s | **9.92~9.94s** | 1.30x |
| decode 段 | 8.60s | 8.60s(未重叠工作量口径;其中 ~1.5-2.5s 与宿主判定链重叠) | 见下 |
| detect | 0.39s | 0.52~0.56s | 重叠后波动 |
| gaze | 1.54s | 1.70~1.74s | 同上 |
| head | 1.06s | 1.08s | — |
| imwrite | 0.16s | 0.12~0.16s | 噪声级 |

- 逐视频 e2e:006 2.44→2.35s(1.04x)/ 099 3.58→2.72s(1.32x)/
  154 4.33→2.96s(1.46x)/ 214 3.07→2.41s(1.27x)。
- 增益来源:①解码与宿主判定链(pose/gaze/head/JPG)块级重叠(核心);
  ②pinned 落地把落地缺口 390-430→~200ms/块;③先建源削 ramp ~0.3-0.5s/视频
  + 首块 16 让消费端提前 ~0.4s 启动。
- **距离解码地板**:006/099/214 的 wall ≈ 源帧数 × 2.13-2.28ms + 尾部,
  已贴 NVDEC 地板;154 剩余 ~0.3s 为 pose/gaze/head 宿主链(60fps stride 3
  人脸密度最高)。再往下只剩动宿主消费语义(3 条已否决:2-stage CE 拥塞、
  双流+event 无收益、IPC 无必要)或 NV12 半宽写帧(等价风险大,未启动)。

### 11.5 本轮新增工程注意点

1. **槽安全(2 组轮换)**:信号量 2 token,pop 时释放上一个 token → producer
   领先 ≤1 块;D2D(j) 覆写第 j%2 组上一用户是块 j-2,其 head 已在第 j-1 轮
   迭代内同步完成,流上 D2D 必然排在其后,无覆写竞态(与池容量 2×(chunk+8)
   自洽;串行臂同池容量连续游标两块驻留)。
2. **pinned slab 复用必须等 JPG futures**:`_emit` 把宿主帧引用交给 imwrite
   线程池,futures 视频结束才统一 wait —— 两套 slab 轮换,覆写第 j%2 套前
   先 `wait(land_futs[set_i])`(块 k 的 JPG futures 在块尾登记)。
3. **落地同步点**:`memcpy_dtoh_async` 入队后一次 `stream.synchronize()`,
   替代逐帧同步 D2H —— 宿主判定链在此与 GPU 前后级会合,是逐块必需的数据
   依赖;pinned 分配失败自动回退可分页逐帧路径(逐字节同)。
4. **2-stage 否决教训**:「落地 D2H(CE)与判定(pose/gaze host 链)并行」
   理论上成立,实测 CE 带宽拥塞(落地 train + producer D2D + 解码器写帧
   三者并发)让双方都慢 30~60%;首块 16 无脸帧时判定窗口更小,ramp 反而
   恶化。吞吐优化的并发度上限受 CE/SM/带宽三方约束,改结构前先测带宽。
5. **producer 线程退出语义**:`close()` 置 stop 事件;有界 put 超时轮询
   stop;内层源由主线程构建/异常同步抛 —— Ctrl-C 时 `finally` 关源不阻塞
   (泄漏检查 3.3s 自行退出),daemon 线程随进程结束。
6. **首块 16 与判定无关**:块大小只影响重叠窗口,不影响送入引擎的 blob
   序列(逐块分批边界不变,FP16 引擎批组合不变性在此再次成立),等价性
   全 PASS 证实。

## 12. 判定链减负与到墙收尾(2026-09-29 第五轮,gaze 批量)

### 12.0 结论摘要

时间线剖析定案(§12.1):10.3s 墙钟里 producer 解码 8.0s(NVDEC 地板 ~7.0
+ 预热/交接),consumer 已大部分隐藏 —— 唯一值得动的是 gaze 段(1.61s,
consumer 最大单项)。**gaze 批量化接入:固定 batch=1 引擎流水化 enqueue +
线程池预处理,`estimate_batch` 块内一次调用;判定逐位一致**。动态 batch
引擎(`--dyn`,min1/opt16/max32)按数据否决:0.09° 级批组合漂移必破坏
「9 项指标 |Δ|=0.0」硬门槛(§12.2)。

- **等价性:on/off 双臂 × 多轮全部 625/625 帧判定逐位一致、9 项指标
  |Δ|=0.0、JPG md5 逐字节**(§12.7)。
- **e2e 10.30~10.44 → 9.90~10.00s(1.03~1.05x);不含探测 9.92~9.94 →
  9.53~9.58s**;gaze 段 1.70~1.74 → 0.87~0.89s(目标 ≤1.1 达成);
  `--overlap off` 臂 13.44 → 12.57~12.63s(批量 gaze 在串行臂全额兑现)。
- **流水深度 2→3 实测无收益**(非首视频变化 ≤0.04s=噪声,首视频多付
  ~0.55s 一次性分配)→ 否决回退(§12.5)。
- **剩余差距物理归因到墙**:producer 解码 8.6s ≈ NVDEC 纯解码 7.4~7.6
  (3171 源帧 × 2.2~2.8ms,含 154 4K)+ 首视频预热 ~0.45 + 逐块 D2D/交接
  ~0.6~0.8;consumer 4.3s 仅暴露 ~1.3s(各视频首块 ramp + 末块判定尾)。
  再往下只剩动解码侧(NV12 半宽、源帧跳解)—— 已否决/超范围(§12.6)。

### 12.1 时间线剖析(`_test/post5_timeline.py`,on 臂 4 视频逐块埋点)

改造前(B 轮末,逐帧 gaze)/ 改造后(C 轮批量)合计(4 视频,s):

| 段 | 改造前 | 改造后 | 说明 |
|---|---|---|---|
| **wall** | **10.21** | **9.90** | 单 VideoFilter 跨视频复用口径 |
| pop 等待(会合) | 4.61 | 5.11 | consumer 等 producer,producer-bound 实证 |
| producer 解码 | 8.00 | 8.63 | NVDEC 拉帧+D2D(后值含 154 与批量 gaze 的资源竞争) |
| detect(SCRFD) | 0.51 | 0.48 | |
| land(swap+D2H) | 0.90 | 0.91 | |
| pose68 | 0.85 | 0.83 | |
| judge(_judge+低头复核) | 0.01 | 0.00 | |
| **gaze** | **1.61** | **0.87** | 批量化目标段(−0.74) |
| head | 1.09 | 1.09 | B 轮已批量化 |
| JPG futures 排空 | 0.13 | 0.13 | |
| CSV/写盘 | ~0.01 | ~0.01 | 可忽略 → 任务「非 GPU 段并行化」仅剩 pose CPU 前处理(估 <0.2s),不做 |

逐块视角:006 consumer 每块仅 0.03~0.09s(kept=0,几乎全 too_small 不落
地),099/214 每块 pop 等待 0.5~0.7s —— 消费端在这些视频早已空转等解码;
154 是唯一 consumer-bound 视频,gaze 批量收益在其全额兑现(§12.4)。

### 12.2 gaze 批量等价探针(`_test/post5_gaze_probe.py`,数据
`runs/post5_gaze_equiv.json`)

308 个真实管线裁剪(4 视频 on 臂收集,≥52 要求的 6 倍)。基准 = 旧固定
FP16 引擎逐帧 `_trt_pass`;attention 判定阈值 = 管线实际值
(|pitch|>15° 或 |yaw|>20°):

| 后端 | 逐位 | logits\|Δ\|max | \|Δyaw\|max | \|Δpitch\|max | attention 翻转 |
|---|---|---|---|---|---|
| B1 dyn 引擎 bs=1 | 否 | 2.734e-02 | 0.0946° | 0.0936° | 0 |
| B2 dyn 引擎 分组 bs≤32 | 否 | =B1 | =B1 | =B1 | 0 |
| **C1 固定引擎 流水 enqueue** | **是(全 0)** | 0 | 0 | 0 | **0** |
| D FP32 引擎(噪声基线) | 否 | 4.608e-02 | 0.2016° | — | 0 |

- dyn 引擎的漂移量级与 FP16-vs-FP32 噪声基线同阶(批组合 tiling 所致),
  且 CSV gaze 列打印 3 位小数 → **0.09° 级漂移必破坏 |Δ|=0.0 硬门槛,
  数据驱动否决**;产物 `models/gaze/resnet34_gaze_dyn_fp16.engine` 保留,
  默认引擎不变(`resnet34_gaze_fp16.engine`)。
- C1 同引擎同 blob 同流顺序执行,无批量算子 → 逐位一致 → **采纳接入**。
- `estimate_batch` 数值等价(mismatch=0)与计时:逐帧 estimate 4.95
  ms/帧 → 固定引擎批量 workers=0 4.71 / **workers=8 2.55 ms/帧**(1.9x)。

### 12.3 改动内容

| 文件 | 内容 |
|---|---|
| `src/gaze_resnet.py` | `estimate_batch(items, workers)`(线程池并行 `_prep` → 按引擎分组 → 逐帧 `_decode` 复用);固定引擎后端 `_infer_group_pipelined`(逐帧 H2D→execute→D2H 连续入队,`_PIPE_GROUP=32` 帧组末一次 sync;pinned slab `_pipe_slabs` 懒建,分配失败回退可分页,逐字节同);动态引擎后端 `_infer_group_dynamic`(否决未接入,代码保留);引擎 batch 维检测(动态/-1 vs 固定 1,非 1 且非动态报错) |
| `filter_video.py` | `_gate_pre` 拆为 `_gate_judge`(阶段A)+`_gate_gaze`(阶段B 逐帧)+`_gaze_apply`(判定应用,单帧/批量共用)——host 臂/dedup 路径行为不变;`_run_pass_gpu` 两阶段循环改三阶段:**阶段A** 逐帧 judge(收集 keep 单脸候选)→ **阶段B** 块内一次 `gaze.estimate_batch` + 按帧序 `_gaze_apply`(dy_all 顺序=帧序)→ **阶段C** 按帧序 head 分发/emit(emit 调用时序变化,按键控无观测影响,§12.8-1) |
| `scripts/build_gaze_engine.py` | `--dyn`:输入 batch 维 reshape 动态 + profile min=1/opt=16/max=32(图 GAP→Flatten→Gemm 天然 batch 安全) |
| `src/gpu_decode.py` | 槽组数提为常量 `OVL_SETS`(实验后维持 2,§12.5) |
| `_test/` | `post5_timeline.py`(逐块埋点)、`post5_gaze_probe.py`(等价探针) |

### 12.4 耗时对比(r1 口径:单进程一臂 4 视频、引擎加载单列)

| 口径 | §10 zerocopy(A 轮) | §11 overlap on(B 轮) | **本轮 C(gaze 批量)** | C vs B |
|---|---|---|---|---|
| **e2e 全链路** | 13.42s | 10.30~10.44s | **9.90~10.00s** | **1.03~1.05x** |
| └ 探测段 | 0.10s | 0.10s | 0.09~0.10s | — |
| **不含探测段** | 12.96s | 9.92~9.94s | **9.53~9.58s** | 1.04x |
| gaze 段 | 1.54s | 1.70~1.74s | **0.87~0.89s** | 1.9~2.0x |
| head 段 | 1.06s | 1.08s | 1.08~1.10s | — |
| `--overlap off` 臂 e2e | — | 13.31~13.44s | **12.57~12.63s** | 1.06x(批量 gaze 串行臂全额兑现 −0.7) |

- 累计加速比:**16.30 → 13.42(1.21x)→ 10.37(1.29x)→ 9.95(1.04x),
  共 1.64x**。
- on 臂逐视频(run_ovl_on_final → run_c_d2_on):006 2.35→2.28 /
  099 2.72→2.72 / **154 2.97→2.64** / 214 2.42→2.30 —— 154(consumer-bound)
  兑现最多;006/099/214 已在 NVDEC 地板,gaze 减负被地板吸收。
- 同会话 on/off 交替 ×2(ab):on 10.00/9.90s、off 12.57/12.63s,波动
  ≤0.1s,可复现。

### 12.5 流水深度 2→3 实验(否决)

槽组/信号量/pool/land slab 四处同步扩到 3(`OVL_SETS` 常量泛化,覆写安全
论证不变:D2D(j) 同组上一用户是块 j−N ≤ j−2)。实测(run_c_d3_on):
**合计 10.57s —— 006 2.82(多付 ~0.55s 一次性池/slab 分配),099/154/214
变化 ≤0.04s=噪声**。producer 并未被 token 饿住(信号量 2 token 下 producer
可领先 1 块 ≈ 0.5~0.9s,已覆盖块解码时长),回退 OVL_SETS=2。

### 12.6 距 7.0s 地板的剩余构成(9.9s → 物理归因)

| 构成 | 量级 | 依据 |
|---|---|---|
| NVDEC 纯解码 | ~7.4~7.6s | 3171 源帧 × 2.13~2.28ms(1080p)+ 154 4K ~2.8ms/源帧 |
| 首视频 NVDEC 预热 | ~0.45s | 006 blk0 662ms/96 源帧(3x 地板),一次性 |
| 逐块 D2D/队列交接 | ~0.6~0.8s | producer 逐块 prod_end 与源帧×地板之差 |
| consumer 暴露(首块 ramp + 末块判定尾) | ~1.0~1.3s | 各视频解码器会话初始化 + 末块 gaze/head 在 producer EOF 后 |
| CSV/排空 | ~0.15s | |

- consumer 合计 4.3s 中 ~3.0s 已隐藏(pop 等待 5.10s 实证 producer-bound);
  gaze 批量 −0.86s 只兑现 −0.4~−0.5s,其余被 NVDEC 地板吸收。
- 154 producer 解码 1.62→2.17s(+0.55):疑与批量 gaze 流水 enqueue 的持续
  H2D/D2H/SM 负载相关(E1b 仅在 1080p 证伪过 SM 限速,4K 未复测);即便
  如此 154 墙钟仍 −0.33s。后续若再压 154,先复测 4K 下的负载-解码曲线。
- 再往下只剩动解码侧:NV12 半宽写帧(等价风险大,§11 已列为未启动)、
  源帧跳解(采样语义改变,超范围)—— 按任务书「不为凑数字引入风险」
  到墙收尾。

### 12.7 等价性与稳定性(全部 PASS)

| 检查 | 结果 |
|---|---|
| on 臂 vs `run_ovl_on_final` 基线 | 625/625 帧判定一致、9 项指标 \|Δ\|=0.0、JPG md5 逐字节(57+160+40+0)—— c_gz154/c_gz_on/c_d2_on 三次复跑全过 |
| `--overlap off` 臂 vs `run_ovl_off` 基线 | 同上逐位一致(c_gz_off + ab 两轮) |
| 同会话 on/off 交替 ×2(`post4_ab.json` 覆写) | 4 轮 × 4 视频全部逐位一致 |
| host pynvvc 臂 vs `run_hostcheck` 基线(`post4_oldarms_check.py`) | 625/625 逐位一致 —— 同时证明 `_run_pass`/`_gate_pre` 重构零行为变化(ffmpeg 管道臂共用该路径与帧源无关的判定链) |
| 回退冒烟(`post4_fallback_check.py`) | ThreadedDecoder 初始化失败 → WARNING 回退串行臂;假 fps 标定外 → 链式回退正常(FALLBACK_PASS) |
| Ctrl-C/强杀(`post4_abort_leak.py`) | 注入 KeyboardInterrupt exit 130 自行退出;强杀后无 python 计算进程残留、显存回落 512MiB 桌面基线(ABORT_LEAK_PASS) |
| 006 的 maxdiff 缺 gaze/nheads 列 | 全帧 None 对 None 按口径跳过,非缺失 |

### 12.8 本轮新增工程注意点

1. **emit 顺序无关性**:批量 gaze 后 `_emit` 调用时序变化(阶段A 先处理
   非候选帧,阶段C 再处理候选),但计数器、JPG 文件名、report 行序、
   `land_futs` 登记全部按帧号/索引键控,输出字节不变 —— 由全量 md5 比对
   证实。若未来加入顺序敏感输出(如流式写 CSV)需回到逐帧序。
2. **dyn 引擎否决要留数据**:CSV gaze 列 3 位小数 → 任何 >1e-3° 漂移都
   会破坏逐位一致;FP16 动态批组合漂移 0.09° 是同精度噪声基线量级,不是
   bug。固定引擎流水化(单流顺序执行、无批量算子)才是零漂移路线。
3. **探针 slab 视图陷阱**:pipelined 后端宿主缓冲按组复用,探针比较必须
   sync 后立即 `.copy()`,否则下一组 D2H 覆写造成假性不一致(生产路径组内
   立即 decode 无此问题)。
4. **`_run_pass_gpu` 里 `g` 是 GPU 资源字典**:新增循环变量避开 `g`
   (本轮踩过:批量 gaze 循环用 `g` 遮蔽 → `g["head"]` TypeError)。
5. **OVL_SETS 泛化四处同步**:信号量 token、slot_base 取模、pool 容量、
   land slab 套数(含 `g` 缓存列表的扩容兼容);实验否决后常量保留,后续
   调深度只改一处。
6. **批量 gaze 的 t_gaze 口径变化**:breakdown 中 gaze 现在按块内整次
   `estimate_batch` 计(含线程池预处理),与逐帧口径直接可比(同段同工作)。

## 13. nsys 系统级 profiling(2026-09-29 第六轮,Nsight Systems 2026.5.1)

### 13.0 结论摘要

用 nsys 对现役配置(`--decode pynvvc-gpu --overlap auto`,e2e 9.9s/4 视频)做系统级
剖析,四问四答(证据全部来自 recipe 输出表,见 13.2 工具链限制):

| 问题 | 实测结论 |
|---|---|
| ① NVDEC 解码总时长 vs 2.13~2.28ms/帧估计 | **稳态解码节奏 1.68ms/源帧**(4K10bit,周期 8.42ms/5 帧:6.3+0.26+1.6+0.26ms),其中 pynvvc 的 SM 转换 0.31ms/帧,NVDEC ASIC ~1.37ms/帧;原估计偏保守,它混入了共享流串行拉伸与预热 |
| ② SM 空闲构成 | SM 忙仅 **1.63s/9.9s(16.5%)**;空闲 ~8.3s = NVDEC ASIC 解码 ~4.3s + 共享流串行拉伸/lockstep ~2.6s + 会话预热 1.07s + 视频过渡/暴露尾 ~0.98s |
| ③ producer/consumer 重叠率 | CPU 线程重叠良好(consumer 4.3s 大部分隐藏,pop 等待 5.1s 不变);但**设备侧单流串行**:消费端 SM 工作以 ~1:1 顶推解码窗口(099 +0.56s、154 +1.65s),真正的并行只发生在 NVDEC ASIC 与 SM 之间 |
| ④ 剩余 >0.3s 可优化项 | 3 项:转换流分流 **0.8~1.5s**(中高风险)、宿主臂引擎懒加载 **0.3~0.4s**(低风险)、pinned slab 预分配 **~0.3s**(低风险);之后到墙(NVDEC ~4.3s + pynvvc 转换结构 1.0s) |

对 §12.6 的复核:9.9s 的主体归因(producer-bound、NVDEC 地板)成立,但
"NVDEC 纯解码 7.4~7.6s"**高估**——正确拆分为 NVDEC ~4.3s + pynvvc SM 转换
~1.0s + 串行拉伸 ~2.6s(详见 13.8)。

### 13.1 采集配置与开销(墙钟膨胀 0%)

```
nsys profile -t cuda,nvtx,nvvideo,python-gil --gpu-video-devices=0
  --sample none --cpuctxsw none -o E:\output\nsys\bench4 -f true
  E:\output\nsys\run_bench.bat E:\output\nsys\bench_vids
  --out-dir E:\output\nsys\nsys_run --decode pynvvc-gpu
```

- Windows 无 osrt 采样(非 admin),故 `--sample none --cpuctxsw none`;
  `--gpu-video-devices=0` 开视频引擎采集(但见 13.2 限制③)。
- 4 视频硬链接到 `E:\output\nsys\bench_vids\`(不动原文件);app 输出经
  ASCII-only `run_bench.bat` 重定向到 `app_nsys.log`。
- **开销对照**:app 内部计时 nsys 下 9.9s == 无剖析基线 ×3 次(9.9s)→ 测量
  窗口内膨胀 0%;nsys 会话墙钟 14.2s vs 裸跑 10.85s(差值 = 注入 + 报告
  生成,在 app 计时之外)。
- **输出等价**:4 份 pose_report CSV + 257 张 JPG 与基线逐字节一致;单视频
  154 冒烟(194 帧)逐位一致。GPU 全程单进程(utilization 表仅 PID 9652)。

### 13.2 工具链障碍与绕行(三个,全部记录在案)

1. **中文 locale 的 nsys 元数据 GBK bug**(产品 bug):报告内
   META_DATA_EXPORT.parquet 时区名是 GBK 字节("中国标准时间"),所有走
   duckdb 的 skill 命令(report-fact / report-query / report-context)在真实
   报告上全部失败;TZ=UTC、--discard-environment=true 均无效。
   **绕行**:分析全部改走 recipe 引擎(原生加载器容忍 GBK),报告副本须放
   C 盘(跨盘 alias 报 WinError 17)。副作用:NVDEC 引擎表/video 事实无法经
   report-fact 读取。
2. **nsys 注入使子进程 stdout 静默变空**:ffprobe exit 0 但输出空(文件重
   定向同样空,-t none 也一样)。**绕行**:probe() 加 SVF_PROBE_CACHE
   门控缓存分支(不设变量时该分支不存在,默认行为逐字节不变;缓存由同机
   真实 probe() 预生成,数值与 §1 一致);此改动在 filter_video.py 工作区
   **未提交**,冒烟 194 帧逐位一致。
3. **无 NVTX runtime DLL**(机器上搜索无结果)→ 未加 NVTX(按任务书"仅在
   必要时"),改用 **kernel 名归因**(自研 kernel 名 + pynvvc 固有
   转换 kernel 名 + TRT 内核),语义等价且零代码侵入。

另:recipe 位置参数必须放 `--` 之后且只接受 live help 里的选项(无
--rows);--filter-time 格式为 start/end 整数 ns;recipe 输出每表上限
100 行(pace 类需配合 count/窗口切分交叉验证)。

### 13.3 GPU 全景:SM/CE 忙闲(证据:cuda_gpu_kern_sum / cuda_gpu_mem_time_sum / gpu_time_util --chunks 200)

设备忙合计 ~2.37s / 9.9s(24%):

| 类别 | 时长 | 明细 |
|---|---|---|
| SM kernels | **1.63s (16.5%)** | pynvvc 转换 **0.99s(61%)** = ConvertP016BLtoP016 3171×155.6µs + YuvToRgbKernel 3171×157.9µs;自研 kernel 0.185s(area_fast_u8 31×39.4ms、rgb2bgr_inplace_u8 508×95.4ms、head_letterbox_u8 268×22.5ms、letterbox_bgr2rgb 41×12.1ms、area_generic_u8 10×15.5ms);TRT 引擎+preproc ~0.45s(permutationKernelPLC3 626×107ms ≈ 625 判定帧各一次等) |
| CE D2D | 96.9ms / 634 次 | = §10.2 拷贝1(625 选中帧 + 9 溢出),14.66GB 对上 |
| CE D2H | 538.1ms / 1637 次 | 508 全帧落地 ≈ 534ms(12.57GB ≈ 508×24.8MB 对上)+ 1129 次小输出拷贝 |
| CE H2D | 89.0ms / 6705 次 | pose/gaze 小块 + 引擎权重(max 114ms) |
| memset | 11.1ms / 848 次 | |

200-chunk 利用率图:解码窗口内 SM duty 仅 10~25%;全程无 >0.2s 的设备全空
段(32-chunk 粗图上的 1.7s"空洞"是低占空比 chunk 被阈值合并的假象)。

### 13.4 NVDEC 解码节奏实测(问题①;证据:cuda_gpu_kern_pace --name YuvToRgbKernel,全 session + 5 个 filter-time 窗口)

YuvToRgbKernel 恰好 3171 次 = 源帧总数:pynvvc 每解一源帧跑一对
ConvertP016BLtoP016(P016→中间格式)+ YuvToRgb(→RGB),逐 launch 间隔即
解码输出节奏:

| 视频 | 分辨率/帧率 | 稳态节奏 | 转换窗口(trace 时轴) | 窗口/帧 |
|---|---|---|---|---|
| 006 | 4K60 | **1.68ms/帧**(周期 8.42ms/5 帧) | 1.956→~3.06s(1.10s) | 1.74ms |
| 099 | 4K60 | 2.23ms/帧(消费端已活跃) | ~3.14→~5.54s(2.40s) | 2.38ms |
| 154 | 4K30 | 3.2~3.9ms/帧,max stall 61ms | 5.606→8.237s(2.63s) | 4.51ms |
| 214 | 1900×3378@60 | 1.57~1.66ms/帧 | 8.98→10.76s(1.78s) | 1.88ms |

- 每帧稳态结构:NVDEC ASIC ~1.37ms + ConvertP016 0.156ms + YuvToRgb
  0.166ms = 1.68ms;006 与 214 节奏几乎相同 → 该档分辨率下解码节奏不受
  像素量影响(NVDEC 管线节拍主导)。
- 006(消费端几乎无工作,gaze/head 均 0)给出**无干扰地板 1.68ms/帧**;
  099/154 的劣化与消费端负载强相关(见 13.5),不是源帧率差异。
- 154 的窗口 2.63s 比地板(583×1.68=0.98s)多 1.65s —— §12.6 的"154 之谜"
  在此定位(13.5/13.8)。

### 13.5 producer/consumer 重叠质量(问题③)

- CPU 侧:producer 等待(decode= 合计 9.25s)覆盖 9.9s 墙钟的绝大部分,
  consumer 的 host 链(detect/judge/gaze/head/imwrite 合计 4.3s)大部分
  隐藏其内 —— 与 §12.6 pop 等待 5.10s 的 producer-bound 结论一致。
- **设备侧是单流串行**(解码器与自研臂共用一条 pycuda 流,§10.5 保序设
  计):pynvvc 的逐帧转换 kernel 与消费端全部 SM kernel(SCRFD/pose/gaze/
  head/自研 kernel)在同一条流上互斥执行。消费负载与解码窗口拉伸:

| 视频 | 消费端 gaze+head+detect(host) | 窗口/帧 vs 地板 1.68ms | 拉伸 |
|---|---|---|---|
| 006 | ~0.1s(kept 0) | 1.74ms | ~0 |
| 099 | 0.64s | 2.38ms | +0.56s |
| 154 | 1.38s | 4.51ms | +1.65s |
| 214 | 0.47s | 1.88ms | +0.13s |

  方向单调一致(量级受 kept 数/JPG/D2H 影响有噪声)。**"重叠"的真实机
  制 = NVDEC ASIC 与 SM 并行;SM 内部的解码路径与推理路径是串行的,消费
  端每 ~1s SM/host 工作换取解码窗口 ~1s 拉伸。**这正是 §12.6 观察到的
  "gaze 批量 −0.86s 只兑现 −0.4~−0.5s"的机理:省下的消费时间一部分被解
  码窗口回吐。

### 13.6 CPU 侧开销(证据:cuda_api_sum / cuda_api_sync;多线程聚合值,不与墙钟直加)

| API | 次数 | 总 CPU 时 | 注 |
|---|---|---|---|
| cuLaunchKernelEx + cuLaunchKernel | 18181+6462 | **2.48s** | 中位数仅 4~9µs,重尾(max 9/119ms)→ 双线程驱动锁争用痕迹,总量大但非单项可治理 |
| **cudaMemcpyToSymbol** | **6342 = 2×3171** | **2.16s** | pynvvc 逐帧转换路径自带,每源帧 2 次(中位 125µs,max 99ms)—— 解码路径 CPU 税,见 13.7-D |
| cuMemcpyDtoHAsync_v2 | 1543 | 1.37s | 中位 6.8µs,max 121ms(重尾个别点) |
| cuStreamSynchronize | 210 | 575ms | 单次 max 72.6ms;§11.2 的同步点符合 |
| cuMemHostAlloc / cuMemFreeHost | 4 / n | **381ms**+141ms | pinned land slab 按分辨率重建(4K 套×2 + 214 分辨率套×2) |
| cuEventSynchronize/Record 等 | — | <60ms | 可忽略 |

单次同步 max 0.71ms(cuda_api_sync)确认无隐藏大同步。

### 13.7 优化机会清单(问题④)

| # | 现象 | 证据 | 预计节省 | 风险 | 触碰 625 帧逐位一致门槛? |
|---|---|---|---|---|---|
| A | 解码转换 kernel 与推理 kernel 同流串行,解码窗口被消费负载拉伸 | 13.5 表(154 +1.65s/099 +0.56s);窗口内 SM duty 仅 10~25% 有并发余量 | **0.8~1.5s** | 中高:转换分流到独立流需逐帧/逐块 event 保序(D2D 必须等该帧转换完成),竞态写错像素是最坏路径 | 可验证:配方一致时输出应逐位不变;必须过 625 帧硬门槛 + 4K 复测 |
| B | 宿主臂 SCRFD/head 引擎双加载:__init__ 急加载(filter_video.py:587/643)+ _ensure_gpu_arm 再建同文件独立实例(:1523/1527) | app 日志 SCRFD/head 各出现两次 engine loaded;预热窗口 0.889→1.956s(1.07s,util 5.4%) | **0.3~0.4s/会话** | 低:宿主臂实例改懒建(回退路径首次用时再建),GPU 臂行为不变 | 无影响(判定逻辑与引擎实例无关) |
| C | pinned land slab 按分辨率重建(4 次 381ms + 释放 141ms) | cuda_api_sum cuMemHostAlloc 4 次 381ms;OVL_SETS=2 × 两档分辨率 | **~0.3s** | 低:按最大分辨率预分配(内存峰值反而更低);需 OVL 槽位几何参数化 | 无影响(纯内存管理) |
| D | pynvvc P016→RGB 结构成本:0.99s SM + 2.16s CPU(13.3/13.6) | kernel/api 表 | 仅改像素管线(NV12/P010 直读 + 自研融合 kernel)才可动,上限 ~1.0s | 极高(§11 已否决同路线) | 会 —— 像素数学改变,本轮不做 |
| E | launch API 聚合 2.48s 的重尾(双线程驱动锁) | cuda_api_sum 中位数 vs max | 无单独治理手段(减 kernel 数/合并已做过) | — | — |

A+B+C 全兑现的期望墙钟 ≈ **7.8~8.5s**;再往下 = NVDEC ASIC ~4.3s +
pynvvc 转换 1.0s + 消费端真实工作,系统级到墙。

### 13.8 对 §12.6 归因的复核结论

§12.6 的框架(producer-bound、consumer 大部分隐藏、预热单列)被 nsys 证实,
两处量化修正:

1. "NVDEC 纯解码 7.4~7.6s"高估。实测拆分:NVDEC ASIC ~4.3s(3171×1.37ms)
   + pynvvc SM 转换 ~1.0s + 共享流串行拉伸 ~2.6s(154/099 为主)+ 预热
   1.07s + 过渡/暴露尾 ~0.98s ≈ 9.9s(闭合,±0.2s)。原 2.13~2.28ms/帧是
   "稳态节奏+拉伸+预热"的混合均值,纯地板是 1.68ms/帧(含转换)。
2. "154 的 +0.55s 疑与批量 gaze 负载相关"证实并放大:154 解码窗口比地板
   多 1.65s,与消费端 1.38s 的 SM/host 负载同源(单流串行顶推);gaze 批
   量化在 154 上净赚 −0.33s 的同时确实拉长了 producer 窗口。若做 13.7-A
   分流,154 是最大受益者。

### 13.9 本轮新增工程注意点

1. **中文 locale 上 nsys 的 duckdb 路径不可用**(GBK 时区名 bug),分析一律
   走 recipe;报告与 skill 工作区必须同盘(C 盘),否则 WinError 17。
2. **nsys 注入会阉割子进程 stdout**(含文件重定向),任何依赖 ffprobe 的
   剖析跑法都要预生成元数据缓存;SVF_PROBE_CACHE 门控分支未提交,留
   用户决定去留。
3. recipe 参数必须放 `--` 后、以 live help 为准;--filter-time 用整数
   start/end(ns);pace 输出每表 100 行上限,长序列用窗口切分 + count
   交叉验证。
4. .bat 包装脚本必须 ASCII-only(GBK 命令报错文本会被收进报告
   StringIds,污染所有 parquet 读取)。
5. YuvToRgbKernel/ConvertP016BLtoP016 的次数 = 源帧数、permutationKernelPLC3
   次数 ≈ 判定帧数 —— 后续 nsys 分析可直接用这三类 kernel 名做免 NVTX 的
   阶段归因。
6. 本轮产物:E:\output\nsys\(bench4.nsys-rep、采集/配方脚本、
   _recipe_tables.txt、_pace_tables.txt、_gpuu200.txt、app_nsys.log);
   判定逻辑/阈值/旧臂零改动;未删除任何 mp4;未提交 commit。

---

## 14. D 轮:流分流与死重清除(2026-09-29)

目标:兑现 §13.7 的 A/B/C 三项。执行顺序 B→C→A;结论先行——**B/C 按预期兑现,
A 机制成立但 §13.5 的收益归因被本轮 trace 证伪(收益≈0)**,r1 口径 e2e
10.05s → 9.70s(同机同会话交替计时,1.036x)。未达 §13.7 期望的 ~8.0-8.5s,
缺口正是 A 项被高估的部分,详见 §14.2/§14.3。

### 14.1 改动清单(全部未提交)

| 项 | 位置 | 内容 |
|---|---|---|
| B | filter_video.py `__init__`(SCRFD 块 / head 块 / 收尾块) | 宿主 SCRFD/head 改 property 懒加载(det :719-728, head :731-754);非 pynvvc-gpu 臂(含 --dedup)在 `__init__` 立即物化(旧行为逐字节同);pynvvc-gpu 臂保持懒建 |
| B | filter_video.py `_ensure_gpu_arm` :1584-1610 | device head 只看 `self._head_on/_head_conf`,不触碰 `self.head`(避免触发宿主无谓构建);device head 构建失败 → WARNING + 闸门全局关闭(与旧宿主失败同语义) |
| B | filter_video.py `_run_pass_gpu` B/C 段 | `self.head is not None` → `g["head"] is not None`(GPU 臂不再触发懒构建;判定语义等价) |
| C | filter_video.py `__init__` 收尾 :697-715 | pinned land slab 一次性预分配:2×(chunk+8)×4K(3840×2160×3)≈3.57GB,仅 `pynvvc-gpu 且非 dedup 且 overlap≠off` 时 |
| C | filter_video.py `_run_pass_gpu` :1666-1692 | slab 获取改三级:已有够大 slab(跨视频复用 grow-only,原语义)→ 吃 `__init__` 预分配 → 原地按需分配;失败 → None → 原可分页逐帧回退路径 |
| A | src/gpu_decode.py `_PynvvcGpuSource` :242-257, :276-281, :336-350 | 新参 `dstream=None`(None→退化为旧单流);解码器(Threaded/Simple)建在 dstream 上;逐帧 D2D 改走 dstream;`read_block` 末尾出口事件护栏:`ev_d.record(dstream)` + `stream.wait_for_event(ev_d)` |
| A | src/gpu_decode.py `_OverlappedGpuSource` :430-446 | `dstream` 透传内层源(producer 线程不变:ctx.push + 解码 + D2D) |
| A | filter_video.py `_ensure_gpu_arm` :1598-1602, `_run_pass_gpu` :1712/:1721 | `g["dstream"] = Stream()` 一次创建跨视频复用;两臂构造处传入 |

### 14.2 实测收益 vs §13.7 估计(诚实口径)

| 项 | §13.7 估计 | 实测(r1 口径) | 差异解释 |
|---|---|---|---|
| B 宿主臂引擎懒加载 | 0.3~0.4s | **r1 口径 ≈0**(宿主引擎加载本就在 run() 墙钟之外);进程 wall −0.30~0.35s(成功走 GPU 臂时 6→4 次引擎加载,host SCRFD/head 全程不建) | 与 §13.7-B 的"预热窗口"口径一致:收益在预热段,不在 e2e 指标 |
| C pinned slab 预分配 | ~0.3s | **r1 口径 −0.35~0.39s**;进程 wall ±0(成本一次性付在 `__init__`) | 两阶段:先做后台线程版,实测 ≈0 收益——cuMemHostAlloc 走驱动全局锁,与 CUDA 宿主 API 天然串行(探针 `_test/post6_pinned_stall_probe.py`:后台分配 1.7GB 期间主线程 4MB 小分配最大停顿 245ms),无法与解码会话建立真并行;改 `__init__` 预分配(r1 口径"预热单列"段)后兑现 |
| A 解码转换流分流 | **0.8~1.5s** | **≈0**(噪声内) | §13.5 归因证伪,见 §14.3 |

**e2e(4 视频,r1 口径,同机同会话交替)**:old(§13 代码快照)10.1/10.0/10.1/10.0
→ mean **10.05s**;new(B+C+A)9.7/9.7/9.7 → mean **9.70s**;**1.036x**。
进程 wall:old mean 11.02s → new mean 10.86s(−0.16s:B 的加载节省被 C 的
init 预分配抵消大半——两笔成本都真实存在,只是位置移动)。
按 §13 机器标定(9.90s 基线)折算:本轮等效 ≈9.55s。系列:16.30 → 13.42 →
10.30 → 9.90 → **9.90 基线机器 ≈9.55**(今日漂移机器实测 9.70),累计 1.71x。
**未达 ≤8.6s 目标**——缺口即 §13.7-A 被证伪的部分(§14.3/§14.6)。

### 14.3 A 项:机制成立、归因证伪(本轮最重要的一手结论)

**机制(已 trace 级验证)**:新旧 trace 各采一轮 4 视频(`_test/post6_nsys_streams.py`
/`post6_nsys_compare.py` + post6_a.nsys-rep/bench4.sqlite):
- old:YuvToRgbKernel/ConvertP016BLtoP016 3171 次全部在主流(stream 74)与消费 kernel 同流;
- new:3171 次全部移到独立 dstream(stream 22),消费 kernel(TRT sm80/sm75_xmma、
  自研 area/letterbox/swap)在主流(stream 20),pose/gaze 引擎仍各在自流(16/21)。

**但解码窗口未变**(按视频分段,转换 kernel 逐帧窗口/间隔分布新旧同量级):

| 段(=视频) | n | old 窗口/帧 | new 窗口/帧 | old gap_p90 | new gap_p90 |
|---|---|---|---|---|---|
| 006+099 | 1641 | 2.52ms | 2.47ms | 5.8ms | 5.9ms |
| 154 | 583 | 3.61ms | 3.67ms | 6.6ms | 6.5ms |
| 214 | 947 | 2.03ms | 2.03ms | 4.9ms | 4.8ms |

且新旧 trace 中转换 kernel 与消费 kernel 的**时间重叠都是 0.00s**——SM 上根本
不存在"被消费 kernel 顶推"的碰撞:重叠臂是 producer-bound,消费端 GPU 段每块
只占几十 ms 且紧跟 sync,与解码窗口几乎不相交。**§13.5"同流 SM 串行拉伸解码
窗口"归因不成立**,§13.5 表格中"拉伸 ∝ 消费负载"的真实耦合是:
1. **消费端 host 链**闸住 token 释放(gaze/head/imwrite 的 host 时间)——重叠臂
   producer 最多领先 1 块,消费块 host 时间 > 解码块时间时窗口被拉长(154:
   gaze+head+imwrite ≈2.9s ÷ 10 块 ≈ 290ms/块,远大于解码 108ms/块);
2. **解码线程自身逐帧 CPU 税**(§13.6 cudaMemcpyToSymbol 2 次/帧 0.68ms +
   launch/sync)——与 §13.7-D 同源,像素管线不动就不可治(§11 已否决)。

**结论:A 保留(机制正确、零回归、为未来消费端 GPU 负载加重的场景留了余量),
但 §13.7-A 的 0.8~1.5s 从"待兑现"改判为"不存在";到墙的最后一块石头是
§13.7-D(解码线程 CPU 税)与消费 host 链,均在本轮否决/已优化过的范围内。**

### 14.4 A 项保序证明(两流后的 happens-before)

数据流(新):

```
[producer 线程]                          [主流 = 消费]
  NVDEC 硬解(YUV, ASIC)                    scrfd_block_detect: area→letterbox→TRT
  ConvertP016/YuvToRgb (dstream)            landing: swap→D2H(主流)→sync
  D2D 入池 (dstream)                        head letterbox→TRT→sync
  ev_d.record(dstream)                      …全部读槽 kernel…
  stream.wait_for_event(ev_d)  ──────────→ 块 j 的消费 kernel(排在 wait 后)
  q_put(块 j)  ───(queue happens-before)──→ pop(j) → 释放块 j-1 token
```

不变式(对每块 j):
1. **D2D(j) 完成 → 消费读帧**:`read_block` 末尾 `ev_d.record(dstream)` 后
   `stream.wait_for_event(ev_d)` 在 `q_put(j)` **之前**入队(producer 程序序);
   consumer 块 j 的 kernel 只能在 pop(j) 之后入队(queue happens-before),在
   主流上恒排在该 wait 之后,执行晚于 D2D(j) 完成。覆盖 no_face 块(area 仍
   读槽)与 EOF 块(guard 照入队,零 D2D 时空等)。
2. **消费读帧(块 j-2)→ D2D(j) 覆写同组槽**:token 协议(OVL_SETS=2)+ 块内
   同步链(scrfd `_trt_pass`、landing D2H、head `stream.synchronize()`)使得
   pop(j-1) 释放块 j-2 的 token 时,块 j-2 全部读槽 kernel 已**设备侧完成**
   (host sync);producer 解码第 j 块(至少 NVDEC 一帧延迟)远在其后。与旧单流
   设计同一论证,事件只加不减。**不加入口护栏**是有意的:入口 guard 会令
   D2D(j) 等待块 j-1(异组,无冲突)的 kernel,破坏重叠。
3. **解码器内部帧缓冲回收**:写它的转换 kernel 与读它的 D2D 同在 dstream,
   同流程序序保护(§10.5 enqueue-order 余量论证原样成立)。
4. **事件复用**:CUDA wait 捕获"wait 调用时刻最近一次 record";producer 单线程
   record→wait 程序序保证本块 wait 恒绑定本块 record,跨块 re-record 不影响
   已入队 wait(探针 `_test/post6_event_probe.py`/`post6_event_probe2.py`:
   pycuda 2026.1 Event/record/wait_for_event 可用且跨流 wait 生效)。
5. **无死锁**:全部等待在设备侧,无新增 host 阻塞;两流 wait 依赖链
   (dstream→main→dstream)按块交替、每段只覆盖此前已入队工作,无环。

串行臂(--overlap off)同路径受益同保护:单线程 read_block,出口 guard 与旧
同流程序序语义等价;`_run_pass` 宿主臂/管道臂/落盘臂不触碰 gpu_decode 模块
(懒导入),零影响(实测 §14.5 #6/#7)。

### 14.5 等价性与稳定性总表(全部 PASS)

| # | 验证 | 对照 | 结果 |
|---|---|---|---|
| 1 | GPU 臂(final)r1 | base_run_r1(§13 基线) | CSV 625/625 逐位一致,JPG 257 md5 逐字节一致 |
| 2 | GPU 臂 r2(复现) | base_run_r1 | 同上 |
| 3 | --overlap off(final) | base_off_r1 | 同上 |
| 4 | 回退链(注入 GPU 臂初始化失败→宿主 pynvvc) | base_run_r1 | 同上;且懒加载断言:run 前 `_det_obj is None`/head pending,run 后 det 已建、head ok(`_test/post6_fallback_check.py`) |
| 5 | 宿主 pynvvc 臂(--decode pynvvc) | base_run_r1 | 同上(§10.3 双臂逐位同再证) |
| 6 | ffmpeg 管道臂 new vs old 快照 | 同臂互证 | CSV 625/625,JPG 260 逐字节一致 |
| 7 | --dedup new vs old 快照 | 同臂互证 | CSV 542/542,JPG 187 逐字节一致 |
| 8 | 100 帧反复起停 ×10(串行)+ ×3(重叠臂) | — | 无死锁/无泄漏/帧数一致(`_test/post6_startstop_smoke.py`) |
| 9 | 强杀泄漏检查:硬杀 ×2(7s 中途/12s)+CTRL_BREAK ×1 | nvidia-smi | 三次均无 python 残留显存(`_test/post6_kill_leak_check.py`) |
| 10 | B 懒加载断言 | — | 见 #4;引擎加载日志 6→4 |

注:早期 B 轮回退冒烟曾报 154 上 14 行 keep→gaze 不一致——为测试脚本直构
VideoFilter 未传 `gaze_yaw_max`(argparse 默认 20.0,`__init__` 默认 15.0),
15<|yaw|≤20 帧误判;对齐参数后 PASS。**非代码缺陷**,已在脚本内注释。

### 14.6 对 §13.6/§13.7 的两处归因修正

1. **"cuMemHostAlloc 4 次 381ms = pinned slab 按分辨率重建"不准确**:land
   slab 本就是 grow-only 跨视频复用(`len(land_slabs[_si]) < land_cap` 判定),
   4 次分配 = 首视频 006 的 2 块 land slab + gaze `_pipe_slabs` 2 块(约 98MB×2,
   小头);不存在"4K 套×2 + 214 分辨率套×2"的重建。C 项的收益来源是"首视频
   0.39s 移出 r1 口径"而非"消除重建"。
2. **§13.5 同流串行拉伸 → 改判**(§14.3):拉伸 = 消费 host 链闸 token + 解码
   线程 CPU 税;SM 流串行贡献 ≈0。§13.7-A 的 0.8~1.5s 不存在,§13.8 的
   "9.9s 闭合拆分"中"共享流串行拉伸 ~2.6s"一项应并入"解码线程 CPU 税 +
   消费 host 链暴露"。到墙判定不变但构成修正:**NVDEC ASIC ~4.3s + 解码线程
   CPU 税(§13.7-D,已否决路线)+ 消费端真实工作(host 链为主)+ 预热**,GPU
   侧调度已无余量可挖——GPU 臂到墙的判定由"NVDEC+SM 串行"修正为
   "NVDEC+解码线程串行+消费 host 链"。

### 14.7 本轮产物与工程注意点

- 脚本(`_test/`,前缀 post6_):event_probe / event_probe2(pycuda Event API
  探针)、pinned_thread_probe / pinned_stall_probe(分配争用探针)、
  equiv.py(比对器)、fallback_check.py(回退链)、startstop_smoke.py、
  kill_leak_check.py、nsys_streams.py / nsys_compare.py(trace 分析)。
- 产物:E:\output\_gpudecode\post6\(old_snapshot_src 旧代码快照+models
  junction、全部对照运行与日志)、E:\output\nsys\(post6_a.nsys-rep/
  post6_a.sqlite/bench4.sqlite 及采集脚本 post6_capture_a.sh)。
- 注意点:1) pycuda `pagelocked_empty` 非主线程调用必须 `ctx.push()`(探针2
  同款),线程退出时 pycuda 打 "could not be cleaned up" 警告为 GC 清理提示,
  持引用内存有效;2) cuMemHostAlloc 与一切 CUDA 宿主 API 经驱动全局锁串行,
  "后台线程分配 pinned 内存"是伪并行;3) 测试脚本直构 VideoFilter 必须显式传
  `gaze_yaw_max=20.0`(argparse 默认与 `__init__` 默认不同);4) nsys 采集沿用
  §13.9 的 SVF_PROBE_CACHE + ASCII bat 包装配方,本轮零改动复用。
- 判定逻辑/阈值零改动;未删除任何 mp4;未跑 225 全量;未提交 commit。
