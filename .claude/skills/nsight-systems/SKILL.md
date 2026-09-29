---
name: nsight-systems
description: 本机 Nsight Systems 2026.5.1 实战手册——nsys profiling 采集命令、.nsys-rep 分析路径、CUDA/NVDEC 时间线归因、优化点排查。含实测过的命令配方、kernel 名免 NVTX 归因法与避坑清单(中文 locale GBK bug、子进程 stdout 阉割等)。凡涉及 nsys / profiling / GPU 时间线 / 性能排查,先读本文件再干活,勿重新调研。
---

# nsight-systems(本机实战手册)

## 0. 环境事实(勿重新探测)

- Nsight Systems **2026.5.1**,`nsys` 已在 PATH(`nsys -V` 核验)。
- 官方 skill 包(自包含,权威证据流程):`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\skills\nsight-systems\SKILL.md`——**第一动作**:读它,然后跑 bootstrap(Git Bash: `sh scripts/determine_local_unix_platform.sh`;cmd: `scripts\determine_local_windows_platform.bat`)拿权威 nsys CLI + Nsys 专属 Python;之后一切分析走 `<该Python> scripts/nsys_skill_cli.py`(doctor / inspect-cli / report-fact / report-query / report-context / run-recipe / lookup-recipes),CLI 语法不确定就 `inspect-cli --target "<cmd>"` 核实,**不凭记忆编 flag**。
- 官方在线文档(离线存档已入库,见下)。
- 本机:Windows 11 **中文 locale(GBK)**、RTX 3060 12GB、被测对象=本仓库 `filter_video.py` 管线。
- 产物一律放 `E:\output\nsys\`,绝不放仓库根目录;GPU 单卡串行,profiling 期间只跑被测进程;基准命令行口径见 `data/e2e_benchmark_2026-09-28.md` §1/§9.4。

离线文档存档(仓库内,内容大,**grep 按问题取节,勿整读**):
- `docs/nsight-systems/User Guide — Nsight Systems.html`(采集/CLI/trace 选项)
- `docs/nsight-systems/Post-Collection Analysis Guide — Nsight Systems.html`(报告分析/recipe)

## 1. 实测采集配方(墙钟膨胀 0%,直接复用)

```bash
nsys profile -t cuda,nvtx,nvvideo,python-gil --gpu-video-devices=0 \
  --sample none --cpuctxsw none \
  -o E:/output/nsys/<报告名> -f true \
  E:/output/nsys/<ASCII-only包装>.bat <app参数...>
```

- Windows 非 admin **无 osrt 采样** → 必须 `--sample none --cpuctxsw none`;`--gpu-video-devices=0` 开视频引擎(NVDEC)采集。
- 被测命令用 **ASCII-only .bat 包装**并重定向 stdout 到 log(见坑 ⑦);基准 4 视频先**硬链接**到采集目录(如 `E:\output\nsys\bench_vids\`),不动原文件。
- **开销对照必做**:app 内部计时(nsys 下)vs 裸跑 ×3;膨胀 >10% 说明 trace 过重,降配重采。实测该配方膨胀 0%(nsys 会话墙钟多出的 ~3s 是注入+报告生成,在 app 计时之外)。
- **输出等价必验**:采集轮的 CSV/JPG 与基线逐字节一致,否则 profiling 改了行为,数据作废。
- flag 以 `nsys profile --help` / `inspect-cli` 现场核实为准(版本可能变)。

## 2. 分析路径(避开坑 ① 的既定 SOP)

**主路径 = recipe 引擎**(report-fact/report-query 在真实报告上不可用,见坑 ①):

1. 把 `.nsys-rep` **复制到 C 盘**(与 skill 工作区同盘,跨盘 WinError 17);
2. `nsys_skill_cli.py lookup-recipes --query "<问题>"` 找 recipe → `run-recipe`(**位置参数必须在 `--` 之后**,选项以 live help 为准,没有 `--rows`);
3. 实测有用的输出表:
   - `cuda_gpu_kern_sum` — kernel 总时长/次数排行
   - `cuda_gpu_kern_pace` — 逐 launch 间隔(算稳态节奏;`--filter-time` 用**整数 ns** 的 start/end)
   - `cuda_gpu_mem_time_sum` — memcpy 总量对账
   - `gpu_time_util --chunks 200` — 利用率图(**必须 200-chunk**,32-chunk 粗图"空洞"多为阈值合并假象)
4. pace 输出**每表 100 行上限** → 长序列按窗口切分 + count 交叉验证;
5. 报告健康存疑先跑 `report-doctor`。

## 3. 免 NVTX 阶段归因(kernel 名法)

本机**无 NVTX runtime DLL**,不加 NVTX;用 kernel 名归因,语义等价零侵入:

| kernel 名 | 含义 |
|---|---|
| `YuvToRgbKernel` / `ConvertP016BLtoP016` | 次数 = **源帧数**(pynvvc 解码管线) |
| `permutationKernelPLC3` | 次数 ≈ **判定帧数** |
| 自研(`area_fast_u8` / `area_generic_u8` / `head_letterbox_u8` / `rgb2bgr_inplace_u8` / `copy_u8`,来自 `src/gpu_decode_kernels.cu`) | 预处理段 |
| TRT 内核 | 推理段 |

NVDEC ASIC 时间**不可直读**(坑 ① 副作用,NVDEC 引擎表读不出)→ 用转换 kernel 的逐 launch 间隔做代理:实测 4K10bit 稳态周期 8.42ms/5 帧 = ASIC 6.3 + 间隙 0.26×2 + 转换 1.6ms。

## 4. 避坑清单(全部实测踩过,勿再踩)

1. **中文 locale GBK bug(产品 bug,最致命)**:报告内 `META_DATA_EXPORT.parquet` 时区名是 GBK 字节("中国标准时间")→ 所有走 duckdb 的 skill 命令(**report-fact / report-query / report-context**)在真实报告上全部失败;`TZ=UTC`、`--discard-environment=true` 均无效。**绕行**:分析全走 recipe 引擎(原生加载器容忍 GBK)+ 报告副本放 C 盘。副作用:NVDEC 引擎表/video 事实拿不到。
2. **报告与 skill 工作区必须同盘(C 盘)**,跨盘报 WinError 17。
3. **nsys 注入阉割子进程 stdout**(含文件重定向,`-t none` 也一样):子进程 exit 0 但输出为空。本项目 `filter_video.py` 的 `probe()` 已内置 **SVF_PROBE_CACHE** 门控缓存分支(设环境变量 `SVF_PROBE_CACHE=1` 用预生成元数据;不设变量时分支不存在,默认行为逐字节不变)。
4. **.bat 包装必须 ASCII-only**:GBK 命令的报错文本会被收进报告 StringIds,**污染所有 parquet 读取**(连 recipe 一起挂)。
5. **无 NVTX runtime DLL** → 别装别加,用 §3 kernel 名归因。
6. recipe 参数必须在 `--` 后、以 live help 为准;`--filter-time` 整数 ns;每表 100 行上限(见 §2)。
7. 利用率粗图(低 chunk 数)的空洞可能是阈值合并假象,结论前用 200-chunk 复核。
8. 采集前后必做开销对照与输出等价性校验(§1),两者任一不过,profiling 数据不可信。
9. GPU 单卡:profiling 期间严禁并行其他 GPU 负载(utilization 表里只应有被测 PID)。

## 5. 本仓库的 nsys 实战记录(证据与脚本)

- 基准数据与结论:`data/e2e_benchmark_2026-09-28.md` §13(profiling 全景、四问四答)、§14.3(D 轮 trace 对比证伪 §13.5 归因)、§14.6(到墙构成)。
- 可复用脚本:`_test/post6_nsys_streams.py`、`post6_nsys_compare.py`(trace 对比)、`post6_capture_a.sh`(采集包装);nsys 工作区配方(SVF_PROBE_CACHE + ASCII bat)在 §13.9。
- 产物样例:`E:\output\nsys\`(bench4.nsys-rep、post6_a.nsys-rep/.sqlite)。
- 经验教训:nsys 数字也要复核——§13.7-A"流分流 0.8~1.5s"被 D 轮前后 trace 对比证伪(转换×消费 kernel 重叠本来就是 0.00s);单份 trace 的归因要等价性+复测双重确认后才能进结论。
