---
name: nsight-systems
description: 本机 Nsight Systems 2026.5.1 的包装指针——凡涉及 nsys profiling、.nsys-rep 报告分析、CUDA/CPU/GPU 时间线、优化点排查,先读官方 skill 包再干活。
---

# nsight-systems(本机包装)

官方 skill 包(自包含,含 scripts/ 与 references/):

    C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\skills\nsight-systems\SKILL.md

**第一动作**:完整读该 SKILL.md,然后立即执行其 session bootstrap(Windows 用 `scripts\determine_local_windows_platform.bat`,Git Bash 可用 `sh scripts/determine_local_unix_platform.sh`)获取权威的 nsys CLI 与 Nsys 自带 Python 路径,之后所有 `nsys_skill_cli` 调用一律用该 Python、所有 nsys 调用一律用该 CLI(override 规则)。

参考文档:
- https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html

离线存档(仓库内,内容很大,grep 检索按问题取节,勿整读):
- `docs/nsight-systems/User Guide — Nsight Systems.html`(使用参考)
- `docs/nsight-systems/Post-Collection Analysis Guide — Nsight Systems.html`(采集后分析参考)

项目约定:
- 被剖析目标 = 本仓库 `filter_video.py` 管线;基准 4 视频与命令行口径见 `data/e2e_benchmark_2026-09-28.md` §1/§9.4。
- `.nsys-rep` 等产物放 `E:\output\nsys\`,绝不放仓库根目录。
- GPU 单卡串行,profiling 期间不要并行跑其他 GPU 负载;不跑 225 全量。
