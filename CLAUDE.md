# short_video_filter 项目

## 铁律(最高优先级,不可违背)

**后续所有任务一律派子代理(subagent)执行,且必须串行:一次只派一个子代理,等它完成、拿到结果后,再派下一个;严禁同一时刻并行派多个子代理。**

- **为什么必须串行**:本机只有一块 12GB GPU(RTX 3060),TensorRT 推理无法并行;并行子代理会争抢同一块 GPU,导致 OOM / 失败 / 变慢。
- 子代理类型用 `general-purpose`(能读写文件、跑命令)。
- 派子代理时,在 prompt 里写清楚:背景、目标、环境(python / CUDA / TRT 路径见全局 CLAUDE.md,子代理不继承主会话对话)、以及明确的**验收标准**。
- 主会话只负责:拆解任务、给子代理写清上下文与验收标准、串行调度、汇总结果。**不要把大量文件读取 / 模型转换 / 实验过程塞进主会话上下文**——那些都交给子代理去做、只把结论带回来。

## 项目约束

- **严禁删除** `E:\output\buding\delete\`、`E:\output\buding\buding_133_68pt1\`,以及任何 `.mp4` 文件。
- 不要跑 225 个视频的全量批处理(除非用户明确要求)。
- 生成的帧/输出放视频所在文件夹或 `E:\output\`,测试脚本放 `_test\`,**绝不放进源码根目录当正式代码**。
- 用户确认最终阈值前,不要删除临时的 score / angle 文件名和 `pose_report.csv`。
- pip 一律用清华镜像:`-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 与用户沟通用**中文**。

## 本机参考资源

- **FFmpeg 7.1.1 完整源码已下载**:`E:\work\FFmpeg-n7.1.1`——需要查 NVIDIA/多媒体相关 API、容器/像素格式定义等参考实现时直接读,勿重新下载调研。
- **NVIDIA Optical Flow SDK 5.0.7(官方 zip)**:`E:\work\Optical_Flow_SDK_5.0.7.zip`——NVOF2/OFA 硬件光流官方头文件与示例(官方页 developer.nvidia.com/optical-flow-sdk);`nvofapi64.dll` 是驱动自带官方运行库,按官方头文件 ctypes 调用属正规用法。
- **DeepStream 官方 Python 示例库**:`E:\work\deepstream_python_apps`——含 deepstream-opticalflow 官方示例(flow vectors 直出 numpy;需 DeepStream 运行时)。

## Nsight Systems(nsys)profiling

凡涉及 nsys / profiling / GPU 时间线 / 性能排查:**先走项目 skill `.claude/skills/nsight-systems/SKILL.md`**(本机实战手册:实测采集配方、分析 SOP、kernel 名免 NVTX 归因法、避坑清单),不要重新调研。三条最致命的坑提前知道:中文 locale 下 report-fact/report-query 必挂(分析走 recipe + 报告副本放 C 盘);被测命令必须用 ASCII-only .bat 包装;nsys 注入会阉割子进程 stdout(本项目用 `SVF_PROBE_CACHE=1` 预生成元数据缓存绕过)。
