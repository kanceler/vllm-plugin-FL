# 真实 BF16 负载的外部 Shape 采样可行性检查

> 日期：2026-10-18  
> 范围：GEMM 基础分析；保持 W8A8 ADR  
> 提交状态：当前分支，无提交或推送

## 执行内容

未找到历史原始 Trace 后，按用户“那先做这个”的授权，用 AGENTS.md 的沐曦官方命令启动 BF16 服务；未额外添加 FULL_DECODE_ONLY、profiler-config、enforce-eager 或其他优化参数。实际日志确认 BF16、quantization=None、TP=1、FULL_AND_PIECEWISE，服务成功启动。

使用原版 `benchmark_throughput_serve.py`，test-cases 保持 `[[4096,1024,64,256],[16384,1024,64,128]]`；脚本原有 RUNS=4、SKIP_FIRST=1 未修改。外部 mcTracer 3.7.0.38 通过 `--mctx --attach 30792` 采集了 4K 中间部分窗口。

4K 前三轮各 256 个请求全部成功；第 1 轮输出 total tokens/s=5039.51、Mean TTFT=3543.36 ms，但其属于脚本跳过的首轮。第 2/3 轮受采样等诊断过程影响，不作为新基线。看到字段缺失后中断第 4 轮，未执行 16K；完整官方测试没有完成，不能报告正式平均成绩或完整 TTFT 基线。

## 结果与限制

- 已确认本次镜像可以按官方原参数启动，且官方客户端前三轮未触发历史 GPU runtime 故障。
- 原始 Trace 为 2,468,438,606 字节，7,987,084 个事件。
- 匹配的 `mm_kernel_nt` 为 462084 次、累计 37,483,971,023 ns；`mm_kernel_splitk` 为 2772 次、累计 56,551,810 ns。扫描只汇总匹配名称的 GPU X 事件，不能据此声称完整 GEMM 覆盖率。
- GPU GEMM 参数全量字段集合包括 grid/block、co_id、queue、资源占用等，未提供 M/N/K、stride、投影名。抽查 CPU API 的 mcModuleLaunchKernel 也未提供 Shape。
- 因此不能归属小 M/大 M，也不能从同名 Kernel 的 grid 或时序强行推断 Q/O、Prefill/Decode。

这次采样只验证现有工具的字段可用性，不重复全框架热点研究，不据此选量化层。没有修改或运行模型量化/Kernel 优化，没有 math_500 精度测试。

## 归档与清理

容器路径：`/workspace/benchmark_results/gemm-shape-20261018-r1/`。

- `serve.log`：原参数服务启动和关闭日志。
- `benchmark.log`：官方脚本的部分执行输出。
- `raw_runs_20261018_210705.csv`：已完成前三轮原始行，复制归档，原始文件仍在官方输出目录。
- `pilot/pilot/trace-30792.json`：完整 pilot Trace，不纳入 Git。
- `pilot_summary.json`：流式解析汇总与字段集合，未按猜测补全 M。
- `status.txt`：中断范围说明。

小型日志/CSV/汇总已复制到本机工作区 `benchmark_results/gemm-shape-20261018-r1/`；大型 Trace 保留在容器。原始文件未删除。自本任务启动的 benchmark、API Server 和 EngineCore 已发送 TERM 并确认 PID 不再运行。

## 下一步

需要独立的模块/Shape 观察入口及事件关联，具体文件、图模式约束和未知归属处理见 [`../plans/gemm-shape-attribution.md`](../plans/gemm-shape-attribution.md)。先讨论诊断代码计划，获同意后实施，不直接改量化/Kernel 或禁用 CUDAGraph。
