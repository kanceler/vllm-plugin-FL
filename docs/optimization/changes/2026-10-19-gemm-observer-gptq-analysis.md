# 独立 Shape 观察器与 GPTQ 候选分析

> 日期：2026-10-19（UTC+8）  
> 分支：继续 `docs/gemm-workspace-rules`，提交前再按用户约定创建目标分支  
> 状态：实现了诊断代码；真实 Shape/耗时分布尚未得到，不是量化或 Kernel 优化成果

## 已授权诊断代码

新增 `benchmarks/diagnostics/collect_gemm_shapes.py`、`summarize_gemm_shapes.py` 及对应关联契约测试。通过独立启动入口生成输出目录内的 sitecustomize 观察接入，只影响该诊断进程及其子进程，不修改安装包、模型权重、官方 Benchmark 或服务参数。

- 原样传递官方 `vllm serve` argv，保存配置和实际权重模块 inventory。
- 通过控制文件按 worker step 限制 torch.profiler 窗口，记录输入 Shape 和步骤阶段。
- 在运行时 CUDAGraph wrapper 外标记 capture/replay、padding 后 token 维度；不向 Linear 添加 hook、不关闭图模式。
- 解析器只用 CPU launch correlation 关联异步 GPU 事件，不按 GPU 壁钟重叠猜测；缺失/冲突保留 unknown。
- 区分实际算子 Input Dims 的 M 与仅有 graph token extent 的情况；后者不宣称为精确 GEMM M。投影组仅按真实 inventory 候选匹配，不宣称能识别每个 block。

目前没有捕获实际输入 stride；权重 stride 可由 inventory 记录。不是全部预期字段已经可用，不据此宣称采样完成。

## 验证与运行限制

本机和容器上初版解析器的 6 项测试通过，覆盖异步关联、padding 与真实 M 区分、关联冲突、同 Shape 不同投影歧义、capture 排除及无关联时禁止按时间强行归属。

初次诊断启动目录为 `/workspace/benchmark_results/gemm-shape-20261018-r2/`，启动仍保持原官方参数。检查发现平台实际使用独立的 `vllm_fl.worker.worker.WorkerFL`，初版只观察原生 GPU Worker，未记录模块 inventory 或采样结果。该次仅完成启动检查，未启动官方负载；已向启动的 API Server PID 37952 发送 TERM。

随后本机观察器已适配 WorkerFL 和普通/Breakable CUDAGraph wrapper，阶段详情复用 `vllm.v1.utils.compute_iteration_details`。这部分尚未完成容器运行复核：后续 SSH/scp 连接连续被服务器关闭，未确认最新脚本已同步，亦未重新确认清理后的远端进程状态。

不能把本机测试通过写成真实服务插桩通过。目前无新的可验证投影 M/耗时表，也不能判断小 M/大 M 主导。恢复连接后先核对进程、脚本哈希、配置和 WorkerFL inventory，再进行少量 step 的关联可行性检查；关联不足则继续保留 unknown，不扩大采样。

## GPTQ 分析

根据用户最新建议新增 [`../research/gptq-feasibility.md`](../research/gptq-feasibility.md)，区分 GPTQ 离线权重量化算法、最终数值格式和推理 Kernel。重点分析 W4A16 的分组 scale、反量化融合成本、校准数据、布局和 TP 接入要求。

本轮没有执行 GPTQ 校准或量化，没有编写反量化/GEMM Kernel，不改变现有 W8A8 ADR。真实负载分析仍是下一步量化策略比较的前提。
