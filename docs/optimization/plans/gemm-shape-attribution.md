# GEMM 实际 Shape 与累计耗时归属：最小补采计划

> 日期：2026-10-18  
> 当前状态：已完成材料检索、官方原参数 BF16 服务启动及外部采样；现有 Trace 缺 M/N/K 和投影关联，暂停扩大采样，待讨论独立诊断代码  
> 主线：ADR-0002 W8A16；本阶段仅分析 BF16 原始工作负载

## 目标

分别在固定 4K、16K 负载中建立 `场景 × 投影 × M/N/K × 阶段 × Kernel × 调用数 × 累计 GPU 耗时`，判断主要 GEMM 时间落在小 M 还是大 M。优先报告精确 M 和按累计耗时排序，不用任意 M 阈值替代原始分布。

不以调用次数、模型参数量、同 M 理论 FLOPs、synthetic M=1 结果或不同并发的分阶段采样占比推断真实总体分布。小 M 是否访存受限、大 M 是否计算受限仍需额外效率证据。

## 已完成检索

- 本机保存历史汇总 `report.md`、`new_report.md`、`profiling/metax-c500.md`，以及 `aggregate_kernels.py`、`explore_phases.py`。
- 旧编排记录指向 `/data/trace_dec16/dec16/trace-894.json`、`/data/trace_o4k/`、`/data/trace_o16k/` 等；当前新容器 `/data` 和本机工作区未找到这些原始 Trace。
- 当前检查的 `/workspace/benchmark_results` 仅有 2026-10-18 synthetic Down 探测，不能用于真实模型 M 分布。
- 原解析器按 Kernel 名聚合，未提取 Shape 或投影；阶段探索按 Attention Kernel 模式估计时段，不能作为逐次 GEMM 的可靠阶段标签。
- 历史独立 Prefill/Decode 组的并发、输出长度和请求数不同，不能比较两组累计时间来推算官方整体场景的小/大 M 占比。
- 当前模型端口未返回服务信息；尚未为本任务启动模型。

本机与当前容器查找范围有限；这些结论不表示旧平台实例或其他存储一定没有原始文件。

## 工具能力核对

当前 vLLM 0.24 profiler 默认关闭，`torch_profiler_record_shapes` 默认 false；worker 的 start_profile 要求预先配置 profiler。不能假设调用 `/start_profile` 就能得到 Shape，也不能未经讨论添加 `--profiler-config` 或修改官方服务参数。

mcTracer 3.7.0.38 提供外部 `--mctx --attach <PID>`，可沿用历史外部采样方式，不改变服务参数。**帮助信息没有证明其输出一定包含 GEMM M/N/K 或模块前缀**；先检查实际 Trace 字段，再决定能否做 Shape 归属。

## 本次授权范围内的最小采样步骤

1. 检查 GPU 和端口占用，在既有 Conda/MACA 环境中按 AGENTS.md 的沐曦官方命令启动原始 BF16 服务，保存完整启动日志，不加 FULL_DECODE_ONLY、profiler-config、enforce-eager 等参数。若遇到历史启动错误，记录并停止，不静默替换参数。
2. 服务可用后使用官方性能命令，保留两场景、并发、请求数和脚本实现不变。脚本原有 RUNS=4、SKIP_FIRST=1 不改动；保存完整控制台输出与 raw/summary CSV。若官方客户端触发历史 runtime 问题，停止并报告，不擅自换 aiohttp 客户端。
3. 首先用 mcTracer 外部 attach 做短时字段可用性检查。只回答能否恢复 Kernel、关联 ID、Shape 或可验证的布局/编译元数据；短片段不能代表总体分布。
4. 若字段足够，再对两个场景各自完整代表性测量轮次采样，并与 Benchmark 日志关联，排除 warmup、编译和不同测试轮次。记录采样是否覆盖完整阶段，不将不等长采样窗口的累计量直接比较。
5. 若外部 Trace 缺 Shape 或模块关联，先展示字段样例和精确缺口，再讨论独立最小模块标记/Shape 采样方案及文件改动；本计划不预先授权修改模型或 profiler 配置。

新增源码文件：本轮无。执行命令和解析记录可以单独归档；若需要新解析器或采样脚本，先列出具体改动再讨论。官方评测脚本不修改。

## 归属与汇总规则

- 以实际关联链连接模块调用、运算 Shape 与 GPU Kernel；一个 GEMM 可能包含主乘法、Split-K 规约等多个 Kernel，不能漏计或重复计数。
- Gate/Up 和 QKV 的 checkpoint 拆分权重与运行时 packed 投影分别标注；Q 与 O 的 N/K 相同，不能仅凭 Shape 区分。
- CUDAGraph replay 可能不产生逐次 Python/ATen 事件。实际活跃 token 数与 padding 后 GEMM M 区分记录；不能以并发数代替 M，也不为采样关闭官方图模式。
- Prefill/Decode/Mixed 需要可靠调度或执行关联；无法确定的事件保留为 unknown，不按 Kernel 名或时间顺序强行分类。
- 按 Kernel 累计时长统计，明确它不是壁钟占比；重叠 Kernel 不相加成端到端时长。每个场景记录可归属覆盖率、未知耗时和采样范围。
- profiling 会影响时延，采样轮次的 throughput/TTFT 不申报正式收益。没有足够归属覆盖率时，不宣布小 M 或大 M 主导，也不据此选量化层。

## 本轮可行性检查结果

官方沐曦原参数服务成功启动，实际配置为 BF16、quantization=None、TP=1、FULL_AND_PIECEWISE。此次没有复现历史启动或官方客户端 runtime 故障。

按原版官方脚本启动了两场景测试，4K 的前三轮各 256/256 成功；外部采样覆盖了中间一段窗口。发现字段缺失后中断第 4 轮并停止后续测试，16K 尚未执行。采样干扰下的部分结果不能替代正式 4K/16K 基线，也不能用于判断量化收益。启动的 benchmark、服务和 EngineCore 已停止，原始日志保留。

pilot Trace 大小 2,468,438,606 字节，含 7,987,084 个事件。全量扫描了匹配 `mm_kernel` 的 GPU 事件参数：仅有 launch grid/block、关联 ID、queue、资源占用等，没有 M/N/K、stride 或模块前缀。CPU API 样例同样没有矩阵维度。不能用 grid 或调用顺序猜测 Q/O、M 或 Prefill/Decode。

| 匹配 Kernel | 调用数 | 累计时间 |
|---|---:|---:|
| mm_kernel_nt | 462084 | 37.483971 s |
| mm_kernel_splitk | 2772 | 0.056552 s |

这是局部窗口中的名称匹配汇总，不保证覆盖所有 GEMM，也不是全场景分布；这些事件的 M/投影归属目前均未知，因此本任务的小 M/大 M 判断仍未完成。详见 [`../changes/2026-10-18-gemm-shape-pilot.md`](../changes/2026-10-18-gemm-shape-pilot.md)。

## 下一步需讨论的代码范围

仅拟新增独立诊断入口 `benchmarks/diagnostics/collect_gemm_shapes.py` 和关联解析器 `benchmarks/diagnostics/summarize_gemm_shapes.py`，不修改官方 Benchmark、模型权重、量化实现或生产 Kernel。

诊断入口需在单独诊断进程中记录标准 Linear 模块前缀、输入/权重 Shape、dtype、stride，并通过 profiler/关联 ID 连接实际 GPU 事件。若需在入口中开启 PyTorch profiler，该观察配置必须明确记录，不能静默添加官方 CLI 参数或改变计算、调度及图模式；诊断吞吐不申报成绩。

先验证少量事件能否准确关联，尤其区分编译/捕获与 CUDAGraph replay 的实际执行、padding M 与活跃 token 数；无法建立关联的部分保留 unknown。若插桩破坏原图模式，停止并重新讨论，不把改变执行模式后的分布当作官方工作负载分布。关联有效后再完整采集两场景，产出覆盖率及按精确 M 分组的累计耗时表。

本轮尚未创建这两个脚本；按照工作区先讨论再修改代码的要求，用户同意上述范围后才实施。

## 预期产物

原始 Trace、环境/命令、服务及客户端日志存于 `/workspace/benchmark_results/gemm-shape-<experiment-id>/` 等独立目录；官方命令产生的原始 CSV 保留路径和校验值，不覆盖历史基线。

在仓库文档内仅保存归属表、证据来源、覆盖率、限制和结论；大型 Trace 不进入 Git。已知 N/K 可预填，但缺失 M/耗时必须保持待测。

| 场景 | 投影 | M | N | K | 阶段 | 调用数 | 累计 GPU 时间 | 证据 |
|---|---|---|---:|---:|---|---|---|---|
| 4K/16K | Down | 待测 | 2048 | 6144 | 待关联 | 待测 | 待测 | 真实 checkpoint，TP=1 结构 |
| 4K/16K | Packed Gate/Up | 待测 | 12288 | 2048 | 待关联 | 待测 | 待测 | 模型实现与 checkpoint |
| 4K/16K | Packed QKV | 待测 | 2560 | 2048 | 待关联 | 待测 | 待测 | 模型实现与 checkpoint |
| 4K/16K | O | 待测 | 2048 | 2048 | 待关联 | 待测 | 待测 | 真实 checkpoint，TP=1 结构 |
