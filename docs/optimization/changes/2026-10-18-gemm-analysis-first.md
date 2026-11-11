# 补充分析优先方案与真实 FFN 权重检查

> 日期：2026-10-18  
> 模块：GEMM  
> 基线提交：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`  
> 结果提交：尚未提交，继续当前分支；提交前再按任务创建分支

## 方案补充

根据用户讨论，执行顺序明确为：已有 BF16 Baseline 与 Profiling → GEMM 专项基础分析 → 在 W8A8 ADR 内细化策略 → 策略与 Kernel 可行性验证 → Kernel 优化、混合精度接入 → 比赛验收。

补充 `research/gemm-quantization-plan.md` 阶段 0，更新工作区 `AGENTS.md`、`CONTEXT.md` 和 `plans/phase-1.md` 的入口。ADR 数值格式不变，未修改 Kernel、量化实现、依赖、模型权重或官方评测。当前不重新做全框架 Profiling，不重复已完成基线。

## 继续完成的只读分析

### 真实模型与投影

读取容器 `/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors` 头部，确认 381 个张量均为 BF16，Down、Gate、Up、Q/K/V/O 各 42 个投影。

| 投影 | checkpoint 权重 [N,K] | 全部 42 层由 BF16 到 INT8 的理论权重节省 |
|---|---|---:|
| Down | [2048,6144] | 504 MiB |
| Gate + Up | 各 [6144,2048] | 1008 MiB |
| Q/K/V | [2048,2048]、[256,2048]、[256,2048] | 210 MiB |
| O | [2048,2048] | 168 MiB |

不包括 scale、预打包、workspace 或保留的 BF16 副本。参数量与同 M 理论乘加量中 FFN 占上述投影的 80%，不是实测 Kernel 时间占比，不能据此推算 total tokens/s。

### 抽样 FFN 权重分布

选择 block 0、21、41 仅作前/中/后结构抽样，不用作最终量化白名单。CPU 读取 safetensors 对应字节区间，将 BF16 位模式精确展开为 FP32 后统计；不运行模型，不修改或导出量化权重。9 个样本全部有限值。

| block | 投影 | absmax | RMS | abs P99 | 通道内 max(abs)/RMS 的最大值 |
|---:|---|---:|---:|---:|---:|
| 0 | Down | 0.65625 | 0.01860 | 0.04956 | 23.83 |
| 0 | Gate | 0.75000 | 0.01629 | 0.04419 | 23.04 |
| 0 | Up | 0.53906 | 0.02112 | 0.05640 | 20.14 |
| 21 | Down | 0.85156 | 0.02013 | 0.05469 | 62.64 |
| 21 | Gate | 0.57813 | 0.01882 | 0.05151 | 25.87 |
| 21 | Up | 1.20313 | 0.02143 | 0.05884 | 27.35 |
| 41 | Down | 1.48438 | 0.02323 | 0.06226 | 30.56 |
| 41 | Gate | 1.83594 | 0.01921 | 0.05396 | 30.61 |
| 41 | Up | 1.79688 | 0.02112 | 0.05713 | 37.16 |

计算定义：每个权重输出行独立求 `max(abs(W[row,:])) / sqrt(mean(W[row,:]^2))`，表中取各行比值的最大值。block 21 Down 最大约 62.64，而该样本行比值的中位数约 4.54，说明至少部分输出通道的分布与典型行差异明显。

这支持后续检查 per-channel scale 下的离群权重与误差，不证明 block 21 必须保留 BF16。权重统计也不能代替激活分布、逐层输出误差或模型敏感性；本轮未计算真实权重量化误差、未选择阈值。

原始 CPU 统计及源码摘录保存在本机工作区 `benchmark_results/gemm-analysis-20261018/static_inspection.txt`，未纳入仓库提交。

### BF16 Linear 源码调用链

在容器安装的 vLLM 源码中确认：

```text
LinearBase: quant_config is None
→ UnquantizedLinearMethod
→ apply
→ dispatch_unquantized_gemm（常规非 CPU/ROCm 分支）
→ default_unquantized_gemm
→ torch.nn.functional.linear
```

相关源码位置：`layers/linear.py` 269、217 行，`layers/utils.py` 332、92 行，均相对于 `/opt/conda/lib/python3.12/site-packages/vllm/model_executor/`。若启用 `VLLM_BATCH_INVARIANT` 且平台 cuda-alike，`apply` 还有单独分支；实际环境开关和编译器改写仍须运行证据。

ColumnParallelLinear 在 `quant_method.apply` 后可执行 gather，RowParallelLinear 在 apply 后可执行 all-reduce。未来量化实现应保持接口和通信语义，不仅替换矩阵乘法函数。

这确认的是源码中的常规可达路径，并未运行实际服务，也未证明生产负载最终落到 vendor、FlagGems、特定硬件指令或某个编译后 Kernel。`mm_kernel_nt/splitk` 与具体投影的对应关系仍缺运行证据。

## 尚缺证据与下一步

| 已有证据 | 缺口 | 后续方法 |
|---|---|---|
| BF16 Baseline 和六组 Profiling 汇总 | 原始 trace 未在仓库归档；投影到 Kernel 映射和 M 分布缺失 | 先查现存日志/trace，不能恢复的信息再计划定向采样 |
| 实际权重形状及 9 个 FFN 分布样本 | 全模型分布、真实激活和量化敏感性缺失 | 经讨论后制定独立代表性输入的分析方案 |
| BF16 与 W8A8 源码调用链 | 真实 runtime/编译后实现证据缺失 | 实际负载下关联模块、Shape、Kernel 与耗时 |
| synthetic Down 存在边界差异和 M=1 慢速 | 全部主要实际 Shape 的正确完整路径性能缺失 | 在基础分析后定位数值契约、低精度 API、布局和指令路径 |

当前不能产出“哪些具体层一定量化”的结论，也未达到阶段 0 完成标准。下一项需讨论的是最小投影/Shape 采样计划；获同意前不新增运行采样代码、不启动模型量化、不修改 Kernel。
