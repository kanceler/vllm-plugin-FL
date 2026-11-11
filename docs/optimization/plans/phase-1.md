# 第一阶段工作方案

## 目标与分工

基于 MetaX C500 Baseline 与 Kernel Profiling，第一阶段拆为三个并行方向。

| 模块 | 负责人范围 | 主要关注点 | 主要代码区域 |
|---|---|---|---|
| 一：GEMM | 当前分支负责人 | Shape 拆分、实现对比、Split-K、投影特化 | Linear/GEMM 调用链及 FlagGems GEMM |
| 二：Attention/KV Cache | 其他成员 | Split-KV、GQA 复用、Cache 布局、调度 | MetaX Attention、KV Cache、调度参数 |
| 三：算子融合 | 其他成员 | GEMM/Attention 相邻融合、残差与归一化 | FlagGems fused ops、RoPE 等路径 |

三个方向可以并行，但框架层对 Batch、Shape、调用频率或 KV Cache 布局的修改会使既有 Kernel 结论失效，届时需要重新 Profiling。

## 模块一：GEMM

### 已知依据

- GEMM 在六组采样中均为主要热点，占比约 47%–94%。
- MiniCPM5-2B 的主要投影形状不同：Q/O 输出维度为 2048，K/V 为 256，Gate/Up 为 6144，Down 的 K 为 6144。
- Prefill 是大 M，Decode 是小 M，两者可能对应不同瓶颈与最优策略。

### 本阶段先回答的问题

1. 实际运行中每类投影对应哪些 M/N/K，调用次数和累计耗时如何？
2. `mm_kernel_nt`、`mm_kernel_splitk` 分别来自哪条调用链？
3. 高频 Shape 中，Vendor 与 FlagGems 实现是否存在稳定差距？
4. Split-K 对大 K、小 M 的收益与规约成本如何变化？

### 赛题约束下的优先级

前置阶段：先补 GEMM 专项基础分析，沿用已完成 Baseline 与六组 Profiling，不重新分析整个框架；核对真实 Shape/耗时、调用链、数值分布和硬件能力，再按 ADR-0002（W8A16）细化策略。详见 [`../decisions/0002-w8a16-mainline.md`](../decisions/0002-w8a16-mainline.md)。下面是通过前置分析后实施的优先级，不代表立即开始量化或写 Kernel。

1. 自研 W8A16 weight-only 量化路径：load-time 权重量化，改版 BF16 GEMM（INT8 权重加载 + upcast + scale epilogue）；
2. BF16/W8A16 Kernel 的 tiling、访存、Split-K 和 Shape 特化；
3. 与 GEMM 相邻的 `RMSNorm/SiluAndMul + activation quant` 融合；
4. Activation-aware 低秩方案仅作为高风险备选，不与量化主线绑定。

仅切换现成算子不满足规则要求；技术报告中的策略必须在最终提交代码和正式运行路径中实际启用。

### 决策顺序

```text
确认调用链
→ 收集并归类 Shape
→ 按累计耗时选高价值 Shape
→ 设计 Vendor/FlagGems 单算子 A/B
→ 决定实现切换、参数调优或 Shape 特化
→ 端到端与精度验证
```

本轮不执行上述实验，只把问题、证据标准和文档入口准备好。

## 模块二：Attention 与 KV Cache

核心问题包括 4K/16K 下 Split-KV 策略、16Q/2KV 的 KV 复用、NHD/HND 布局及 Prefill/Decode 调度。整体 16K 中 Attention 约占 46.5%，但分阶段与整体并发不同，参数选择必须在对应负载上验证。

## 模块三：算子融合

融合围绕两类主热点展开，优先考虑 GEMM epilogue 与 Attention 前后处理。候选包括 SiluAndMul 与 Gate/Up、RMSNorm 与投影、RoPE 与 Attention、KV Cache 写入以及 residual + norm。任何融合都必须通过端到端收益与精度回归，不能只以 Kernel 数量下降作为成功标准。

## 协作与集成约定

- 每人使用独立分支，分支名体现模块与目标。
- 每个可独立评审的阶段新增一份 `docs/optimization/changes/<日期>-<主题>.md`；普通 push 和中间小提交不强制新增。
- 摘要中记录基线提交和结果提交，便于 AI 精确读取差异。
- 跨模块修改在摘要中注明影响的 Shape、布局、调度或调用链。
- 无稳定收益的尝试也应简要记录结论，避免其他成员重复投入。
- 原始 Trace、日志等大文件不直接提交；记录受控存储位置、校验值及生成命令。

## 完成标准

每项性能改动需同时满足：

- 单算子或功能正确性通过；
- 对目标 Shape 有可重复的 Micro Benchmark 收益；
- 4K/16K 至少目标场景存在稳定端到端收益；
- TTFT、请求成功率和稳定性无不可接受回退；
- 精度满足比赛要求；
- 已重新 Profiling 并解释瓶颈变化。
