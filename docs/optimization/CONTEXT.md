# MiniCPM5-2B 优化上下文

> 最后更新：2026-10-29（W8A16 Down 已完成精度验证与官方 4K/16K 同环境对照，结果为负收益；详见阶段总结）

## 当前目标

本工作区负责模块一 GEMM，复用已有 MetaX C500 BF16 Baseline 与六组 Profiling，主线已调整为 weight-only W8A16（见 ADR-0002）。当前先补真实 Shape/耗时、调用链/布局、数值分布及硬件支持等 GEMM 专项基础分析，再细化策略、验证可行性和优化 Kernel，之后进行混合精度接入与赛事验收。Attention/KV Cache 由其他成员负责，不主动扩展范围。

已完成最小 synthetic Down 探测：M=1 完整路径慢于 BF16，其余测试 M 存在量化码边界差异；当前已完成模型精度验证（math_500 Level 3 103/105、98.1%）与官方 4K/16K 同环境对照（Down-only 吞吐与 TTFT 均负收益、未达标），详见阶段总结。详见 [`changes/2026-10-18-down-w8a8-capability-probe.md`](changes/2026-10-18-down-w8a8-capability-probe.md)。这些结果不代表模型精度不达标；结合无原生高效 INT8 收益证据，主线已调整为 W8A16。

## 2026-10-26 路线与工作方式更新

> 2026-10-27 补充：在用户新提供的 C500 实例完成原 `weight_int8pack_mm` 和有限配置算子实验，见[运行报告](changes/2026-10-27-flaggems-w8a16-baseline-configs.md)。显式对照为 FlagGems MetaX BF16 mm；FP32 scale、边界/真实 Shape 已验证，现成融合已生成 A/B 双缓冲。配置候选仅改善部分 Shape 相对原融合的耗时，尚未超过 BF16，不新增生产开关、不接入模型；没有官方 Benchmark 或新模型精度结果。本轮测试进程已退出、GPU 已释放。下方“无资源”记录保留为 10/26 状态，不阻止用户授权的新实验。

- 用户已明确确认：当前 MiniCPM5-2B 主要投影的官方 BF16 基线 GEMM 使用 **FlagGems MetaX Triton 内核**，不是直接调用 vLLM 自带或 mcBLAS/mcoplib GEMM。`F.linear` 只是上层入口，Inductor/CUDAGraph 描述编译执行机制，不能据此否定后端归属。
- 仍待补齐的是具体 Shape 对应的 Kernel、配置及流水线归档，不再把主要投影的后端路线写成未确认。此前独立 Microbenchmark 未显式启用 FlagGems 的 `F.linear` 计时不能自动标为官方基线实现。
- 后续 W8A16 优先基于 FlagGems 现有 `weight_int8pack_mm`（INT8 权重、浮点激活、FP32 累加后乘 channel scale），复用其融合和 autotune，再按证据做 MetaX 适配；不继续优先扩展旧手写融合。用户已允许评估 scale-after-dot，数值契约改变后重新验证精度，尚未完成新路径运行验证。
- 旧手写实现和历史实验保留归档。实验候选不新增生产开关；仅有效候选才讨论生产接入，且仍须精度、吞吐和 TTFT 验收。无效候选记录原因和复现证据，不进入正式路径。
- 当前无 GPU 资源，只进行本地分析和明确授权的文档工作。详见 [复用评审与后续计划](research/2026-10-26-flaggems-w8a16-reuse-review.md)。
- 已补充 MetaX 专用 weight-only 后端适配计划及天数本地源码核查：两平台可共用数学契约、数据格式和通用融合骨架，但 tile、流水线、资源剪枝及实际模型路径必须分别验证。天数有专用 mm/linear 和名为 matmul_int8 的 BF16 dot 实现，本地未发现 weight_int8pack_mm 专用覆盖；不能因名称或源码存在就宣称原生 INT8 性能、完整 W8A16 融合或跨平台收益。

## 模型与软件环境

| 项目 | 值 |
|---|---|
| 模型 | MiniCPM5-2B |
| vllm-plugin-FL | `flagos-2026-s2`（原始 Profiling 所用分支） |
| FlagGems | `v5.3.5` |
| dtype | bfloat16 |
| 硬件 | MetaX C500 |
| hidden size | 2048 |
| intermediate size | 6144 |
| layers | 42 |
| attention heads | 16 |
| KV heads | 2 |
| head dim | 128 |

模型使用 16Q/2KV 的 GQA，Q:KV head ratio 为 8:1。

## 已验证事实

- MetaX 环境使用 `FULL_DECODE_ONLY` cudagraph 时服务可以稳定运行；当时默认的 `FULL_AND_PIECEWISE` 会导致 EngineCore 异常退出。
- 官方 benchmark 在当时环境中会再次初始化 GPU runtime，因此 Profiling 使用不导入 torch/vLLM/transformers 的纯 aiohttp 客户端产生负载。
- 已完成 4K/16K Baseline，以及 Prefill、Decode、整体场景共六组 Kernel Profiling。
- 六组数据中 GEMM 均为主要热点；整体 16K 中 Attention 与 GEMM 占比接近。
- 目前缺少 4K/16K TTFT 基线。
- 2026-10-18 按官方沐曦原参数成功启动 BF16 服务，官方 4K 客户端前三轮成功；外部 pilot Trace 无 M/N/K/投影字段，已停止后续负载。部分诊断结果不替代正式基线。详见 [`changes/2026-10-18-gemm-shape-pilot.md`](changes/2026-10-18-gemm-shape-pilot.md)；真实 Shape 归属仍待完成。

详细数据与采样限制见 [`profiling/metax-c500.md`](profiling/metax-c500.md)。

## 分析判断

- 模块一应先按实际 M/N/K 和投影类型拆分 GEMM 成本，再决定优化哪种实现或配置。
- 当前 GEMM 主线为 weight-only W8A16：静态 per-output-channel INT8 权重、激活保持 BF16、kernel 内 upcast 后 BF16 计算、epilogue 乘 weight scale 恢复 BF16 输出。决策见 [`decisions/0002-w8a16-mainline.md`](decisions/0002-w8a16-mainline.md)。
- 低秩资料仅保留为历史备选，不与当前主线并行推进；重新选择路线须有 ADR 止损证据并经讨论。
- Prefill 与 Decode 的 M 维差异很大，不宜默认共用同一套最优策略。
- K/V、Q/O、Gate/Up、Down 的 N/K 维差异明显，按 Shape 特化可能比全局替换更稳妥。
- 主热点优化后瓶颈可能迁移，端到端结果需要重新 Profiling 才能解释。

## 待验证事项

- 已确认主要投影使用 FlagGems MetaX Triton；待补齐具体 Shape、`mm_kernel_nt`/Split-K 等实现与投影的准确映射和运行证据。
- Vendor GEMM 与 FlagGems Triton GEMM 在六类投影及 Prefill/Decode 高频 Shape 上的差异。
- 当前 Split-K 选择是否分别适合大 M Prefill 与小 M Decode。
- Decode GEMM 的有效带宽及其距硬件带宽上限的差距。

特别说明：仓库的 MetaX dispatch 配置明确让 Attention 优先使用 vendor backend，但普通 GEMM 并未作为相同的高层 dispatch op 注册。这不代表 GEMM 绕过 FlagGems；用户已确认主要投影走 FlagGems MetaX Triton，“所有线性层由 vendor 路径绕过 FlagGems”不再作为当前路线假设。

## 回归约束

- 精度 Baseline 为 0.962，最低要求 0.950；绝对下降空间只有 0.012，且不得出现明显核心能力下降或异常输出；
- 4K total tokens/s Baseline 4990.64，最低门槛 4940.73，超过 1% 的参考线为 5040.55；
- 16K total tokens/s Baseline 6566.87，最低门槛 6501.20，超过 1% 的参考线为 6632.54；
- TTFT 不得高于对应 Baseline 1% 以上，但当前缺少正式 TTFT Baseline；
- 服务稳定性与请求成功率；
- 实验需固定版本、启动参数、负载参数和随机种子，并报告多次运行的均值或中位数。

完整合规要求见 [`competition-requirements.md`](competition-requirements.md)。

## 当前工作边界

- 负责人范围：模块一 GEMM 优化。
- 当前包含：补充阶段 0 方案、复用既有结果、只读检查真实模型与相关源码、整理证据缺口。
- 当前不包含：新增运行采样代码、模型量化、Kernel 修改、实现切换、参数调优、提交或远端推送；需要实施时先讨论并获得同意。
