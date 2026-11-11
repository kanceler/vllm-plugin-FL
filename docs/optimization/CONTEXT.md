# MiniCPM5-2B 优化上下文

> 最后更新：2026-10-30（三算法联合搜索、覆盖与调优审计计划归档；未实施新框架）

## 当前目标

本工作区负责模块一 GEMM，复用已有 MetaX C500 BF16 Baseline 与六组 Profiling，主线已调整为 weight-only W8A16（见 ADR-0002）。当前先补真实 Shape/耗时、调用链/布局、数值分布及硬件支持等 GEMM 专项基础分析，再细化策略、验证可行性和优化 Kernel，之后进行混合精度接入与赛事验收。Attention/KV Cache 由其他成员负责，不主动扩展范围。

早期W8A8探测见[能力报告](changes/2026-10-18-down-w8a8-capability-probe.md)，其synthetic M1完整路径慢于BF16，其他M有量化码边界差异。随后W8A16 Down版本取得math_500 Level 3 103/105（98.1%）；最终小M Split-K＋大M独立scale-after-dot组合的官方4K/16K吞吐与TTFT均负收益。该最终组合未重新完整测精度，不能将历史103/105写成其重测成绩。详见[阶段总结](research/2026-10-29-w8a16-stage-summary.md)。

## 2026-10-30 当前执行入口与分工

以[三算法分发与分层调优计划](plans/2026-10-30-w8a16-dispatch-autotuning.md)为当前执行入口。本人负责W8A16 Dense/Split-K/Separate候选、真实Shape与分支覆盖、离线GPU搜索和Runtime；原生BF16 MM底层及大M配置优化由队友负责，Separate复用其接口。旧BF16/LM Head专项不自动纳入本任务，LM Head保持BF16。

当前小M Split-K是固定参数，Dense有原8组autotune，Separate的转换/scale固定而内部MM沿用其调优；尚无跨算法统一搜索。历史42/42只说明Down层覆盖，尚不能恢复正式负载的完整M和三分支频次，不将Dense认定为负收益主因。

后续按历史复用/Shape采集 → 三算法能力及正确性 → O1/O2联合搜索 → 版本绑定manifest与实际分派验证 → 全局模型验收推进。覆盖全部观测Shape并补边界、交界和留出验证；明确固定/调优/缓存来源、候选失败及实测数量。大M最终须测完整Separate，但可待队友版本稳定后密集调优。分界与候选库由实测共同确定，Dense可仅作回退。当前只归档计划及工作区规则，无新增源码或GPU实验。

## 2026-10-29 后续方案（历史范围，分工以10-30更新为准）

以[全局W8A16后续计划](plans/2026-10-29-global-w8a16-followup.md)为执行入口：Down、Packed Gate/Up、Packed QKV、O分别开关，共用底层实现；先做离线源码/证据/理论分析，再分组测完整调用micro，依据实测、Down历史官方结果及容量/开销模型选择组合，统一接入后做全局精度和官方双场景评测。取消逐组完整模型评测，失败后再必要消融。C500不适用的有价值设计保留供天数独立验证，不声称已有跨平台收益。

用户最新补充两项：核对autotune实际配置、候选/剪枝/缓存与真实M/N/K匹配情况；研究LM Head通用linear的MetaX适配和内部优化。LM Head先保持BF16，不自动加入W8A16量化组。附件报告的采样路径与本地源码佐证分别标注；新增优化与实际收益仍待验证。本次仅文档和只读分析，不代表已运行新实验。

## 2026-10-26 路线与工作方式更新

以下为10/26～10/27历史状态；当前分工与计划以本页10/30更新为准，历史实验状态见阶段总结。

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
- 早期Profiling未归档正式TTFT；2026-10-29同环境官方对照已记录BF16 Mean TTFT：4K 3180.14 ms、16K 27258.78 ms，仅用于该版本/环境的可比分析。
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
- 历史4K/16K吞吐4990.64/6566.87只作历史参考；2026-10-29同环境BF16为4989.25/7004.75，完整证据见[官方对照](changes/2026-10-29-w8a16-official-comparison-negative.md)。新环境须建立可比基线；
- 2026-10-30口径更正：±1%（含边界）为正常波动，不计提升分；吞吐相对基线>+1%才算有效提升，TTFT相对基线<−1%才算有效改善。反向超1%是有效退化，逐场景单列；“吞吐≥0.99倍、TTFT≤1.01倍”不代表有效优化。total tokens/s计分，TTFT不自动单独计分，具体公式不自行推导；
- 服务稳定性与请求成功率；
- 记录版本、实际开关、官方命令与负载、原始轮次及随机性设置；不为固定seed修改官方命令。正式性能按原脚本RUNS=4、SKIP_FIRST=1，各场景后三轮平均，保留全部raw与异常。

完整合规要求见 [`competition-requirements.md`](competition-requirements.md)。

## 当前工作边界

- 负责人范围：模块一 GEMM 优化。
- 本次交付：整合全局方案及两个补充专项、维护文档入口、只读核对源码和已有证据。
- 后续代码任务按计划逐阶段明确文件、假设与验证，遵守用户确认范围；新路线或明显扩大范围先讨论。本次未改kernel/运行配置、未跑新测试、未提交或推送。
