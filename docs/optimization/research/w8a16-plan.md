> 2026-10-23 融合方案更正：本次用户批准方案一，融合 Kernel 内必须执行 `BF16(FP32(q) * scale)` 后再参与 BF16 GEMM，FP32 累加；不采用文中旧版 scale epilogue 方案。旧版涉及累加后乘 scale 的文字仅为历史设计，已被本更正取代。本次跳过模型精度评测，算子正确性及完整官方 4K/16K benchmark 仍须验证，不能宣称完成全部赛事验收。

# MiniCPM5-2B GEMM W8A16（weight-only）实施计划

> 日期：2026-10-22
> 状态：第一阶段原型已实现；C500 9 项测试通过、42 层 Down 启用且无回退、独立输出检查通过；按用户要求暂不执行 Level 3 精度评测
> 目标：在不修改 `vllm serve` Baseline 参数的前提下，通过框架内自研 weight-only 量化路径提升 4K/16K total tokens/s

## 1. 决策摘要

### 当前执行顺序

```text
复用已有 BF16 Baseline 与六组 Profiling
→ 补齐 GEMM 专项基础分析（阶段 0）
→ 在 ADR-0002 内细化 W8A16 策略
→ 第一阶段：加载时量化 + 独立反量化 + 现有 BF16 GEMM，验证 Down 正确性与 Level 3 精度
→ 第二阶段：融合反量化 GEMM，重新验证精度、完整路径性能并逐步扩展
→ 官方精度、4K/16K 吞吐和 TTFT 验收
```

W8A16 是当前已决定的研发主线（见 [`../decisions/0002-w8a16-mainline.md`](../decisions/0002-w8a16-mainline.md)），不是已验证的最优策略。本次按用户明确调整，先做独立反量化精度验证；ADR-0002 的 kernel 内 upcast + scale epilogue 保留为后续融合目标，不代表第一阶段实现。根目录旧 W8A8 描述与此不一致时，以本次明确指定的 W8A16 范围为准。基础分析用于确认适用投影、Shape、布局和安全回退。若证据触发 ADR-0002 的改线条件，先讨论并记录新决策，再改实现。

### 阶段 0：GEMM 专项基础分析（当前优先）

沿用已有 Baseline 和热点结论，不重做全框架瓶颈分析。现有 4K/16K total tokens/s 为 4990.64/6566.87；TTFT 尚缺。

| 分析项 | 方法与产物 | 证据边界 |
|---|---|---|
| 模型结构、实际权重 | 核对 checkpoint 张量名、dtype、形状、packed/TP 映射；给出投影清单 | 结构和参数量不等于实际耗时或敏感性 |
| 真实 GEMM 工作负载 | 复用现有 trace；缺失时只补采投影与 M/N/K、stride、调用数、累计耗时、Prefill/Decode 阶段的关联 | 不用 synthetic M 冒充实际分布，不重复采全框架热点 |
| BF16 调用链与 Kernel | 追踪原 Linear method、torch/后端入口、Kernel 与布局，区分源码可达路径和实际执行证据 | 不凭 Kernel 名认定 vendor 或硬件指令类型 |
| 数值分布及敏感性 | 先检查真实权重统计；后用独立代表性输入分析投影输出误差与累积影响 | 权重范围不等于激活分布，不能凭"FFN 鲁棒"认定具体层可量化 |
| 硬件与计算开销 | 对实际 Shape 查证 INT8 权重加载、转置/预打包、upcast、scale 及通信成本 | 权重减半不等同端到端加速，需实测权重带宽主导的 Decode 场景 |

校准输入须与目标任务类型和长短分布相近，但不能包含正式 math_500 评测题/答案用于阈值选择。

先形成候选表：`投影/Shape → BF16 成本 → 权重数值敏感性 → 完整路径预期与实测 → 保留 BF16 或进入候选`。有缺失项标为待验证，不提前生成全模型量化白名单。

策略需同时考虑精度与收益：低误差但更慢的投影不应仅为量化覆盖率进入低精度；多个投影各自通过也不代表组合误差可接受。混合精度选择以自动分析产生的可复现 manifest 表达，而不是手写固定层号。

第一阶段每次 forward 独立反量化为 BF16 权重，再沿用原 BF16 GEMM；scale 已在反量化中应用，GEMM 输出不得重复乘 scale。后续融合版再在 GEMM epilogue 应用 scale。若量化层需要不支持 Shape 的 BF16 回退，先明确保留副本或恢复权重的成本。显存节省单独核算，不通过修改官方 batch/并发或服务参数兑现收益。

阶段 0 完成标准：已有证据、剩余缺口、目标投影和下一项最小实验清晰；实际 Shape/敏感性尚缺时不能宣称分析全部完成。只读检查先进行；需要新增采样代码、启动正式负载或修改实现时，先列出文件、命令、开销与风险，讨论同意后实施。

### 既有 W8A16 方案

GEMM 量化作为模块一主线，第一版选择：

```text
权重：静态、对称、per-output-channel INT8 + FP32 scale
激活：BF16（不量化）
反量化：INT8 → FP32，与 FP32 scale 相乘，最后转 BF16 权重
计算：现有 BF16 GEMM，输出 BF16（不再乘 scale）
范围：先支持 Down；其他投影、Embedding、LM Head、Norm、Attention 主体及 KV cache 保持 BF16
```

加载时量化一次，不等于提前导出离线量化 checkpoint。定义权重 `W=[N,K]`，使用 FP32 计算：

```text
amax[n] = max_k(abs(FP32(W[n,k])))
scale[n] = amax[n] / 127；全零通道设 scale[n] = 1
Q[n,k] = INT8(clamp(round_to_nearest_even(FP32(W[n,k]) / scale[n]), -127, 127))
```

持久保存 `Q=[N,K] INT8` 和 `scale=[N] FP32`，无 group、无 zero point、无位解包，不做 GPTQ 二阶误差补偿。非有限权重不进入量化路径，需记录并保留原 BF16 路径。TP 分片下按实际存储分片的输出通道计算 scale，并记录分片语义。

每次 forward 独立恢复：`W_deq = BF16(FP32(Q) * scale[:, None])`。INT8 转浮点可精确表示，但乘 scale 后转 BF16 会有舍入。第一阶段不长期缓存完整 BF16 反量化权重；若后续增加加载时恢复并缓存模式，需单独记录其用途和显存成本。

即 **weight-only W8A16**。

不优先选择：

- **W8A8**：已按 ADR-0002 放弃（量化/反量化写显存开销 + 无原生高效 INT8 收益证据，synthetic M=1 慢约 7.8x）。
- **FP8**：C500 公开资料显示不支持原生 FP8。
- **W4A16 / GPTQ4**：搁置（反量化 + 位解包复杂、精度风险高，见 [`gptq-feasibility.md`](gptq-feasibility.md)）。
- **低秩 / 截断 SVD**：历史备选，不与当前主线并行。
- **启动参数开启现成量化**：赛题禁止，且没有体现框架开发量。

## 2. 为什么 W8A16 适合当前项目

### 2.1 硬件方向匹配（低风险）

计算仍是 BF16 tensor core（C500 最成熟、最稳的路径），不依赖"原生 INT8 是否高效"这一未验证前提。持久权重采用 INT8；第一阶段 GEMM 实际仍读取反量化后的 BF16 权重，硬件执行与性能需验证。

### 2.2 针对 Decode 权重带宽

- 后续融合版：若 Decode 小 M 的实际瓶颈是权重带宽，INT8 权重加载有望降低权重读取量；收益待验证。
- 第一阶段：独立反量化增加 INT8/scale 读取、完整 BF16 权重写回及随后 GEMM 读取，不预期有性能收益；端到端 benchmark 会计入这些开销。
- Prefill：大 M、计算主导，计算仍是 BF16，无计算吞吐收益——这是 W8A16 的已知边界；
- 不做激活量化，绕开 A8 的全部问题（±1 舍入 bug、`default.flagos` 实现失败、独立量化 kernel 写显存往返）。

### 2.3 已有代码基础

- `vllm_fl/quantization/w8a8/reference.py`：per-channel 权重 scale 规则可复用；
- FlagGems `_metax/ops/mm.py`：第一阶段沿用 BF16 GEMM；`mm_kernel_nt` 的 INT8 加载、upcast 和 scale epilogue 留待第二阶段；
- `vllm_fl/quantization/w8a8/packed.py`：packed 权重适配思路可参考（只复用权重打包部分，不含激活量化）。

## 3. 第一阶段改动与调用链

只新增权重量化、独立反量化 kernel 和 W8A16 Linear 接入，不修改现有 BF16 GEMM。

```text
加载完成并取得本地 TP 分片：
  BF16 权重 → per-channel 对称 INT8 权重 + FP32 scale（一次性）

每次 forward：
  独立反量化 kernel：Q(INT8) × scale(FP32) → W_deq(BF16)
  → 原 BF16 Linear/GEMM(x, W_deq, bias)
  → 原有 TP 通信与 BF16 输出
```

现有 BF16 源码调用链参考：`ParallelLinear.quant_method.apply → torch.nn.functional.linear → mm/addmm`；FlagGems 是否接管、最终选择哪个 BF16 kernel，需在目标环境确认，不能仅凭静态源码认定实际执行路径。

| 拟修改位置 | 动作 |
|---|---|
| `vllm_fl/quantization/w8a16/`（新增） | FP32 scale 的加载时量化、数值 reference、独立反量化 Triton kernel、Linear 适配器 |
| 插件初始化/Linear 接入位置（实施前定位） | 接入受支持 Down，保留原选择逻辑、接口、bias 和通信语义，不依靠启动参数开启已有量化 |
| 相关单元测试与诊断 | 覆盖量化边界、TP 分片、反量化、Linear 输出以及启用/回退记录 |

`w8a8/reference.py` 的规则可核对复用；`w8a8/linear.py` 只参考适配结构，它接收已量化 INT8 权重，并不提供 BF16 加载时量化。不得为接入 W8A16 改坏现有 W8A8 路径。实际支持范围、导入路径、缓冲区生命周期和 CUDAGraph 兼容性需验证；不支持的情况保留原 BF16 路径并记录原因及比例。

## 4. 第一阶段验证策略

1. 核对目标环境提交、差异、实际 import 路径和模型版本，补齐与 Down 接入直接相关的基础证据。
2. 验证量化与反量化：全零通道、正负极值、舍入边界、scale dtype/shape、非连续输入及 TP 分片。
3. 使用真实 Down Shape 检查独立反量化与数值 reference 一致性，再检查 BF16 GEMM 输出误差；分别报告量化误差与实现误差。
4. 确认模型实际启用 Down 量化路径，记录覆盖范围、回退、显存/临时缓冲和异常；不以成功启动代替启用证据。
5. 使用官方原始启动及精度命令，运行 MiniCPM5-2B 的 math_500 `Level 3`，不得改数据、生成参数或忽略失败样本对结果完整性的影响。归档得分、有效/失败样本数及输出异常。
6. 精度门槛为 `≥0.950` 且核心能力及输出行为无明显异常；历史 `0.962` 仅作参考，跨环境比较需说明可比性。CUDA 可作前置验证，不能代替 C500 最终实现的重新评测。

第一阶段可以记录反量化及完整路径耗时，但不以速度提升为通过条件，也不宣称完成赛事性能验收。正式 4K/16K 吞吐和 TTFT 留待后续性能版本；命令保持官方原样。

## 5. 第一阶段 Down MVP 通过条件

- 加载时量化一次，持久权重为 INT8、scale 为 FP32，每次 forward 独立恢复 BF16 权重并调用现有 GEMM。
- Down 的单算子正确性及模型路径启用证据完整，其他投影保持 BF16，不硬编码固定层号或评测长度。
- 官方 math_500 Level 3 精度 `≥0.950`，无明显能力异常，并完整记录错误与回退。
- 结果明确标注为 Down MVP；通过不代表全模型 W8A16 或后续融合版已通过。

独立反量化先舍入权重再 GEMM，而融合版先对 INT8 值进行 BF16 GEMM、最后乘 scale，两者浮点舍入顺序不同。后续融合或扩展 Gate/Up 后，必须重新做算子正确性和 Level 3 精度验证。

## 6. MVP 之后的优化阶梯

第二阶段再实现 INT8 权重加载 → kernel 内 BF16 upcast → FP32 累加 → FP32 scale epilogue → BF16 输出，消除完整 BF16 中间权重写回。潜在收益来自降低权重访存及后续 kernel 优化，必须实测。以下均为后续候选，不属于第一阶段改动。

### 优化 1：目标 Shape 的 GEMM 特化

围绕 MiniCPM5 高频 Shape 优化：

- Packed Gate/Up：`K=2048, N=12288`；
- Down：`K=6144, N=2048`；
- Packed QKV：`K=2048, N=2560`；
- O：`K=2048, N=2048`。

调优项包括 tile M/N/K、warp/block 数、pipeline stage、K 分片、权重布局和预打包。配置由 Shape capability 表驱动，不写死模型层号。

### 优化 2：weight-scale epilogue 融合

把 `acc × per-channel weight scale + optional bias → BF16` 放进一次 Kernel，避免 FP32 累加结果或完整 BF16 反量化权重作为独立中间张量写回显存。

### 优化 3：SiluAndMul → GEMM epilogue 融合

MiniCPM5 MLP 的 Gate/Up 输出经过 `SiluAndMul(SiLU(gate) × up)`。当前这是独立 kernel：GEMM 先写 `[M, 12288]` BF16 中间张量，再读回做激活。把 SiluAndMul 融进 Gate/Up GEMM epilogue：

```text
gate_up GEMM epilogue:
  对 gate 半区：SiLU(x)
  对 up 半区：x
  相乘 → 直接输出 [M, 6144]
```

预期收益（待实现验证）：省掉一次 kernel launch + 一次 `[M, 12288]` 中间张量写/读往返。历史 profiling 中 `triton_poi_fused_mul_silu` 在部分场景占约 1.4%。

### 优化 4：权重预打包与加载缓存

- 在加载阶段转换为 Kernel 最优布局；
- scale 与 INT8 权重连续存储或按访问模式排列；
- 避免每次 forward transpose/contiguous；
- 量化与预打包只执行一次；
- cache key 包含 runtime、device、Shape 和 layout 版本。

### 优化 5：Prefill/Decode 双策略

保持相同 W8A16 数学契约，但允许不同 Kernel：

- Prefill：大 M、计算吞吐优先；
- Decode：小 M、权重读取与启动开销优先；
- mixed batch：由 Shape/capability 选择，不修改调度语义；
- 不复制两套权重。

### 优化 6：自动混合精度

若全量 W8A16 精度不足，通过离线工具生成 manifest：

```text
默认 W8A16
+ 少数敏感层/投影 BF16
```

选择目标是用最少 BF16 FLOPs 恢复到 0.950 以上。Manifest 是数据产物，不修改 Kernel 或模型代码。

### 优化 7：outlier channel BF16 side path

仅在普通 W8A16 + mixed precision 无法同时达到精度和吞吐时启用。它会增加权重表示复杂度，不应抢在基础 W8A16 性能验证之前。

## 7. Go/No-Go 决策

### 第一阶段 Go

满足第 5 节 Down MVP 条件即可进入后续融合讨论；独立反量化较慢本身不是量化路线止损证据。

### 第二阶段性能 Go

- 最终目标平台的实际融合实现通过 math_500 Level 3（≥0.950 且无明显能力异常）；
- Down 与 Packed Gate/Up 至少一个主要 Decode Shape 在完整路径下显著快于 BF16；
- Prefill 不回退或回退成本可控；
- 4K/16K 至少有一个超过 1%，另一个不突破回退线；
- 两场景各自满足 `candidate TTFT <= baseline TTFT × 1.01`；正式 TTFT 基线目前未归档，不编造数值。
- 正式比较使用相同环境、官方命令和负载建立可比 BF16 基线；历史吞吐不能自动充当新环境正式基线，最终报告两场景平均提升。

### No-Go / 转向

- 第一阶段精度不达标：先排查实现误差，再用独立输入分析敏感性并讨论混合精度，不使用 math_500 题目或答案调阈值；
- 第二阶段融合版在 MetaX 落地后，完整路径性能仍无收益；
- 只有极少、正式负载不出现的 Shape 加速；
- 精度恢复后需要保留太多 BF16 层，端到端收益不足 1%；
- 量化改变服务行为或被规则方认定不合规。

## 8. 参考资料

- MetaX C500 Kernel 优化指南：<https://github.com/MetaX-MACA/mcoplib/blob/main/Skills/optimized-cuda-kernels/references/c500-optimization-guide.md>
- MetaX 部署资料（C500 支持 BF16/W8A8/W4A16、非原生 FP8）：<https://developer.metax-tech.com/api/client/document/file/278/preview/?file_type=pdf>
- FlagGems MetaX/C500 端到端性能问题记录：<https://github.com/flagos-ai/FlagGems/issues/5144>
- NVIDIA INT8/weight-only 行为只能作算法参考，不能替代 C500 实测。
