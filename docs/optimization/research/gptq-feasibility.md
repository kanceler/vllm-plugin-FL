# GPTQ 可行性分析：先确认真实工作负载与推理契约

> 日期：2026-10-19（UTC+8）  
> 状态：分析候选，不是路线变更；未执行 GPTQ 校准、权重转换或 Kernel 开发  
> 当前主线：ADR-0002 W8A16（GPTQ/W4A16 搁置）

## GPTQ 与现有路线的关系

GPTQ 是利用近似二阶信息进行离线权重量化的算法，解决在校准输入分布上如何降低权重量化误差；它不直接规定推理时激活位宽、数据布局或芯片 Kernel。原始实现用输入构造二阶统计，并通过逐块量化和误差补偿处理权重，提供分组、阻尼与 activation-order 等选项。

依据：[原论文](https://arxiv.org/abs/2210.17323)、[作者的 gptq.py](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)、[量化与打包实现](https://github.com/IST-DASLab/gptq/blob/main/quant.py)。论文在其他模型和 NVIDIA GPU 上的收益不能视为 MiniCPM5-2B/C500 的收益。

必须分别讨论三个问题：

1. 权重如何得到量化值：朴素 round-to-nearest 或 GPTQ 等离线算法。
2. 最终数值契约：例如 W8A8、W8A16 或 W4A16，以及 group size、对称性、scale、zero point、输出与累加精度。
3. 推理实现：如何读取 packed 权重、融合转换、完成矩阵计算并保持框架接口与通信语义。

因此 GPTQ 不是与 INT8/INT4 同层级的选项，也不天然等于 W4A16。若只用 GPTQ 改善 W8A8 的 INT8 权重，仍需保持既有 per-output-channel 对称权重契约，并处理动态 A8 的额外误差；不能把 grouped/asymmetric GPTQ 文件直接交给现有 W8A8 adapter。

## 反量化有代价，应分析在哪里支付

现有 W8A8 概念路径：

```text
BF16 activation → per-token INT8
INT8 × INT8 → INT32
在输出 epilogue 乘 activation/weight scale → BF16
```

典型 W4A16 候选路径：

```text
BF16 activation + packed INT4 weight
→ 在片上解包、转换、应用 group scale/zero point
→ BF16/浮点矩阵计算与累加
→ BF16 输出
```

这是待实现和验证的候选设计，不能据此断言 C500 已有高效接口。若按 K 分组，权重表示为 `W[n,k] ≈ s[n,g(k)] * (q[n,k] - z[n,g(k)])`，scale 随 K 组变化，不能像 per-channel W8A8 一样在整个 K 累加结束后只乘一个输出通道 scale。

“没有免费的反量化”应落实为成本核算：解包、scale/zero 读取、类型转换、寄存器/共享内存占用和计算流水线。融合能够避免完整 BF16 中间权重写回和额外启动，但不会让这些工作变成零成本。

每次推理先把整个权重反量化到显存再调用 BF16 GEMM，会增加中间读写；预先长期保留完整 BF16 副本又会抵消部分显存收益。二者可以作诊断对照，不能未经测量作为正式优化方案。是否需要自研新 Kernel，要结合平台现有接口、实测效率和赛事实质性代码要求决定，而不是仅凭压缩算法名称判断。

## 结合真实 M 分布再判断价值

| 情况 | GPTQ + W4A16 的潜在价值 | 主要风险/验证项 |
|---|---|---|
| 小 M 权重读取占主导 | 压缩权重降低读取量，同时可避免 A8 量化 | 解包和转换、并行度、启动成本可能抵消收益 |
| 大 M 计算占主导 | 依赖融合、布局与浮点计算效率 | 压缩权重不代表获得原生 INT4×INT4 计算吞吐；需与 BF16/W8A8 完整路径比较 |
| 敏感投影 | GPTQ 可能降低同位宽下的权重误差 | 校准误差和模型精度仍须验证，不能假设全部层适合 INT4 |

当前 synthetic M=1 的通用 W8A8 路径慢，既不能证明 W4A16 更快，也不满足推翻 W8A8 的全目标 Shape 性能证据条件。当前真实工作负载 M/投影的累计耗时表仍待完成，优先完成该基础分析。

## 校准与框架集成的分析清单

- 独立、可复现的校准/保留输入，覆盖推理、问答及长短分布；不使用正式 math_500 题目或答案选层、调阈值。128～512 条仅是起点，需要明确实际 token 数和覆盖率。
- 自动输出层/投影敏感性、模型指纹、数据版本及选择证据；与 BF16 输出及最终 math_500 分数区分。
- 明确位宽、group size、signed/unsigned、zero-point 编码、scale dtype、打包顺序、是否 act-order、g_idx/置换方式及 TP 分片语义。
- packed QKV/Gate-Up 的边界、scale 对应关系和分片不能混淆；原模型 LM Head、Embedding 当前仍按 ADR 保留 BF16。
- Down 的 K=6144，单个 FP32 二阶矩阵占 144 MiB；K=2048 时占 16 MiB。这只是二阶矩阵本身，不包括副本、分解、激活等峰值开销。逐层处理，不先假定可以同时保留所有矩阵。
- 搜索到仓库有 GPTQ/Marlin op schema 和通用候选 registry，不等于 C500 实际支持或存在高效实现；目前未获得 C500 GPTQ 路径的实际执行证据。
- 保持官方服务参数和 Benchmark 不变。不能仅导出 GPTQ 权重并调用原生加载功能申报优化；需要最终框架实际执行的实质性算子/转换集成与可复现收益。

## 实现落点（调用链与 kernel 改动）

### 离线（打包）

```text
GPTQ 校准（可选，外部工具，FlagGems 无现成校准 kernel）→ INT4 权重 + group scale/zero
+ _convert_weight_to_int4pack.py  (int32 → uint8 byte-pair 打包)
+ _dyn_quant_pack_4bit_weight.py  (权重 + scales_zeros + bias → 单 tensor)
```

注意：`_convert_weight_to_int4pack` 注释明确"未实现 Marlin tiled 格式，只有 byte-pair packing"——高性能 Marlin 布局需另补。

### 推理（现有 = naive fallback，慢）

```text
A(BF16) → _weight_int4pack_mm_with_scales_and_zeros.py
  _weight_int4pack_mm_kernel：逐 K rank-1 外积（解包 nibble + (q-zero)*scale → FP32 FMA）
  ← 没有 tl.dot，是 correctness fallback，非高性能 kernel
```

说明：W4A16 片上计算可走 BF16（解包 + 反量化到 BF16 后 BF16 MMA），不依赖原生 INT4/INT8 Tensor Core；真正成本是位解包 + group scale/zero + 4bit 精度。

### kernel 改动清单

| 位置 | 动作 |
|---|---|
| FlagGems `ops/_weight_int4pack_mm_with_scales_and_zeros.py` | 重写 kernel：naive 逐 K rank-1 → tiled `tl.dot(BF16×BF16)`（解包 + dequant 到 BF16 后走 BF16 MMA） |
| FlagGems `ops/_convert_weight_to_int4pack.py` | 补 Marlin tiled 布局（现只有 byte-pair packing） |
| GPTQ 校准 | FlagGems 无现成离线校准 kernel，需外部 AutoGPTQ 或自写 |

核心工作量：大（重写 GEMM kernel + 打包布局 + 校准 + 精度校验）。

## 建议的后续决策顺序

先完成已批准的 GEMM Shape/耗时诊断 → 确定真实高价值投影与小/大 M 分布 → 对比 W8A8 与明确契约的 GPTQ 候选（必要时包括 W4A16） → 对真实权重和独立输入评估误差及完整路径成本 → 与用户讨论是否调整 ADR → 再实现对应 Kernel 和混合精度接入。

本文件不授权量化或更换技术主线，也不表示已证明 GPTQ 在 C500 或正式 4K/16K 场景中更优。
