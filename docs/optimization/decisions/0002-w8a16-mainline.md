# ADR-0002：量化主线从 W8A8 调整为 W8A16（weight-only）

> 状态：已决定（取代 ADR-0001 的 W8A8 主线）
> 日期：2026-10-21
> 范围：模块一 GEMM 优化

## 背景

ADR-0001 将主线固定为通用 dynamic-token/channelwise W8A8。经 2026-10-18 Down 能力探测（`changes/2026-10-18-down-w8a8-capability-probe.md`）与 2026-10-21 运行时调用链核查（`profiling/runtime-chain-20261021.md`），现有证据不足以支撑 W8A8 的收益前提，故调整主线。

## 决策

```text
weight-only INT8（W8A16）
+ 激活保持 BF16（不做动态激活量化）
+ 计算仍走 BF16 tensor core（kernel 内将 INT8 权重 upcast 回 BF16）
+ BF16 安全回退
```

量化契约：

```text
权重：静态、对称、per-output-channel INT8 + FP32 scale
激活：BF16（不量化）
计算：BF16 × BF16 → FP32 累加（权重在 kernel 内 upcast）
输出：乘 per-channel weight scale → BF16

> 简化点（相对 GPTQ/W4A16）：推理范式对齐 weight-only + kernel 内反量化，但算法大幅简化——per-channel 对称量化，无 group、无 zero point、无位解包；反量化只是 kernel 内 INT8 → BF16 的精确 upcast（INT8 值域 [-128,127] 可被 BF16 无损表示），scale 在 epilogue 末尾乘一次，不做 GPTQ 的二阶误差补偿。
```

## 为什么不采用 W8A8

### W8A8 的 kernel 调用路径

```text
激活量化（运行时，每次 forward）:
  _dynamic_per_token_quant_int8_kernel    # 读 BF16 激活 → 写 INT8 激活 + scale 回显存
GEMM:
  flag_gems scaled_mm → scaled_mm_kernel   # INT8 × INT8 → INT32 → scale → BF16
```

### 原因 1：量化/反量化写显存开销

激活量化是独立 kernel，每次 forward 都要「读 BF16 激活 → 写 INT8 激活回显存 → scaled_mm 再读回 INT8」，多一趟写+读的显存往返。前驱融合（SiluAndMul/RMSNorm → quant）可部分消除写回，但实现复杂度高，且量化本身不产生计算收益。

### 原因 2：C500 上 INT8 计算路径不优于 BF16

当前唯一可比数据点（synthetic Down，M=1 / K=6144 / N=2048）：

| 路径 | 中位耗时 |
|---|---:|
| BF16 Linear | 0.021 ms |
| INT8 scaled_mm | 0.165 ms（慢约 7.8x） |

且原生 INT8 能力尚未验证（"Kernel 名不能证明底层是否使用原生高效 INT8 指令"）。在缺乏原生高效 INT8 证据、唯一数据点反而慢 7.8x 的情况下，W8A8 的收益前提不成立。

> 说明：此对比仅覆盖 M=1，M=8..4096 因激活量化码契约不一致被跳过、无数据；但综合「写显存开销 + 无 INT8 收益证据 + 精度余量仅 0.012」判断，不上 W8A8。

## 采用 W8A16

### 优势

1. 权重 BF16 → INT8 减半，Decode（小 M、权重带宽主导）权重读取减半；
2. 激活保持 BF16，**完全不用激活量化 kernel**，绕开 A8 的全部问题（±1 舍入 bug、`default.flagos` 实现失败、独立量化 kernel 写显存往返）；
3. 计算仍是 BF16 tensor core，不依赖「原生 INT8 是否高效」这一未验证前提，风险低。

### kernel 调用路径

```text
权重预处理（加载时一次性离线，非每步 kernel）:
  BF16 权重 → per-channel absmax → INT8 权重 + FP32 scale

GEMM（每次 forward）:
  改版 mm_kernel_nt（在现有 BF16 mm_kernel_nt 上改）
    → 权重按 INT8 加载 → kernel 内 upcast 成 BF16 → BF16×BF16 → FP32 累加
    → epilogue 乘 per-channel weight scale → 输出 BF16

Embedding / LM Head / Norm / Attention / KV cache：保持 BF16 不变
```

### 需要更改的 kernel 部分

1. B 矩阵加载：`tl.load` 由 BF16 改为 INT8；
2. kernel 内 upcast：`b = b.to(tl.bfloat16)`；
3. epilogue：累加器乘 per-channel weight scale（`acc * w_scale[rn]`）后再写回；
4. 权重布局/预打包：权重以 INT8 + 连续 scale 存储，避免每次 forward transpose/contiguous。

### 验证策略

W8A16 需改 kernel，若改动较复杂，先不直接在 MetaX 上做完整实现：
1. 先在 CUDA（NVIDIA）侧用等价实现跑 math_500 精度校验，确认 weight-only INT8 精度合格（≥0.950 且核心能力无异常）；
2. 精度合格后再回到 MetaX 实现 kernel 改动，避免浪费时间。

## GPTQ4 / W4A16 搁置

### 原因

1. 反量化 + 位解包（bit unpack）实现复杂；
2. 4-bit 精度很可能不达标（基线 0.962、最低 0.950，绝对下降空间仅 0.012）；
3. 必须先在 CUDA 侧做精度校验，不能直接在 MetaX 上实现。

### 触发条件

- W8A16 必须先通过精度门槛（否则 W4 也无意义）；
- 放在最后，有时间且 W8A16 落地后有余量再做；
- W8A16 通过后，优先做 kernel 内优化（tiling / Split-K / epilogue / 布局），而非先做 W4A16。

## 修改本决策的条件

以下任一成立时重新评估：

- W8A16 在 CUDA 侧精度校验不达标，或 MetaX 落地后 4K/16K 吞吐无收益；
- 出现可证明 C500 原生高效 INT8 GEMM 的新证据（重新评估是否回到 W8A8）；
- 规则方对 weight-only 量化的合规性有明确判定。
