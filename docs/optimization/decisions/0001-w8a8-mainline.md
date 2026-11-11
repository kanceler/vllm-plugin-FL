# ADR-0001：GEMM 量化主线采用通用动态 W8A8

> 状态：已取代（见 [ADR-0002](./0002-w8a16-mainline.md)）  
> 日期：2026-10-12  
> 范围：模块一 GEMM 优化

## 决策

模块一主线固定为：

```text
通用 dynamic-token / channelwise W8A8
+ MetaX C500 原生 INT8 scaled GEMM
+ activation quant 前驱融合
+ BF16 安全回退
```

量化契约：

```text
权重：静态、对称、per-output-channel INT8
激活：动态、对称、per-token INT8
累加：INT32
scale：FP32
输出：BF16
```

第一轮不把 W4A16、FP8、截断 SVD 或其他有损分解作为并行主线。它们只在 W8A8 触发明确止损条件后重新评估。

## 目标

- 用同一量化机制覆盖 Prefill 与 Decode；
- 提升正式 4K/16K 场景 total tokens/s；
- 精度保持 `>=0.950`，且核心能力和输出行为无明显异常；
- 不修改 Baseline 的 `vllm serve` 参数或 Benchmark；
- 量化策略体现在最终提交代码和实际执行路径；
- 支持标准 vLLM ParallelLinear，而不是只对一个层号或测试集生效。

## 架构边界

量化实现拆成三层，避免模型、数据集或运行时变化时反复修改同一份代码。

### 1. Quantization policy

负责：

- BF16 权重到 INT8 的转换；
- weight/activation scale 规则；
- 数值 reference；
- mixed-precision policy；
- manifest schema。

不依赖 MetaX Kernel 细节，不硬编码层号或评测题目。

### 2. Backend kernel

负责：

- MetaX INT8 scaled GEMM；
- Shape、dtype、alignment、runtime 与 CUDAGraph capability；
- tiling、pipeline、预打包与 epilogue；
- unsupported case 回退 BF16。

不决定哪些语义层必须量化。

### 3. Model integration

负责：

- vLLM QKVParallelLinear、MergedColumnParallelLinear、RowParallelLinear 接入；
- packed weight 与 TP 分片；
- load-time quantization；
- 可选 manifest 读取；
- 保持原 Linear 接口和通信语义。

## 泛化原则

以下内容不得硬编码进 Kernel 或量化算法：

- `math_500` 题目或 Prompt；
- 4K/16K 固定长度；
- MiniCPM5 某个固定 layer id；
- 手工复制的 scale；
- 单次实验得到的 M 值；
- 某个 checkpoint 的特殊权重名称。

模型差异通过标准模块类型、Shape capability 和可重新生成的 manifest 表达。新模型或新 checkpoint 应重新运行转换/校准工具，而不是改量化代码。

## 默认策略

第一版不依赖校准数据：

- 权重在加载时自动 per-output-channel 量化；
- 激活运行时动态 per-token 量化；
- 满足 capability 的目标 Linear 才启用；
- Embedding、LM Head、Norm 和 Attention 主体保留 BF16；
- 未支持、无法判断或发生错误时回退 BF16。

若精度不足，再按顺序引入：

1. 自动敏感层分析与 mixed-precision manifest；
2. SmoothQuant 类 activation/weight rescale；
3. outlier channel BF16 side path。

这些扩展不得改变基础 W8A8接口。

## 不采用的替代方案

### FP8

C500 没有明确的原生 FP8 scaled GEMM主路径，软件解码到 BF16不能作为性能主线。

### W4A16

潜在带宽收益更大，但精度、打包、反量化和 Kernel 风险更高。留作 W8A8 后续扩展。

### Prefill-only 截断 SVD

会引入两次 GEMM、额外中间张量、mixed batch 与双份权重问题，且精度空间只有 0.012。保留调研，不作为主线。

### 单纯切换现成算子

不满足赛题对算子开发和最终代码落地的要求，也不能保证端到端收益。

## 修改本决策的条件

只有满足以下任一条件才重新选择主路线：

- C500 trace 证明当前软件栈不存在原生或高效 INT8 GEMM；
- 包含 activation quant 和 epilogue 后，主要正式 Shape 均不优于 BF16；
- 自动 mixed precision 后仍无法达到 0.950；
- 规则方明确判定 load-time 权重量化或当前形式的自研量化不合规；
- 最终4K/16K吞吐长期无法越过1%波动线。

任何路线变化都需要新增 ADR，记录证据、影响和迁移方案。
