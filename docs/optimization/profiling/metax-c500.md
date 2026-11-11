# MetaX C500 Baseline 与 Kernel Profiling

> 数据状态：历史实测归档。本文由既有 `report.md`、`new_report.md` 和阶段工作方案合并整理；未在本轮重新运行实验。

## 1. 环境与方法

| 项目 | 值 |
|---|---|
| 硬件 | MetaX C500 |
| vllm-plugin-FL | `flagos-2026-s2` |
| FlagGems | `v5.3.5` |
| 模型 | MiniCPM5-2B |
| dtype | bfloat16 |
| cudagraph | `FULL_DECODE_ONLY` |

采样使用镜像内置工具：

- `mcTracer`：采集 Kernel、API 和 Memory event；
- `cut_stat`：汇总 Trace 中的 Kernel/API/Memory 数据。

采样命令为：

```bash
mcTracer --mctx --attach <EngineCore PID>
```

Profiler attach 到已运行的 EngineCore，进程外使用纯 aiohttp 客户端产生负载。解析时过滤单个 Kernel `dur` 为 INT64 溢出哨兵值的异常事件。

### 已知限制

- `FULL_DECODE_ONLY` 只对 Decode 使用 Full CUDAGraph，Prefill/Mixed Batch 不使用 CUDAGraph，因此 Prefill 可能包含额外 launch 与调度开销。
- 分阶段组与整体组的并发不同，占比不可直接横向比较。
- Decode 组在 Prefill 后进入稳态 Decode 时 attach，但仍应结合原始时间线判断采样边界。
- 当前只有汇总统计，没有在仓库中归档原始 Trace JSON。
- TTFT 基线尚未记录。

## 2. Baseline

| 场景 | total tokens/s | output tokens/s | 成功率 | 平均时延 |
|---|---:|---:|---:|---:|
| 4K（4096/1024/64/256） | 4990.64 | 997.90 | 256/256 | 65.31 s |
| 16K（16384/1024/64/128） | 6566.87 | 386.66 | 128/128 | 166.89 s |

括号内依次为 input length、output length、concurrency 和 prompt 数。

## 3. Profiling 负载

| 组 | input length | output length | concurrency | prompts | 用途 |
|---|---:|---:|---:|---:|---|
| 16K Prefill | 16384 | 1 | 4 | 8 | 隔离 Prefill 大矩阵计算 |
| 16K Decode | 16384 | 1024 | 4 | 8 | 长 KV Decode 稳态 |
| 4K Prefill | 4096 | 1 | 8 | 16 | 隔离 Prefill 大矩阵计算 |
| 4K Decode | 4096 | 1024 | 8 | 16 | 长 KV Decode 稳态 |
| 整体 4K | 4096 | 1024 | 64 | 256 | 官方口径整体场景 |
| 整体 16K | 16384 | 1024 | 64 | 128 | 官方口径整体场景 |

Decode 保留真实 4K/16K 上下文，不用短输入代替；Prefill 使用 `output=1` 隔离 Prompt Processing。

## 4. 总体结果

| 场景 | GEMM 占比 | Attention 占比 | 主要特征 |
|---|---:|---:|---|
| 16K Prefill | ~64%（nt 55.6% + splitk 8.7%） | ~29% | GEMM 与大块 Prefill Attention 双热点 |
| 16K Decode | ~70%（nt 63.0% + splitk 6.7%） | ~21% | GEMM 主导，Decode Attention 次之 |
| 4K Prefill | ~94% | ~3.6% | GEMM 绝对主导 |
| 4K Decode | ~80%（nt 72.4% + splitk 7.3%） | ~9.5% | GEMM 主导 |
| 整体 4K | 59.6% | ~31.8% | 高并发下 Attention 占比上升 |
| 整体 16K | 47.6% | ~46.5% | GEMM 与 Attention 接近对半 |

## 5. 各组 Top Kernel

以下保留决定当前优先级所需的主要 Kernel。`pct` 为各自采样窗口内 GPU Kernel 累计时间占比。

### 5.1 16K Prefill（总 GPU Kernel 12.39 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 74769 | 6883.2 | 92.1 | 1110.0 | 55.55% | `mm_kernel_nt` |
| 2 | 2688 | 3644.0 | 1355.6 | 2549.2 | 29.41% | `flash_fwd_splitkv 128x128x64x4` |
| 3 | 56287 | 1072.2 | 19.1 | 33.5 | 8.65% | `mm_kernel_splitk` |
| 4 | 2735 | 169.7 | 62.0 | 65.0 | 1.37% | `triton_poi_fused_mul_silu_1` |
| 5 | 56288 | 152.8 | 2.7 | 3.9 | 1.23% | `zero_persistent_kernel` |

### 5.2 16K Decode（总 GPU Kernel 25.57 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 310479 | 16112.5 | 51.9 | 1077.2 | 63.01% | `mm_kernel_nt` |
| 2 | 43974 | 2605.9 | 59.3 | 65.8 | 10.19% | `flash_fwd_splitkv 128x16x16x1` |
| 3 | 1260 | 1805.0 | 1432.5 | 2548.5 | 7.06% | `flash_fwd_splitkv 128x128x64x4` |
| 4 | 85344 | 1699.3 | 19.9 | 33.6 | 6.65% | `mm_kernel_splitk` |
| 5 | 43974 | 875.8 | 19.9 | 24.8 | 3.42% | `flash_fwd_splitkv_combine` |

### 5.3 4K Prefill（总 GPU Kernel 13.15 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 232595 | 12351.5 | 53.1 | 1097.5 | 93.91% | `mm_kernel_nt` |
| 2 | 1344 | 467.9 | 348.1 | 502.8 | 3.56% | `flash_fwd_splitkv 128x128x64x4` |
| 3 | 1350 | 84.7 | 62.8 | 64.8 | 0.64% | `triton_poi_fused_mul_silu_1` |

### 5.4 4K Decode（总 GPU Kernel 23.44 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 346161 | 16968.6 | 49.0 | 1102.3 | 72.40% | `mm_kernel_nt` |
| 2 | 85175 | 1708.7 | 20.1 | 34.5 | 7.29% | `mm_kernel_splitk` |
| 3 | 43303 | 1614.9 | 37.3 | 43.3 | 6.89% | `flash_fwd_splitkv 128x16x16x1` |
| 4 | 42757 | 623.0 | 14.6 | 16.4 | 2.66% | `flash_fwd_splitkv_combine` |

### 5.5 整体 4K（总 GPU Kernel 26.53 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 113901 | 15798.3 | 138.7 | 1106.4 | 59.55% | `mm_kernel_nt` |
| 2 | 26281 | 6022.6 | 229.2 | 250.2 | 22.70% | `flash_fwd_splitkv 128x16x16x1` |
| 3 | 2646 | 1163.9 | 439.9 | 622.1 | 4.39% | `flash_fwd_splitkv 128x64x64x4` |
| 4 | 2856 | 977.9 | 342.4 | 511.2 | 3.69% | `flash_fwd_splitkv 128x128x64x4` |

### 5.6 整体 16K（总 GPU Kernel 29.88 s）

| rank | calls | total ms | avg us | max us | pct | Kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 30120 | 14233.1 | 472.6 | 1144.1 | 47.64% | `mm_kernel_nt` |
| 2 | 6586 | 9019.4 | 1369.5 | 2557.4 | 30.19% | `flash_fwd_splitkv 128x128x64x4` |
| 3 | 7530 | 3056.7 | 405.9 | 530.7 | 10.23% | `flash_fwd_splitkv 128x16x16x1` |
| 4 | 944 | 1743.5 | 1846.9 | 2951.9 | 5.84% | `flash_fwd_splitkv 128x64x64x4` |

## 6. 当前结论及证据边界

### 已验证

- GEMM 是六组采样中的主要热点，覆盖面最大。
- Attention 随上下文长度和高并发的重要性上升，整体 16K 中已接近 GEMM。
- RMSNorm、SiluAndMul、RoPE 和 KV Cache write 等单项占比明显低于前两类热点。

### 分析判断

- 当前优先级应为 GEMM，其次是长序列 Attention，再根据重新 Profiling 结果选择融合目标。
- GEMM 需要进一步按 M/N/K、投影类型和 Prefill/Decode 拆分，现有 Kernel 名称汇总不足以指导具体优化。

### 待验证

- `mm_kernel_nt` 与 `mm_kernel_splitk` 到 Q/K/V/O/Gate/Up/Down 投影的逐项映射。
- GEMM 的实际上层 dispatch 路径，以及 Vendor/FlagGems 各自在目标环境中承担的调用。
- 各 Shape 的调用次数、累计耗时、有效带宽与最优 Split-K 策略。

因此，本文支持“先研究 GEMM”的方向，但不支持在缺少 Shape 与 dispatch 证据时直接全局切换实现。
