# 天数 final 候选基线确认 + 实际算子调用路径审计

## 1. 基本信息与摘要

- 日期：2026-11-04
- 设备平台：天数 BI-V150（Iluvatar CoreX，`COREX_VERSION=4.5.0`）
- 涉及文件：本次**无源码/Kernel 改动**，仅归档天数 final 候选的官方性能基线 + 实际算子调用路径审计结论
- 目标：
  1. 确认天数 final 候选（`tianshu-final-20261104`，commit `f82fa16`）的官方 4K/16K 基线，作为后续 W8A16 GEMM 量化的**新 BF16 对照基线**
  2. 审计实际运行时的算子调用路径（GEMM / attention / 小算子 / 量化），为 W8A16 接入方式提供依据
- 结论：
  - 基线（BF16，无量化）：4K **3073.11** tok/s、16K **2790.28** tok/s
  - 相对官方天数基线（2028.01 / 915.15）提升 **+51.5% / +204.9%**（队友框架优化成果，非 GEMM 量化）
  - 实际调用路径：**GEMM 走 vendor BLAS（非 FlagGems）、attention 走 vLLM Triton（非 FlagGems）、小算子走 FlagGems、量化未启用**

## 2. 问题与原因分析

此前没有建立天数平台的可比 BF16 基线。历史 4990.64 / 6566.87 是**沐曦 C500（MetaX）**的 Profiling 基线（`metax-c500.md`），与天数（Iluvatar BI-V150）是两块不同的卡，不可直接比较。技术报告列出的官方天数基线为 4K 2028.01、16K 915.15 tok/s。

队友在 `tianshu-final-20261104` 分支（#002~#010）对 attention 分块/流水/瓦片、decode warp 数、Prefill chunk 预算、调度器等做了设备属性推导式优化。需要以官方命令在同一会话跑一次完整 4K/16K，确认当前候选的真实吞吐与 TTFT，作为后续 W8A16 量化叠加的对照点。

同时，W8A16 接入前必须弄清：实际 GEMM 到底走哪条路径（FlagGems 还是 vendor BLAS）、框架是否已有通用量化算子、attention 是否走 FlagGems。这决定 W8A16 的 dequant+GEMM 应挂在哪个入口。

## 3. 具体实现（本次为审计，无代码改动）

本次不改动源码，仅做两件事：(a) 归档官方性能基线；(b) 用运行时证据（serve.log、`enabled-ops.txt`、`computation_graph.py`）审计实际调用路径。

### 3.1 两层 dispatch 机制（源码结构）

- **A 层 —— vLLM 自带 GEMM/linear 内核选择**（`vllm.model_executor.kernels.linear`）
  - BF16 未量化：`RowParallelLinear.forward()` → `UnquantizedLinearMethod.apply()` → `dispatch_unquantized_gemm()` → 从 `_POSSIBLE_KERNELS` 选 vendor kernel
  - INT8 量化：`Int8ScaledMMLinearKernel.apply_weights()` → 从 `_POSSIBLE_INT8_KERNELS` 选
  - `quantization/quant_linear.py::add_oot_quant_kernel()` 把 CUDA 的 `INT8/FP8/FP8_BLOCK/MXFP8` 候选复制进 `PlatformEnum.OOT`，供天数(OOT/iluvatar)复用
- **B 层 —— vllm_fl 自研 dispatch 系统**（`vllm_fl/dispatch/`，registry + policy）
  - 管 builtin op：activation / normalization / rotary / quantization / attention_backend / fused_moe
  - 天数 vendor backend（`vendor/iluvatar/`）只覆盖 activation / normalization / rotary，**不含** attention、**不含** quantization

### 3.2 实际调用路径（运行时证据，非静态推断）

| 算子 | 实际路径 | 证据 |
|---|---|---|
| **GEMM/linear** | iluvatar **vendor BLAS**（corex），**非 FlagGems** | `enabled-ops.txt` 无 mm/matmul/gemm/linear/scaled；`linear_backend='auto'` |
| **attention** | vLLM **Triton attention** + 队友 tile_policy，**非 FlagGems** | serve.log 无 "Using FlagGems attention backend" 打印；`attention_backend` 走 `default.flagos` 但内部返回 `TRITON_ATTN` |
| **rms_norm / rotary / silu 等小算子** | **FlagGems** | `enabled-ops.txt` 有；`rms_norm` 在捕获图出现 423 次 |
| **量化** | **无** | serve.log：`quantization=None, quantization_config=None`；`fuse_norm_quant/act_quant/attn_quant` 全 False |

### 3.3 框架已有通用量化算子（源码，非本次启用）

| 组件 | 位置 | 作用 |
|---|---|---|
| `dynamic_per_token_quant_int8` | `dispatch/backends/flaggems/impl/quantization.py` | 激活**动态 per-token** INT8 量化（`flag_gems.scaled_int8_quant` + in-tree Triton fallback） |
| `FLW8A8DynamicLinearKernel` | `quantization/w8a8/linear.py` | W8A8 动态线性层，调 `flag_gems.scaled_mm`（**原生 INT8×INT8 GEMM**） |
| `add_oot_quant_kernel()` | `quantization/quant_linear.py` | 注册 INT8/FP8/MXFP8 kernel 候选 |
| `w8a8/{moe,moe_experts,packed,reference,int8_mode}` | `quantization/w8a8/` | 完整 W8A8 方案 |

**关键**：这套是 **W8A8（原生 INT8×INT8）**，靠 `scaled_mm`；而 W8A16 是 **weight-only INT8 + BF16 激活**（权重转 BF16 → BF16 dot → FP32 累加 → scale），是**另一条独立路径**。且 W8A8 的 `scaled_mm` 需要 FlagGems 接管 GEMM，而当前实际运行里 FlagGems **未接管 GEMM**，两条线在当前 vendor-BLAS 主路上互不重叠。

## 4. 实验与验证

### 4.1 环境与软件

- 软件：PyTorch + Iluvatar CoreX 4.5.0；插件 `tianshu-final-20261104`（commit `f82fa16`）；FlagGems `4df52d91`
- 官方命令（逐字，未改）：serve 用 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`；benchmark 用赛题 `benchmark_throughput_serve.py`
- 负载：4K `[4096,1024,64,256]`、16K `[16384,1024,64,128]`；`RUNS=4, SKIP_FIRST=1`，取后三轮平均

### 4.2 吞吐与 TTFT（BF16，无量化）

| 场景 | total tok/s | Mean TTFT (ms) |
|---|---:|---:|
| 4K | **3073.11** | 5115.36 |
| 16K | **2790.28** | 177074.49 |

四轮明细（4K：2934.8 / 2892.52 / 3007.18 / 3319.62；16K：2573.03 / 2797.09 / 2792.59 / 2781.15），4K 首轮低于后三轮（"越跑越快"），16K 稳定。`benchmark.exit=0`，8 轮全 SUCCESS。

### 4.3 相对官方基线的提升

| 场景 | 官方天数基线 | 本次 | 提升 |
|---|---:|---:|---:|
| 4K | 2028.01 | 3073.11 | **+51.5%** |
| 16K | 915.15 | 2790.28 | **+204.9%** |

> 上述提升是队友框架优化（attention 分块/调度，非 GEMM 量化）的成果。技术报告自己警告跨会话噪声可达 48%，故本次作为一次实测基线，不宜当最终定论；后续 W8A16 对比需在同一会话同命令下做 A/B。

### 4.4 实际运行证据（确认无量化、GEMM/attention 归属）

- serve.log：`dtype=torch.bfloat16, quantization=None, quantization_config=None`
- serve.log：`pass_config: {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, ...}`
- serve.log：`OpManager initialized: 11 ops with 23 implementations`；`Op 'attention_backend' using 'default.flagos'`
- `enabled-ops.txt`：FlagGems 实际接管 softmax/where/sort/cumsum/mul/add 等，**无 mm/matmul/gemm/linear/scaled**
- `computation_graph.py`（7713 行捕获图）：`linear: "bf16[...`、`torch.ops.vllm.unified_kv_cache_update`、`rms_norm` 423 次、`rotary_embedding` 84 次

## 5. 与已有改动的关系

- 依赖队友 `tianshu-final-20261104`（#002~#010）框架优化，本次不叠加任何 GEMM 量化。
- 与本地 W8A16（`vllm_fl/quantization/w8a16/`，未接入）**尚未接通**：本次基线是 W8A16 的对照 BF16，不是 W8A16 自身成绩。
- 与历史 10-29 Down-only 负结果（沐曦 C500，4K −5.19% / 16K −2.62%）是不同平台：那次是沐曦，本次是天数；不能直接比较。

## 6. 合规性与风险

- 未改官方 serve/benchmark 命令、数据或 Benchmark 实现；量化开关未启用（无规则风险）。
- 本次只有 BF16 基线与路径审计，**无模型精度、无 W8A16 量化收益、无 TTFT 改善验证**，不构成赛事验收。
- **待裁决冲突**：队友技术报告 `已排除方向.md` 第 9 条已将「量化」标记为"未采用"，与本工作区 AGENTS.md 的 W8A16 主线（ADR-0002）冲突；继续 W8A16 前需先与队友对齐是否纳入最终交付。
- 待验证：本基线仅单次实测，需同会话 A/B 复核；W8A16 的 dequant+GEMM 入口需按 3.2 的 vendor-BLAS 路径设计。

## 7. 复现方式

容器内实验目录：`/workspace/experiments/tianshu-final-20261104-01/`（`final-sha.txt = f82fa1670509f081aa13b58464cb6e5627fc10c4`，`benchmark.exit = 0`）。

```bash
# serve（官方命令，未改）
/usr/local/bin/python3.12 /usr/local/bin/vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072

# benchmark（官方命令，未改）
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

关键环境：`PYTHONPATH=/workspace/tianshu-final-20261104`，`VLLM_PLUGINS=fl`，`COREX_VERSION=4.5.0`。

正式 summary：`/workspace/benchmark_results/summary_20261104_170016.csv`（及 `raw_runs_20261104_170016.csv` 四轮明细）。本次仅归档状态与路径审计，未提交或推送 Git。

