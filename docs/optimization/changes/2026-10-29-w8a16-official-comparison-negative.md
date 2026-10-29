# W8A16 Down 官方 4K/16K 同环境对照（负结果）

## 1. 基本信息与摘要

- 日期：2026-10-29
- 目标平台：MetaX C500（容器主机名 `c1a85143a091`）
- 涉及文件：本次不改动源码，仅归档 `benchmarks/diagnostics/run_w8a16_official_comparison.py` 串行控制产出的官方性能对照结果
- 目标：在相同环境、相同官方命令、同一 commit 下，比较 BF16 基线与 W8A16 Down-only 候选在 4K/16K 的 total tok/s 与 Mean TTFT
- 结论：**候选在两个场景的吞吐与 TTFT 均劣于 BF16 基线，且超过 ±1% 波动门槛，属于负结果**。本次仅记录实验证据，作为 W8A16 Down-only 组合的止损依据。

## 2. 问题与原因分析

此前算子级 microbench（`2026-10-29-w8a16-large-scale-after-diagnosis.md`）显示当前统一入口多数 M 点仍慢于 MetaX BF16 GEMM：大 M 独立 scale-after-dot 路径比纯 BF16 多一次 INT8→BF16 权重转换与一次输出 scale 的额外读写，小 M Split-K/整除特化虽快于通用融合但仍慢于 BF16。端到端需确认该组合路径相对 BF16 基线的真实吞吐与 TTFT 变化。

赛事门槛为吞吐不低于基线 1%、TTFT 不高于基线 1%。因此本次用同一 commit、同一官方命令、仅切换 W8A16 开关做同场对照，避免把历史不同实例基线（FULL_DECODE_ONLY 配置）当作可比基线。

## 3. 具体实现

本次不改动任何量化/GEMM 源码，仅通过 `benchmarks/diagnostics/run_w8a16_official_comparison.py` 串行控制两轮官方评测：

- **候选**：`VLLM_FL_W8A16_DOWN=1, VLLM_FL_W8A16_DOWN_INT8PACK=1, VLLM_FL_W8A16_DOWN_FUSED=0`，Down 42/42 层按 shape 分派到小 M Split-K/整除特化、大 M 独立 scale-after-dot 或通用 int8pack 融合。
- **基线**：三个开关显式置 0，纯 BF16 权重走 FlagGems MetaX GEMM。

两轮均为同一官方 `vllm serve` 参数（无 FULL_DECODE_ONLY）与同一 `benchmark_throughput_serve.py` 命令，负载 4K `[4096,1024,64,256]`、16K `[16384,1024,64,128]`。每场景四轮、跳过首轮后取后三轮平均（RUNS=4、SKIP_FIRST=1，未改动）。

## 4. 实验与验证

### 4.1 吞吐（total tok/s）

| 场景 | BF16 基线 | W8A16 候选 | 吞吐变化 | 基线×0.99 门槛 |
|---|---:|---:|---:|---:|
| 4K | 4989.25 | 4730.11 | −5.19% | 4939.36（未达标） |
| 16K | 7004.75 | 6821.13 | −2.62% | 6934.70（未达标） |

### 4.2 TTFT（Mean，ms）

| 场景 | BF16 基线 | W8A16 候选 | TTFT 变化 | 基线×1.01 门槛 |
|---|---:|---:|---:|---:|
| 4K | 3180.14 | 3428.53 | +7.81% | 3211.94（未达标） |
| 16K | 27258.78 | 28161.24 | +3.31% | 27531.37（未达标） |

### 4.3 完整性与请求成功数

两轮请求全部成功：4K 256/256、16K 128/128，四轮 Run Status 均为 SUCCESS，无异常。summary 为后三轮平均口径，raw_runs 四轮明细完整保留。

候选四轮 total tok/s：4K = 4931.92 / 4595.96 / 4624.12 / 4970.25；16K = 6827.55 / 6824.43 / 6820.16 / 6818.81。

基线四轮 total tok/s：4K = 5063.40 / 4752.09 / 5108.80 / 5106.85；16K = 7034.51 / 7005.24 / 7006.79 / 7002.22。

## 5. 与已有改动的关系

延续 `2026-10-28-w8a16-int8pack-production.md`（小 M Split-K/整除特化）与 `2026-10-29-w8a16-large-scale-after-diagnosis.md`（大 M 独立 scale-after-dot）接入的组合路径。算子级 microbench 已显示多数点慢于 BF16，本报告端到端结果与之一致，确认组合路径在官方负载下无收益。此前 `2026-10-23-w8a16-down-benchmark.md` 的旧独立反量化负结果（4K −5.47%）与本报告量级相近。

## 6. 合规性与风险

不改官方命令、数据、启动参数或 Benchmark 实现；量化开关为自研 W8A16 Down 路径的既有能力开关。结果与历史不同实例基线不可直接比较：历史基线（4990.64 / 6566.87）使用 FULL_DECODE_ONLY，本报告可比基线为同容器 BF16（4989.25 / 7004.75）。W8A16 Down-only 在 4K/16K 均负收益且 TTFT 超 1% 门槛，按 ADR 止损条件应记录并暂停该 Down-only 组合，保留 BF16 回退；后续路线需结合能力证据另行讨论，不能据此直接推定 W8A8 可获得收益。

## 7. 复现方式

容器内任务目录与原始文件：

```text
/workspace/benchmark_results/w8a16-official-comparison-20261029-02/
├── candidate/
│   ├── summary_20261029_141125.csv   # 候选 summary
│   └── raw_runs_20261029_141125.csv  # 候选四轮明细
├── bf16/
│   ├── summary_20261029_150308.csv   # 基线 summary
│   └── raw_runs_20261029_150308.csv  # 基线四轮明细
├── complete.txt
└── ...（commands.json / server.log / benchmark.log / source-hashes.json 等）

/workspace/benchmark_results/w8a16-official-controller-20261029-02/run.log  # 控制日志（exit-code=0）
```

版本：插件基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，FlagGems 基点 `a7620cc191a0b42e040194622c5758b22a7a25dc`，均含未提交实现；复现需结合实际源码/差异与实验指纹。
