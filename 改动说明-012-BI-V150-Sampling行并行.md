# 改动 #012：BI-V150 Sampling top-k/top-p 行并行

> 环境：天数 BI-V150（16 SM / warp 64），MiniCPM5-2B，FP32 logits，词表 130560。
> 开发基线：`perf/BI-V150` 的 `8f17b5d5668082a10f765c6dc8e972862c754447`（已含 #011）。
> 状态：算子及真实路径验证通过；与 #013 联合完成双场景简化端到端测试。默认关闭，联合 MATH 尚未运行。
> 编号说明：本改动在实验阶段记为 #014，本次提交按正式编号 #012 归档，不是另一项新增优化。

---

## 1. 为什么这样改

实际随机采样走 vLLM `TopKTopPSampler.forward_native` →
`apply_top_k_top_p_triton` → `_topk_topp_kernel`，而不是根据源码中存在某个
reference 实现就推断热点。实测常见条件为 `top_p≈0.95`、`k=None`。

上游 Kernel 使用 `min(SM数, batch)` 个程序。BI-V150 只有 16 SM，B64 时每个
程序串行处理 4 行 logits，长词表的过滤工作缺少同一 SM 上多个独立 CTA 的并发机会。
此前实际 Decode profiling 中，此过滤占 4k B64 约 6.35ms/步、16k B31 约 3.20ms/步。

## 2. 改动内容

复用原 vLLM `_topk_topp_kernel`，只把 grid 改为每行一个程序。保留 block 参数、
pivot、mask、重复值处理、过滤算法和后续随机采样，不重新实现排序或 RNG。

| 文件 | 作用 |
|---|---|
| `vllm_fl/patches/iluvatar_sampling.py` | 设备/形状保护、原 Kernel 行并行启动、device/stream 独立 scratch |
| `vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py` | 在天数 backend 导入时安装 opt-in 包装 |
| `tests/unit_tests/patches/test_iluvatar_sampling.py` | ABI、设备/形状回退与过滤数值回归 |

开关：`VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING=1`，默认 0；关闭或 unset 后重启 worker 回退。

适用条件：实际厂商/型号为 Iluvatar BI-V150、FP32 logits、B17–64、V130560、
vocab 连续、k/p 元数据形状和设备合法。其余情况调用原函数；上游 Kernel ABI 不匹配不安装。
既有算子选择逻辑不删除，不扩展到其他硬件。独立 scratch 为每 stream 31.875MiB，
上游自身的缓存可能仍保留；一次过滤仍是一发 GPU Kernel。

## 3. 收益与正确性验证

### 3.1 单独算子 A/B

真实模型 logits、同环境、20 次预热、40 对 AB/BA，CUDA Event 中位数：

| 来源 | B | 原 GPU ms | 本改动 GPU ms | 加速 |
|---|---:|---:|---:|---:|
| 4k | 64 | 6.632 | 2.191 | 3.03× |
| 16k | 30 | 3.384 | 1.821 | 1.86× |

16k 算子 A/B 的实际批次是 B30，与 profiling 的 B31 区分。模型释放后使用同一 logits
复核最终保护路径：B64 为 6.634→2.200ms，B30 为 3.369→1.816ms；CPU 提交增加约 12µs/次。

过滤结果逐位相同；相同 GPU RNG 起点下，native sampler 的 token IDs 及结束 RNG 状态一致。
两边 profiler 均只有一次 `_topk_topp_kernel`。此前本改动与原 Mixed Attention 合计 56 项回归通过；
4k/16k 的短模型检查每场景均 64/64 请求成功。提交前在隔离的拟提交代码树上联合复核
本改动、#013 和 Mixed 回归，**79 项全部通过**，Ruff/格式及差异检查通过。

### 3.2 与 #013 联合端到端结果

2026-11-11，同一服务依次执行 4k 预热→4k 正式→16k 预热→16k 正式。
按用户指定，每场景完整预热一轮、正式一轮，**只取正式轮，不平均、不挑最好值**。
原版 benchmark 的 `run_once` 由外部管理器调用两次，脚本文件、请求参数和统计未改。

| 场景 `[输入,输出,并发,请求数]` | 历史 #011 后 total tok/s | #012+#013 正式 total tok/s | 增量 |
|---|---:|---:|---:|
| `[4096,1024,64,256]` | 3330.59 | **3456.39** | **+3.78%** |
| `[16384,1024,64,128]` | 2781.10 | **2865.16** | **+3.02%** |

| 场景 | 历史 Mean TTFT ms | 联合正式 Mean TTFT ms | 变化 | 正式 Mean TPOT ms |
|---|---:|---:|---:|---:|
| 4k | 4471.86 | 4481.68 | +0.22% | 87.72 |
| 16k | 176468.06 | 172305.75 | −2.36% | 160.91 |

正式轮 256/256、128/128 请求成功，输入/输出 token 数分别为
1048576/262144 和 2097152/131072；时长 379.22s、777.70s。
预热轮 3064.74、2821.19 total tok/s 仅存档，不计入正式结果。

**口径限制**：这是两项同时启用的结果，不能单独归因为本改动。对照是 2026-11-09 #011
后三轮历史均值，本次按用户要求没有重测基线，不能排除跨时段环境/调度波动。
本次不是官方四轮/后三轮均值，也没有完成联合 MATH，不能据此宣称精度验收通过。

## 4. 与已有改动的关系

- 与 #013 的 Attention 索引/网格优化正交，联合收益见上表，未测两者各自 E2E 增量。
- 保留 #006/#009/#010 的 Attention 参数及 #011 的 Mixed 拆分；不重复 #007 的 GEMM 配置工作。
- 不启用 KV-aware Admission Guard、Shape-aware Split/Tile、GEMM-SiLU 融合等 NO-GO 方向。

## 5. 复现与回退

在本分支对应依赖环境启动服务，以下同时开启 #012/#013 以复现联合配置：

```bash
VLLM_PLUGINS=fl TORCHINDUCTOR_COMPILE_THREADS=1 \
VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING=1 \
VLLM_FL_BIV150_DECODE_DIRECT_INDEX=1 \
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

另一终端运行简化两轮；不修改官方脚本的 `RUNS=4`，只直接调用其单轮函数：

```bash
VLLM_PLUGINS= python3 - <<'PY'
import json
import runpy

b = runpy.run_path("benchmarks/benchmark_throughput_serve.py")
common = b["build_common_args"]("/workspace/MiniCPM5-2B", 9031, "minicpm")
for case in [(4096, 1024, 64, 256), (16384, 1024, 64, 128)]:
    for run_id, label in [(1, "warmup"), (2, "formal")]:
        metrics = b["run_once"](case, run_id, common)
        assert metrics["successful_requests"] == case[3]
        print(json.dumps({"case": case, "round": label, "metrics": metrics}))
PY
```

原函数的日志分母仍显示 `/4`，实际这里只调用两次。不要将预热计入最终值。
可分别将两个独立开关设 0 并重启服务回退。此次没有修改默认开关或固定 serve 参数。

工作区完整记录：`天数BI-V150-非Attention算子Profiling与优化报告.md`、
`天数BI-V150-Sampling与Decode直接索引联合端到端Benchmark报告.md`；联合原始证据在
`artifacts/tianshu_combined_e2e_20261111/evidence/`。这些路径属于项目工作区，未随插件源码提交。

## 6. 合规性

只改既有 Kernel 的启动并行度，保留框架算子选择、模型权重、采样算法和 RNG。
不改比赛服务参数、Benchmark、题集或评分器，不启用量化或投机采样。
设备/形状来自真实运行元数据，不识别测试题目或 Benchmark 场景名称。
