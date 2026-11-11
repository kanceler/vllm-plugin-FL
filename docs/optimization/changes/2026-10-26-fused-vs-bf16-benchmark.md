# 新实例 BF16 与 Down 融合候选官方性能评测记录

## 1. 基本信息与摘要

日期：2026-10-26（UTC+8）。任务标识：`fused-vs-bf16-20261026-01`。

本次用户授权在新实例执行既有 Down 融合实现与无自研量化 BF16 的官方 4K/16K 性能对比。本次未修改量化源码或 Kernel；将本机既有实现同步到原始干净远端后，使用环境开关区分模式。

**实际结果：BF16 完整完成 8 轮；融合候选在模型初始化的动态形状编译阶段失败，未开始候选 Benchmark，因此没有可计算的融合收益。**

平台：单卡 MetaX C500，GPU bus `0000:0e:00.0`，显存 65536 MiB；容器 `5c374b08eca5`；主机/容器内存限额 128 GiB（137438953472 bytes），swap 为 0。Python 3.12.11、PyTorch `2.8.0+metax3.7.0.7`、vLLM 0.24.0、驱动 3.8.30、MACA 3.7.0.38。

仓库基点：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。本机分支 `perf/gemm-w8a16-down-fused`，原有未提交源码保留；远端同步前干净。同步范围：`vllm_fl/__init__.py`、`vllm_fl/quantization/w8a16/`、对应单元测试。源码归档、逐文件指纹及依赖清单见证据目录。本次未提交、未推送。

## 2. 问题与原因分析

既有融合实现将 INT8 权重和 FP32 per-output-channel scale 在 Kernel tile 内相乘并舍入 BF16，再执行 BF16 dot、FP32 累加，避免完整临时 BF16 权重的写回与再读。真实 Down 本地权重为 `[2048,6144]`，42 层，TP=1。

独立 GPU 测试可以通过，但不代表与 vLLM 默认动态形状图编译兼容。本次模型启动在 `torch.compile` 的 `(1,2048)` 编译范围失败，日志明确报告：动态 `input_ids/positions` 的 token 维度被特化为常数 2048，触发 `torch.fx.experimental.symbolic_shapes.ConstraintViolationError`。错误的用户调用栈包含 `fused.py:63` 的 Triton launch。

该错误发生在健康检查通过之前，并非 OOM。不能通过修改官方 compilation 参数、禁用编译或改用其他客户端绕过后声称完成官方候选测试。需要另行获准修复融合调用与动态形状编译的集成。

## 3. 本次执行流程

1. 记录原始进程、GPU、内存限额、端口、仓库版本与脚本指纹；确认无 GPU 进程，9031 空闲，空闲显存 826 MiB。
2. 归档远端原始接入文件，再同步既有融合实现；显式指定 `PYTHONPATH=/workspace/vllm-plugin-FL`，核对实际模块路径。
3. 执行单元测试。首次错误地设置 `DOWN=0`，3 项接入测试失败、14 项通过；纠正测试环境为 `DOWN=1` 后 17 项全部通过（41.19s）。两次日志均保留。
4. 新建 BF16 服务，`DOWN=0,FUSED=0,OFFLINE=0`，确认启用数 0、健康检查 200；完整执行官方性能命令。
5. BF16 结束后停止本任务进程组，记录 GPU 无进程、显存 826 MiB、端口空闲；新建 `DOWN=1,FUSED=1,OFFLINE=0` 候选服务。
6. 候选加载日志为 42/42 层 INT8、FP32 scale、`fused_bf16_dequant`，随后编译失败。未启动候选评测客户端，失败服务已退出，最终 GPU 空闲、9031 可绑定。

只读资源采样每约 15 秒记录主机内存、进程 RSS 和 mx-smi；不运行额外 GPU 负载。两场景之间未人工清缓存或重启，没有删除持久编译缓存。

## 4. 实验与验证

官方脚本 SHA256：`0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`，与已有归档版本一致。保持 `RUNS=4,SKIP_FIRST=1`，summary 为每场景第 2/3/4 轮平均。

| 场景 | BF16 total tokens/s | BF16 Mean TTFT (ms) | 请求成功数（每轮） | 融合候选 |
|---|---:|---:|---|---|
| 4K | 4186.32 | 4730.26 | 256/256，4 轮全部成功 | 未开始 |
| 16K | 5713.89 | 37423.01 | 128/128，4 轮全部成功 | 未开始 |

| 场景 | 第 1 轮 total tok/s | 第 2 轮 | 第 3 轮 | 第 4 轮 |
|---|---:|---:|---:|---:|
| 4K | 3962.40 | 3697.49 | 4378.18 | 4483.28 |
| 16K | 6427.85 | 5509.52 | 5518.37 | 6113.78 |

| 场景 | 第 1 轮 Mean TTFT (ms) | 第 2 轮 | 第 3 轮 | 第 4 轮 |
|---|---:|---:|---:|---:|
| 4K | 4527.27 | 4411.97 | 4937.11 | 4841.71 |
| 16K | 28799.28 | 39179.36 | 38336.31 | 34753.35 |

本地已核对 raw 共 8 行、summary 共 2 行、请求成功数及后三轮平均一致。轮次波动明显，保留全部明细，不择优。新实例结果不与历史实例直接计算优化收益。

模型 math_500 精度未测试；候选模型健康检查、运行时融合覆盖和正式性能待验证。42 层加载启用不等于模型推理成功，也不等于已验证运行时零回退。

## 5. 与已有改动的关系

复用 `2026-10-23-w8a16-down-fused.md` 的现有实现；不改其数值契约、分块参数或其他算子。原根目录规则描述的 W8A8 与仓库后续 W8A16 ADR 存在历史差异；本次按用户明确要求测试已有 W8A16 融合候选，不重新选择路线。

本次补充了新实例 BF16 完整结果，并暴露了此前独立测试未覆盖的模型编译集成缺陷。17 项独立测试通过不能替代模型启动、精度或端到端性能验证。

## 6. 合规性与风险

未改官方服务参数、性能脚本、负载、数据集或正式统计口径。未使用旧实例 BF16 数据代替本次基线，未编造候选成绩。所有性能门槛与精度验收仍未完成候选验证。

建议另行批准最小集成修复：以具备 symbolic-shape fake 实现的自定义算子封装现有融合调用，将运行时 M 分块与 Triton launch 留在算子实现内；保持量化算法及 BF16 舍入位置不变。需要同时检查 capability 条件是否额外产生动态形状 guard，并新增动态 M 编译回归验证。该建议尚未实施；修复成功后，只有确认原 BF16 实际路径和环境未受影响，才能复用本次同实例基线，否则重新建立基线。

## 7. 复现方式与证据

远端：`/workspace/benchmark_results/fused-vs-bf16-20261026-01/`。

本地：工作区 `artifacts/fused-vs-bf16-20261026-01/`，包含原始归档 `remote-evidence.tar.gz`、源码归档/指纹、环境、单元测试、`bf16/` 与 `fused/`。本地编排工具 `run_comparison.py` 原样调用下列命令；失败停止，不改参数重试。

官方服务命令（加载已有 Conda/MACA 环境）：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL VLLM_PLUGINS=fl
export VLLM_FL_W8A16_OFFLINE=0
unset VLLM_FL_W8A16_CHECKPOINT
# BF16：两个开关均为 0；融合候选：两个开关均为 1。
export VLLM_FL_W8A16_DOWN=0 VLLM_FL_W8A16_DOWN_FUSED=0
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

官方评测命令：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

正式 BF16 文件：`summary_20261026_025344.csv`、`raw_runs_20261026_025344.csv`；完整 stdout 见 `bf16/benchmark.log`，启动和清理证据见 `before.txt/healthy.txt/after.txt`。候选失败见 `fused/serve.log` 第 158 行起及 `status.json`；没有候选 raw/summary。
