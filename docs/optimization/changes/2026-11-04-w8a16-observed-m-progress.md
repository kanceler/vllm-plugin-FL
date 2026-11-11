# W8A16 已观测 M 增量矩阵：完成状态与待复验项

## 1. 基本信息与摘要

2026-11-04 核验 11 月 3 日后台实验。阶段为离线三算法完整调用矩阵，尚未完成交界精调或生产分派接入。测试于 `2026-11-03T19:01:13+08:00` 正常完成，76 个问题、927 个候选全部通过既定正确性门槛，保存 2,660 条终测计时样本。没有新增生产 Kernel 改动。

容器隔离目录：`/root/experiments/2026-11-03-w8a16-boundaries/`。原始容器仓库基线为插件 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`、FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc`。隔离目录覆盖 11 月 1 日已有的显式算法后端、调优脚本和 reference，不覆盖原始工作目录。BF16 提供者 SHA256 为 `2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec`；显式后端 SHA256 为 `41397bd1101e400126dd288183c68577e7ebf7742fac6c31cc4fb46d98f926b4`。

## 2. 问题与原因分析

真实 Shape 补采已取得 Mixed、Decode 和尾部执行 M，需要确认历史候选在这些 M 下的正确性和性能，避免继续采集或重复整个历史矩阵。不同 N/K 的赢家不同，不能预设 Split-K/Separate 单阈值。算法内部赢家也不等于优于原始 BF16。

## 3. 具体实现与执行范围

复用 `tune_metax_w8a16_algorithms.py`，新增实验执行脚本及结果分析脚本，存于工作区 `experiment-artifacts/2026-11-03-w8a16-boundaries/tools/`。四组 `(N,K)` 为 Down `(2048,6144)`、Gate/Up `(12288,2048)`、QKV `(2560,2048)`、O `(2048,2048)`。

每组 M：`4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,160,320,1554,2048`。包含全部当前已观测 M，160 为人工代表点。每个点均筛选候选，因此本轮不能称为“留出点仅验证最终固定策略”；该项仍待后续。

激活 BF16，权重静态对称 per-output-channel INT8，scale FP32。Dense/Split-K 在 dot 前将权重转 BF16，FP32 累加后 scale；Separate 包含权重转换、BF16 MM、中间 BF16 输出与 scale。不是原生 INT8×INT8 GEMM。数据为 synthetic，量化层覆盖与模型算法调用覆盖未测试。

## 4. 实验与验证

详细的 76 点原生 BF16 基准、三算法耗时、加速比、配置、误差和五轮范围见[明细补充](2026-11-04-w8a16-observed-m-details.md)。按用户 11 月 4 日明确决定，scale-before-dot 因精度计算契约不一致弃用，后续不再测试或纳入候选；历史结果保留。

- 软件：PyTorch `2.8.0+metax3.7.0.7`、Triton `3.0.0`；设备 MetaX C500，MACA `3.7.0.38`。
- screening warmup/rep：20/60 ms；终测 warmup/rep：50/150 ms，五轮交错，汇总中位数。此为诊断计时协议，不是官方评测轮次。
- 正确性：927/927 通过，最大 relative L2 `0.004732837434858084`，门槛 relative L2 `<0.005`、rtol `0.02`、atol `0.125`。量化误差另存 summary，不能与 Kernel 误差混同。
- 样本为重复完整调用的稳态计时；不能据此宣称新编译数、底层 autotune 缓存未命中数或图重放性能。Separate 仅单个顶层候选，内部 BF16 提供者可使用自己的调优缓存。

| 投影 | 本轮已测点中的三算法赢家 | 下一步 |
|---|---|---|
| Down | 4～64 Split-K；72～96 Separate；104/112 Split-K；160/320/1554/2048 Separate | 104/112 两算法差仅约 0.20%/0.60%，不固化为真实反转；精测交界 |
| Gate/Up | 4～16 Split-K；24～96 Dense；104 起所测点 Separate | 精测 16～24、96～104，保留 Dense |
| QKV | 4～64 Split-K；72 起所测点 Separate | 精测 64～88；1554 的 BF16 对照存在异常差异 |
| O | 4～64 Split-K；72～96 Dense；104 起所测点 Separate | 精测 64～72、96～104；复核极小 M 的 BF16 对照 |

以上仅描述已测点，不是连续区间全部通过的分派规则，也不是全局最优。

**收益复验风险：**QKV M=1554 的 Separate 中位数约 `0.231424 ms`，原 BF16 `0.308992 ms`，表面快约 25.10%；但量化后 BF16 权重 MM 为 `0.208128 ms`，Separate 反而慢约 11.19%。相同 Shape 的两个 BF16 控制项有显著差异，尚未确定配置、地址/布局或缓存原因，不能将表面收益认定为量化收益。O M=8/16 本轮相对原 BF16 分别慢约 1.98%/6.93%，未复现 11 月 1 日的正收益；M=4 约快 0.88%，处于 ±1% 波动范围。

结束时 GPU 空闲、无计算进程、显存 826 MiB，见 `gpu-after-screen.txt`。11 月 4 日登录后测试进程已退出，`mx-smi` 返回 `No available devices were discovered`；容器主机名与上一轮不同，但无法仅凭这一点确定设备不可见原因。未启动新 GPU 测试。

## 5. 与已有改动的关系

依赖 11 月 1 日显式算法和 11 月 3 日 Shape 采集，未改写历史结果。原始生产 dispatch、官方服务和 Benchmark 均未改动。旧报告的小 M 收益本轮未复现，保留两份证据，下一阶段核对 BF16 控制项配置。

## 6. 合规性与风险

本轮为独立 synthetic 算子实验，无模型精度、官方吞吐或 TTFT 结果；不构成赛事验收。尚待交界局部精调、代表 Shape 的宽候选审计、固定策略留出验证、真实布局/图重放与多层工作集验证。当前 GPU 不可见，待资源恢复后再执行。生产接入、manifest 与模型级评测尚未开始。

## 7. 复现方式与证据

容器在已加载 Conda/MACA 的交互环境中执行：

```bash
bash /root/experiments/2026-11-03-w8a16-boundaries/tools/run_stage.sh
```

该脚本使用新结果目录，原目录已存在时不覆盖；重测应另设实验目录。本地完整原始证据：

```text
experiment-artifacts/2026-11-03-w8a16-boundaries/
├── metadata/{preflight,effective-pythonpath,gpu-before,gpu-after-screen,screen-complete}.txt
├── results/observed-plus-anchors/{metadata,screening,raw_final,summary}.json
├── screen.log
└── tools/{run_stage.sh,analyze_results.py,local-backend.patch}
```

分析命令：

```bash
python3 experiment-artifacts/2026-11-03-w8a16-boundaries/tools/analyze_results.py \
  experiment-artifacts/2026-11-03-w8a16-boundaries/results
```

此次仅归档状态、取回证据并分析，未提交或推送 Git。
