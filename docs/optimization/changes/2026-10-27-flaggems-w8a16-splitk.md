# FlagGems W8A16 小 M 两段 Split-K 诊断实验

## 1. 基本信息与摘要

日期：2026-10-27（UTC+8，远端实验日志）。任务 ID：`flaggems-w8a16-splitk-20261027-02`。用户明确批准小 M、Split-K=1/2/4/8 的诊断实验；不修改生产分派、开关，不跑官方模型 Benchmark。

新增文件：`benchmarks/diagnostics/benchmark_flaggems_w8a16_splitk.py`。基于 FlagGems MetaX `mm_kernel_splitk_partial/mm_kernel_splitk_reduce` 的现有结构，适配 INT8 权重转换和最终 scale epilogue，仅作为诊断代码；不是另写整套框架，也没有注册为正式算子。

沿用本机 W8A16 任务分支 `perf/gemm-w8a16-down-fused`，vLLM 插件基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems 基点 `a7620cc191a0b42e040194622c5758b22a7a25dc`。远端生产仓库没有修改，本机已有工作保留；未提交、未推送。

主要结果：M=1/16 的 Split-K=8 分别约为 0.041984/0.045056 ms，明显优于独立反量化和原融合，已接近同场 MetaX BF16，但仍慢 3.14%/5.39%。M=64 则 Split-K=4 比 8 更快。该方向有相对已有量化路径的实际增量收益，但尚无端到端吞吐或精度结论，本轮不新增生产开关。

## 2. 问题与原因分析

上轮配置调整使 M=1/16/32 的融合相对原实现约快 2 倍，但仍慢于 MetaX BF16；编译缓存证据显示 BF16 在 M=1/16 使用 Split-K=8，而现成 weight-only 接口没有对应 K 分片执行策略。因此本轮固定上一轮小 M tile，单独验证 K 并行能否降低权重转换和 dot 的串行循环成本。

不把“已有双缓冲”误当成已经达到足够并行度；也不直接照搬 BF16 原子累加到输出的数值组织。W8A16 使用统一 INT8 权重和 per-output-channel FP32 scale，与是否 Split-K 无关；最终先合并 FP32 partial 再乘 scale，避免每个分片先写 BF16。

## 3. 具体实现

第一段沿用 MetaX 两段 Split-K 的输出 tile 映射、K 迭代拆分、边界 mask 和 FP32 partial 布局：

```text
total_iters = ceil(K/BK)
iters_per_split = ceil(total_iters/SPLIT_K)
每个 pid_k 处理其连续 K 迭代范围
partial[pid_k,M,N] = FP32 accumulate(x_bf16 × BF16(q_int8))
```

相对原 BF16 partial 的核心适配是在 dot 前将 INT8 B 转为 BF16。第二段沿用逐分片 FP32 累加结构，在最终写回前新增 per-channel scale：

```text
y = BF16(FP32(sum(partial, split_axis)) × FP32(scale))
```

不重新量化、不生成完整 BF16 临时权重、不保留 BF16 原权重副本、不使用 BF16 atomic。保持 scale-after-dot 数值契约；不同归约顺序仍可能产生浮点差异，需数值验证。

固定 `(BM,BN,BK)=(16,64,128)`、warps=4、stages=2、partial pipeline=basic；reduce BLOCK_SIZE=256、warps=4。只改变 Split-K=1/2/4/8。Split-K=1 也走两段，作为两段结构控制项；上一轮单 Kernel 配置候选另列对照，避免混淆额外归约成本。

M=1/16/32/64，N=2048、K=6144。完整路径分配 FP32 `[S,M,N]` workspace 和 BF16 输出，计入两次 Kernel 启动、partial 写读、归约和 scale。最大测试 workspace 为 S=8、M=64 的 4 MiB。

## 4. 实验与验证

### 环境与执行隔离

同一容器 `f3a0dc142860`、单卡 MetaX C500 64 GiB、bus `0000:d4:00.0`，驱动 3.9.35、MACA 3.7.0.38；48 GiB RAM、swap 0。包和生产源码沿用上轮已核对版本，实际 FlagGems 路径 `/workspace/FlagGems/src`。

开始前确认无本任务残留服务、评测客户端或 GPU 测试，GPU 无进程、显存 860 MiB、9031 空闲。串行运行单个诊断进程，不清理持久编译缓存、不启动模型服务。首次独立 SCP 会话在完成传输前断开，随后重试传输成功；失败传输没有启动 GPU 测试。

脚本 SHA256：`421f039deb106bccc27fab24550f2fe4f84345ee396f0ec1b40735b678e3cf92`，本地与远端一致。既有 helper 复制到本实验目录，未覆盖旧结果。

### 计时条件

seed=20261027。每条路径三轮，轮换顺序，`do_bench(warmup=100,rep=300,return_mode='median')`，汇总取三个延迟中位数的中位数。先完成编译/调优/正确性再计时，首调用另存。所有路径是同场比较，不与前一实例或前轮绝对值拼接收益。

对照包括显式 MetaX BF16、独立 Triton 反量化＋同一 MetaX mm、原 FlagGems weight-only、上一轮配置单 Kernel，以及四个 Split-K 候选，共八条路径。以下单位均为 ms。

| M | MetaX BF16 | 独立反量化＋MetaX | 原融合 | 配置单 Kernel |
|---|---:|---:|---:|---:|
| 1 | 0.040704 | 0.085504 | 0.172800 | 0.085248 |
| 16 | 0.042752 | 0.086784 | 0.177920 | 0.087040 |
| 32 | 0.050688 | 0.096512 | 0.178176 | 0.086784 |
| 64 | 0.072192 | 0.116736 | 0.178688 | 0.143104 |

| M | Split-K=1（两段） | Split-K=2 | Split-K=4 | Split-K=8 | 已测较快项 |
|---|---:|---:|---:|---:|---|
| 1 | 0.087040 | 0.052224 | 0.048896 | 0.041984 | 8 |
| 16 | 0.088064 | 0.053248 | 0.050432 | 0.045056 | 8 |
| 32 | 0.088320 | 0.081408 | 0.067584 | 0.061440 | 8 |
| 64 | 0.144128 | 0.113408 | 0.099072 | 0.103680 | 4 |

| M | 较快 Split-K 相对独立路径加速 | 相对原融合加速 | 相对 BF16 耗时增加 |
|---|---:|---:|---:|
| 1 | 2.04× | 4.12× | +3.14% |
| 16 | 1.93× | 3.95× | +5.39% |
| 32 | 1.57× | 2.90× | +21.21% |
| 64 | 1.18× | 1.80× | +37.23% |

M=1 的 S=8 workspace 为 64 KiB，M=16 的 S=8 为 1 MiB，M=32 的 S=8 和 M=64 的 S=4 均为 2 MiB。它们保留同一 INT8 权重格式及持久权重节省，但节省显存不自动等于已获得模型吞吐收益。

### 正确性与编译证据

- 对每个 S 验证 `(M,N,K)=(1,17,33)` 和 `(3,129,257)`，包括零权重通道、尾块及部分分片没有 K 迭代的情况，共八项边界检查，全部通过。
- 真实 Down 四种 M、四个 S 共 16 个结果按 FP32 matmul 后乘 FP32 scale 再转 BF16 的参考验证，L2 相对误差全部低于 0.005，并通过 rtol=0.02、atol=0.125 检查；实际 Split-K 误差范围为 0～约 `6.72e-5`。原融合及配置候选也核对自身契约误差，均通过。
- 独立反量化权重与 reference 完全相等。该算子正确性不等于模型量化精度，未运行 math_500。
- 保存每个 M/S 的 partial 与 reduce TTIR/TTGIR、mcfatbin 和资源字段。较快项的 partial 后端 n_regs=82、n_spills=0、shared=24576 bytes；这不是实测占用率或带宽证据。
- 正常退出码 0。本地审计四个 Shape、八条路径、三轮共 96 条原始计时，全部中位数与汇总一致；本机/远端脚本指纹一致。

## 5. 与已有改动的关系

本轮基于 FlagGems MetaX 两段 Split-K 实现做最小 W8A16 适配，保留现成普通融合和上一轮配置作为对照，不恢复旧手写 scale-before-dot 路线。

相较 `2026-10-27-flaggems-w8a16-baseline-configs.md`，主要变化是执行策略从完整 K 循环变为 K 分片及 FP32 合并；tile 固定，不把本轮结果归因于再次搜索 BM/BN/BK。适配只存在诊断脚本，FlagGems 生产文件和原 autotune 配置保持不变。

该结果支持继续研究小 M 策略，而非证明所有 M 应使用 Split-K=8。当前 M=64 的较快项为 4，说明 workspace、归约与并行度之间存在取舍。尚未测试更大 M，不外推收益范围。

## 6. 合规性、风险与后续判断

所有实验为独立 synthetic GEMM，未修改模型输出策略、数据集、正式命令、框架主要选择逻辑或依赖。无并发 GPU 任务，不新增生产开关或 M 分派，不运行官方 Benchmark。

Split-K 相对原融合和独立反量化已有明确的算子级增量收益，M=1/16 接近 BF16，是可继续分析的候选；不因单点略慢就宣称整条量化路线无效，也不把 INT8 存储节省自动当成时延收益。尚未建立模型级吞吐、TTFT、精度和量化覆盖结果，不能宣称赛事验收通过。

下一步若继续，优先拆分 partial、reduce 与完整路径耗时，检查小 M 剩余启动/归约成本及编译资源，再决定是否调整归约或分片配置。任何生产后端覆盖、M/Shape 分派、进一步算法变化及模型测试另按已讨论流程确认；本轮只完成批准的诊断范围。

## 7. 复现方式与证据

远端目录：`/workspace/benchmark_results/flaggems-w8a16-splitk-20261027-02/`。本地：工作区 `artifacts/flaggems-w8a16-splitk-20261027-02/`，原始归档 `remote-evidence.tar.gz` 已下载并展开。

`results/summary.json`、`raw_timings.json`、`correctness.json`、`first_calls.json`、`metadata.json` 保存原始数据；`m*-s*-partial/reduce.*` 保存编译产物；`audit.json` 为本地审计。`preflight.txt/run-environment.txt/postflight.txt/run.log/exit.txt` 记录环境、命令、退出和资源状态。

辅助脚本 `benchmark_flaggems_w8a16.py`、`benchmark_flaggems_w8a16_configs.py` 以及 `reference.py/dequant.py` 需与新脚本一起准备。本次使用：

```bash
export PYTHONPATH=/workspace/FlagGems/src
unset VLLM_FL_W8A16_DOWN VLLM_FL_W8A16_DOWN_FUSED VLLM_FL_W8A16_OFFLINE VLLM_FL_W8A16_CHECKPOINT
python /workspace/benchmark_results/flaggems-w8a16-splitk-20261027-02/benchmark_flaggems_w8a16_splitk.py \
  --helpers-dir /workspace/benchmark_results/flaggems-w8a16-splitk-20261027-02 \
  --output-dir /workspace/benchmark_results/flaggems-w8a16-splitk-20261027-02/results
```

复现前按任务要求归档并核对进程、GPU、端口、源码和环境；现有目录拒绝覆盖，重跑使用新实验 ID。语法、diff 空白检查及实际 GPU 运行通过。结束后进程完全退出，GPU 无进程、利用率 0%、显存 860/65536 MiB，9031 可绑定，远端生产仓库仍干净。
