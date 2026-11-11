# FlagGems W8A16 现成实现验证与 MetaX 配置实验

## 1. 基本信息与摘要

实验日期：2026-10-27（UTC+8，按远端实验日志）。实验 ID：`flaggems-w8a16-20261027-01`。用户授权连接新 C500 实例，按已讨论的 W8A16 方案执行算子验证及后端优化实验；不做官方模型 Benchmark。

新增诊断代码：

- `benchmarks/diagnostics/benchmark_flaggems_w8a16.py`：显式 MetaX BF16、现成 weight-only、独立反量化三路径基准，保存选中配置和编译产物。
- `benchmarks/diagnostics/benchmark_flaggems_w8a16_configs.py`：复用原 FlagGems JIT 函数体的有限配置筛选与独立输入复验。

沿用本机 W8A16 任务分支 `perf/gemm-w8a16-down-fused`，框架基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems 基点 `a7620cc191a0b42e040194622c5758b22a7a25dc`。新容器两仓库开始和结束均无未提交差异；本机已有修改保留。本次未修改任何 FlagGems 生产 Kernel 或 vLLM 接入、未新增生产开关、未提交/推送。

结论：现成接口支持本次 BF16 激活、INT8 权重、FP32 scale 及测试形状；大 M 实际生成 A/B 双缓冲。配置调整使 M=1/16/32 相对原融合约快 2 倍，但所有复验点仍慢于 MetaX BF16，大 M 选中候选还慢于原融合。因此没有达到生产接入门槛，无效配置和改进有限的候选均保留为诊断证据。

## 2. 问题与原因分析

此前独立手写融合在 K 循环内乘 scale 并舍入 BF16，编译产物中 B 侧缓冲结构较弱。本次转为已有 `flag_gems.ops.weight_int8pack_mm`，使用 FP32 累加后乘 scale；不把旧路径的舍入契约或性能结论直接迁移过来。

同时修正对照：本轮主 BF16 项直接调用 `_metax.ops.mm.mm(x,w.t())`，独立反量化后也调用同一接口；不用原生 `F.linear` 代替 FlagGems 后端。`F.linear` 仅在 FP32 数值参考中使用，不参与主性能基线。

新实例驱动、GPU bus 和主机内存与前次不同，不用旧耗时计算收益。源码配置中有 pipeline 候选，但编译是否形成双缓冲、最终选择什么分支，均按本轮产物记录。

## 3. 具体实现

### 数学契约与路径

加载时复用已有 `reference.py` 的权重量化算法：对称 per-output-channel，INT8 `[N,K]`、FP32 `[N]` scale。辅助模块直接从本实验目录加载，不修改插件或依赖。N=2048、K=6144，激活/输出 BF16。

```text
主 BF16：MetaX mm(x_bf16, original_weight_bf16.T)
原融合：FlagGems weight_int8pack_mm(x_bf16, q_int8, scale_fp32)
独立路径：Triton dequant(q,scale) → 完整 BF16 权重 → 同一个 MetaX mm
配置候选：原 weight_int8pack_mm_kernel 的 JIT 函数体，指定另一组配置
```

新融合数值参考为 `BF16(FP32_matmul(x,q) * scale)`，FP32 参考关闭 TF32，避免先得到 BF16 matmul 输出再乘 scale 带来的额外舍入。旧路径 `sum(x*BF16(q*scale))` 作为另一契约，不要求逐位等价。

独立路径每次重新生成 BF16 权重，计入分配、反量化和 GEMM；候选计入输出分配和完整 launch。量化预处理不计入每次推理延迟。测量是事件 GPU 延迟，不代表完整 CPU/API 端到端时延。

### 配置实验

固定使用 `pipeline=basic`，在未修改 JIT 函数体的前提下测试 14 组 `(BM,BN,BK,warps,stages)`。包含小 M 的 BM=16、BK=128/256，较大的 tile，以及 stages=2/3/4；不改库内 tuner 配置、不覆盖默认选择。

每个 M 筛选 14 组，共 126 项，筛选 seed=73、warmup=50 ms、rep=150 ms，选该有限集合中较快的一组。随后换 seed=314159，重新验证边界形状和数值，并对九种 M 的四条路径各测三轮，warmup=100 ms、rep=300 ms，取三轮中位数的中位数。只称“已测配置中选中候选”，不称全局最优。

14 组列表、失败记录及选中项在 `config-results/metadata.json/screening.json/chosen.json`，原始三轮计时全部保留。该集合没有包含原 tuner 的全部 8 组配置，因此大 M 的筛选赢家仍可能慢于原实现；最终独立对照如实揭示了此情况，没有将其自动替换到生产。

## 4. 实验与验证

### 环境和指纹

容器 `f3a0dc142860`；单卡 MetaX C500，65536 MiB，bus `0000:d4:00.0`；驱动 3.9.35、MACA 3.7.0.38。主机 48 GiB RAM，无 swap，开始/结束可用约 43 GiB。Python 为已有 `/opt/conda/bin/python`，未重装依赖。PyTorch 为 `2.8.0+metax3.7.0.7` 系列、FlagGems 5.3.5、MetaX Triton 3.0.0 系列，详细包信息见 `packages.txt` 和运行 metadata。

通过 `PYTHONPATH=/workspace/FlagGems/src` 显式加载核对过的源码。安装版与仓库版 weight-only 和 MetaX mm 文件 SHA256 一致：

| 文件 | SHA256 |
|---|---|
| `ops/weight_int8pack_mm.py` | `a7f27d8030f4a4fa9a47a9d3d71f3bd710e269895d8ae30e4e4397f4167c4def` |
| `_metax/ops/mm.py` | `2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec` |

开始前没有服务、评测或 GPU 测试进程，GPU 利用率 0%、无 GPU 进程、显存 860 MiB、9031 空闲。基准进程完全退出后再次确认释放，再启动配置实验；未并发运行测试。没有清理持久编译缓存。

### 未修改现成接口的首次基准

seed=42，单位 ms，三轮中位数的中位数：

| M | MetaX BF16 | 原 weight-only 融合 | 独立 Triton＋MetaX BF16 |
|---|---:|---:|---:|
| 1 | 0.038400 | 0.173312 | 0.086784 |
| 16 | 0.038656 | 0.177664 | 0.086784 |
| 32 | 0.048896 | 0.177920 | 0.095744 |
| 64 | 0.071424 | 0.178688 | 0.116992 |
| 128 | 0.078848 | 0.285184 | 0.125184 |
| 256 | 0.130560 | 0.299520 | 0.176384 |
| 512 | 0.186624 | 0.522496 | 0.235776 |
| 1024 | 0.303360 | 0.770560 | 0.348416 |
| 2048 | 0.582656 | 1.241088 | 0.626688 |

首调用 wall time 单独记录于 `results/first_calls.json`，包含编译和 tuner；不纳入稳态表。持久缓存保留，首调用耗时不是“空缓存编译时间”。

### 独立随机输入的配置复验

单位 ms。括号中的配置为 `(BM,BN,BK,warps,stages)`：

| M | MetaX BF16 | 原融合 | 配置候选 | 独立 Triton＋MetaX BF16 | 候选配置 |
|---|---:|---:|---:|---:|---|
| 1 | 0.039168 | 0.173056 | 0.085504 | 0.087040 | (16,64,128,4,2) |
| 16 | 0.039936 | 0.177920 | 0.086528 | 0.086784 | 同上 |
| 32 | 0.052992 | 0.179968 | 0.088832 | 0.098816 | 同上 |
| 64 | 0.075264 | 0.180224 | 0.144896 | 0.120320 | 同上 |
| 128 | 0.082688 | 0.287488 | 0.207360 | 0.128000 | 同上 |
| 256 | 0.130816 | 0.299008 | 0.328960 | 0.176640 | 同上 |
| 512 | 0.186624 | 0.521216 | 0.538880 | 0.235776 | (128,128,32,8,2) |
| 1024 | 0.303104 | 0.764416 | 0.989440 | 0.348160 | 同上 |
| 2048 | 0.581632 | 1.232640 | 1.468416 | 0.629248 | 同上 |

M=1/16/32 候选相对原融合约加速 2.02/2.06/2.03 倍，但仍慢于 MetaX BF16；M=1/16 相对独立反量化仅有小幅差别，不据此宣称稳定获益区间。M≥256 的候选比原融合更慢，全部记录为不适合接入。没有任何一组候选超过同场 BF16。

### 实际配置、调用和流水线

原融合选中配置：M=1/16/32/64 为 `(32,32,64,2,3)`；M=128 为 `(64,64,64,4,3)`；M≥256 为 `(64,128,64,4,2)`。

显式 MetaX BF16 调用的新缓存项显示：M=1/16 使用 `mm_kernel_splitk`，`SPLIT_K=8`、`(16,64,128,4,3)`、pipeline=null；M≥32 为 `mm_kernel_nt`。其中 M=128 的本次选中项为 cpasync、stages=3，其他 NT 点为 basic、stages=4。见各 M 的 `dense-entries` 及编译文件。这是本次独立算子直接调用的结果，不冒充某次官方模型运行的全 Shape trace。

**重要新证据：现成融合已经形成 A/B 双缓冲。** M=2048 的 TTGIR 有 `2x64x64xbf16` A 缓冲和 `2x64x128xi8` B 缓冲；每轮先读取下一块 A/B，再从 shared 载入当前 INT8 dot fragment，转换为 BF16，执行两个 K=32 dot 完成逻辑 BK=64；scale 在循环后读取并乘到 FP32 accumulator。此前旧手写版本的“B 侧没有同样双缓冲”不能继续当作现成 Kernel 的缺陷。

原融合 M=2048 后端资源为 n_regs=174、n_spills=0、shared=32768 bytes、pipeline=basic；M=1 为 n_regs=120、n_spills=0、shared=12288 bytes。缓冲结构改善不等于达到 BF16 性能；具体瓶颈仍可能涉及 tile 并行度、INT8 fragment 转换与布局、指令开销，当前没有硬件 stall/带宽计数器证据，不能给出因果占比。

日志出现 `no avalilable pipeline for maca` 提示；完整流程仍退出 0，M=128 也保存了 cpasync 选中项。它表明存在调优/编译提示，不能据此宣称整个 cpasync 不可用或所有配置均有效。未修改 pipeline 逻辑绕过提示，原日志、候选配置和生成 metadata 均已保留；本轮额外配置实验固定 basic。

### 正确性、统计完整性及限制

- FP32 scale、零权重通道、`(M,N,K)=(1,17,33)/(3,129,257)/(16,128,256)` 和真实 Down 形状验证通过。
- 新融合按其 scale-after-dot 参考验证：首次基准九点 L2 相对误差低于 0.005；复验候选的实际误差约 `2.07e-7～8.22e-5`，边界/零通道也通过。
- Triton 独立反量化权重与 reference 完全相等；BF16 与独立 GEMM 对各自 FP32 累加参考的相对误差均低于预设 0.005。首次 M=1/16 的 BF16 和独立项约 0.0043～0.0045，应与其分段/原子 BF16 输出累加路径一起记录，不当作精度逐位等价证据，也不修改基线来消除差异。
- 新融合相对原 BF16 的 synthetic 量化输出差异约 0.0092～0.0099（L2 相对误差），与 Kernel 对自身契约的计算误差分开；不据此推断 math_500 精度。
- 两进程退出码均为 0。本地审计 9 个原始基准 Shape、81 条原始计时、126 个配置筛选（本轮无失败项）、9 个复验 Shape 和 108 条复验计时；全部中位数与汇总一致，脚本指纹与远端一致。
- 调优候选集有限，筛选与复验使用不同输入，结果只适用于已测 Shape、dtype、布局和本实例，不称“最优配置”；没有测量模型级精度、TTFT 或吞吐。

## 5. 与已有改动的关系

落实 `research/2026-10-26-flaggems-w8a16-reuse-review.md` 的复用路线。本轮没有调用旧自写融合或新增 GEMM 算法；配置候选直接调用原 `weight_int8pack_mm_kernel.jit_function`，不修改其 dot、scale epilogue、分组或存储契约。

与旧手写实现相比，数学舍入位置不同；本轮独立验证了新契约，不能沿用旧模型精度。与旧 Microbenchmark 相比，主 BF16 也改为显式 MetaX 入口，因此不以历史原生 F.linear 耗时计算本轮收益。

新增诊断脚本是验证工具，不等同于已经交付生产优化。原融合开关、模型接入和历史报告均保留，不新增 M 分派，不让效果不足的候选进入正式路径。

## 6. 合规性、风险与后续判断

未运行模型、官方 Benchmark、精度数据集；未改官方参数、数据、缓存统计方式或依赖。仅 synthetic GEMM 实验，保持 GEMM 任务边界，不扩展 Attention 等模块。认证信息不进入代码和归档。

没有观察到超过 MetaX BF16 的完整路径收益，本轮配置候选不启用。下一步可依据已确认事实另行制定计划：小 M 的 BF16 已使用 Split-K，而现成 weight-only 未覆盖同类分派；大 M 的双缓冲已有，不应继续泛称“补双缓冲就能解决”。可讨论基于现有 FlagGems Split-K/归约或 fragment 布局做有针对性的最小改动，先评估完整归约成本和数值契约，再决定实施范围。

不在本轮无限扩展配置或新增生产路径。后续任何算法/分派扩大按工作区规则讨论；只有算子候选有效、模型接入和精度验证通过，才考虑官方 4K/16K 吞吐和 TTFT。

## 7. 复现方式与归档

远端目录：`/workspace/benchmark_results/flaggems-w8a16-20261027-01/`；本地：工作区 `artifacts/flaggems-w8a16-20261027-01/`，含原始 `remote-evidence.tar.gz` 及展开文件。

- `preflight.txt/run-environment.txt/packages.txt/postflight.txt`：环境、资源及源码证据。
- `provenance-source/`：原 weight-only、MetaX mm、配置和 libentry 源码快照。
- `results/`：原版计时、首调用、正确性、选中配置、TTIR/TTGIR、mcfatbin 和资源字段。
- `config-results/`：126 项筛选、选中项、独立输入复验、全部原始计时和候选编译产物。
- `audit.json`：本地完整性核对；`exit.txt/config-exit.txt`：正常退出码。

按任务前置要求先检查 GPU、进程、端口和源码，加载已有 Conda/MACA；本次显式源码路径为：

```bash
export PYTHONPATH=/workspace/FlagGems/src
unset VLLM_FL_W8A16_DOWN VLLM_FL_W8A16_DOWN_FUSED VLLM_FL_W8A16_OFFLINE VLLM_FL_W8A16_CHECKPOINT
python /workspace/benchmark_results/flaggems-w8a16-20261027-01/benchmark_flaggems_w8a16.py \
  --helpers-dir /workspace/benchmark_results/flaggems-w8a16-20261027-01 \
  --output-dir /workspace/benchmark_results/flaggems-w8a16-20261027-01/results
```

等待基准进程完全退出、确认资源释放后，再串行执行配置实验：

```bash
python /workspace/benchmark_results/flaggems-w8a16-20261027-01/benchmark_flaggems_w8a16_configs.py \
  --helpers-dir /workspace/benchmark_results/flaggems-w8a16-20261027-01 \
  --output-dir /workspace/benchmark_results/flaggems-w8a16-20261027-01/config-results
```

目录已归档存在，复现须使用新的实验 ID/输出目录；辅助 `reference.py/dequant.py` 和两个诊断脚本需一起准备。脚本语法、diff 空白检查与实际 GPU 运行通过。最终 GPU 无进程、利用率 0%、显存 860/65536 MiB，9031 可绑定；两仓库生产源码未改变。
