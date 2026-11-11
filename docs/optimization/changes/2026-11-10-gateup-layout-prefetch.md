# Gate/Up M56 同 tile 转换诊断与显式预取实验

## 1. 基本信息与摘要

2026-11-10，MetaX C500。用户批准定位 Gate/Up 转换成本、优化融合 Dense，并在有收益后评估推广到其他 M。本次沿用 `perf/gemm-w8a16-shape-dispatch`；新增诊断脚本 `benchmarks/diagnostics/benchmark_metax_gateup_layout.py` 与实验 Kernel `metax_gateup_prefetch.py`。没有修改生产分派，未提交、未推送。

本机插件 HEAD `f93489fa1afb74969b76701c8b551552a55ae3bf`，FlagGems HEAD `f6a4257e3d042acd26a7d3c496848025a4b32c07`；两仓库原有未提交改动保留。远端工作仓库 HEAD 分别为 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`、`a7620cc191a0b42e040194622c5758b22a7a25dc`，实际测试使用前轮隔离 overlay，不能用远端 Git HEAD 代替运行源码指纹。

## 2. 问题与原因分析

Shape 为 M=56、N=12288、K=2048。现有 Dense INT8 转 BF16 在 dot 前执行，FP32 累加后乘 FP32 通道 scale。相同源码 tile 不保证编译器生成相同流水线。诊断以同一 JIT 函数分别读取连续 INT8、预转换 BF16(Q) 和物理转置 INT8；scale 均保留输出端。BF16(Q) 排除转换但增加权重读取字节数，因此耗时差不能称为纯转换指令成本。

## 3. 具体实现

诊断覆盖六组 tile、两个随机种子，每个合法候选五轮交错 eager 计时，并单列侧流 CUDA Graph replay。记录原生注册 `flag_gems.mm`、数值相对 L2、寄存器/spill、TTIR/TTGIR 与编译 metadata。每次调用包含输出分配和完整融合计算。

预打包候选仅一次性 `q.t().contiguous().t()`，仍是 INT8 静态权重，不持有正式 BF16 权重缓存；转换时间不伪装成量化调用收益。诊断 BF16(Q) 是独立实验控制。

显式预取原型在进入循环前加载第一块 A/B 并转换 B；循环提前加载和转换下一块，再计算当前块，最后计算剩余一块。数学契约、累加顺序及末端 scale 保持，测试结果决定是否继续。该原型没有接入生产。

## 4. 实验与验证

启动前 GPU 无进程，显存 826 MiB；9031 端口绑定成功。仅运行独立 GPU 测试；上一轮已经退出。Python `/opt/conda/bin/python`，MACA `/opt/maca`。环境、GPU、进程、仓库差异与命令归档于 `/data/gateup-layout-20261110-01/`。`ss` 不存在，改用 socket bind 检查 9031，失败事实保留。

第一轮诊断已完成。以下是 seed20261110 五轮 eager 中位数，单位 μs；不是模型性能：

| 路径 | BM/BN/BK、warps/stages | 延迟 |
|---|---|---:|
| 原生 BF16 | 注册提供者 | 64.72 |
| 融合 INT8 | 64/64/64、8/4 | 87.99 |
| 同 tile BF16(Q) | 64/64/64、8/4 | 64.27 |
| 同 tile BF16(Q) | 64/128/64、4/4 | 62.21 |
| 简单转置打包 INT8 | 64/64/64、8/4 | 109.57 |

同 tile INT8 为 74 registers、24 KiB shared，BF16(Q) 为 102 registers、32 KiB shared，两者 n_spills=0。不能归因于已证实的 spill。图重放单次计时较 eager 高，单独保存，不能混用两种模式宣称收益。BF16(Q) 一组候选失败，保留错误，不删除失败后宣称全部通过。

显式预取六组配置、两种子已完成，合法候选相对 L2 约 0.001656，通过预设 0.005 门槛。领先配置仍为 64/64/64、8/4，完整调用结果如下：

| seed | 同场原生 BF16 μs | 同场原 Dense μs | 显式预取 μs | 相对原 Dense 延迟降低 |
|---|---:|---:|---:|---:|
| 20261110 | 65.47 | 88.40 | 86.55 | 约 2.09% |
| 20261111 | 65.69 | 89.03 | 86.67 | 约 2.65% |

图重放同场原 Dense 为 103.36/103.25 μs，预取 101.08/101.90 μs，方向一致，但图重放目前每个候选仅一组计时，不夸大稳定性。预取仍慢于 BF16，不能作为成功推广的依据。

TTGIR 已核对：原 Dense 的 INT8 路径有一个 `2x64x64xbf16` 持久 shared 缓冲（A 侧），循环内 B 执行 sitofp 后 local_alloc；BF16(Q) 有两个持久缓冲。该证据支持 B 侧流水线差异，但不能定量分解转换与访存成本。

基于安装版本编译器源码确认的能力，追加有界调度实验：fullstage、dot_operands_out_loop、cvt_shm_no_pad、roll、pipeline_load_num=2、unprefetch 与 cpasync-mixed，各三组 tile、两种子；仅用于诊断，不改安装编译器或生产默认。显式预取组合在第一个 fullstage 候选触发 LLVM `dyn_cast on a non-existent value` 断言，进程退出；该次不完整，其他预取调度组合未测，不列作成功。之后原 Dense 的同组调度实验完成 46 条记录（42 候选加 4 控制），全部数值检查通过；最佳调度候选约 89.30/89.23 μs，对照原 Dense 88.83/89.36 μs，没有稳定超过 1% 的收益。

最终保留显式预取作为有限改善的实验原型，尚未达到超过 BF16 的目标，不推广全 Gate/Up、不接入生产，不把有限搜索称为全局最优。所有实验进程已结束，GPU 最终无进程、826 MiB。

其他 M、真实权重、模型精度、官方 4K/16K、TTFT 均待验证。本次不宣称达到全模型验收或所测范围外最优。

## 5. 与已有改动的关系

复用当前 MetaX W8A16 Dense 和已存在的 pipeline 实验 overlay。历史 Down 同 tile 诊断只提供假设，不能直接替代 Gate/Up 数据。新实验不覆盖原结果；原始脚本保留为 `diagnostic-v1.py`。纯 BF16 专属配置分支不修改。

## 6. 合规性与风险

只新增独立诊断和实验 Kernel，不更改官方 serve/benchmark、数据集、模型超参数或主要分派逻辑。W8A16 权重 per-channel INT8、scale FP32、激活与输出 BF16，dot FP32 累加。相对量化参考的 kernel L2 门槛预先固定为 0.005；该门槛不替代模型精度。无全层量化覆盖声明；推广需验证边界、不同 M 和真实布局。

## 7. 复现方式

```bash
export PYTHONPATH=/data/metax-m56-pipeline-20261110-01/overlay
python benchmark_metax_gateup_layout.py --output <新的结果目录>
python benchmark_metax_gateup_layout.py --prefetch --output <另一个新的结果目录>
```

首次结果 `results/`，预取结果 `prefetch-results/`；`results.jsonl` 保存筛选及失败，`finalists.jsonl` 保存各合法候选五轮原值、中位数与图重放，`complete.txt` 表示脚本完成而非全部候选成功。AST 检查通过；本机 py_compile 因系统缓存写权限失败，未改权限或重装环境，改用不落盘 AST 校验。

本机完整归档：`artifacts/gateup-layout-20261110-01/gateup-layout-20261110-01.tgz`，已解压核对完整标记、原始计时和最终 GPU 状态。调度复现增加 `--schedule-scan`，显式预取组合的编译失败也保留原始日志。
