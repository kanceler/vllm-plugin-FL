# 006 大M权重转换优化与scale epilogue融合

## 1. 基本信息与摘要

目标：降低Down Separate完整调用成本。采用转换BLOCK4096/warps4和scale direct融合；权重仍INT8、scale FP32、激活BF16，INT8运行时转换为BF16参与dot，FP32累加后scale、输出BF16。不是原生INT8×INT8 GEMM。

## 2. 问题与原因分析

原Separate每次分配BF16权重暂存、转换INT8权重、调用BF16 MM、输出中间BF16结果，再独立scale。转换与独立scale都真实发生，会抵消权重压缩带来的收益。只比较MM单核或把转换移出计时，会高估量化收益。

## 3. 具体实现

转换Kernel按连续INT8[N,K]展开，搜索block/warps，在完整调用里选BLOCK4096/warps4。未建立持久BF16副本；权重暂存每次调用仍分配。

在MetaX MM增加scale epilogue变体，读取每输出通道FP32 scale。保留两种数值契约：rounded版本先模拟原BF16中间舍入再scale；direct版本在FP32 accumulator上直接scale再写BF16，减少舍入点。采用direct前分别检查量化后FP32参考误差和图行为，不能将两者称为完全逐位等价。

Dense属于前期整体融合算法能力；此项是Separate内部的转换＋融合scale MM，不把二者混为一条新增INT8计算算法。

## 4. 实验与验证：microbenchmark

转换独立约58.59/56.80μs降至43.70/41.52μs；阶段数字仅诊断。固定MM、独立scale不变的五轮完整调用结果：

| seed | 原转换1024/4完整 μs | 转换4096/4完整 μs | 降低 |
|---|---:|---:|---:|
| 20261110 | 477.210 | 461.637 | 3.26% |
| 20261111 | 479.235 | 462.340 | 3.52% |

单独scale block调整收益低于1%，未作为独立有效优化采用。更大转换block8192/16384未稳定超过4096，不继续扩大。后续default/wide交叉验证也支持转换4096，但百分比依提供者而异，不跨进程绝对值归因。

scale direct在M2048烟测eager及图相对FP32量化参考L2约0.001659362，低于0.005；这是算子检查，不是模型精度。融合性能原始数据在scale-direct、scale-epilogue归档，汇总时不编造单独端到端归因。

### 详细数据与原始依据

**来源：[2026-11-10-down-splitk-separate-pipeline.md](../2026-11-10-down-splitk-separate-pipeline.md)。各表仅在原实验内比较，不跨版本合并。**

**4. 实验与验证**

| 配置（转换block/warps；scale block/warps） | seed20261110 μs | seed20261111 μs |
|---|---:|---:|
| 原1024/4；1024/4 | 477.210 | 479.235 |
| 仅转换4096/4；1024/4 | 461.637 | 462.340 |
| 转换4096/4；scale2048/8 | 460.053 | 459.508 |
| 仅scale1024/4；2048/8 | 475.793 | 475.265 |

**来源：[2026-11-11-convert-block-provider-cross.md](../2026-11-11-convert-block-provider-cross.md)。各表仅在原实验内比较，不跨版本合并。**

**实测结果与结论**

| MM提供者 | seed | 原调用 | 新调用 | 延迟降低 |
|---|---:|---:|---:|---:|
| 官方默认 | 20261110 | 642.83 | 627.09 | 2.449% |
| 官方默认 | 20261111 | 643.80 | 628.46 | 2.383% |
| wide覆盖 | 20261110 | 475.446 | 461.183 | 3.000% |
| wide覆盖 | 20261111 | 477.508 | 460.791 | 3.501% |

**scale rounded/direct同场完整调用：M2048/N2048/K6144，单位μs，5轮eager中位数；graph为原实验单组计时，不写成5轮图中位数。**

| seed | 版本 | 完整μs | 图μs | 量化后FP32参考L2 | 相对rounded L2 | 不同输出元素数 |
|---|---|---|---|---|---|---|
| 20261110 | rounded | 449.884 | 469.223 | 0.002346978 | 0.000000000 | 0 |
| 20261110 | direct | 445.599 | 460.169 | 0.001659761 | 0.002807545 | 1092696 |
| 20261111 | rounded | 451.394 | 466.099 | 0.002344134 | 0.000000000 | 0 |
| 20261111 | direct | 447.318 | 462.291 | 0.001658862 | 0.002806681 | 1090744 |

原始依据：工作区 `artifacts/scale-direct-20261111-01/results/results.jsonl`。direct并非逐位等价；每组完整调用包含相同转换。

## 5. 与已有改动的关系

消费004的Separate分支，复用既有BF16 MM实现；007将在转换与融合固定后重新选择MM配置。其他三投影BF16专用配置是独立组成，不计为本项量化收益。

## 6. 合规性与风险

旧非融合转换策略曾仅对113≤M≤4096选择4096；后续符合条件的direct融合helper实际固定4096，不能将旧报告边界直接复制到新路径。仅对Down已支持dtype/布局启用融合，未知提供者与范围外保留回退。direct数值契约改变后需最终math_500复验，旧精度不能沿用。

## 7. 复现方式与分支汇总清单

依据：[转换与预取实验](../2026-11-10-down-splitk-separate-pipeline.md)、[提供者交叉验证](../2026-11-11-convert-block-provider-cross.md)、[direct组合记录](../2026-11-11-scale-direct-default-mm-evaluation.md)。

汇总FlagGems `mm.py` 的scale/scale_direct Kernel与 `weight_int8pack_mm.py` 的转换及融合helper；保留插件BF16提供者接口。诊断用 `benchmark_metax_convert_provider_cross.py`，融合原始脚本与实际参数见工作区 `artifacts/scale-direct-20261111-01/`、`artifacts/scale-epilogue-20261111-01/`。复现需相同提供者与数值参考。
