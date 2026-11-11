# W8A16 完整调用有限配置调优

## 1. 基本信息与摘要

日期2026-10-27，实验 `flaggems-w8a16-full-20261027-05`。用户同意完整算子调优：小M有限配置、大M适用性对照，正确性优先，再复测优胜项。仅新增 `benchmarks/diagnostics/benchmark_flaggems_w8a16_full.py`，复用原FlagGems单kernel及已有Split-K partial/reduce，不改生产算子或开关，不启动模型/官方bench。沿用分支 `perf/gemm-w8a16-down-fused`，插件基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，FlagGems基点 `a7620cc191a0b42e040194622c5758b22a7a25dc`，已有工作未覆盖。主实验退出码0，52项初筛全部通过，9个M的五路径三轮复测完成；另做M64新旧配置同场确认，结果随后归档。

## 2. 问题与原因分析

阶段诊断显示partial计算值得优先研究，但独立阶段耗时不可相加，也不能替代完整调用。两段Split-K保留INT8权重转换与dot融合，新增FP32部分和写读及归约；并行收益必须覆盖这些额外成本。原算子只一个kernel，本实验验证其候选实现，不把partial作为独立生产算子接入。

## 3. 具体实现

单kernel配置 `(BM,BN,BK,warps,stages)` 三项：`(16,64,128,4,2)`、`(32,64,128,4,2)`、`(128,128,32,8,2)`，原JIT主体不改。Split-K配置格式在末尾增加S；M1/16固定BM16/BN64/BK128/S8，比较(warps,stages)=(4,2)/(2,2)/(4,1)/(4,3)。M32/64比较BM16/32/64×S4/8，固定BN64/BK128/warps4/stages2。M128/256/512/1024/2048仅测BM32/BN64/BK128/warps4/stages2/S4作为适用性探针，不进行穷举。共52组shape/config初筛（27单kernel＋25 Split-K），精确数量以归档核对。

每候选完整调用分配输出；Split-K额外分配 `[S,M,N]` FP32 partials、启动partial及reduce+scale。计时为GPU事件包围Python完整调用，包含设备执行及可能的提交间隙，不等同于精确CPU端到端wall time，也不等同于模型图捕获表现。初筛seed73、warmup50ms/rep150ms；每M各保留一个single与split优胜项，seed314159独立输入复测。最终同场路径BF16 MetaX、Triton独立反量化＋MetaX、原融合、所选single、所选split，轮换顺序三轮，warmup100ms/rep300ms，取三轮中位数。raw跨M累积，不覆盖。没有手写原始融合路径加入新比较。

## 4. 实验与验证

每配置真实Down与masked `(3,129,257)` 输入，对FP32 dot后scale参考检查relative L2<0.005和rtol0.02/atol0.125，masked权重首通道置零并检查输出精确零。复测量化候选再用独立输入检查。该容差不代表模型精度通过。失败候选记录错误，不纳入优胜；未预期异常停止。分配/访存未拆为可相加占比。

## 5. 与已有改动的关系

依赖现有baseline/configs/splitk诊断helper；相对stage-audit回到完整调用做主指标，不继续优化组件独立延迟。旧报告、结果目录、生产开关保持历史状态。

## 6. 合规性与风险

N2048/K6144 Down shape，M1/16/32/64/128/256/512/1024/2048。静态INT8权重 `[N,K]`、FP32每输出通道scale `[N]`，BF16激活与输出，FP32累加/partial，最终scale-after-dot。无原生INT8×INT8计算声明；无全模型覆盖、模型精度、TTFT、官方吞吐结论。不因配置胜出就新增生产开关。大M只测一项Split-K，失败或较慢不能推导全部Split-K无效。未提交未推送。

## 7. 复现方式

远端新目录 `/workspace/benchmark_results/flaggems-w8a16-full-20261027-05`；实际源码通过PYTHONPATH指向 `/workspace/FlagGems/src`，清除四个历史W8A16开关，使用已存在Conda/MACA环境，不清持久缓存。

```bash
python benchmark_flaggems_w8a16_full.py --helpers-dir . --output-dir results
```

保留preflight/environment/run.log/run.pid/exit.txt/postflight、脚本与依赖helpers、screening/chosen/raw_final/summary/metadata JSON，运行结束后归档到本机artifacts。

### 同场完整调用结果

单位为微秒，三轮延迟中位数的中位数。single/split 是各路径初筛优胜项，不保证胜过原融合或独立反量化。

| M | BF16 MetaX | 独立反量化＋MetaX | 原融合 | 所选single | 所选split |
|---|---:|---:|---:|---:|---:|
| 1 | 43.008 | 88.576 | 174.848 | 87.040 | 45.056 |
| 16 | 42.240 | 87.296 | 178.944 | 87.808 | 46.336 |
| 32 | 50.688 | 96.000 | 178.944 | 88.064 | 63.744 |
| 64 | 72.960 | 117.760 | 178.944 | 144.128 | 85.248 |
| 128 | 80.896 | 125.696 | 286.464 | 207.360 | 173.568 |
| 256 | 131.072 | 176.896 | 300.544 | 328.704 | 324.864 |
| 512 | 188.416 | 238.080 | 524.288 | 540.672 | 620.032 |
| 1024 | 304.640 | 348.928 | 764.672 | 990.976 | 1206.528 |
| 2048 | 586.240 | 629.504 | 1232.384 | 1462.528 | 2357.760 |

### 配置与分析

- M1/16/32 仍选中 BM16/BN64/BK128/warps4/stages2/S8，本轮没有找到更好的新配置。
- M64 选中 BM64/BN64/BK128/warps4/stages2/S8。独立 seed271828 同场确认：旧 BM16/S4 为101.376微秒，旧 BM16/S8 为105.984微秒，新 BM64/S8 为84.224微秒；相对旧S4耗时下降16.92%，加速1.204倍。同场BF16为72.448微秒，独立反量化为116.736微秒；新配置仍比BF16慢16.25%，比独立反量化快1.386倍。不能写成模型吞吐收益。
- M128 所测split虽比原融合快，但仍慢于独立反量化；M256及以上本轮single/split也未胜过独立反量化，不接入这些有限候选的大M分派。大M单kernel初筛优胜项在独立输入复测未胜过原融合，记录失败收益证据，不新增开关。
- 较大BM可以增加同CTA内权重复用，减少M方向重复加载和program数；M64对照支持改变BM有价值，但不是硬件stall的直接证明。S8/BK128每分片仅6次K循环，stages/warps探针未找到小M优胜，不支持盲目增stage。
- 结论只覆盖N2048/K6144和所测M；其他Linear、范围边界、动态图编译及模型精度仍待验证。

### 归档核验与状态

52项初筛全通过；135条最终计时和15条确认计时完整，逐项重算中位数与summary一致；主脚本本地与远端一致。主实验与确认均退出0，所有量化候选通过独立输入数值校验。运行后GPU无进程，util0、显存860MiB、9031空闲，两生产仓库无差异。

本地证据 `artifacts/flaggems-w8a16-full-20261027-05/` 包含脚本、指纹、环境、进程、原始及汇总JSON；`audit.json` 为本机归档核验。增加 `benchmarks/diagnostics/confirm_m64.py` 保存同场确认脚本，在含helpers的实验目录执行 `python confirm_m64.py`。权重量化发生在计时外，不计入每次推理调用；完整调用包含输出和workspace分配、全部kernel。不将GPU事件计时当作模型CPU端到端时延。未修改生产算子、未提交未推送。
