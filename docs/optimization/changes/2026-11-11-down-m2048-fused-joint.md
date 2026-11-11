# Down M2048 固定转换与融合路径的配置联合调优

## 1. 基本信息与摘要

2026-11-11，MetaX C500。用户批准在现有量化路径上进行M2048联合调优，并进一步明确仅与已有专用最优配置比较。新增 `benchmarks/diagnostics/benchmark_metax_down_joint.py`，生产Kernel、dispatch及配置表未修改。实验目录 `/data/down-joint-fused-20261111-01/`。

## 2. 问题与原因分析

之前的MM专用配置在独立BF16路径选出；转换与scale融合之后，完整调用的较优分块及流水线参数仍需验证。本任务不重新搜索转换参数，也不调优原生BF16底层。将转换优化和FP32累加后直接scale的融合实现固定为既定路径，只研究MM配置对其完整耗时的影响。

用户最新比较口径优先：不另跑默认配置或纯BF16基线。已有专用配置 `BM64/BN128/BK128/warps4/stages4/cpasync` 为唯一验收对照，其转换和融合与候选完全一致。候选集合可包含64×64等tile，但不是新增默认路径基线。

## 3. 具体实现

Shape为M2048/N2048/K6144，激活BF16，权重INT8[N,K]连续布局、FP32 per-output-channel scale。每次完整调用分配BF16权重暂存与输出，执行 `BLOCK4096/warps4` 转换，再运行 `mm_kernel_nt_scale_direct`，BF16 dot、FP32累加和scale、BF16输出。转换不移出计时、不保留持久BF16权重副本。权重物理布局和epilogue源码不变。

有限候选：8组tile `(64,64,64)/(64,128,128)/(64,128,64)/(128,64,64)/(128,128,64)/(128,128,128)/(64,64,128)/(128,64,128)`，warps4/8，stages2/3/4，basic/cpasync，共96项。显式传入完整配置直接调用Triton JIT，避免LibEntry启动参数缓存别名或依赖MM tuner的旧配置。复用已存在的融合Kernel，不重新实现MM。记录所有失败；非法访存立即退出，不继续污染后续结果。

## 4. 实验与验证

搜索seed20261120，预设相对FP32量化参考L2<0.005。随机顺序初筛按完整调用选前六，并无条件保留专用对照。使用未参与筛选的seed20261121/20261122，七轮随机交错完整调用、五轮随机交错CUDA Graph replay；每个候选均验证eager和图输出。独立seed、两类测量都稳定降低耗时超过1%才考虑采用，不择优拼接各seed的赢家。此门槛为micro候选筛选标准，不等于官方模型验收。

本地AST检查通过，脚本两端SHA256一致：`7ea40b0d670d76cf59187f8fd0bf10eaea6117b7c08e0aac3e2f8cc624f7183c`。启动前GPU无进程、826MiB可解释驱动保留；控制器复核GPU和9031端口。隔离overlay复制自刚完成的默认M2048融合组合，源码哈希、仓库状态、前后进程及命令归档。本轮搜索与独立复验均已完成。

96项初筛完成，69项通过数值检查，27项失败并保留原因。领先配置为 `(128,128,128,4,4,cpasync)`，只将原专用对照的BM64改为128。初筛后的两种子复测均领先。

追加 `verify_metax_down_joint.py`，在 `/data/down-joint-fused-accept-20261111-01/` 使用实际 `_shape_dispatch`，仅离线注入配置选择，原转换与融合调用保持一致。新seed20261123/20261124，每组9轮完整调用、7轮图重放：

| seed | 原完整μs | 新完整μs | 耗时降低 | 原图μs | 新图μs | 图耗时降低 |
|---|---:|---:|---:|---:|---:|---:|
| 20261123 | 445.356 | 351.850 | 21.00% | 462.605 | 368.022 | 20.45% |
| 20261124 | 446.251 | 352.284 | 21.06% | 461.948 | 367.340 | 20.48% |

四条独立复验的eager及graph相对L2均小于0.001661，通过0.005门槛。已保存绑定提供者/Kernel哈希、设备版本、Shape和dtype的 `candidate.json`，状态明确为micro候选、尚未接入运行时。收尾GPU无进程、9031端口空闲。两个诊断脚本AST与空白检查通过。

## 5. 与已有改动的关系

复用11-11转换与scale direct实现，转换选择沿用已验证BLOCK4096/warps4，不重复其独立搜索。小M56/M64配置保持不变。与此前partial/shared实验独立，不将历史吞吐或历史精度作为本次结果。稳定上下文CONTEXT中的旧“未实现”状态不用于否定本轮实际已有dispatch；以本次overlay与最新报告为准。

## 6. 合规性与风险

只新增离线诊断，不修改官方启动/Benchmark、模型、数据集、量化范围或依赖。96项是有限搜索，不能声称全局最优。数值验证相对于量化后参考，不代替原始模型量化精度。只测M2048不能推断其他M或16K端到端TTFT改善；如有稳定候选，已通过生产接口离线复验及版本绑定，后续需实际接入并进行模型级验收。当前未接入、未执行新的模型评测，未提交或推送。

## 7. 复现方式

```bash
PYTHONPATH=/data/down-joint-fused-20261111-01/overlay \
VLLM_FL_W8A16_SHAPE_DISPATCH=1 VLLM_FL_W8A16_FORCE_SEPARATE=0 \
python /data/down-joint-fused-20261111-01/benchmark_metax_down_joint.py \
  --output <新的不存在的结果目录>
```

原始运行命令见 `commands.json`；实际启动为外围 `run.py`，其负责资源检查、进程等待及状态归档。结果包括 `results/environment.json`、`controls.jsonl`、`screen.jsonl`、`selected.json`、`finalists.jsonl`；保留编译失败、完整原始轮次和图重放结果，不以阶段耗时相加代替完整调用。

完整原始归档已下载到 `artifacts/down-joint-fused-20261111/down-joint-fused-20261111-results.tgz`，两端SHA256一致：`7ab2e72a4ab1da33b6cbcf4143864a36d1c66a0e357c0df74571c1b51685edf0`。包含搜索与独立验收脚本、命令、状态、源码指纹及所有计时轮次，排除重复overlay副本；容器原目录保留。
