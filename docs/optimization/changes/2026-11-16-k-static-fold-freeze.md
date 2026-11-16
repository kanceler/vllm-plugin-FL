# 已采用静态K联合方案：算子版本固化

## 1. 基本信息与摘要

2026-11-16，BI-V150，feat/tianshu-k-top4，HEAD f82fa1670509f081aa13b58464cb6e5627fc10c4。用户明确要求采用上一轮方案并固化。本轮固化版本为k_static_fold_biv150_v1：固定四通道保护＋其余通道静态scale折Q、Q融合KV写入、固定离线分段配置；保留旧动态joint对照和已有修改。未提交或推送。

新增入口benchmarks/diagnostics/k_static_fold_frozen.py、验证脚本validate_k_static_fold_frozen.py，以及frozen/k_static_fold_biv150_v1下的清单、五份实现快照、README和SHA256SUMS。40个唯一调用键来自此前最终88例实测，每个键跨长度配置完全一致。采用的是算子实现与调用策略，尚不是全模型已启用静态量化。

## 2. 问题与原因分析

此前已测v2参数仍位于诊断搜索文件中，状态包含创建时pending字段，后续实验可能修改原型。仅口头采用或改JSON状态不能固定可执行版本，也不能防止设备、provider和实现漂移。需要固定证据绑定、明确入口和前置校验，而非推理时重新搜索或按KV长度择优。

目前每层static scale与固定通道只有5/42层、按类别留一的研究参数；不能填充未知层或把四折参数当统一生产校准。当前模型KV update hook没有Q输入，也不能把微测中的联合writer直接替换hook声称完成模型接入。

## 3. 具体实现

- 固定page32、token-major、separate_meta、276B/token/head；RoPE后Q/K同步固定置换，前4维保留原BF16，其余Q乘静态scale后BF16，INT8 K转BF16参与dot，V保持INT8 scale-on-V。没有修改原Kernel数学形式。
- manifest.json固定40个(batch,Q长度,input_threshold)调用键及融合模式、tile、warps、segments、writer warps。键不含KV长度；72/80采用内部8段和预分配workspace，不改变官方参数或并发上限。
- 固定manifest SHA256写入入口。校验相关实现与五份快照、设备/torch/Triton及provider源文件。未知键、重复键或漂移明确拒绝，不隐式切换静态缓存量化语义；现有模型后端的选择/回退保持原样。
- make_call在capture前验证Q/K/V和slot同序、BF16/INT32/FP32参数格式、真实通道置换、正有限scale、缓存布局和workspace非别名；复制参数表由调用对象持有，分配连续Q及显式FP32归约workspace。
- 返回FrozenCall，运行时只执行选定融合Q/K/V writer、Attention和归约，无文件读取、调优或GPU分配。所有每步新增处理仍在完整调用中；初始化的静态表与workspace不移出实际发生的逐步转换。

快照用来审计/恢复；正常入口调用经checksum核对的源码树实现。以后改实现应另立版本，不改写本版本清单和历史证据。

## 4. 实验与验证

沿用torch2.10.0、Triton3.6.0、驱动4.5.0和BI-V150。远端独立目录/workspace/experiments/tianshu-k-static-fold-frozen-20261116-01，基于已验证final目录复制源码，不覆盖主仓库。开始与结束均归档GPU、进程和9031端口状态；任务串行，未启动模型服务。

本机无需torch的源码验证通过；40个键逐项选择一致，未知键拒绝、源码hash漂移拒绝、manifest篡改拒绝均通过。所有快照SHA256SUMS在远端核对通过，入口/验证脚本/快照/清单与本机逐字节一致。Python语法检查、git diff --check通过。

GPU seed20261217八例入口验证全部通过，覆盖Q-head writer1/2warps、KV-head writer4warps、32段与8段、72/80内部3D、短KV和三种Prefill。不是仅再次直接调用旧诊断函数：variant_1实际由make_call生成并捕获FrozenCall。数值门槛沿用实现max abs≤0.08/NRMSE≤0.02、总误差max abs≤0.20/NRMSE≤0.05；字节、负slot、Q变换、长度1捕获后增长重放均通过。GPU函数和归约被纳入图计时，8/8，0失败。

| batch / KV / Q | 固化入口完整调用 ms | 实现NRMSE | 总NRMSE |
|---|---:|---:|---:|
| 1 / 257 / 1 | 0.030917 | 0.002600 | 0.016346 |
| 56 / 4096 / 1 | 0.679500 | 0.002771 | 0.017844 |
| 64 / 16384 / 1 | 2.929320 | 0.002808 | 0.017853 |
| 72 / 4096 / 1 | 0.841491 | 0.002746 | 0.017804 |
| 80 / 257 / 1 | 0.123737 | 0.002648 | 0.017538 |
| 4 / 257 / 17 | 0.041914 | 0.002695 | 0.018945 |
| 1 / 128 / 128 | 0.027313 | 0.002520 | 0.015622 |
| 4 / 2048 / 512 | 1.004072 | 0.002835 | 0.020017 |

本轮只是固化入口验证，不声称新增性能收益。之前最终88例的4K35/35和16K≤64的11/11正收益，以及真实90快照回放结论，仍以2026-11-16-k-static-fold-q-fusions.md及原始归档为准。短KV和小Prefill负结果保留，不套用官方±1%规则抹掉算子差异。

## 5. 与已有改动的关系

依赖上一轮已验证静态候选及融合实现，不修改其原代码。manifest绑定最终确认归档8cd9349793dd58aa2aaa97b864d9f0bba786ad9073c872c5f60efad28b4383b9和融合归档b93b44c2c6a9b525237f812f73812f4d118a5fea845ca4570c8d10fc18022fd4；旧pending诊断参数文件不改写，新增明确adopted_operator_snapshot状态。

当前已冻结动态joint服务及其开关保持原样，没有静态服务注册或启动开关。入口只用于已测算子联合调用；后续模型接入可复用固定实现，但须提供独立42层参数、联合Q/K/V入口和真实capture/padding验证。

## 6. 合规性与风险

未改官方serve/benchmark、数据集、采样或模型权重；未使用math_500校准。没有用默认scale补齐未知层，没有把5层CPU筛查或旧K官方精度转移给新方案。manifest显式production_ready=false。未知键拒绝只属于本诊断入口，不删除框架原有后端选择逻辑。

算子固化不等于模型采用完成：统一42层校准、新独立文本/量化反馈、模型接入以及官方精度和双场景性能仍待验证。当前KV缓存容量没有因固化增加，不能声称解决16K满载。未宣称全输入最优或所有shape快于V-only。

## 7. 复现方式

本机只读源码/清单校验：

```bash
python3 benchmarks/diagnostics/k_static_fold_frozen.py
```

GPU沿用原实验环境，PATH=/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin，LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib，PYTHONPATH指向下列实验source；VLLM_PLUGINS=fl、VLLM_FL_INT8_KV_REPAIR=1、VLLM_FL_K_TOP4_JOINT=0、VLLM_FL_K_TOP4=0、VLLM_FL_V_ONLY=0。

```bash
cd /workspace/experiments/tianshu-k-static-fold-frozen-20261116-01
/workspace/tianshu-mixed-env-20261109/bin/python \
 source/benchmarks/diagnostics/validate_k_static_fold_frozen.py smoke-01 \
 /workspace/experiments/tianshu-k-static-fold-final-20261116-01/original-vonly-source/vllm_fl/attention/v_only.py \
 20261217 selection.json source/benchmarks/diagnostics/k_static_fold_fusion_screen_v2.json
```

输出目录必须不存在。smoke-01/raw.json保存9轮×5次重放的完整调用、原V两种状态和旧K对照；run-smoke-01.log、checksum-verification.txt、源码SHA和GPU/进程/端口前后状态保留。当前本机归档experiment-artifacts/2026-11-16-k-static-fold-frozen-01/evidence.tar.xz，SHA256 44eb87d5a9edde01959155faad92b17b7e7d08cb8cb30f00b299e5a85f1321e0，两端一致。
