# 已采用的静态K算子固化版本 v1

本版本固定采用：RoPE后固定四通道保护、其余通道静态scale折Q、Q处理融合KV写入，以及离线选定的分段配置。设备为BI-V150；不是原生INT8×INT8 dot。

## 固定内容

- 缓存全生命周期使用page32、token-major、separate_meta、276B/token/head；保护值为BF16，K/V主体INT8，V为scale-on-V。
- manifest.json包含88例最终确认导出的40个唯一调用键。键为batch、Q长度、输入分支阈值，不包含KV长度。同一键跨所有实测KV长度使用相同模式、tile、warps和segments。
- 72/80采用内部8段Decode及对应预分配workspace；没有修改官方serve参数或并发上限。
- 五份同目录Python文件是已测实现的逐字节快照，用于审计和恢复。它们不是独立服务插件；正常入口校验源码树中的相应实现后调用。
- 绑定设备、torch/Triton、provider及相关源码SHA256。清单校验值固定在入口中，漂移即拒绝；升级应另建版本，不改本目录历史文件。

## 入口

源码树中执行：

```bash
python3 benchmarks/diagnostics/k_static_fold_frozen.py
```

该命令只验证本地源码与清单，不访问GPU，不证明设备/模型验收。

GPU调用使用`k_static_fold_frozen.make_call(...)`，在capture前提供Q/K/V、同序slot_mapping、统一缓存、不可变的每层permutation/scale和Attention元数据。入口校验设备、provider、数据契约及已测键，分配Q和归约workspace并复制参数表；返回可捕获的完整调用。重放中包含Q处理、实际KV写入、Attention和归约，无调优或GPU分配。

独立验证使用`validate_k_static_fold_frozen.py`。完整命令及结果见`docs/optimization/changes/2026-11-16-k-static-fold-freeze.md`。

## 接入边界

采用并固化的是算子实现与调用策略。当前只具有5/42层的研究校准，尚无统一42层生产参数；manifest明确production_ready=false，不得自动把类别留一参数当作全模型配置。

现有模型`do_kv_cache_update`没有Q输入，写入融合需要联合Q/K/V调用入口，仍待模型接入与capture/padding验证。当前服务保持已冻结动态joint版本；不存在已启用静态服务的新开关。未知调用键在本诊断入口明确拒绝，不修改框架原有选择与回退。

历史88例、真实90快照回放和本轮入口smoke均不能替代官方math_500及双场景性能验收。短KV与部分Prefill负收益仍保留，详见优化报告。
