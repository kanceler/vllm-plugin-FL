# 改动 #001：天数 GEMM Down 热点 tiling 配置

> 类型：BF16 GEMM tiling 配置优化
> 平台：天数 BI-V150；基线插件 `f82fa167`、FlagGems `4df52d91`
> 分支：插件归档分支 `perf/tianshu-gemm-tiling-optimization`（FlagGems实现通过补丁提供），用于后续天数 GEMM tiling 优化；当前改动随本分支提交

## 1. 为什么这样改

在4K诊断Mixed窗口，四类投影MM占已记录GPU Kernel时间54.01%；Gate/Up占MM时间52.44%，Down占27.12%。16K Mixed中对应为31.89%、58.79%、22.93%。这是有限采样窗口事件耗时，不能当成整场官方壁钟占比。4K采到的Down为M/N/K=2048/2048/6144，BF16、转置权重视图stride=[1,6144]；16K采到M=4096。

对Gate/Up和Down的M=2048/4096各扫描518组，M=2048 Down原配置256/128/32、8 warps、2 stages约0.7284ms；256/256/32、16 warps、2 stages约0.5542ms，五轮交错复测的Kernel加速1.314×。该配置原本在候选池，但历史性能模型排名115，未进入top15实际计时。完整扫描其余三组当前配置已经是本轮赢家。

## 2. 改动内容

修改FlagGems `_iluvatar/ops/mm.py`：仅当M/N/K=2048/2048/6144、A/B/C均BF16且stride分别为[K,1]、[1,K]、[N,1]时，固定BM/BN/BK=256/256/32、warps=16、stages=2、Split-K=1，调用相同JIT MM主体。其它Shape和布局继续原有autotune路径。此分支是天数GEMM tiling优化的起点，后续改动独立编号，不把当前配置推广到未经验证的Shape。

## 3. 收益验证

独立Kernel数值、mm/mm_out和CUDA Graph重放通过；Trace显示固定分支grid=64、block=1024。官方A/B在同一插件和依赖版本、官方命令下完成；每场景各四轮均成功，summary为后三轮平均：

| 场景 | 基线 total tok/s | 候选 total tok/s | 吞吐变化 | 基线 Mean TTFT ms | 候选 Mean TTFT ms | TTFT降低率 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 3132.40 | 3170.60 | +1.22%，有效提升 | 4889.40 | 4736.03 | +3.14%，有效改善 |
| 16K | 2714.58 | 2714.89 | +0.011%，正常波动 | 181392.97 | 181315.98 | +0.042%，正常波动 |

分类依据原始CSV第2/3/4轮均值计算，未先四舍五入；4K吞吐+1.21962%、TTFT降低3.13679%，16K吞吐+0.011297%、TTFT降低0.042446%，均远离分类边界。上述是一次完整A/B；4K基线轮间波动明显。候选模型实际进入固定分支的次数和耗时占比尚未直接记录，不能将端到端变化确定归因于该改动；模型精度待验证，不能称完成赛事验收。

## 4. 与其他改动的关系

基于队友天数优化插件f82fa167，而非回退赛题原始版本。独立于W8A16，量化关闭。后续BF16提供者配置变化时需复验Separate和算法交界。

## 5. 合规性、风险与复现

保留原MM主体和非命中选择路径，不改官方serve/Benchmark参数、数据或数据集。官方脚本RUNS=4、SKIP_FIRST=1。精确Shape固定是实测性能策略，不作为GEMM通用能力限制；图内实际命中、其他布局及模型精度待验证。两场景逐项报告，不将16K正常波动写为有效提升。

完整命令、环境、原始四轮、日志及验证详见[七部分改动报告](2026-11-06-tianshu-down-fixed-config-official-ab.md)；原始证据在工作区`experiment-artifacts/2026-11-06-tianshu-down-fixed-01/`。代码及本文随本分支提交；原始实验材料保存在工作区归档目录。

基线来源：`kanceler/vllm-plugin-FL` 的 `s2-track2-final` 分支，固定提交 `f82fa1670509f081aa13b58464cb6e5627fc10c4`；本归档分支从该提交创建。测评使用该插件上的队友已有优化，未退回赛事原始基线。实际算子代码改动位于FlagGems，补丁见 `patches/001-iluvatar-down-fixed-tiling.patch`，需应用至FlagGems后才会生效。
