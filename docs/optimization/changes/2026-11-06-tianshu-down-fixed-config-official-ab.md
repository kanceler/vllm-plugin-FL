# 天数Down热点固定配置与官方A/B（官方A/B完成）

## 1. 基本信息与摘要

2026-11-06，Iluvatar BI-V150。用户授权将已验证的Down热点配置落地并执行官方对照。FlagGems独立分支perf/tianshu-gemm-tiling-optimization（原名perf/tianshu-down-fixed-config），本地worktree为worktrees/FlagGems-tianshu-down-fixed，远端为/workspace/FlagGems-tianshu-down-fixed。生产安装包不覆盖，使用隔离包副本。

当前：实现、独立正确性/图验证和官方双场景A/B已完成。4K吞吐有效提升，16K吞吐正常波动；模型级实际命中仍需进一步取证。

## 2. 问题与原因分析

完整518组扫描表明，BF16转置权重布局的M,N,K=2048,2048,6144使用256/256/32、16 warps、2 stages优于原256/128/32、8 warps、2 stages，独立Kernel约0.728→0.554ms。原赢家属于原候选池，但历史性能模型rank115、未进top15。微基准不等于模型收益，需要官方双场景验证。

## 3. 具体实现

仅修改FlagGems src/flag_gems/runtime/backend/_iluvatar/ops/mm.py的_launch_mm。命中精确Shape、A/B/C均BF16，stride分别[K,1]、[1,K]、[N,1]时直接以固定配置启动相同JIT Kernel；其他Shape、dtype、布局沿用原autotune。mm与mm_out共享入口。固定Shape下明确EVEN_K及32位offset合法，保留原Kernel主体和主要选择逻辑。无量化，无模型权重格式变化，不改变Attention或调度。

## 4. 实验与验证

seed42，FP32参考关闭TF32；rtol=.02、atol=.125。M=2048固定分支、M=16与4096回退均通过公开mm与mm_out验证，relative L2约6.10e-5/7.34e-5/6.14e-5；mm_out与mm输出完全一致。固定分支CUDA Graph capture/replay输出一致。Trace确认mm_kernel grid=[64,1,1]、block=[1024,1,1]。

官方脚本SHA256为0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565，已核对RUNS=4、SKIP_FIRST=1。原状基线与候选均使用同一安装包复制来源和插件f82fa16，候选只替换上述文件；实际import保存在各模式imports.txt。服务启动参数及官方bench参数保持不变。

官方A/B均完整执行，每场景四轮全部SUCCESS，4K每轮256/256、16K每轮128/128；取后三轮均值：

| 场景 | 基线 tok/s | 候选 tok/s | 吞吐变化 | 基线TTFT ms | 候选TTFT ms | TTFT降低率 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 3132.40 | 3170.60 | +1.22%，有效提升 | 4889.40 | 4736.03 | +3.14%，有效改善 |
| 16K | 2714.58 | 2714.89 | +0.011%，正常波动 | 181392.97 | 181315.98 | +0.042%，正常波动 |

比例由原始CSV第2/3/4轮均值计算，未先将均值四舍五入，远离1%分类边界；正式原始CSV及日志保留。4K后三轮基线3006.35/3184.33/3206.52，候选3122.76/3163.71/3225.34 tok/s，轮间有波动；当前为一次完整A/B。模型级实际命中与耗时覆盖未验证，不能将端到端变化完全归因固定配置；本任务未安排模型精度评测，不宣称完整赛事验收。

## 5. 与已有改动的关系

承接2026-11-06-tianshu-bf16-full-config-scan.md，仅落地2048 Down赢家；4096 Down原配置已为赢家，保持原状。与W8A16独立，不启用量化。原主工作树已有修改保留，不提交无关内容。

## 6. 合规性与风险

新开发的固定选择针对精确数值/布局条件，保留其他选择路径；不是仅切换现有算子。未改官方参数、Benchmark或数据。先基线后候选、每项新服务进程，完整双场景之间不插入负载或清缓存；两模式间检查GPU/端口释放。独立验证结束GPU空闲68MiB。官方两项任务均完整结束，候选结束后GPU无计算进程，68MiB、利用率0%。

## 7. 复现方式

远端/workspace/experiments/tianshu-down-fixed-20261106-01/，本地控制脚本experiment-artifacts/2026-11-06-tianshu-down-fixed-01/run_ab.py。使用baseline-python/flag_gems和candidate-python/flag_gems隔离副本，PYTHONPATH同时指向/workspace/tianshu-final-20261104；VLLM_PLUGINS=fl，清理W8A16遗留变量并明确关闭Down/shape dispatch量化。

run_ab.py按官方参数启动服务并依次执行两项完整官方benchmark，保存命令JSON、脚本指纹、进程/GPU状态、健康检查、原始日志、退出码及新生成CSV；候选结束后自动停止本任务服务。完整A/B证据同步至本地experiment-artifacts/2026-11-06-tianshu-down-fixed-01/；未提交/推送Git。
