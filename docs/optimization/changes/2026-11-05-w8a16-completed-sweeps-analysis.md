# W8A16 Split-K、Dense 完整归档与入口诊断审计

## 1. 基本信息与摘要

2026-11-05，MetaX C500。Split-K 60/60点、Dense 16/16点、入口诊断全部正常结束，退出码均0。原始数据已完整取回本地并逐项重算中位数、核对源码指纹。没有新增GPU实验或修改运行源码，本次只做归档与分析，未提交、未推送。

重要修正：完成后发现显式候选调用共用的LibEntry缓存键未纳入num_warps/num_stages，导致同一BM/BN/BK/S组合更换launch参数时可能命中先前Kernel。本轮原始计时与正确性事实保留，但不能认证完整warps/stages搜索，也不能把名义赢家配置直接固化进生产。脚本complete表示声明点执行结束，不代表配置实传审计通过或赛事验收完成。

## 2. 问题与原因分析

历史Down小M正收益曾出现，但宽搜索期间同名配置耗时明显更高。原先提出“新增能力检查造成CPU提交间隙”的假设，已通过独立入口对照检查：三组共享同一Kernel，结果未呈现当前公共入口稳定更慢。因此不能从宽搜索耗时直接减去推测的检查成本，也不能将本轮差异归因于检查。

进一步检查实际归档的flag_gems/utils/libentry.py：LibEntry.run只从JIT函数参数收集spec_args/dns_args/const_args，再生成entry_key；num_warps/num_stages为启动选项，不属于这些JIT参数。缓存命中后直接使用缓存CompiledKernel。因此显式启动同一JIT函数、同tile/S、不同warps/stages时，后一次可复用先前配置。Dense/Split-K显式入口都共用同一LibEntry对象，受到该风险影响。

使用归档源码提取原始key/run方法，以模拟JIT编译器和设备作CPU复现：先BLOCK64/warps2/stages1，再BLOCK64/warps8/stages3，仅发生一次模拟编译，两次返回同一Kernel对象，第二次实际选项仍2/1。此复现证明缓存逻辑问题，不是GPU性能测量，也不将每个历史Kernel的实际参数都推断为2/1。

现有compiled-resources快照未完整关联每个候选的实际cache key，因此不能逐行可靠恢复实际warps/stages。BM/BN/BK/S作为constexpr进入键，与launch参数的情况不同。本轮更宽tile/S仍提供候选线索，但完整联合调优结论需要修复后复验。

## 3. 归档内容与可追溯性

Split-K及入口诊断：工作区experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/completed-evidence/，包括splitk/results全部候选与raw、源码/环境指纹、编译资源、原始stdout、退出状态、运行后GPU状态以及entry-audit的helper、排队命令、原始80条计时、误差、布局与summary。

Dense：experiment-artifacts/2026-11-05-w8a16-dense-mid-01/completed-evidence/，包含16点全部候选、raw、正确性、源码快照和环境/显存接受条件。归档包均保留，未覆盖历史实验。

归档SHA256：

- Split-K＋入口诊断：47b8884a67e02968c1dd5d74a3d3c3c93f167e59a993b4c86d22b881553e253b。
- Dense：3370b477cb7e8a785073e97f43efe82dbf72862dc42b7152ba2d6e01fd411806。

两实例使用相同显式后端SHA256 41397bd1101e400126dd288183c68577e7ebf7742fac6c31cc4fb46d98f926b4、BF16提供者2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec、LibEntry 67c3318ff94c602b01622fb0b5900e17e6ac449e61c1c6165265ecc6cf33622a。仍为不同实例，不拼接绝对延迟形成分界。

## 4. 实验核验与性能结果

| 实验 | 完成点 | 初筛通过记录 | 编译/资源/执行失败记录 | 终测原始样本 | 重算候选/seed中位数 |
|---|---:|---:|---:|---:|---:|
| Split-K | 60 | 13188 | 105 | 4770 | 714 |
| Dense | 16 | 2731 | 697 | 1590 | 190 |
| 入口诊断 | 4个M/seed组合 | 不适用 | 0 | 80 | 16 |

共6440条终测原始样本，920行汇总中位数全部与raw一致。初筛记录数不等于独立配置实际编译数；发现缓存别名后尤其不能宣称每条记录都代表不同launch参数被执行。

Split-K最终记录的最大Kernel relative L2约0.000197918；Dense约0.000054193。均指本轮结果中记录的终测Kernel误差，不包括量化权重相对原权重误差；不能外推模型精度或未实际执行的名义launch组合。

### 4.1 本轮实测的正收益点

在脚本既定两seed稳定性及>1%规则下，只有O极小M出现两套输入都优于同场原始BF16的执行路径。配置标签受缓存风险影响，以下不认证为可部署的最终配置。

| O的M | BF16 A/B µs | Split-K A/B µs | 耗时降低率 A/B |
|---:|---:|---:|---:|
| 3 | 28.672 / 28.672 | 26.368 / 26.880 | 8.04% / 6.25% |
| 4 | 28.160 / 28.160 | 27.648 / 27.648 | 1.82% / 1.82% |
| 8 | 28.672 / 28.672 | 26.624 / 26.624 | 7.14% / 7.14% |

Dense 16点没有两套输入都稳定优于原始BF16的点。Gate/Up中段Dense快于同场历史固定Split-K及Separate，但还未与真实生效的新Split-K赢家同机对照；O中段没有稳定的Dense优于Separate优势。不能据此固化算法分界。

Split-K BF16对照不稳定点为Down M2、O M2/16/96、QKV M1/2；Dense为Gate/Up M16。不稳定只按本次脚本的两seed门禁定义，不把五轮事件分辨率离散样本当作严格置信区间。

### 4.2 入口对照

固定Down N2048/K6144、BM16/BN64/BK128/warps4/stages2/S8，共用同一Kernel。完整调用含workspace、输出分配、partial及reduce/scale，单位µs：

| M | seed | 原始BF16 | 旧式入口 | 当前配置入口 | 公共显式入口 |
|---:|---:|---:|---:|---:|---:|
| 1 | 20261028 | 39.936 | 41.984 | 41.984 | 41.984 |
| 1 | 20261104 | 39.936 | 43.520 | 43.264 | 43.008 |
| 16 | 20261028 | 39.936 | 44.544 | 44.544 | 44.544 |
| 16 | 20261104 | 39.936 | 45.568 | 45.568 | 45.568 |

三个M/seed组合中入口中位数完全一致；另一个M1组合公共入口反而低0.512µs（约1.18%），不支持公共入口检查导致稳定额外开销。此诊断仅覆盖历史固定配置，不等价于所有候选、精确CPU壁钟、纯Kernel或图重放测试。

独立进程中固定配置恢复到约42～46µs，而宽搜索期间同名配置约55～57µs。缓存别名提供了具体可复现的执行配置差异机制；但没有做“同一实际二进制、同一状态”的完整消融，不能将差值全部归到缓存，也不能改写原始计时。

## 5. 与已有改动的关系

本报告是11月4日搜索工具、11月5日执行修复和入口诊断的完成后审计。先前“每组配置均显式实测”“最优warps/stages”应收窄为名义候选标签下的实际调用计时；原始数据不作废、不覆盖，但联合调优有效性未完全成立。已有报告追加提示，详细表仍展示名义配置并显著标注风险。

之前提出“若入口差异明显，只补赢家即可”是条件方案。入口复验未发现这种稳定开销，反而发现launch参数缓存别名，因此下一步不能只将名义赢家拿出来一次计时就宣布完成。应先修复诊断启动隔离与实际配置证据，再对代表tile/S重新验证warps/stages，之后进行同机finalist完整对照。

## 6. 合规性、风险与后续建议

本次未改变生产源码、官方参数、Benchmark、依赖、模型或量化路线；数据为synthetic eager算子诊断，非模型精度/官方吞吐/TTFT结果。INT8权重按通道静态对称量化，FP32 scale，BF16激活与INT8→BF16 dot、FP32累加/归约后scale，不是原生INT8 GEMM。

建议后续修复仅针对显式候选启动：每个完整launch配置使用独立LibEntry，或采用经验证能区分启动选项的专用调用；不优先改全局LibEntry，避免影响其他算子。必须记录requested与actual的warps/stages、tile/S、二进制/IR哈希及缓存来源。先做小范围配置切换回归，再决定代表点重搜与finalist复验范围。修复和新GPU任务仍需用户确认具体计划；当前只归档分析，没有启动重测。

两个容器的同一N/K/M的微秒值不能直接拼接选择算法；Gate/Up M80和O M88/112等Dense点不在本轮Split-K清单，后续对齐需补这些点，而不是伪称已有同机三算法最优对照。纯BF16回退保持合法；GPU结果不能外推天数。

## 7. 复现与详细数据入口

在工作区执行：

```bash
python3 experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/reproduce_libentry_launch_cache_cpu.py
python3 experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/analyze_completed.py
```

分析产物在experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/analysis/：

- detailed-performance.md：60＋16点逐点BF16、候选、百分比、名义配置和对照稳定性。
- selected_timings.csv：全部76点两seed详细结果，包含Dense同场历史Split-K/Separate、Split-K旧控制项对比。
- all_candidate_final_timings.csv：全部904行候选/seed终测明细及原始五轮。
- entry_timings.csv：入口对照16行及原始五轮。
- audit.json：完整性、计数、中位数与源码指纹核验、缓存CPU复现。
- winner_tables.json：机器可读逐点汇总。

原始JSONL、候选失败、边界数值、完整编译快照和环境均在completed-evidence目录中。证据已取回，不依赖远端持续在线。
