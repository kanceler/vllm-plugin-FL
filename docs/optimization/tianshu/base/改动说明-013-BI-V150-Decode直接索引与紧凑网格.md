# 改动 #013：BI-V150 Decode 直接索引与紧凑网格

> 环境：天数 BI-V150（16 SM / warp 64 / 128KiB SMEM），MiniCPM5-2B，BF16。
> 开发基线：`perf/BI-V150` 的 `8f17b5d5668082a10f765c6dc8e972862c754447`（已含 #011）。
> 状态：算子及真实路径验证通过；与 #012 联合完成双场景简化端到端测试。默认启用（仅命中已验证设备/形状时生效），联合 MATH 尚未运行。
> 编号说明：本改动在实验阶段记为 #016，本次提交按正式编号 #013 归档，不是另一项新增优化。

---

## 1. 为什么这样改

当前实际 Attention 是 vLLM Triton unified，Decode 使用 3D Split-KV head＋Reduce。
#006/#009/#010 已覆盖分块/流水/warp，#011 已将 Mixed 中足够长的单 Query 前缀
拆为独立 Decode 调用；这些优化仍保留。

通用 head 为 query block 构造保守上界 `floor(q.shape[0]/BLOCK_Q)+num_seqs`。
本模型 GQA=8、BM16、BLOCK_Q2，单 Query Decode 实际每序列只需一个 block，
却额外启动约三分之一空 CTA，且每个 CTA 仍通过二分查找定位序列。

| 实际形状 | 通用 head grid | 紧凑 grid | CTA 数变化 |
|---|---|---|---|
| 4k B64 | `(96,2,16)` | `(64,2,16)` | 3072→2048 |
| 16k 有效31/padding32 | `(48,2,16)` | `(32,2,16)` | 1536→1024 |

空 CTA 较轻，不能据其数量直接声称省三分之一时间。真正收益由完整 head＋Reduce 测量。

## 2. 改动内容

新增 `vllm_fl/attention/decode_index_policy.py`，由 `tile_policy.py` 在 #011 安装后接入。

1. metadata 构造阶段复用已有 CPU query 前缀和，每步确认是密集单 Query 前缀、仅尾部 padding；
   GPU 前缀身份标记各层复用，不增加逐层 D2H 同步。Mixed 复用 #011 已验证的前缀计划。
2. 从当前上游 head 的源码做一次精确局部替换，把 query→sequence 二分查找换成直接索引；
   用内存源码注册生成对应 JIT，不修改安装包、不维护整个计算 Kernel 的独立 fork。
3. 只对合格调用使用 `(num_seqs,2,16)` 紧凑 grid；其余功能保留原 Kernel 和原 grid。

| 文件 | 作用 |
|---|---|
| `vllm_fl/attention/decode_index_policy.py` | CPU 资格、head 局部源码转换、受保护的启动分发 |
| `vllm_fl/attention/tile_policy.py` | 在 #011 后接入带独立开关的安装 |
| `tests/unit_tests/attention/test_decode_index_policy.py` | 前缀布局、参数/功能回退、metadata 身份与未知源码保护 |

Attention 的 QK/PV、softmax、KV 读取、归约顺序和原 Reduce 全部不改。保持 S16/T16、
BM16/BLOCK_Q2、warp2/stage1、head 85 寄存器/线程和 12KiB SMEM；仍是两次 launch，
没有新增 GPU 中间 Tensor。Reduce 直接索引原型仅多省约 1µs，不纳入本改动。

## 3. 开关、范围与 Graph 安全

`VLLM_FL_BIV150_DECODE_DIRECT_INDEX` 默认 1，不设置即可启用；显式设为 0 并重启 worker 回退，unset 则恢复默认启用。
关闭时不安装新分发或构造专用 JIT。仅实际 Iluvatar BI-V150 启用，源码/API 不匹配不安装。

限定当前已验证的 BF16 Q/K/V/out、16 Q heads/2 KV heads/D128、页16、3D、B≤64、
S16/T16、普通 causal 配置。特殊窗口、PrefixLM、量化/输出量化、bias、sinks、descriptor、
多 Query/内部空洞等不支持路径保留原实现；不改 #011 的 D≥16 门槛和 Prefill 后缀。

Graph 资格依赖当前 ModelRunnerFL“删除未调度请求→正 token 数密集 batch→仅尾部 padding”
的不变量。Python 资格检查发生在捕获时，Graph 回放不重新执行；不能把图外内部洞回退测试
当作任意修改 Graph metadata 都安全的证明。未来改 Runner 或新增调度模式须重新验证。
诊断字段 `force_original` 只用于展开 A/B，不能用于改变已捕获 Graph 的节点。

## 4. 收益与正确性验证

### 4.1 完整 Attention A/B

真实短运行捕获的长度形状、合成 BF16 Q/KV、实际 NHD stride、随机物理页、
不等长 KV 和真实 padding；充分预热，40 组同环境交替 Graph，计完整 head＋原 Reduce：

| 形状 | 原 ms | 本改动 ms | 耗时下降 |
|---|---:|---:|---:|
| 4k B64，KV4105–4233 | 1.000506 | 0.959881 | 4.06% |
| 16k 有效31/padding32，KV16385–16486 | 1.868921 | 1.800698 | 3.65% |

两组均 40/40 对更快，有效输出和修改 Q 后的 Graph 回放输出逐位一致。
补充 B16/B8/B1 未见退化。并非只计 head 或只用等长、连续 KV 的合成形状。

真实模型短检查中，4k64/64、16k32/32 请求成功；Mixed/Decode 展开每层 42 次新 head 命中，
Prefill 后缀仍是原 Kernel。同输入经过 42 层后的最终 hidden/logits 与原路径逐位相同。
模型检查后仅补输出量化/非 BF16 排除保护，最终保护路径另做算子及回归复核，计算主体未改。
此前本改动与 Mixed 合计 45 项回归通过；提交前在隔离的拟提交代码树上联合复核
本改动、#012 和 Mixed 回归，**79 项全部通过**，Ruff/格式及差异检查通过。
默认启用修正后，补充未设置开关、显式启用/关闭及设备回退验证，**83 项全部通过**；
真实 BI-V150 上取消两个开关后加载正常 backend，确认两项优化均自动安装。

### 4.2 与 #012 联合端到端结果

2026-11-11，每场景完整预热一轮、正式一轮，只取正式轮；未运行四轮。
固定场景、原版 benchmark `run_once`、固定天数 serve 参数与 `FULL_DECODE_ONLY` 均保留。

| 场景 `[输入,输出,并发,请求数]` | 历史 #011 后 total tok/s | #012+#013 正式 total tok/s | 增量 |
|---|---:|---:|---:|
| `[4096,1024,64,256]` | 3330.59 | **3456.39** | **+3.78%** |
| `[16384,1024,64,128]` | 2781.10 | **2865.16** | **+3.02%** |

| 场景 | 历史 Mean TTFT ms | 联合正式 Mean TTFT ms | 变化 | 正式 Mean TPOT ms |
|---|---:|---:|---:|---:|
| 4k | 4471.86 | 4481.68 | +0.22% | 87.72 |
| 16k | 176468.06 | 172305.75 | −2.36% | 160.91 |

正式轮 256/256、128/128 请求成功，输入/输出 token 数分别为
1048576/262144、2097152/131072；时长 379.22s、777.70s。
预热 3064.74、2821.19 total tok/s 仅存档。两个候选都确认命中，正式计时前一次性观测包装已移除，
无模型 profiler 或 Graph 展开。Decode 命中记录来自捕获前 warmup，不冒充 Graph 内部 trace。

**口径限制**：这是联合启用结果，不能单独归因给本改动。对照是 2026-11-09 #011
后三轮历史均值，本次没有重测基线，不能排除跨时段环境/调度波动；不是官方四轮口径。
联合 MATH 尚未运行，不以吞吐和逐位检查代替正式精度验收。

## 5. 复现与回退

本分支已默认开启 #012/#013，无需设置额外开关，直接沿用固定天数服务参数。

完整启动命令及“不修改原脚本、外部调用 run_once 两次”的简化复现步骤，见
[改动说明 #012](改动说明-012-BI-V150-Sampling行并行.md)。只关闭本项时将
`VLLM_FL_BIV150_DECODE_DIRECT_INDEX=0` 并重启；#012 和 #011 独立保留。unset 恢复默认启用。

上述 Benchmark 是显式设 1 时的实测结果；随后默认值修正为 1，计算 Kernel、分发保护和已测路径相同，未重复长测，不声称默认值修正后重新运行过 Benchmark。

工作区完整记录：`天数BI-V150-Decode直接索引小实验报告.md`、
`天数BI-V150-Sampling与Decode直接索引联合端到端Benchmark报告.md`；联合证据在
`artifacts/tianshu_combined_e2e_20261111/evidence/`，算子/模型证据在
`artifacts/tianshu_decode_index_20261111/evidence-final/`。这些路径属于项目工作区，未随插件源码提交。

## 6. 合规性与保留范围

只消除已验证单 Query 路径的通用定位和空 CTA，保留原 vLLM Attention 与框架算子选择。
不改模型权重、计算精度、KV 格式、采样、serve 参数、Benchmark、评测数据或评分器。
不加入无收益的 Reduce 改写或 Shape-aware Split/Tile，不改变 #006/#009/#010/#011。
设备/布局来自真实运行元数据，不识别题目或 Benchmark 场景名称。
