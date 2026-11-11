# 沐曦 MiniCPM5-2B：BF16 GEMM 邻接融合落地方案

日期：2026-11-04。状态：设计文档，未实现、未启动新测评。本文获授权的工作仅为编写方案，源码实施仍须按工作区 AGENTS.md 讨论并获得同意。

> 平台隔离约定（2026-11-04）：本文为沐曦独立计划。天数另见 [天数审计与实验计划](2026-11-04-tianshu-gemm-audit-and-experiments.md)。初期分别管理调用证据、实现、配置、实验和结果；仅实际路径与接口重合且两端验证后，才提取共享优化。本文候选不自动适用于天数。

## 1. 目标、范围与成功定义

在不引入量化的条件下，研究 MiniCPM5-2B 的 BF16 GEMM 与邻近算子融合，减少中间张量读写和 Kernel 启动，同时保持模型行为与数值要求。权重、激活、对外输出保持 BF16，GEMM 的累加和舍入行为须与实际基线核对。本文不改变 W8A16 主路线，也不将 BF16 融合与量化收益混报。

第一候选为 Gate/Up GEMM+SwiGLU；第二候选为当前 TP=1 下 O/Down GEMM+模型缩放/残差。第二候选必须先完成实际调用链审计，若无法减少完整链路开销则停止。QKV+RoPE、GEMM+RMSNorm 暂缓；Attention 主体、KV Cache、调度、Embedding、LM Head 不在本次实现范围。

成功分为三级，不能混淆：算子正确性通过；完整调用在所测 Shape 上获得稳定收益；模型精度和官方双场景性能验证完成。未完成最后一级，不宣称赛事验收完成或端到端收益。

## 2. 已有证据与待补充事实

证据来源均须标注日期和平台，不以天数捕获图代替沐曦实际运行。模型采用 GQA，已有形状证据为 16Q/2KV、head_dim=128；本次接入前仍核对实际模型配置。

| 事项 | 当前证据与结论 | 实施前还需补充 |
|---|---|---|
| Packed QKV | 沐曦 11/3 层清单：QKVParallelLinear，BF16 权重 [2560,2048] | 当前提供者、布局和实际 GEMM Kernel |
| Packed Gate/Up | 沐曦 11/3 层清单：MergedColumnParallelLinear，BF16 权重 [12288,2048] | Gate/Up 行顺序、bias、stride、分片及加载约定 |
| O、Down | 层清单：RowParallelLinear，权重分别 [2048,2048]、[2048,6144] | 当前执行分支、缩放/残差/Norm 完整数据流 |
| TP | 沐曦 11/3 服务日志 tensor_parallel_size=1 | 每次复验记录实际 TP；层类型不等于存在 all-reduce |
| SiLU×Up | 沐曦 GPU Trace 实际出现 triton_poi_fused_mul_silu_1 | 与层、GEMM、M 对齐，确认当前版本仍如此 |
| RoPE | 尚无本方案所需的沐曦提供者及最终 Kernel 核验 | 不宣称已走 FlagGems，不列为首轮任务 |
| residual+Norm | 尚未逐项确认沐曦最终 Kernel 边界 | 不把图中 op 数或其他平台证据当作实际启动数 |

本地证据：

- `experiment-artifacts/2026-11-03-w8a16-shape-coverage/local-delivery/metadata/worker-5954-inventory.json`。
- `experiment-artifacts/2026-11-03-w8a16-shape-coverage/local-delivery/logs/bf16-observer-pilot-v3-service.log`，启动配置 TP=1。
- 沐曦远端 `/root/experiments/2026-11-03-w8a16-shape-coverage/results/bf16-observer-conditional/` 下三份 Trace：`trace-14fab7a26bf845fabf66a031557ef4cf-pid11217.json`、`trace-422df6d962464c43b8dd8a287c3b0e34-pid11217.json`、`trace-ed56e516b117463ab1121aef0313f606-pid11217.json`，融合 SiLU/Mul Kernel 事件数依次 1344、42、504。计数仅针对对应采样窗口，不推导未来覆盖率。
- [已观测 M 明细](../changes/2026-11-04-w8a16-observed-m-details.md)可复用 M 集合及历史 BF16 诊断资料；其中量化候选时间不作为本文 BF16 融合性能结论。

11/4 只读检查时指定沐曦容器没有正在运行的 vLLM 服务。历史 Trace 证明采样窗口的实际执行，不代表当前实时请求状态。

## 3. 阶段 A：实际调用链审计

先复用既有沐曦 Trace、编译缓存产物、模型源码和层清单，缺失字段再补采，禁止重做全框架瓶颈分析。为四种投影建立以下记录：

`模型层/投影 → 实际 M/N/K → 输入/权重/输出 dtype 与 stride → BF16 GEMM 提供者及配置 → 编译后 GPU Kernel → 后处理 Kernel → 输出消费者`。

重点核查：

1. Gate/Up 是否按 [Gate;Up] 排列，两个半区各 6144；split 是否仅为 view，是否引入复制。
2. SiLU 与 Mul 已融合的事实在当前版本是否保持；不能把 FX 图中的两个节点算作两个 GPU Kernel。
3. O/Down 的 MiniCPM 缩放常数、施加位置、BF16 cast、残差更新、alias 和 Norm 输出；不要用简化 Transformer 公式覆盖实际模型逻辑。
4. TP=1 时是否直接省略通信，TP>1 时的归约边界；不通过更改官方启动参数验证候选性能。
5. 编译/capture/实际负载 replay 分别归档；Python 调用计数不当作图重放次数。

阶段交付为逐投影调用链表、带关联标记的 Kernel 清单和缺口列表。若缺少关联信息，明确写“待补充”，不从 Kernel 名字推测层归属。

## 4. 阶段 B：Gate/Up GEMM+SwiGLU

### 4.1 基线与目标数据流

基线：`X[M,2048] × W[12288,2048]^T → Y[M,12288] → SiLU(Y_gate)×Y_up → Z[M,6144]`。

候选：配对计算同一输出通道的 Gate/Up 累加结果，在 program 内完成与基线一致的 cast、SiLU 和 Mul，直接写 Z。保留原 BF16 层及原后处理作为合法回退，不改变加载、TP 通信及模型接口。

若完全消除 Y 的写回与读取，逻辑流量减少上限约为 `2 × M × 12288 × 2 = 49152M bytes`；这不是实测 HBM 流量或收益预测。Z 的写出仍然需要。

### 4.2 配对 tile 设计

普通 packed GEMM 的 Gate/Up 常由不同输出 program 计算，因此不能仅向现有 epilogue 添加一个表达式。候选 program 处理相同通道范围的两个权重 tile，复用 X tile，分别维护 Gate 与 Up 的 FP32 累加器。逻辑输出宽度为 6144，权重读取分别偏移 0 和 6144 行；确切行顺序以阶段 A 为准。

先与 BF16 GEMM 提供者负责人确认可复用接口和分块，避免重复实现通用 BF16 GEMM。配对 tile 的 BM/BN/BK、warps、stages 在真实 M 上选择，记录寄存器、spill、并行度与尾块效率。双累加器可能降低 occupancy，省流量不保证更快。

若采用 Split-K，必须先完成所有 K 分片归约，再执行非线性；可研究“最终 reduce+SwiGLU”。partial workspace 读写、分配/复用和 reduce 全部计入完整调用，不能与单 Kernel 方案混算。

### 4.3 数值契约

先从当前实际融合 SiLU/Mul 的编译代码确认计算与舍入顺序。GEMM 原输出为 BF16时，候选必须在对应位置模拟 BF16 舍入；不能直接对 FP32 累加结果做激活并默认数值等价。SiLU 内部是否使用 FP32、Mul 前是否舍入也须核查。

正确性参考使用原始 BF16 权重及实际基线路径，不用量化后权重，不涉及 scale-before-dot 或量化 scale。scale-before-dot 已因量化数值契约不一致弃用，不恢复、不测试。

### 4.4 接入约束

采用显式 BF16 融合接口，输入为 X、packed W 及必要的 bias/元数据，输出为 Z。只对已验证的 Gate/Up 模式调用，同时消除其对应的旧激活调用，避免重复执行。不能仅替换 Linear 输出而让旧消费者仍将 Z 当作 [M,12288]。

优先使用插件支持的编译模式匹配/模型适配接口；具体接口待阶段 A 确认。必须兼容动态图、fake/meta shape 和 CUDA Graph。未匹配、dtype/布局不支持、资源不足时，完整回退到原 GEMM+激活。推理/capture 期间不做大规模搜索。

## 5. 阶段 C：O/Down 缩放与残差融合

此阶段依赖审计结果，不预先承诺实现。当前 TP=1 没有跨卡归约屏障，允许研究 GEMM 输出 epilogue 中的模型缩放和残差，但须逐位置复现真实算式、cast、alias 和输出消费关系。

如果基线已经把残差更新与 RMSNorm 放在同一 Kernel，候选将加法移进 GEMM 后仍可能需要单独 Norm，Kernel 数和流量可能不降。因此必须比较：

- 原 GEMM + 实际缩放/残差/Norm 完整链路；
- 候选 GEMM epilogue + 剩余 Norm 完整链路。

仅测 GEMM epilogue 或仅测残差时间不构成收益证据。需记录更新后的 residual 是否必须保留，以及 normalized output 是否另存。TP>1 默认回退；不设计通信融合，也不将单卡结果推广到多卡。

GEMM+RMSNorm 不作为首轮实现：Norm 需要整行归约，跨 N tile 数据不在同一 program 内；若需改变 GEMM 主体分块或新增归约，单独讨论计划。

## 6. 拟修改位置与职责

以下是拟议位置，均未创建；阶段 A 后确定准确路径，再申请源码实施同意。

| 位置 | 拟议职责 |
|---|---|
| `FlagGems/src/flag_gems/runtime/backend/_metax/` 下专用 BF16 融合实现 | 复用提供者计算主体，新增配对 Gate/Up epilogue；具体目录依现有后端组织确定 |
| FlagGems 对应 tests/benchmark | 数值、布局边界和完整调用测试 |
| `vllm-plugin-FL/vllm_fl/compilation/` 或现有模型适配入口 | 识别并替换完整 Gate/Up→SwiGLU 模式，支持回退 |
| `vllm-plugin-FL/benchmarks/diagnostics/` | 独立融合 microbenchmark、原始采样和 Shape 归档 |
| `vllm-plugin-FL/docs/optimization/changes/` | 按工作区七部分规范记录实现、证据、局限及复现方式 |

不改官方 serve 参数、Benchmark 实现和正式评测参数，不以开关切换现成算子冒充实质性优化。实现应包含新的融合计算及真实运行接入。

## 7. 验证矩阵与测评口径

### 7.1 算子正确性：待补充

- 当前已观测 M：4、8、16、24、32、40、48、56、64、72、80、88、96、104、112、320、1554、2048；160 为人工代表点，另覆盖 M=1、非整除及配置交界两侧。
- Gate/Up 本地 N/K、packed 顺序、真实权重、随机输入、极值和 SiLU 饱和区；O/Down 如实施则覆盖各自实际语义。
- 连续及非连续输入、非整除尾块、bias/无 bias、不支持 dtype、TP 回退、动态 M、图捕获与重放、workspace 生命周期和 alias。
- 保存最大绝对误差、relative L2、逐元素差异和多层累积误差。数值门槛须在首次性能搜索前依据实际 BF16 路径及模型容差明确；**门槛当前待补充，不能借用 W8A16 阈值或看到性能后放宽**。

### 7.2 算子性能：待补充

基准为同环境、同提供者版本的原始 BF16 GEMM 加实际后处理。同时记录原生 BF16 GEMM 单项，完整链路才用于判断融合收益。原生 BF16 提供者如更新，基线及候选一并复测。

预热、编译、调优、稳态和图重放分别记录。稳态初步沿用历史诊断协议：warmup 50ms、rep 150ms、五轮交错、汇总中位数并保留全部样本；这是 microbenchmark 协议，不是官方模型评测轮次。补做多层工作集和实际布局验证，避免只测单权重热缓存。

| 投影 | M/N/K、布局 | 基线 GEMM μs | 基线完整链路 μs | 候选完整链路 μs | 加速比 | Kernel 数/流量证据 | 正确性 |
|---|---|---|---|---|---|---|---|
| Gate/Up | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| O（条件实施） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| Down（条件实施） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

完整链路加速比 = 基线完整链路时间/候选完整链路时间。降低率 = 1−候选/基线。所测点超过 1% 且重复证据稳定才列为有效改善，±1% 含边界列为波动；噪声明显时复测，不据小差异固化分界。保留留出 Shape，只验证固定策略；搜索结果称“所测范围内较优”。

### 7.3 模型与官方评测：待补充

先验证实际启用覆盖和每层输出，再做 math_500 Level 3 与官方 4K/16K。原始 BF16 基线与候选分别使用新服务进程，相同官方命令、软件、模型及 BF16 提供者；按根 AGENTS.md 完成归档、进程清理、GPU/端口核查及环境记录。独立 GPU 测试、精度与性能串行，不在 4K/16K 之间插入清理或额外负载。

官方 Benchmark 保持 RUNS=4、SKIP_FIRST=1，summary 使用后三轮平均，保留 raw_runs、TTFT、成功请求数与异常。精度最低 0.950，且核心能力和行为正常。total tokens/s 与 TTFT 按工作区 ±1% 规则逐场景分类，不自行新增两项都必须改善的计分规则。

| 场景 | BF16 total tokens/s | 候选 total tokens/s | 吞吐变化及分类 | BF16 TTFT | 候选 TTFT | TTFT变化及分类 | 完整性 |
|---|---|---|---|---|---|---|---|
| 4K | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 16K | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

模型精度、异常输出检查、实际融合层数、算法调用覆盖和耗时覆盖：均待补充。不得将历史量化精度或性能作为本文候选结果。

## 8. 实施顺序、停止条件与复现归档

1. 完成阶段 A 调用链审计，确定 BF16 提供者、数值契约、准确接入文件及预设误差门槛。
2. 与用户讨论上述具体代码计划，取得明确同意后，在独立用途分支/隔离目录实现 Gate/Up 候选。
3. 通过正确性后做完整调用 microbenchmark；按 M 选择已验证候选，范围外保留原路径。同步提交改动报告。
4. Gate/Up 有证据后再判断是否实施 O/Down；不因文档列出候选而自动扩大实现范围。
5. 固定策略通过留出点、真实布局、图重放与多层工作集后，再进行模型精度及官方双场景验证。

停止条件：数值契约无法满足；配对 tile 导致完整链路退化且有限候选搜索无改善；替换破坏 alias/动态图/图捕获；残差重组没有减少实际开销。记录负结果，不无限调参。允许仅部分 M 获益，但必须有明确 dispatch 和合法回退。

每个实验使用新目录，记录两仓库 commit/diff、提供者及 Kernel 哈希、硬件/驱动/Python/Triton、输入/权重布局、命令、配置来源、原始逐轮时间、正确性、编译及缓存状态、图重放证据。复现命令待工具实现后补充实际命令，不提供假装已执行的占位结果。密码、令牌和 SSH 临时连接信息不进入文档或 Git。
