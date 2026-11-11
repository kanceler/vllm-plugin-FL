# W8A16 三算法分发、共享候选与分层调优实施计划

> 日期：2026-10-30（UTC+8）。状态：讨论确认后的计划归档；未实施本计划的新代码、采集、调优或GPU实验。
> 本次授权：写入后续计划和工作区AGENTS规则。后续代码任务仍按根AGENTS第八节确认实施范围。
> 当前本机审查版本：FlagGems `f6a4257e3d042acd26a7d3c496848025a4b32c07`；插件 `f93489fa1afb74969b76701c8b551552a55ae3bf`。它们不是历史实验的自动复现版本，执行前重新核对提交、差异、实际import和指纹。

## 1. 目标、分工与优先关系

本计划作为当前W8A16算法分发和候选搜索的执行入口，细化[全局后续方案](2026-10-29-global-w8a16-followup.md)。历史实验事实不改写；范围、流程冲突以本计划及用户最新指令为准。

- 本人负责W8A16的三算法统一接口、Shape感知候选生成/剪枝、离线GPU调优、Runtime Dispatch、完整调用Benchmark和模型接入。
- 原生BF16 MM底层及其大M候选优化由队友负责，本任务复用接口，不重写或另起BF16调优主线。旧计划中的BF16/LM Head专项不自动列入当前实施范围；LM Head保持BF16。
- 三算法确定为Fused Dense、Fused Split-K、Separate，不为创新点无依据增加第四种算法。
- 算法分界与配置联合确定。共享候选按N/K、布局和设备等计算特征组织，不按固定层号复制Kernel，也不使用官方输入长度、题目或答案分派。
- 三算法参与竞争，不要求三算法都进入正常分发。如果Dense没有胜出，可以只保留为合法回退；需先验证合理候选、边界与未参与调优的Shape。
- 大M先保留Separate的正确性和少量完整调用基线；队友BF16版本稳定后补齐真实大M及交界验证。暂缓密集调优不等于免测；未完成前不宣称整个框架验收通过。
- 离线调优仍在真实目标GPU上完成。运行时可固化结果，前提是可追溯、版本匹配和合法回退；不能把手工固定、单候选或缓存命中描述为本轮充分搜索。

## 2. 当前实现与证据边界

源码入口（相对各仓库）：FlagGems `src/flag_gems/ops/weight_int8pack_mm.py`、`src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py`、`src/flag_gems/runtime/backend/_metax/ops/mm.py`；插件 `vllm_fl/quantization/w8a16/{linear,int8pack,reference}.py`。

| 路径 | 当前策略 | 调优现状 | 完整调用与临时张量 |
|---|---|---|---|
| Dense | 不命中特化时使用通用融合 | 原8组libtuner，M/N/K采用align32配置键策略 | 1个计算Kernel；无完整BF16权重或partial中间张量 |
| Split-K | M∈{1,16,32,64}、N%64=0、K%1024=0及布局/dtype条件 | 固定S=8、BN64、BK128、warps4、stages2、basic；M64用BM64，其余BM16 | partial＋reduce/scale；FP32 workspace约4SMN字节 |
| Separate | M≥128、相同N/K整除条件及连续BF16/INT8/FP32 scale | 转换/scale固定BLOCK1024；内部BF16 MM沿用其分派和调优 | Convert＋BF16 MM＋Scale；约2NK字节转换权重、2MN字节未缩放输出，另计内部MM及最终输出 |

当前没有跨三算法的统一调优。14组历史配置是同一Dense Kernel上的手工有限集合，未完整包含原8组，不是全局搜索，也未作为三算法生产配置库。BF16 MM内部可能包含清零、partial和归约，因此Separate的Kernel启动次数不能一律写成3。

### 2.1 能力条件与性能条件

- M白名单、M≥128是现有策略，不是算法数学限制。
- 当前Split-K缺少N/K尾部处理；N/K整除要求对现实现有必要性。放宽前必须实现或验证mask、分片覆盖与边界，不能只删判断。
- Separate的转换/scale有mask，整除限制从源码看可能过严，但必须经边界测试后放宽。
- 上层连续布局与bias限制需要显式处理；通用Dense的stride能力不等于整个插件入口已支持任意布局。
- 量化加载后原始BF16权重不保留，内部反量化回退不等于原始BF16基线。若设计双表示或持久转换缓存，须另行核算容量、生命周期和数值语义。

### 2.2 已有结论不能扩大解释

[同场官方负结果](../changes/2026-10-29-w8a16-official-comparison-negative.md)：旧Down-only组合4K/16K吞吐分别−5.19%/−2.62%，Mean TTFT分别+7.81%/+3.31%。已测Split-K和Separate多数点仍慢于BF16，不能把负收益全部归因于Dense。

[阶段总结](../research/2026-10-29-w8a16-stage-summary.md)中Dense对照显著较慢，但未提供正式负载逐Shape分支统计。42/42是Down层覆盖，不代表42层每次调用均走优化分支，也不证明Dense的实际调用比例。历史103/105精度不自动适用于最终组合、全层量化或后续数值变体。

## 3. P0-A：历史复用、Shape与分支覆盖采集

### 3.1 先查已有资料

本机历史计算图可核对packed关系与N/K，M为符号维度；coverage文件主要记录启用层和fallback日志。10月18日pilot汇总明确缺少M/N/K、stride和投影字段，原大型Trace当时留在容器。日志中的图捕获尺寸是配置，不能当作正式负载实际执行集合。优先查原容器资料，缺失字段再补采集，不重复所有实验。

TP=1结构清单如下；实际执行从本地权重/模块核对，其他TP不能机械地把所有维度除以TP：

| 组 | N | K | 重点 |
|---|---:|---:|---|
| Packed QKV | 2560 | 2048 | Q/K/V宽度不同，通常是一次融合投影 |
| O | 2048 | 2048 | 输入布局、TP通信 |
| Packed Gate/Up | 12288 | 2048 | packed顺序和scale行对应 |
| Down | 2048 | 6144 | 已有加载和三路策略 |

### 3.2 两种采集各有用途

1. BF16服务：同官方服务参数和4K/16K负载，得到主要Shape集合。
2. 旧W8A16组合：恢复或补采实际分支覆盖，用于检验Dense回退假设。
3. 最终新组合：重新核对新M、算法分支和回退。运行速度、显存和动态批次可能改变分布，不能把BF16频次直接当候选频次。

复用 `benchmarks/diagnostics/collect_gemm_shapes.py`、`summarize_gemm_shapes.py`。先少量step验证关联，再扩大采集。不得为采集更改官方服务参数、关闭图模式或修改官方Benchmark；诊断结果不申报正式成绩。

记录场景、阶段（prefill/decode/mixed）、模块/投影、TP、本地M/N/K、dtype、stride、布局、实际执行M、padding、算法、配置、调用数、可关联GPU时间和关联方式。

必须区分初始化、编译、warmup、图捕获和负载执行。Python包装或分支计数可能只在捕获发生，不能计作图重放次数。捕获阶段可建立图节点到Kernel/Shape映射，必须验证重放关联后再统计实际覆盖；无法关联的保留unknown。GPU累计时间不直接等同壁钟时间。

### 3.3 完成条件

- 有每组N/K/布局的精确观测M集合、频次和采样窗口说明；覆盖4K/16K、Prefill/Decode/Mixed及尾部请求。
- 关联覆盖率、unknown数及原因明确；采集不足时不声称全量分布或主要瓶颈。
- 两类回退分别统计：算子内部Dense回退、插件级反量化回退。
- 不穷举所有理论M；对全部观测M建立后续测试清单，再增加交界、边界与保留验证点。未来新M有明确回退。

## 4. P0-B：三算法接口与正确性门禁

统一Problem描述包括M/N/K、dtype、stride、对齐、设备、运行模式；同类层可共享调优，但不能省略真实权重、层/packed/TP接口检查。

每算法提供：`is_legal(problem, config)`及拒绝原因、`workspace_bytes`、`generate_configs`、显式配置调用入口。配置中只允许真正实现的参数；不能把尚无Kernel支持的layout名称列作可执行候选。保留旧接口和旧策略作为基线。

| 测试类 | 必须覆盖 |
|---|---|
| 真实Shape | 全部观测M与各投影本地N/K、实际布局 |
| 边界 | tile前后、奇数M、N/K非整除、K尾分片、有效地址/offset范围 |
| 数值 | 零通道、INT8端点、不同scale幅度、正负抵消、独立随机输入；量化器对非有限值明确拒绝或按已定义契约处理 |
| 接口 | 空输入、bias、非连续输入、不支持dtype、设备不匹配的支持/拒绝/回退，不得静默算错 |
| Split-K | K分片无漏算/重算、workspace各元素先写后读、归约误差和不同S |
| 接入 | packed行顺序、TP本地scale、原通信语义、返回shape |
| 执行 | eager、动态编译、图捕获、多次重放与输入/scale变化、workspace并发隔离 |

Dense/Split-K以FP32累加或归约后乘scale再BF16输出；Separate先输出BF16再scale，多一次舍入。分别报告相对量化高精度参考的计算误差、量化相对原BF16权重的误差、算法之间的舍入差异。不能仅用含中间BF16舍入的现有通用测试参考判断全部算法。

数值阈值及异常处理在筛选前记录并固定；历史relative L2<0.005、rtol=0.02/atol=0.125是特定测试口径，不自动成为所有数据和形状的充分标准。不得因候选快而事后放宽。新增算法复用既有量化格式，不使用math_500校准。

完成条件：候选能独立强制调用；正确性不通过者不进入性能赢家；不支持情况可解释且回退有验证。

## 5. P0-C：共享候选与O1/O2联合搜索

### 5.1 搜索空间和剪枝

候选来源保存为manifest：原Dense全部8组、历史14组及Split-K候选、当前生产配置、按设备/Shape生成的扩展项；去重且保留来源。旧生产配置必须参与对照，不强制它胜出。

- Dense：BM/BN/BK、warps、stages、支持的pipeline/layout、必要的tile ordering。
- Split-K：上述参数、SPLIT_K及reduce配置。S=1的两阶段控制仍含归约，不能冒充单Kernel Dense。
- Separate：转换和scale参数；内部BF16提供者固定版本，记录入口和配置，不重复队友的底层搜索。

不盲目枚举全参数笛卡尔积。用输出块数、尾块浪费、K分片长度、workspace、资源预算筛选。合法性/资源不满足可以硬剪枝；“小M适合Split-K”等性能估算只用于排序与预算分配，每条合法算法保留探索候选。shared/register估算用编译资源校验；已知编译器崩溃组合采用隔离进程，记录失败与版本，不重复无依据重试。

#### 2026-10-31补充：官方MetaX Dense与M=1候选研究

状态：已完成源码分析，纳入后续实施计划；以下变体尚未实现、未做新GPU验证，不表示已有性能收益。参考为官方FlagGems已拉取提交 `ed8a6e0d8bc36604e5a9843fda33a13bad44debc` 的 [MetaX weight_int8pack_mm](https://github.com/flagos-ai/FlagGems/blob/ed8a6e0d8bc36604e5a9843fda33a13bad44debc/src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py)，属于官方分支源码，不以未合并PR或issue作为实现证据。只借鉴并独立实现、验证必要设计，不直接合并最新上游获取收益。

**数值与已有实现的关系。** 官方Dense和M=1专用kernel均将A、INT8权重转成FP16进行dot，FP32累加后乘per-output-channel scale，再转输出dtype；BF16输入也不例外。本计划主线仍为BF16激活、INT8转BF16、FP32累加/归约、FP32 scale、BF16输出。原官方FP16计算仅可作为单独标记的诊断对照，不进入主线赢家比较；检查BF16转FP16的范围损失，不能凭输出BF16认为契约相同。融合转换、scale-after-dot和GROUP_M=8在已有通用Dense中已存在，不计为新的优化成果。旧scale-before-dot融合和Separate的中间舍入继续分别验证。

**1. 纳入官方tile来源，仍联合选择算法和配置。** 将以下配置去重加入共享Dense候选池，保留原8组、历史候选和当前固定配置；来源和原始选择条件写入manifest。下表是官方BF16输入分支的手工固定策略，不是已验证最优策略，也不是本计划的新合法性限制。

| 官方原条件 | BM/BN/BK | warps/stages |
|---|---|---|
| M≤16 | 16/64/256 | 4/3 |
| 16<M≤32，K≥N | 32/32/128 | 4/3 |
| 16<M≤32，K<N | 32/64/128 | 4/3 |
| 32<M≤64，K≥N | 64/64/128 | 4/3 |
| 32<M≤64，K<N | 64/128/64 | 8/3 |
| M>64 | 32/128/64 | 2/3 |

官方FP16输入分支还使用 `64/128/64, warps=8, stages=3` 处理M>64；可将此配置作为BF16候选探索，但必须标记为迁移候选并重新验证。相同stages不保证相同编译流水线；按每个tile计算并行块数、K循环、尾块效率和资源，O1/O2实测后再选择，不照搬M或K/N阈值。

**2. 整除专门化。** 官方Dense用 `M%BM==0 && N%BN==0 && K%BK==0` 选择无mask路径，否则M/N/K均带mask。计划先验证BF16计算下的整除/尾块两路径，再研究独立EVEN_K及M/N边界专门化，避免仅因M尾块就给全部K循环附加不必要的K mask。保留通用合法回退，限制编译变体数量，保存编译资源及关键IR；无mask条件必须是安全的能力条件，不能因预期更快而省略边界检查。

**3. M=1广播候选。** 官方仅在 `M==1 && N>=16384` 时启用：读取A的一行，广播为 `[16,BK]`，与INT8转FP16的权重dot，生成 `[16,BN]` 累加器，只写第0行；固定BN=64、BK=256、warps=4、stages=3，无Split-K。计划研究保持BF16 dot的对应Dense变体，归入Dense内部候选，不新增第四条顶层算法。源码有重复行计算，不能假定编译器自动消除；检查生成指令、寄存器及访存后，与普通BM=16 Dense、Split-K和Separate测完整调用。

按现有文档TP=1的N，Down/O为2048、QKV为2560、Gate/Up为12288，均不满足官方N≥16384；实际packed/TP尺寸以审计为准。BN=64时分别为32、32、40、192个输出program，只是并行度特征，不是性能结论。优先用Gate/Up探索放宽N条件，保留Down的Split-K对照；阈值由实测决定，不能为了命中宽N路径量化LM Head。

**4. 增补验证矩阵与计时约束。** 在第4节通用门禁上增加以下检查，按代表点组织而非全笛卡尔积：

- M=1及16/32/64两侧，N=16383/16384/16385，K=BK−1/BK/BK+1及真实K附近尾块；所有真实投影N/K、独立留出点均纳入适用范围验证。
- FP32 scale、INT8端点、零通道、正负抵消、激活/scale范围、非连续A/Q/scale、设备/dtype不匹配及空输入；明确支持、拒绝或回退，不能照抄官方入口未检查的假设。
- 单独覆盖M=1专用路径、mask和无mask、普通Dense、Split-K、Separate；核对实际分支、动态图与多次图重放、输入变化和workspace生命周期。
- 官方入口对A/Q/scale调用contiguous；若候选实际发生复制，必须计入完整调用。scale在FP32累加/归约后应用，不能把Separate每次转换移出计时或默认持久BF16副本。
- 官方该算子测试文件仅列M=16/64及规则小N/K，scale为FP16/BF16；这些测试不构成M=1、FP32 scale和尾块覆盖证据。官方benchmark的Torch转换＋MM＋scale对照也不等于原始BF16权重基线，并含中间输出舍入。保持第6节同版本BF16和完整调用对照，正确性门槛预先固定。

实施顺序：先在P0-B补齐契约和显式配置入口，P0-C吸收tile候选；整除专门化和M=1变体按第8节逐项做独立消融，复测算法交界。新增候选或变体通过算子门禁后，才参与Runtime策略选择及模型接入。本节是计划补充，具体代码实施仍按根AGENTS的计划讨论与授权范围执行。

#### 公式驱动的候选筛选（本次讨论明确采用）

默认流程是“计算特征缩小候选 → O1代表点粗测 → O2交界局部精测”，不逐一穷举所有M/N/K和所有配置。真实Shape统计用于核对N/K/布局、测试范围和优先级；完整动态M频次暂未取得时，可以先对已确认的N/K进行代表点探索，但不能把这些代表点称作实际分布。

对每个候选tile计算，不能把下面的量当成只由Shape决定的常量：

```text
P_dense = ceil(M / BM) * ceil(N / BN)
L_K = ceil(K / BK)
eta_MN = (M * N) / (ceil(M / BM) * BM * ceil(N / BN) * BN)
eta_K = K / (ceil(K / BK) * BK)
```

P_dense反映输出program数量，L_K反映每个program的K循环长度，eta反映tile的有效覆盖比例。这些是排序特征，不等于实测占用率、带宽或延迟。

对支持通用尾分片的Split-K候选，估算 `P_split ≈ S * P_dense`、`L_split_max ≈ ceil(L_K / S)`、workspace约 `4*S*M*N` 字节。按实际Kernel实现校验分片，排除无效空分片和资源超限；现有整除Kernel仍须满足 `K % (BK*S) == 0`，直到尾部支持通过验证。

可以用设备SM数量和保守目标并行度Q生成少量S：围绕 `ceil(Q/P_dense)` 的相邻2次幂取候选，再施加每片最少K迭代、workspace和编译限制。Q与最小迭代数是待校验启发式参数，不是所有GPU通用常数，也不据此硬排除其他算法。P_dense偏少且L_K较长时优先探索Split-K；并行度充足时优先Dense/Separate，但Separate与Dense的胜负还受混合dtype编译流水线、转换及scale开销影响，不能由这两个公式单独判定。

BM明显大于M时通常优先选择更小tile，例如M=64、BM=128的M维有效覆盖只有50%；在有正确mask的Kernel中它不必然非法，M=1也常需使用最小可用BM>1。此类配置可因浪费降低优先级或从主要预算中软剪枝，但须记录原因；默认基线赢家和少量审计候选保留例外，防止把经验剪枝当成正确性限制。

### 5.2 O1粗筛

- 按已确认的N/K、布局、设备分组，以公式筛选后的少量候选测试代表M；真实分布可用时补入高频M、最小/最大M和旧分派空档。
- 代表点可包含1/16/32/64/128/256/512/1024/2048；空档可包含2/8/24/48/80/96/112/127。它们是补充测试点，不冒充实际分布，也不限制实际更大的M。
- 各算法少量代表配置短测；保留Top 3～5作为起点，测量接近或参数结构不同的候选可增加保留。
- 每个候选先正确性再计时；保存失败项而非只留赢家。

### 5.3 O2细筛与候选集评价

- 精细搜索集中在O1发现的算法交界附近，例如按实际结果增加80/96/112/144/160/192等点；不把全部观测M都做一轮完整精调。对排名异常、噪声明显或已选策略明显退化的点允许追加局部搜索，不假设曲线单调或只有一个交点。
- O2之后对其余观测M进行覆盖验证：测最终分派、原策略/BF16对照及必要的少量保留候选，核对合法性和回退；发现退化再触发局部搜索。明确区分“搜索覆盖”和“最终策略验证覆盖”。
- 调整领先配置邻域并重新跨算法比较；配置变化后重新检查交界。
- 预先留出邻近/边界Shape和独立输入作验证集。若据其结果再次调参，改列训练/调优点并另设验证集。
- 在少量代表Shape上做更宽候选审计，必要时对明确限定的小集合完整枚举，评价精简候选的延迟差距、编译数、GPU测量数和搜索总时间；不要求全Shape穷举，也不将有限审计称为全局最优证明。
- 共享候选不足时按Shape类别补充，不按固定层号不断加特例。最终可只选Split-K与Separate，Dense保留合法回退。

### 5.4 BF16提供者协作

原生BF16大M候选不足是队友提供的问题线索，不在本任务中替代其实测归因。现阶段大M进行少量三算法对照和Separate完整基线，密集搜索可待提供者稳定。版本更新后，同时更新纯BF16对照与Separate，重测受影响Shape和交界；不能用新版Separate对比旧版BF16申报收益。

## 6. 完整调用Benchmark与调优审计

### 6.1 固定对照与计时协议

每组Shape比较：同版本原始BF16、原始Dense、当前固定分派、三算法复验赢家、最终Runtime入口。

计时包含Dense完整调用、Split-K workspace处理与partial/reduce、Separate转换/内部MM/scale；同时记录临时显存峰值、Kernel数和实际配置。量化加载/预打包计时单列；每次推理的转换不能移出计时。预转换BF16(Q)只作诊断。

- 编译、第一次调优、预热、稳态GPU执行、CPU调用和图重放分别记录，不互相替代。
- 粗筛预算在实验前保存；最终复验至少三轮交错顺序，使用独立输入，保存各轮及离散程度，不取最快轮。近似并列时增加诊断测量预算或保留原方案，不凭噪声宣称收益。
- 单独分量只用于定位，不能求和替代完整调用，也不能把全部差值归给某一分量。
- 除固定权重热缓存测试外，按真实多层工作集补验证；预分配workspace只有实际部署实现其生命周期、并发与图兼容后才能用作生产计时口径。
- 资源清理、串行GPU实验、环境初始化和前后归档遵守根AGENTS。默认保留持久编译缓存；不通过只清理一方缓存制造差异。

### 6.2 防止“名义autotune、实际固定/缓存复用”

复用现有libtuner能力，但统一调优以多Kernel完整调用为评价对象；不能只调partial内部再声称完整Split-K最佳。

本机LibTuner的配置键可归一化，而BenchmarkCache保存原始key和测量协议。强制运行policy或exhaustive模式也可能复用候选计时，不等于重新测量。独立实验使用可追溯的计时记录/命名空间，保留旧数据；必须分别统计新测和缓存复用，并记录命中来源，不默认删除全局缓存。

每条结果必须含：

```text
experiment_id、source/device/compiler/provider fingerprint
problem、algorithm、config、workspace、measurement protocol
generated/pruned/compile_failed/correctness_failed/measured/cache_hit counts
raw_shape_key、normalized_config_key、cache source
selection_source = offline_tuned / runtime_tuned / manual_fixed /
                   singleton / cached_selection / fallback
compile_time、warmup、latency_samples、error、selected_reason
```

验证显式配置确实传入launch，Runtime真正选择了相应算法；关键候选保存编译资源/IR，不能凭相同stages或pipeline字符串断言生成相同流水线。报告给出每Shape的候选数量、拒绝原因、实测/缓存数、赢家及原始证据链接。

完成条件：每个生产选择均可追溯；“离线调优后固定”明确标注；未测、单候选或手工选择不得标为充分调优。

## 7. P0-D：Runtime Dispatch、回退与接入

先生成精确Shape表，再按数据决定是否压缩为区间。配置身份包括设备型号/架构、编译器及后端版本、Kernel版本、BF16提供者版本、M/N/K、dtype、stride/对齐、运行模式及必要资源条件。

```text
输入能力检查 → 验证manifest版本 → 查找Algorithm + Config
 → 检查能力和workspace预算 → 直接执行
未匹配/失效 → 合法保底路径 → 在诊断中记录原因与待补测项
```

区间规则只依赖通用输入特征；需通过区间内部留出点、两端及邻域验证。无单调优势时保留查表，不强求统一M阈值。捕获/重放和正式请求期间不发起大规模GPU搜索；manifest加载与必要编译准备在捕获前完成，回退也必须图兼容。

接口和回退测试包含：冷启动与热命中、缺记录、损坏/不兼容记录的安全处理、版本变化、workspace受限、未见M、布局变化、图重放。实现可以合法拒绝未承诺支持的输入，不能静默算错或吞掉编译异常。

模型量化组仍为Down、Packed Gate/Up、Packed QKV、O，先micro再统一接入选定组合。不是要求四组全开，也不是只有Down胜出才允许分析其他组。支持相同Shape共享调优，仍逐组验证量化、packed/TP和实际覆盖。无偏置支持时明确回退；原始BF16层选择在加载阶段完成，切换组合重启服务。

## 8. P1内部优化与P2迁移

先建立可靠的三算法基线和搜索，再逐项验证Split-K/归约、Grouped Tile Ordering、layout、pipeline、tile、转换和scale优化。每项有独立前后对照，若影响性能重新调优相关Shape与交界。

Scale epilogue需要与BF16提供者协作，确认能在FP32 accumulator写BF16前乘scale；不能声称修改外层包装即可完成。不默认保留常驻BF16权重或增加新算法。

天数复用接口、候选生成框架、数值测试、搜索和manifest格式；必须重新验证Triton支持、设备资源、layout和编译能力，重新调优，不复用C500性能分界和配置作为已验证结果。

## 9. 最终模型测评与完整性门禁

### 9.1 接入审计

新组合下复核实际dtype、量化组/层数、全部观测Shape的选择或回退原因、新M、workspace峰值、动态编译/图重放、健康与输出。初始化/capture/warmup/负载分开统计。可归属范围内的算法调用数应有可核对分母；unknown明确列出。不因服务启动或42/42层启用就宣称优化覆盖完整。

若要解释旧负结果是否来自Dense，需要旧策略与新策略同版本对照、真实分支覆盖及对应完整算子耗时。不能仅靠synthetic表或分量相加给出端到端因果比例；正式成绩仍以完整测评为准。

### 9.2 正式验收

- 原样使用根AGENTS所列对应平台服务、精度和性能命令；不修改启动参数、数据或官方Benchmark。
- 基线与候选同环境、相同BF16提供者和官方脚本指纹。分别新服务启动，切换前归档、清理本任务进程并核对GPU/显存/端口。GPU测试、精度、性能串行。
- 精度完整math_500 Level 3 ≥0.950，检查请求成功数、异常及核心能力；历史103/105不替代新组合验证。
- 性能一次完整官方命令包含4K `[4096,1024,64,256]`、16K `[16384,1024,64,128]`。保留 `RUNS=4、SKIP_FIRST=1`，每场景第2/3/4轮平均；保留全部四轮raw、TTFT、成功请求和异常、原始summary。不得加外层择优循环或在两场景间重启/清缓存。
- 性能口径（2026-10-30用户更正）：每场景吞吐提升率 `candidate/baseline−1`、TTFT降低率 `1−candidate/baseline` 分别计算。±1%（含边界）为正常波动，无有效提升、不计提升分；沿优化方向>1%才认定有效改善，反向超过1%标为退化。吞吐有效提升需>baseline×1.01，TTFT有效改善需<baseline×0.99。
- 原“吞吐≥baseline×0.99、TTFT≤baseline×1.01”不能作为优化通过条件，只能表示未越反向波动界限。使用未提前四舍五入的数据比较，逐场景保留结果分类，不能只用平均值掩盖退化。total tokens/s计分，TTFT改善不自动单独得分，也不新增两指标必须同时提升>1%的规则；具体分值按明确赛事口径。
- 正式运行不带影响性能的重型诊断；诊断时间与正式成绩明确分开。失败再定向消融，不预先逐量化组跑完整模型测评。

“全覆盖”仅指已声明的观测Shape集合、边界测试矩阵、留出验证集和范围外回退行为全部有结果，不能保证未来所有动态Shape或其他GPU都已覆盖。部分采样、缺图重放归属、未测大M或未完成模型硬门槛时，报告明确标为未完成。

## 10. 文件级实施范围与阶段产物

以下为拟议文件，不表示已创建或获得所有代码阶段的实施授权；不复制整套框架。

| 文件/目录 | 计划内容 |
|---|---|
| FL `benchmarks/diagnostics/collect_gemm_shapes.py`、`summarize_gemm_shapes.py`及解析测试 | 补stride、投影/TP、图重放与分支关联；先验证可行性 |
| FlagGems `src/flag_gems/ops/weight_int8pack_mm.py` | 保留原Dense基线/回退，提供明确配置的候选调用 |
| FlagGems `src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py` | 拆能力与策略、三算法入口、分阶段尾部支持；旧固定策略可复现 |
| 拟新增 FlagGems `src/flag_gems/utils/w8a16_tuning/` | Problem/Config、后端能力、候选、完整调用搜索、manifest/dispatch；复用libtuner缓存/计时能力 |
| 新增 FL `benchmarks/diagnostics/`统一工具 | Shape清单、强制候选、O1/O2、完整/阶段/图计时、审计和汇总 |
| FL `vllm_fl/quantization/w8a16/int8pack.py`、`linear.py` | 保留opaque与TP语义，后续接入manifest及受支持量化组 |
| FlagGems算子测试、FL量化/诊断测试 | 边界、数值、配置实传、缓存失效、回退、动态/图执行 |
| FL `docs/optimization/changes/` | 每项独立代码修改同步七部分中文报告；待测不得写成通过 |

| 阶段 | 必交付 | 进入下一阶段的条件 |
|---|---|---|
| A | 历史基线指纹、Shape/布局清单、采样与分支覆盖/unknown | 观测字段可信、缺口明确 |
| B | 三算法独立调用、能力/拒绝矩阵、正确性原始结果 | 候选合法且数值通过 |
| C | O1/O2全记录、搜索成本、宽搜索与留出复验、候选来源 | 精简候选有对照、每赢家可追溯 |
| D | 版本绑定manifest、Runtime及回退/图回归 | 实际入口使用目标算法与配置 |
| P1 | 单项优化对照、受影响Shape重新调优 | 完整调用和数值有证据 |
| 模型验收 | 全局精度、官方双场景raw/summary/TTFT、覆盖与异常 | 精度/合规通过，双场景数据完整，按±1%口径分类并确认有效收益 |
| P2 | 新GPU能力、独立调优和验收 | 不沿用C500结论充当验证 |

首项代码任务建议限定为A及B的诊断入口/测试，具体文件和范围确认后实施；不直接大规模重写Kernel。当前归档不修改源码、依赖、运行开关或历史结果，不提交/推送。

## 11. 证据索引

- [W8A16路线ADR](../decisions/0002-w8a16-mainline.md)
- [全局后续计划与容量/回退分析](2026-10-29-global-w8a16-followup.md)
- [历史Shape采集限制](../changes/2026-10-19-gemm-observer-gptq-analysis.md)
- [14组有限配置实验](../changes/2026-10-27-flaggems-w8a16-baseline-configs.md)
- [Split-K实验](../changes/2026-10-27-flaggems-w8a16-splitk.md)
- [阶段计时审计](../changes/2026-10-27-w8a16-stage-audit.md)
- [dtype及编译流水线证据](../changes/2026-10-28-w8a16-nt-dtype-recovery.md)
- [大M Separate接入与数值差异](../changes/2026-10-29-w8a16-large-scale-after-diagnosis.md)
- [官方负结果](../changes/2026-10-29-w8a16-official-comparison-negative.md)
- [阶段总结](../research/2026-10-29-w8a16-stage-summary.md)
