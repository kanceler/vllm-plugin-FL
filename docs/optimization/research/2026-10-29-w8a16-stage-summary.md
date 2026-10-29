# W8A16 GEMM 优化阶段同步

2026-10-29｜MiniCPM5-2B｜MetaX C500｜当前 Down-only，42/42 层。官方 4K/16K 对照已完成，吞吐与 TTFT 均未达标，详见第 8 节。

## 1. 方案演进

本阶段从 W8A8 调整为 W8A16，先验证独立反量化的精度与模型性能，再研究反量化与 GEMM 融合。旧手写融合性能不足，后续改为基于 FlagGems `weight_int8pack_mm` 增加 MetaX 后端优化。当前分别采用小 M Split-K、大 M 独立转换与 GEMM，并统一到同一个算子入口。

```text
W8A8 能力验证
  → W8A16 独立反量化，验证精度与模型接入
  → 手写融合，尝试消除临时 BF16 权重读写
  → 复用 FlagGems weight_int8pack_mm，转为后端专项优化
      ├─ 小 M：配置筛选 → Split-K → 整除特化 → 接入
      └─ 大 M：融合配置/MetaX NT 诊断 → 独立 scale-after-dot → 接入
  → 统一入口与模型精度验证 → 当前官方 Down-only 推理对照
  → 后续算子优化 → 扩展全局受支持 Linear
```

| 阶段 | 测试结论与处理 |
|---|---|
|W8A8 → W8A16|激活量化有正确性及额外读写问题，且唯一可比 M1 点 scaled_mm 165 µs、BF16 21 µs，缺少高效原生 INT8 证据，因此暂停该路线。|
|独立反量化 → 融合|旧 scale-before-dot 版本精度可用，但4K模型吞吐−5.47%、TTFT+2.96%，转向融合以减少临时权重读写。|
|旧手写融合 → 复用 int8pack|旧融合大 M 完整调用明显慢于独立路径，保留旧实现作对照，后续基于现有 FlagGems 算子修改。|
|通用 int8pack → MetaX 特化|通用 autotune 配置不足以弥补小 M 并行度和大 M 混合 dtype 流水线差距，参考已有 MetaX BF16 mm 做针对性优化。|
|小 M → Split-K＋整除特化|先选配置和 K 维分片，再消除可知边界与分片控制；整除特化单独贡献2.42%～6.50%完整耗时下降。|
|大 M → 独立 scale-after-dot|已测通用融合及 MetaX NT 融合仍慢，当前复用成熟 BF16 GEMM，加独立转换与输出 scale，完整耗时约为通用融合的一半。|

**当前方案**：静态、对称、per-output-channel INT8 权重＋FP32 scale，激活 BF16、FP32累加、输出BF16。加载时一次量化，推理不量化激活；Down持久权重数据约减半，容量收益与推理加速分开报告。

上层统一调用 `custom op → flag_gems.weight_int8pack_mm`，内部按实际shape分派：

- **小 M 已接入**：M=1/16/32/64，N%64=0、K%1024=0及连续布局等条件下，Split-K=8＋整除特化；BN64/BK128、warps4/stages2，M64用BM64，其余BM16。FP32 partial归约后乘scale。
- **大 M 已接入**：M≥128且满足能力条件，Triton INT8→BF16 → MetaX BF16 GEMM → 输出乘scale。临时BF16权重不持久缓存。
- 其他受支持shape保留通用融合；不受支持输入明确回退。custom op是共同动态图入口。

**小 M 优化尚未结束**：当前接入的是配置筛选、Split-K 与整除特化；INT8→BF16转换位置、dot operand/shared布局的专项优化尚未实现，reduce展开或树形归约也未实现。之前的partial/reduce分析属于诊断，不能写成这些优化已经落地；未验证候选不包含在当前性能结果中。

## 2. BF16 基线调用路径

官方服务的主要投影 GEMM 使用 **FlagGems 的 MetaX Triton 实现**，不是直接调用 mcBLAS/mcoplib GEMM。以无 bias 的 Down 投影为例：

```text
MiniCPM5-2B：LlamaMLP.down_proj
  → vLLM RowParallelLinear
  → UnquantizedLinearMethod.apply
  → dispatch_unquantized_gemm（默认 F.linear）
  → FL / WorkerFL 注册的 FlagGems 算子路径
  → flag_gems.mm 的 MetaX 后端实现
      FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py
  → 按 M/N/K、stride、dtype 和设备资源选择 GEMM kernel
```

`F.linear` 是上层入口，Inductor/CUDA Graph 是编译执行机制；最终 GEMM 后端为 FlagGems MetaX，三者不是互斥路径。输入为 BF16 X[M,K]，权重为 BF16 W[N,K]，计算 `X @ W.T`，输出 BF16。

MetaX `mm` 保留以下选择逻辑，并非所有 shape 都走同一个 kernel：

| 条件/路径 | 实现 |
|---|---|
|N=1|GEMV，按条件使用 K 并行分片/归约|
|满足两阶段 Split-K 条件|`mm_kernel_splitk_partial → mm_kernel_splitk_reduce`|
|满足普通 Split-K 条件|`mm_kernel_splitk`，K 维分片累加|
|连续 NN/NT 布局满足专项条件|`mm_kernel_nn` / `mm_kernel_nt`|
|其他情况|通用 `mm_kernel`|

普通 BF16 `mm` 已包含 MetaX 配置选择、资源剪枝、分组 tile、整除寻址及流水线支持。**W8A16 对比的是这条已有后端优化的 BF16 路径**；当前大 M 独立方案也复用同一 MetaX `mm`，只增加权重转换和输出 scale。

Down N=2048、K=6144，不能把 N=1 GEMV 或 N=256 两阶段 Split-K 条件直接套用到它。官方服务每个具体 shape 的 kernel/config 应以实际生成代码或 trace 为准，不将上述源码分派表写成所有 shape 都已逐项追踪。依据：本地历史资料 `2026-10-26-flaggems-w8a16-reuse-review.md`（未随本分支提交）。

## 3. 独立反量化的模型性能

2026-10-23 官方命令，Triton scale-before-dot 独立反量化，Down 42/42、回退 0。

| 场景 | BF16 total tok/s | 独立反量化 total tok/s | 吞吐变化 | BF16 TTFT ms | 独立反量化 TTFT ms | TTFT 变化 |
|---|---:|---:|---:|---:|---:|---:|
|4K|4937.19|4667.20|−5.47%|3309.85|3407.75|+2.96%|
|16K*|7009.97|6816.29|−2.76%*|27241.61|28187.46|+3.47%*|

4K 双方均四轮、后三轮平均。*16K 候选仅首轮，BF16 为完整平均，只能参考。两次 GPU bus 不同，不能把差值全部归因于量化。结论：旧方案已测、精度可用，但未证明性能达标，故继续研究融合。

证据：本地历史资料 `../changes/2026-10-23-w8a16-down-benchmark.md`（未随本分支提交），工作区 `benchmark_results/bf16-baseline-comparison-20261023.md`、`artifacts/bf16-baseline-20261023-01/`。

## 4. 算子性能

以下各表均为表内同场比较；N=2048、K=6144，单位 **µs**。不同实验绝对值不拼接计算收益。完整调用包含转换、临时分配、GEMM 和必要 scale；权重量化在加载阶段，不计每步耗时。均非端到端推理成绩。

### 4.1 旧手写融合与独立反量化对比

| M | MetaX BF16 | 独立 Triton 反量化＋GEMM | 同分块 BF16 对照 | 旧手写融合 |
|---|---:|---:|---:|---:|
|1|32.768|79.616|77.056|141.568|
|64|52.736|99.072|86.528|157.952|
|256|75.520|119.808|150.528|262.144|
|2048|291.328|334.848|832.512|2018.560|

同分块 BF16 已慢于成熟 MetaX BF16，INT8 转换进一步增加成本；去掉包装和输出分配后差距仍在。不能仅靠减少读写保证融合更快。来源：本地历史资料 `../changes/2026-10-26-w8a16-same-tile-diagnosis.md`（未随本分支提交）。

### 4.2 小 M 整除特化

| M | 原 Split-K | 整除特化 Split-K | 完整耗时下降 |
|---|---:|---:|---:|
|1|42.496|40.192|5.42%|
|16|47.360|44.544|5.95%|
|32|62.976|58.880|6.50%|
|64|84.480|82.432|2.42%|

72 项边界检查通过。简化合法整除条件下的 K 分片和边界判断，M1 保留行 mask；未改变 dot 顺序、转换位置或缓冲层数。来源：本地历史资料 `../changes/2026-10-28-w8a16-small-even.md`（未随本分支提交）。

### 4.3 当前实现与基线对比

| M | MetaX BF16 | 原独立反量化（before-dot） | 通用 int8pack 融合 | 当前统一入口 |
|---|---:|---:|---:|---:|
|1|39.936|86.016|173.824|43.776|
|16|40.192|86.016|178.944|48.640|
|32|49.664|94.720|179.456|64.512|
|64|70.656|114.944|178.944|83.200|
|128|77.824|122.880|280.576|125.696|
|512|188.160|231.680|518.912|237.568|
|2048|576.000|619.520|1235.712|634.368|

- 当前入口全部已测点快于通用融合；小 M 快于原独立路径。
- 大 M 为统一 scale-after-dot，多一个输出 scale kernel及输出读写，比旧独立路径约慢 2%～3%，但明显快于通用融合。
- 当前多数点仍慢于 BF16，不宣称已经获得端到端收益。
- 大 M 转换：PyTorch cast 约 **186 µs**，改为 Triton 后约 **55 µs**；输出 scale 约 **14～42 µs**（M128～4096）。分量不可直接相加作为完整耗时归因。

三轮交错顺序，每轮 warmup100ms/rep300ms，取三轮中位数，完整退出码0。原始数据：`artifacts/w8a16-int8pack-production-20261028-09/w8a16-unified-microbench-20261029-01/{metadata,raw,summary}.json`。来源：[大 M 补测与接入](../changes/2026-10-29-w8a16-large-scale-after-diagnosis.md)。

## 5. 精度与回归

| 验证项 | 结果 | 适用版本 |
|---|---|---|
|旧独立反量化 Level 3|103/105，98.1%|scale-before-dot 历史版本|
|2026-10-29 完整 Level 3|**103/105，98.1% ≥ 95%**；105 请求成功、0 错误，退出码0|小 M Split-K＋大 M 通用融合，scale-after-dot|
|最终组合回归|**22 passed in 34.40s**，退出码0|小 M Split-K＋整除特化＋大 M 独立 scale-after-dot|
|最终生产入口数值|小 M relative L2：0～7.36e-5；大 M：约0.00281～0.00283；检查通过|FP32 dot 后 scale、最终 BF16 为参考|

完整精度使用官方命令、math_500 Level 3、batch8；服务按官方参数启动，42/42 Down 启用，图编译/捕获通过，检查期间无回退。结果目录 `/workspace/evalscope-datasets/level3/20261029_120823`；本地精度归档 `artifacts/w8a16-int8pack-production-20261028-09/w8a16-int8pack-newhost-20261029-01/`。

**数值区别**：小 M 在 FP32 累加/归约后乘 scale；大 M 先由 GEMM 输出 BF16，再以 FP32 乘 scale，多一个舍入点。按用户决定，最终组合不重复完整模型精度测评，只补算子验证；不能把103/105写成该组合重新实测的结果。

22项回归覆盖动态图、CUDA Graph 重放、量化边界、分派与接入；此前21项通过是接入大M前版本。最终日志位于 `artifacts/w8a16-int8pack-production-20261028-09/w8a16-unified-20261029-02/pytest.log`。

## 6. 待验证的算子优化

| 优先级 | 候选方案 | 要解决的问题 | 验证方式 |
|---|---|---|---|
|1，大 M|复用 MetaX GEMM，增加 FP32 accumulator 的 channel-scale epilogue|省掉当前独立输出 scale kernel、输出一次读写与 BF16 中间舍入|对比当前完整大 M 路径、通用融合和 BF16；核对数值及图兼容|
|2，大 M|INT8→BF16 转换的访存、向量化和布局优化|当前转换约55 µs，仍是额外成本|先单项消融，再比较完整调用；不以分量改善代替整体收益|
|3，小 M|INT8转换位置、dot operand/shared布局优化|进一步改善partial的数据组织；此前整除特化未改这部分|固定已有Split-K/tile配置，对照编译产物及完整调用|
|后续|reduce展开或树形归约|可能减少归约控制成本，但会改变加法顺序和资源使用|有证据再推进，数值与完整收益同时检查|

以上均为**待验证候选**，不计入已实现收益。旧手写融合、有限配置无收益项只保留记录；不能用stages/pipeline参数相同推定实际流水线相同，也不再用partial/reduce单独计时相加归因。

## 7. 扩大量化范围

**当前42/42仅指Down覆盖，不是全模型量化完成。最终方向是扩大到全局受支持Linear，并按投影/shape保留混合路径。**

| 阶段 | 工作 |
|---|---|
|当前|完成W8A16与BF16官方4K/16K同环境对照，确认Down方案的吞吐和TTFT；不中途修改候选。|
|下一步算子优化|优先验证大M scale epilogue等候选，有完整调用收益才接入统一入口；保留小M现有有效优化。|
|扩展一|先接入packed Gate/Up，核对权重分片、scale对应关系、真实M/N/K和性能。|
|扩展二|逐步覆盖QKV、O等受支持Linear；不同shape采用适合的配置/策略，不直接复用Down参数。|
|全局验收|统计实际量化覆盖、fallback、权重/临时显存；重新完成模型精度及官方4K/16K吞吐与TTFT验证。Embedding、LM Head等先保留BF16。|

当前已有完整模型精度证据不能直接迁移为全局量化精度；扩展必须独立验证。不保证所有层量化都更快，无收益或敏感投影保留BF16。团队Attention/调度优化尚未整合，组合收益后续另测。

## 8. 官方推理性能

### 本次实际被测实现

```text
MiniCPM5-2B，Down-only 42层
  权重加载时：BF16 → per-channel INT8 Q＋FP32 scale
  推理时：BF16激活 → custom op → MetaX weight_int8pack_mm
    ├─ 小M满足特化条件：Split-K=8＋整除特化
    │    INT8 tile转BF16 → FP32 partial → FP32归约并乘scale → BF16
    ├─ 大M满足独立路径条件：
    │    INT8→临时BF16权重 → MetaX BF16 mm → 独立输出scale → BF16
    └─ 其他受支持shape：通用FlagGems int8pack融合
```

当前测的是上述组合实现，不是旧scale-before-dot手写融合，也不是所有M都走通用融合。小M后续布局/转换优化、大M GEMM内scale epilogue、全局Linear量化及团队Attention/调度改动均未包含。

### 执行顺序与统计口径

| 顺序 | 模式 | 开关与范围 |
|---|---|---|
|1|W8A16候选4K、16K|`DOWN=1, DOWN_INT8PACK=1, DOWN_FUSED=0`；42层Down使用上述组合路径|
|2|停止候选服务、确认资源释放|保留原始结果，不在4K/16K之间额外清理或重启|
|3|BF16基线4K、16K|新服务，三个开关显式为0；原始BF16权重走FlagGems MetaX GEMM|

两模式使用同环境、相同官方服务参数及性能命令。每场景四轮、跳过首轮后取后三轮平均；独立GPU测试和精度测评不与本任务并行。请求全部成功：4K 256/256、16K 128/128，四轮 Run Status 均 SUCCESS。

#### 吞吐（total tok/s）

| 场景 | BF16 基线 | W8A16 候选 | 吞吐变化 | 基线×0.99 门槛 |
|---|---:|---:|---:|---:|
| 4K | 4989.25 | 4730.11 | −5.19% | 4939.36（未达标） |
| 16K | 7004.75 | 6821.13 | −2.62% | 6934.70（未达标） |

#### TTFT（Mean，ms）

| 场景 | BF16 基线 | W8A16 候选 | TTFT 变化 | 基线×1.01 门槛 |
|---|---:|---:|---:|---:|
| 4K | 3180.14 | 3428.53 | +7.81% | 3211.94（未达标） |
| 16K | 27258.78 | 28161.24 | +3.31% | 27531.37（未达标） |

**结论**：W8A16 Down-only 组合在 4K/16K 两个场景的吞吐与 TTFT 均未达标（负收益），按 ADR 止损条件记录，保留 BF16 回退；后续路线需结合能力证据另行讨论，不能据此直接推定 W8A8 可获得收益。

任务目录：`/workspace/benchmark_results/w8a16-official-comparison-20261029-02`（候选 `candidate/summary_20261029_141125.csv`、基线 `bf16/summary_20261029_150308.csv`，raw_runs 四轮明细同目录保留）。

版本：插件基点`13eb9be`，FlagGems基点`a7620cc`，均含未提交实现；复现需结合实际源码/差异与实验指纹。历史路线依据保存在本地 `docs/optimization/decisions/0002-w8a16-mainline.md`，未随本分支提交。
