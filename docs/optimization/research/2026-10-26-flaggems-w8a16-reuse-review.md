# BF16 调用路径与 FlagGems W8A16 复用评审

> 2026-10-27 后续进展：[小 M 两段 Split-K 诊断](../changes/2026-10-27-flaggems-w8a16-splitk.md)已完成。基于 MetaX 现有分片/归约结构适配 W8A16，M=1/16 的 S=8 完整路径较独立反量化约快 2.04/1.93 倍，分别较 BF16 慢 3.14%/5.39%；M=64 已测较快项为 S=4。保留为后续研究候选，不新增生产开关，不宣称模型收益。

> 2026-10-27 执行进展：已在新 C500 实例完成[现成实现与有限 MetaX 配置实验](../changes/2026-10-27-flaggems-w8a16-baseline-configs.md)。FP32 scale 和测试形状通过；现成融合已确认 A/B 双缓冲，不再将旧手写的 B 侧缓冲问题推广到该实现。小 M 配置候选相对原融合约快 2 倍，但九个复验点均未超过同场 MetaX BF16；未修改生产后端或新增开关，未跑模型评测。下文“无卡/待测”描述的是 10/26 制定计划时的状态，当前执行证据以新报告为准。

日期：2026-10-26。状态：结合用户确认的项目事实、本地源码与既有证据核查；当前无卡资源，不连接远端、不运行 GPU 测试、不修改生产代码。

> 路线确认：用户已明确确认，当前 MiniCPM5-2B 主要投影层的官方 BF16 基线 GEMM 使用 FlagGems 的 MetaX Triton 内核，不是直接采用 vLLM 自带 GEMM 或 mcBLAS/mcoplib 厂商 GEMM。此前将此后端归属列为未确定的表述撤回。具体 Shape 命中的 Kernel、配置和编译流水线尚未完整归档，这是细节证据缺口，不再表述为基线路线不明。

## 1. BF16 GEMM 是通用还是 MetaX 调优实现

必须区分“框架入口”“可用的后端实现”和“某次测量实际命中的 Kernel”。

| 对象 | 已确认事实 | 不能据此推断 |
|---|---|---|
| FlagGems `mm` API | 接口通用；本地有 `_metax/ops/mm.py` 专用实现，含 NN/NT、Split-K、两段归约、GEMV 及配置选择 | 所有 `F.linear` 都自动经过它 |
| 10/26 独立 Microbenchmark 的 BF16 列 | 诊断脚本调用 `F.linear(x,w)`，未显式调用 `flag_gems.mm` 或进入 `use_gems()` | 该列就是 FlagGems MetaX mm 性能，或已经确认具体厂商 GEMM 符号 |
| 官方 BF16 模型基线 | 用户确认主要投影 GEMM 使用 FlagGems MetaX Triton；`F.linear` 是上层入口，Inductor/CUDAGraph 是编译与执行机制 | 将所有 Shape 指定为同一个 `mm_kernel_nt`，或认为量化接口已经继承普通 mm 的全部后端调优 |

因此：**官方模型 BF16 基线确定以 FlagGems MetaX Triton GEMM 为对照，其普通 mm 已有 MetaX 专项适配。** 不应将 `F.linear`、Inductor 与 FlagGems 当成互斥路线：前两者描述上层入口及编译执行机制。独立 Microbenchmark 的 BF16 列没有显式启用 FlagGems，仍须与官方模型基线区分，不能据此改变其历史数据标签。

本地证据：`FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py`；归档入口证据 `benchmark_results/runtime-chain-20261021-r1/source-evidence.txt`；FL 注册逻辑 `vllm_fl/worker/worker.py`。MetaX 普通 mm 的 `N==1` GEMV 分派按算子矩阵视角判断，不能把它直接解读为 Linear 的 token M==1 专用路径。

### 官方 BF16 测评服务路径专项核对

以正式命令完整运行的归档为准，不用独立 Microbenchmark 推断服务端 Kernel：

- 10/23：`artifacts/bf16-baseline-20261023-01/serve-command.json` 记录原官方沐曦命令，`VLLM_PLUGINS=fl`、`PYTHONPATH=/workspace/vllm-plugin-FL`、Down=0、OFFLINE=0。
- 模型链路源码为 `LlamaMLP.down_proj → RowParallelLinear → UnquantizedLinearMethod.apply → dispatch_unquantized_gemm`；默认实现调用 `F.linear`。FL 平台选择 `WorkerFL`，其代码按 USE_FLAGGEMS 和算子名单注册 FlagGems。这些是源码入口和条件，不等同于最终设备 Kernel trace。
- 10/23 服务日志第 74 行明确为 BF16、quantization=None、linear_backend=auto；使用 Inductor 编译、默认 FULL_AND_PIECEWISE。第 93～96 行记录从 `/root/.cache/vllm/torch_compile_cache/82fac44c39/rank_0_0/backbone` 加载 `(1,2048)` 范围的编译图。
- 10/26 完整 BF16 对照的 `artifacts/fused-vs-bf16-20261026-01/bf16/serve.log` 第 94～97 行记录相同标识的缓存目录和图编译过程，配置同样为 Inductor。缓存标识相同不代表可以省略两次实际 Kernel 核对。
- 日志明确的 `vendor.metax` 选择属于 `attention_backend`，不能用它证明 GEMM 使用厂商库。注册冲突警告也不能作为普通 mm 实际命中的证据。
- 这两份 BF16 完整运行归档没有保存对应 computation graph、Inductor 最终生成代码、GEMM 分派注册快照或该运行的设备 trace。但用户已明确确认主要投影层使用 FlagGems MetaX Triton；缺少这些归档不再作为否定该项目事实的理由。后续补证针对每个 Shape 的具体 Kernel、选中配置和流水线，不扩展为“所有模型算子都使用 FlagGems”或“所有 Shape 都使用同一 Kernel”。

后续细节核对必须面向原官方服务：在不改官方参数的前提下保存有效环境/注册状态、实际图缓存生成代码和必要的代表性 trace，按 Down、Shape、Prefill/Decode 对应起来。此前独立测试的 BF16 耗时和 FULL_DECODE_ONLY 历史 Profiling 都不能替代相同官方参数下的具体配置证据。当前无卡资源，保留细节归档缺口，不连接远端或补造 trace。

版本：本机 FlagGems 提交 `a7620cc191a0b42e040194622c5758b22a7a25dc`，10/21 远端仓库记录相同提交，当时安装版与仓库 MetaX mm.py 指纹一致；10/26 包清单记录 FlagGems 5.3.5。尚未归档安装版 `weight_int8pack_mm.py` 指纹，因此版本记录不能替代该文件的运行环境一致性检查。

## 2. 旧手写实现归档及已观察问题

旧实现保留原路径，没有删除、停用或改写开关。当前本地快照归档至工作区 `artifacts/handwritten-w8a16-archive-20261026/source.tar.gz`，配套 `manifest.json` 记录源码 SHA256 与仓库基点。快照包含旧融合、加载接入、reference、独立 dequant、测试及诊断脚本。

归档仅是当前本地源码快照；每次历史实验仍以自身源码指纹为准。前一轮 small-M 候选实验的远端最终结果尚未取回，不能记为完整完成或已验证生产收益；旧实验记录不覆盖。

已观察问题：

1. 首版直接 Triton launch 将动态 token M 特化成 2048，vLLM 启动失败；后通过 opaque custom op 与 symbolic fake 修复，19 项测试及模型启动通过。这属于模型集成问题，不等同于 GEMM 性能问题。
2. 同分块对照中，基础 BF16 GEMM 本身已经慢于测量用的 `F.linear`；不能把全部性能差距归因于量化。
3. M=2048 的旧融合 TTGIR 显示 A 侧双缓冲；B 侧在 K 循环中执行 INT8 load、FP32 转换、scale 乘法、BF16 舍入，再暂存到共享内存并参与 dot。B 侧没有与同分块 BF16 对照相同的跨迭代双缓冲结构。
4. 后端资源报告 spill 为 0，融合寄存器数和 shared 占用还更少；没有证据把差距解释成已证实的寄存器溢出。实际占用率和 stall 占比未测量。
5. 九种 M 的补测中，旧融合均慢于独立 Triton 反量化＋GEMM，二者均慢于原 BF16。先前“小 M 融合更快”只针对 PyTorch reference 反量化，不适用于独立 Triton 对照。

同分块诊断（ms，三轮中位数）：

| M | F.linear BF16 | 同分块 BF16 | 独立 Triton＋GEMM | 旧融合 |
|---|---:|---:|---:|---:|
| 1 | 0.032768 | 0.077056 | 0.079616 | 0.141568 |
| 64 | 0.052736 | 0.086528 | 0.099072 | 0.157952 |
| 256 | 0.075520 | 0.150528 | 0.119808 | 0.262144 |
| 2048 | 0.291328 | 0.832512 | 0.334848 | 2.018560 |

证据见 `changes/2026-10-26-w8a16-same-tile-diagnosis.md` 及对应工作区 artifacts。预分配输出、直接 launch 后差距仍在，调用封装不是主要性能差距来源。以上没有 FlagGems weight_int8pack_mm 的性能数据。

## 3. 与现成 weight_int8pack_mm 的差异

| 设计项 | 旧手写融合 | FlagGems 现成接口 | 后续处理 |
|---|---|---|---|
| 数值契约 | sum(x×BF16(q×scale)) | FP32 accumulate(x×BF16(q))，最后乘 scale 再输出 | 用户已允许评估 scale-after-dot；重新验证数值与模型精度 |
| 循环内转换 | INT8→FP32→乘 scale→BF16 | INT8→输出计算 dtype；scale 在 epilogue | 直接复用，不重写融合骨架 |
| 分块 | 固定两组；BK=64、BN=64 | 8 组候选：BM=32/64/128，BN=32/64/128，BK=32/64 | 先记录原 tuner 选中项，再判断是否补 MetaX 候选 |
| autotune | 生产实现无配置搜索 | libtuner，M/N/K key、align32 strategy、warmup/rep | 复用框架；不假设能自动发现候选集合之外的实现 |
| M 分组 | GROUP=8，已有 | GROUP_M=8，已有 | 不重做；缓存命中效果及组大小是否合适待测 |
| 访存表达 | 固定连续布局，M/N/K mask | stride 参数、M/N 取模加载、对齐/连续性提示，K mask | 复用；提示不是性能保证 |
| dot 与输出 | BF16 dot，FP32 累加，支持可选 bias | BF16 激活时 BF16 dot、FP32 累加、融合 channel scale；接口无 bias 参数 | Down 无 bias 可先测；其他层扩展另议 |
| 缓冲 | 已归档特定 C500 编译结果，B 侧流水线存在差异 | 源码未显式写双/三缓冲，候选 stages=2/3 | 需要新 TTGIR 验证，不把 stages 数当作缓冲证据 |
| 小 M 专用路径 | 原生产融合仍按 BM=16 计算 | 本接口最小 BM=32，无单独 GEMV/Split-K | 只有实测存在缺口时才设计后端特化 |
| 动态编译 | 已有适配与回归测试 | 本地源码有 tuner/包装，模型兼容性未测 | 先试直接接入；必要时复用自定义算子封装，内部调用现有接口 |

因此旧实现并非缺少所有基础结构：分组、dot 循环、stages 配置都已有。现成实现的关键优势是不同数值组织、更完整的调优与访存表达；不是已证明它拥有更好的 C500 双缓冲或已经更快。

## 4. 在现有 Kernel 上还需要做什么

执行顺序如下；当前只提出方案，未改生产实现。

### A. 先建立现成实现对照

- 有卡后核对实际包路径、源码指纹及后端选择，确认现有版本可调用该接口，不为获取收益合并最新 FlagGems。
- 保持 BF16 激活、INT8 `[N,K]`、FP32 `[N]` scale；测试 Down N=2048/K=6144 及不整除形状。现有测试 scale 使用激活 dtype，FP32 scale 需要补覆盖。
- 正确性参考必须按 FP32 累加后乘 scale 的契约构造；不能将 PyTorch 先产出 BF16 matmul 结果再乘 scale 作为逐位相等标准。记录与旧 scale-before-dot 及原 BF16 的差异，模型精度重新验证。
- 主要性能基准为与官方模型路线一致的 FlagGems MetaX BF16 mm，比较现成 weight_int8pack_mm 和独立反量化＋相同 BF16 GEMM；原默认 F.linear 和旧手写融合仅作辅助对照。记录实际命中路径，避免把独立脚本默认 F.linear 当成官方基线。autotune/首次编译时间和稳态耗时分开。

### B. 参考 MetaX BF16 优化，检查生成结果再决定最小修改

普通 BF16 mm 已有 NN/NT 专用寻址、整除特化、对齐提示、MetaX 配置与 basic/cpasync 流水线选项、资源剪枝和形状分派；weight_int8pack_mm 是通用融合实现，本地未发现同名 MetaX 覆盖。复用后者的 dot、scale epilogue、分组及 libtuner 骨架，按证据补齐后端适配，不全部手写。num_stages 不保证 A/B 实际双/三缓冲。普通 mm 的 N==1 GEMV 与 N==256 两段 Split-K 条件不直接对应标准 Down 的 M==1/N==2048，不能机械迁移。

- 记录真实 Shape 下 tuner 选中的 BM/BN/BK、warps、stages。
- 检查 B 侧转换、跨迭代缓冲、dot layout、资源字段及可获取的硬件计数器。INT8→BF16 仍存在，scale 移出循环只缩短依赖链，不保证流水线自动理想。
- 若原实现有效但配置不适合 C500，优先按 FlagGems 后端扩展方式补配置/剪枝和测试；若配置不足以解决，再基于该 Kernel 做最小加载、布局或流水线改动。保留通用接口和框架选择逻辑，不直接覆盖全部设备实现。
- 对 M=1 及小 M，仅在现有配置缺口被确认后考虑专用路径；按实际 Shape 表达能力，不硬编码评测长度或层号。

### C. 有效后才接入生产

- Microbenchmark 必须覆盖完整路径和回退开销，收益需相对 BF16 稳定成立；不只比较旧自写 Kernel。
- 实验候选不新增生产开关。仅调用既有算子是对照，不等于实质性赛事优化；最终需要代码改动、真实启用和可复现收益。
- 模型接入保持 Down 范围及 TP 语义，验证动态 M、图编译、CUDAGraph、dtype/覆盖与回退，再执行 math_500 和官方 4K/16K 吞吐、TTFT 验收。
- 无效候选记录配置、误差、耗时及原因，不放入正式运行路径。旧路径和历史结果保持可复现，不静默删除或改写。

## 5. 专用后端优化计划补充

当前方案明确为：**基于通用 `weight_int8pack_mm`，参考普通 BF16 mm 的后端适配方式，逐步做专用优化**，不重新手写整套融合，不把普通 BF16 全部配置原样搬入量化 Kernel。

### MetaX：先补配置和剪枝，再按证据改流水线

拟实施位置（尚未修改）：

- 诊断工具：先直接调用原 `flag_gems.weight_int8pack_mm`，补真实 Down Shape、FP32 scale、独立数值契约与原配置选中项记录。
- FlagGems `_metax/ops/weight_int8pack_mm.py`：若需要后端覆盖，继承现有接口、INT8 转计算 dtype、FP32 累加、scale epilogue、分组和 tuner 骨架。按实际后端注册机制补导出，不能只新增文件后假设已生效。
- `_metax/tune_configs.yaml` 及必要的独立扩展配置：为 weight-only 建立自己的候选与剪枝，不直接改普通 mm 的既有配置制造对照变化。
- 后续测试、诊断与报告：每项改动独立验证；有效候选才进入 vllm-plugin-FL 的 Down 接入，不提前增加生产开关。

参考普通 mm 的优化顺序：NT/整除寻址表达 → BM/BN/BK、warps、stages 和后端 pipeline 选项 → 资源剪枝 → 编译产物检查 → 必要的权重转换/预取/布局修改。Split-K、GEMV 或 M 分派仅在现成实现暴露具体缺口时另行计划，不一次堆叠所有策略。

需要重新估算的是 INT8 全局读取、转换后 BF16 缓冲和中间寄存器，而不是简单把 BF16 的 shared 估算除以二。是否真的形成 A/B 双缓冲、异步预取或更高占用率，只能按实际 C500 编译及运行证据判断。

### 天数 Iluvatar：本地现状核查

以下仅描述当前本地 FlagGems 提交，不代表天数容器安装版本或官方模型运行路径已经核实；本轮无卡、未远程访问。

| 组件 | 本地实现现状 | 与 W8A16 的关系 |
|---|---|---|
| `_iluvatar/ops/mm.py` | 专用 mm/mm_out；`libtuner` 按 M/N/K 调优，early prune＋性能模型，top_k=15 | 可参考调优组织、寻址、尾块和后端配置，不直接把 BF16 mm 改成混合输入接口 |
| `_iluvatar/tune_configs.yaml` 的 mm | BM/BN=32/64/128/256，BK=32/64/128，warps=8/16，stages=1/2，SPLIT_K 候选仅 1 | 不能照搬 MetaX warps/stages/pipeline 配置；参数相同也不代表两平台物理资源或代码生成相同 |
| mm 的 pid 映射 | 当前使用行优先映射，GROUP_M 分组代码被注释 | 不能说天数普通 mm 已启用 MetaX 式分组；weight-only 自带分组效果应独立测 |
| mm Split-K | Kernel 有 atomic_add 分支，但当前配置只有 SPLIT_K=1 | 不能把有代码分支写成当前已启用优化；启用还需输出初始化、数值及完整成本验证 |
| `_iluvatar/ops/linear.py` | 已导出专用 linear，独立 `linear` tuner 配置，支持 bias | 普通 BF16 路径有 linear 和 mm 两类入口，模型实际命中及图编译 lowering 待核对 |
| 通用 `weight_int8pack_mm` | 本地 `_iluvatar` 下未找到同名覆盖/导出 | 与 MetaX 一样先作为现成 W8A16 对照；BF16 激活、INT8 权重和 scale 后乘逻辑可共用 |
| `_iluvatar/ops/matmul_int8.py` | 已导出；输入在 dot 前都转 BF16，FP32 累加，输出 BF16，无 scale 参数；包装层执行 `b.t().contiguous()` | 名称不证明原生 INT8 GEMM，也不是完整 per-channel W8A16 scaled GEMM；每次权重布局复制成本不能忽略 |
| 通用 scaled_mm/dequantize | 未发现天数同名专用覆盖 | scaled_mm 同 dtype 约束、通用 dequantize 的 per-tensor/FP32 输出限制仍在，不因换平台就匹配当前契约 |

天数 mm 内部还包含：整除 K 的无尾块 mask 路径、索引规模决定的 int32/int64 分支，以及输出写回时重新计算行列索引（源码明确用于减少寄存器）。这些是可以参考的实现方法，不代表适用于 INT8 权重后仍是最优。其性能模型是否准确预测 weight-only 转换成本，也要重新验证，不能直接当成已校准模型。

### 两平台可共用与不能照搬的部分

**可共用：** W8A16 数学定义、INT8 `[N,K]`＋FP32 `[N]` 数据格式、per-output-channel 量化算法、加载/TP 分片语义、通用 weight-only Kernel 骨架、scale-after-dot 正确性参考、测试 Shape/边界数据生成、诊断工具和实验记录格式。加载时权重量化继续复用已存在的算法；不拿 fake-quantize 返回浮点张量的接口冒充 INT8 权重生成。现成接口缺少的 bias、dtype/布局校验和动态编译适配按需求单独处理。

**必须单独验证：** 后端实际可用算子和注册路径、BF16 基线具体实现、tile/warps/stages、流水线参数、资源剪枝阈值、线程布局、Split-K 取舍、图编译/CUDAGraph 和最终性能。MetaX 的 cpasync 配置不能直接假定为天数可接受的编译选项；C500 收益和精度记录不能冒充天数实测。

**建议执行顺序：**

1. 当前只补本地方案，不改算子；资源恢复后分别核对安装版本与源码指纹。
2. 两平台各自先测原 `weight_int8pack_mm`、各自实际 BF16 后端和独立反量化完整路径，分开建立对照。
3. 先推进当前主要目标 C500 的后端特化；天数优先验证通用实现和数据/接入层复用，是否追加天数专用覆盖由其自身证据决定，不机械复制 MetaX 优化。
4. scale-after-dot 与旧舍入路径不同，两平台都重新验证算子数值及模型精度。动态编译兼容按实际调用验证，必要时共享 symbolic fake 封装方法，而非绑定旧手写 Kernel。
5. 算子阶段有效后才接入模型并增加必要开关；端到端按各平台官方原参数分别执行。天数保留官方 `FULL_DECODE_ONLY`，沐曦不额外添加该参数，不用统一命令抹平平台要求。

仅切换现成算子是对照，最终仍需可复现的实质性代码优化与实际启用。没有通过的新候选仅记录，不进入正式路径。

## 6. 当前交付边界

本次完成旧代码快照归档、BF16 路径证据澄清、两类融合源码对照和后续方案。没有新增 Kernel、修改生产开关、运行 GPU 测试或宣称现成接口已解决性能问题。卡资源恢复前仅可继续本地分析或经确认的诊断工具准备。

## 2026-10-28 无卡阶段更新

小M配置、partial/reduce IR及资源证据的离线复核见 [小M离线分析与补测计划](2026-10-28-w8a16-small-m-offline.md)。当前无GPU，不重复调参、不连接远端、不改生产源码。优先候选为同算法整除路径特化，转换布局属后续待证实方向；M64新BM64缺少编译产物，需资源恢复后补齐。大M已执行但远端原始归档和dtype归因未完成，状态保持原样。
