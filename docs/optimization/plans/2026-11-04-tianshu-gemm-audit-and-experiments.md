# 天数：队友 BF16 基线上的 GEMM 路径审计与后续实验

日期：2026-11-04。平台：Iluvatar BI-V150 / CoreX 4.5.0。状态：只读审计及实验计划，未修改源码、未启动新 GPU 测试。

## 1. 独立记录原则

本文仅适用于天数。沐曦的调用链、配置、实验目录、性能基准、计划和验收结果分别管理，不将沐曦 tile、算法分界、量化误差或收益迁移为天数结论。BF16 融合与 W8A16 分别对照，不同时开启来掩盖独立贡献。

只有两平台实际调用接口、dtype/布局、数值契约及实现依赖确实重合，且各自有正确性与性能证据，才将对应改动整理成共享实现。共享前仍分别归档计划和结果；设备 tile、资源剪枝、缓存和分派策略保持独立。数学公式或函数名相同不构成可直接复用证据。

## 2. 已核对的基线与证据强度

- 队友版本 `/workspace/tianshu-final-20261104`：`f82fa1670509f081aa13b58464cb6e5627fc10c4`，本次检查工作树干净。
- 历史实验 `/workspace/experiments/tianshu-final-20261104-01/`；官方双场景完整基线已归档，无需仅为审计重新运行。
- 4K total tokens/s 3073.11、Mean TTFT 5115.36ms；16K 2790.28、177074.49ms。来源为既有报告及归档 CSV，不是本次新测。4K 原始轮次离散较大，后续候选需可比环境复验。
- 检查时无运行中的 vLLM 服务。交互环境 COREX_VERSION=4.5.0；默认 import 来自 site-packages，不是队友工作目录。历史服务使用 PYTHONPATH 指向队友目录；后续启动必须再次核对，不能因目录有修改就认为生效。

本次远端已安装 vLLM 源码核对的上层路径：

`UnquantizedLinearMethod.apply → dispatch_unquantized_gemm → default_unquantized_gemm → torch.nn.functional.linear`。

已归档 `artifact_compile_range_1_2048_subgraph_1` 的生成代码包含：

| 投影及邻接操作 | 生成代码中的调用 | 证据边界 |
|---|---|---|
| O | extern_kernels.mm → triton_red_fused_fused_add_rms_norm_0 | 已编译分支，不等于全负载 GPU 计时 |
| Gate/Up | extern_kernels.mm → triton_poi_fused_mul_silu_slice_1 | SiLU/Mul/slice 已合成后处理 Kernel |
| Down | extern_kernels.mm → triton_red_fused_fused_add_rms_norm_2 | 需继续核对 alias、舍入与多输出 |
| QKV | extern_kernels.mm → triton_poi_fused_3 | 后处理生成注释包含 split、cos/sin 索引、旋转算式；具体 GPU 耗时待采样 |

这比 enabled-ops 白名单更直接，但 `extern_kernels.mm` 仍不能单独证明具体厂商库 GPU Kernel、tile 或硬件效率。先前基线报告将“GEMM vendor BLAS”“小算子 FlagGems”整体写为已确认，证据强度过高：当前准确结论为外部 MM 调用、编译生成的融合后处理；具体提供者与运行 Kernel 待补充。

第一步归档/静态审计结果见[实际调用路径审计报告](../changes/2026-11-04-tianshu-gemm-path-audit.md)，实际负载Kernel与M覆盖仍待短时补采。

## 3. 第一阶段：实际分支与配置适配审计

1. 核对两个 compile range 的生成代码、模块权重/alias映射、Prefill 路径与 Decode capture/replay 路径，分别记录覆盖窗口和 unknown。
2. 对四种投影记录实际 M/N/K、dtype、stride、连续性、bias、TP、输出消费者；不能用沐曦观测 M 或 capture 尺寸列表替代天数实际 M。
3. 追踪外部 MM 到实际 GPU Kernel；区分框架默认、厂商库、FlagGems 和编译器生成实现。
4. 对 tile 配置记录“定义位置→平台选择→候选来源→剪枝→实际选中配置→调用证据”。源码中存在配置不等于生效。
5. 检查 dtype 意外转换、额外 contiguous/copy、错误后端、错误缓存键、未生效补丁及回退。不修改官方 serve 参数和原始 Benchmark。

目前在队友插件和已安装 FlagGems Iluvatar 后端 Python 源码内未搜索到 Intel 字样；不是全依赖审计结果，不能据此认定没有沿用其他设备配置。当前外部 MM 也不自动使用 FlagGems MM 的配置。必须定位具体文件及实际调用，才能将“Intel tile”定性为错误。

发现明显错误后，整理准确文件、当前行为、原因、最小修复及验证计划，与用户讨论后修改；当前授权覆盖审计和计划，不默认授权未讨论的源码修复。

## 4. 第二阶段：定向采样与瓶颈排序

复用已有编译产物，不重复完整框架 Profiling。若缺少实际 Kernel/Shape/耗时，则单独制定短时 GEMM 专项采样：沿官方服务参数，按规则归档并核对进程、GPU、端口、环境和 import 路径；采样属于诊断，不替代官方成绩。

按四类 GEMM 和相邻后处理分组统计 Prefill、Decode、Mixed 的调用数、耗时、M 分布和覆盖率。Graph replay 按实际 GPU 事件归档，不用 Python capture 计数推算。记录 Kernel launch、复制、权重读取与后处理开销，避免继续沿用队友优化前的热点占比。

阶段输出：天数独立调用链表、配置适配问题表、定向采样原始证据、按耗时覆盖排序的真实 Shape。GPU 可用性及空闲状态待执行前核对。

## 5. 第三阶段：融合与量化各自验证

### BF16 融合

优先评估 Gate/Up GEMM+SwiGLU。当前对照为外部 BF16 MM+已融合激活，不能再将 SiLU+Mul 两算子融合算作新增收益。先判断提供者是否允许配对 Gate/Up epilogue；新计算主体若慢于外部 MM，可能抵消节省的中间流量。实现前明确 BF16 中间舍入与非线性数值契约。

O/Down 仅在确认现有残差/Norm链路后评估。残差已与 Norm 融合，拆分重组不一定减少 Kernel 和流量。QKV+RoPE、GEMM+Norm 暂缓，不修改 Attention 主体。

### W8A16

沿用 ADR-0002 的静态对称 per-output-channel INT8 权重、FP32 scale、BF16 激活、INT8 转 BF16、FP32 累加后 scale 的契约。独立验证 Dense、Split-K、Separate；Separate 包含转换+实际天数 BF16 MM+scale，记录其中间 BF16 舍入。scale-before-dot 已弃用，不进入候选。

先用少量天数真实 Shape 验证编译能力、正确性和完整调用收益，再决定候选搜索规模。沐曦算法赢家、tile、M 分界及 manifest 不直接复用。既有天数 FlagGems weight_int8pack_mm 源码存在不代表已被模型调用或已有性能优势。

两个方向均先给出具体源码计划再实施，不把队友历史“未采用量化”当作用户授权后的阻塞条件；最终组合和交付由实测决定。

## 6. 后续测评与交付

待补充：天数真实 M、具体 GPU GEMM Kernel/提供者、各分支选中配置、后处理耗时占比、数值门槛及正确性、微基准收益、图重放与多层工作集、模型精度及候选官方双场景结果。

微基准基准固定为天数本次版本原生 BF16 GEMM；融合还需测完整原始后处理链路，W8A16 测完整调用。不把 BF16(Q) 诊断项替代原始权重基准。误差门槛须在搜索前明确，不因性能放宽。

正式验收按根 AGENTS.md：原样官方命令、RUNS=4/SKIP_FIRST=1、后三轮平均、保留全部明细；精度最低0.950且输出行为正常；逐场景报告吞吐与TTFT，±1%含边界为波动。阶段样本或旧基线不写成完整新验收。

每项实质性修改另写七部分中文改动报告，绑定天数的代码、提供者和环境版本。当前无源码修改，无新 GPU 数据，无 Git 提交。

## 2026-11-04 短时补采状态

见[天数短时运行采样报告](../changes/2026-11-04-tianshu-gemm-bounded-profile.md)：Prefill确认M=2048及四类投影耗时；Decode取得8步FULL上下文但图内Kernel不可见，不能宣称完整运行路径审计完成。服务已停止、GPU资源已释放。
