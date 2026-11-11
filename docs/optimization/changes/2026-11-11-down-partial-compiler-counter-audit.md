# Down M56/M64 partial 编译与计数器审计

## 1. 基本信息与摘要

2026-11-11，MetaX C500，仅审计量化内部的 Down Split-K partial，不做 BF16 MM 调优。新增 `benchmarks/diagnostics/audit_metax_down_partial.py`。实际被测源码来自最近官方融合组合的隔离副本，Kernel SHA256为 `ae55d507d8b4d7502ee41c6bada539094d9da5357b61705bc37cbe5a2ec30cf6`。远端 `/data/down-partial-audit-20261111-01/`。

结论：没有观察到寄存器spill；当前循环累加已被编译器融合。INT8转BF16与shared布局准备/同步有明确编译证据，shared bank-conflict事件也有可重复增量，值得优先进行针对性消融。但本轮无法把总时延唯一归因于转换、shared冲突、显存带宽或计算调度，不报告未经校准的Tensor利用率或stall占比。

## 2. 问题与原因分析

此前有限配置与写法搜索没有稳定改善热点M64，因此本轮从实际生产partial的编译结果和硬件事件入手。M56/M64均从 `_shape_dispatch` 截获实际配置，确认BM/BN/BK=64/64/128、4 warps、1 stage、Split8、basic；随后只测原partial Kernel，预分配其输入和FP32输出，不包含reduce/scale、workspace分配或模型时间。

这不是复采当前官方Decode占比，也不把独立计时当成模型replay耗时。历史4K采样的partial占比10.90%仍只适用于原窗口。

## 3. 具体实现与证据

导出原Kernel的TTIR、TTGIR、metadata和mcfatbin；使用MACA随附clang-offload-bundler提取fatbin内LLVM bitcode，llvm-dis生成 `.ll`。这是实际产物中嵌入的LLVM IR，不是最终机器ISA反汇编。

TTGIR显示循环每片K有6个BK128迭代，循环携带FP32累加器。原源码 `acc += tl.dot(...)` 已变成 `tt.dot A,B,acc`，因此改写显式累加通常不能消除额外一次加法——该额外加法在此IR中已不存在。

循环内有：A加载→shared准备，INT8 B加载→`sitofp`→BF16→shared准备→dot布局加载→dot。LLVM IR中可见32个静态 `sitofp i8→float` 与32个 `llvm.mxc.cvt.f32tobf16` 调用点，以及32个MMA intrinsic调用点、整个函数8个barrier静态调用点。静态数量不是动态执行次数，更不能当作最终ISA指令条数；barrier还包括循环外输出布局处理。不能将其直接乘循环次数来报告总barrier数。

当前IR没有通过跨迭代A/B缓冲表达独立双缓冲；这不排除后端机器调度重排。确认实际重叠程度仍需最终指令调度/可信stall指标。

## 4. 实验与验证

环境：Torch 2.8.0+metax3.7.0.7、Triton3.0.0、MACA3.7.0.38、驱动3.8.30；GPU报告104个SM、warp64、每SM2048线程、131072寄存器、64KiB shared，L2为8MiB。原始硬件属性与源码指纹见environment.json/source-hashes.json。任务开始与切换计数器采样前检查GPU空闲，9031可绑定，使用新进程；未清理JIT缓存。

| 项目 | M56 | M64 |
|---|---:|---:|
| 编译寄存器数 n_regs | 140 | 136 |
| 编译 n_spills | 0 | 0 |
| 每CTA shared | 16KiB | 16KiB |
| grid | 32×8=256 CTA | 32×8=256 CTA |
| partial独立计时中位数（7轮） | 57.632μs | 56.509μs |
| 单partial图replay中位数（5轮） | 76.400μs | 71.808μs |
| partial求和相对FP32参考L2 | 1.987e-6 | 2.002e-6 |

图replay值包含此独立图执行的测量开销，不能混用其与eager的差值解释Kernel内部瓶颈。M56仍使用BM64，M方向数学有效行比例87.5%；不能期待工作量按56/64等比例减少。M56多4个寄存器与mask路径同时存在，仅说明相关性，不单独证明其时延来源。

每次逻辑权重12MiB；A唯一数据量分别0.65625/0.75MiB，FP32 partial写出3.5/4MiB。A可能被不同N tile重复读取、由cache复用；这些唯一字节数不是实测DRAM流量，不据此判定带宽饱和。

### MCPTI采样与自校验

本机安装的MCPTI可枚举106条事件。先对20事件×2形状逐事件采3次，再做0/1/2/1次启动的读前读后验证，全部40条验证记录返回成功。原绝对值存在巨大累计偏移，Reset/Read表现与统一清零假设不一致，因此保留原值并按实际变化解释：

- `sm_cta_launched`：0、256、512、256增量，吻合实际grid，证明采样窗口确实包含目标partial。
- `local_load/local_store`：0/1/2次启动增量均为0，与编译n_spills=0一致。没有spill证据，但不能排除寄存器数量限制并发。
- `shared_load`：单次约100000～100800增量，两次约200800；`shared_store`单次约56000～56448，两次约112448。
- M64 `shared_ld_bank_conflict`单次约32737/32877、两次64754；`shared_st_bank_conflict`单次约31513/33123、两次66681。M56也有非零且近似随启动数增长的读写冲突事件。支持shared访问存在冲突活动，不等于这些事件占用多少周期，不能将事件比值直接当作耗时比例。
- `active_warps`增量约1000～1008，不能直接当成驻留warp时间或achieved occupancy。
- cycles/tensor及L2事件表现为最近一次Kernel的快照而非累计增量，0次启动也可能保留上次值。缺少足够的计数归一化与事件映射验证，暂不使用这些数据声称Tensor饱和或DRAM带宽瓶颈。原始L2读取大致有miss/query≈43.53%（M56）、41.22%（M64），仅为接口报告的sector比值，不是已校准的带宽利用率。

枚举事件没有直接的warp stall分类。本轮完成实际计数器读取与自校验，但“唯一主瓶颈”的证据仍不充分。

### 并发约束的静态估算

按4×64=256线程/CTA和报告的寄存器容量粗算，寄存器允许约3个CTA/SM，shared允许4个，线程数允许8个；这是忽略分配粒度及架构分区的资源估算，不是实测occupancy。总grid只有256CTA，相对104SM平均约2.46CTA/SM，也限制可提供的并行工作量。此前Split16完整调用并未稳定改善M56/64，不能据此直接增加Split数。

## 5. 与已有实验的关系

复用最新量化组合作为对照；本轮没有新增性能策略或生产Kernel修改。编译证据解释了上一轮仅改累加写法为何无明显收益；shared准备与转换链提供了比继续泛化扫tile更具体的研究目标。先前预取原型无稳定收益仍保留，不把本次IR观察当作预取方案已经有效。

## 6. 结论、合规性与剩余风险

优先级建议：先针对权重转BF16后的shared布局、bank冲突与同步做单因素验证，再研究向量化转换及加载/转换/dot调度。寄存器压力列为并发约束，spill不列为已证实瓶颈；显存带宽、Tensor吞吐是否饱和仍待可信指标确认。没有依据承诺具体加速比。

只审计Down量化内部partial。无BF16 MM调优、无权重持久BF16副本、无官方参数修改、无模型级收益宣称。数值为原partial核对，不替代模型精度。GPU任务均结束，最终GPU无进程。生产分发不改，未提交或推送。

## 7. 复现与归档

原路径：`/data/down-partial-audit-20261111-01/`。主命令详见commands.json：在该overlay的PYTHONPATH下执行 `audit_metax_down_partial.py --output <新目录>`。`counters.py`执行初采，`counter_validation.py`执行0/1/2/1次校验，均仅启动已编译的原partial；运行前检查GPU空闲，不能与模型评测并发。

关键文件：`results/m56.ttgir`、`m64.ttgir`、`m56.ll`、`m64.ll`、`m*-timing.json`、`events.json`、`counters.jsonl`、`counter-validation.jsonl`、`ir-summary.json`及`gpu-final.txt`。编译cache复用导致IR source location指向此前隔离目录，实际源码内容哈希相同，不能把该路径误认成运行错误版本。

归档 `/data/down-partial-audit-20261111-01-results.tgz`，本地 `artifacts/down-partial-audit-20261111-01/`。包含脚本、编译产物、原始计数与环境证据；overlay原件留在容器，归档仅排除重复overlay和__pycache__。
