# 天数Down M4096：阶段诊断、INT8直接加载与分组调度实验

## 1. 基本信息与摘要

日期2026-11-13；Iluvatar BI-V150、CoreX/驱动4.5.0，Torch2.10.0、Triton3.6.0。按用户批准的三步计划完成阶段诊断、复用天数MM主体的直接INT8加载、tile分组调度，以及同tile dtype编译审计。结论：精度通过，未找到超过原生BF16的量化路径，新增实验不替换默认策略。

插件工作树 `worktrees/vllm-plugin-FL-tianshu-down-adapt` / 分支 `feat/tianshu-down-w8a16-adapt`；FlagGems工作树 `worktrees/FlagGems-tianshu-down-adapt` / 分支 `feat/tianshu-down-scale`。修改插件 `vllm_fl/quantization/w8a16/backend.py`、FlagGems `_iluvatar/ops/mm.py`；新增两个诊断脚本 `diagnose_tianshu_down4096_fused.py`、`audit_tianshu_down4096_dtypes.py`。源码未提交/未推送。

## 2. 问题与原因分析

前轮M4096 Separate完整图约1181μs，原生固定BF16约1121μs；两布局各518组搜索后仍选256/256/32、16warps/2stages。继续无边界扩tile的价值有限，需定位差距来自转换、scale还是MM本体，并尝试移除独立转换。

现有天数MM主体在整除K循环中已有 `b.to(AB_DTYPE)`，理论上可直接加载INT8 B并转换BF16。它省去一次全权重BF16暂存，却会在每个M tile重复转换相同权重；M4096、BM256时有16个M tile组，因此转换指令的重复量与独立转换一次不同。此为源码层工作量分析，实际成本还受缓存、流水线与编译布局影响，不能仅由读写字节推断收益。

## 3. 具体实现

- 新增显式 `fused_native` 算法实验入口：直接向天数scale-after-dot MM变体传Q.T，INT8加载后转BF16、FP32累加后乘FP32通道scale、BF16输出。没有原生INT8 dot，也没有持久BF16权重缓存。
- 新增 `scale_mm` 辅助接口，复用天数MM主体而非另写BF16 GEMM；INT8入口要求K可被BLOCK_K整除。原因是原尾块分支没有同样的类型转换，实验入口对不支持K明确拒绝，auto仍保留原Dense/Separate选择，不自动调用此实验路径。
- 新增独立 `mm_kernel_scale_grouped`，只改变pid到M/N tile的映射，不改原生MM与旧scale变体。以GROUP_M个M tile为一组，组内让M方向先变化，提高B tile跨CTA复用的机会。0表示旧行序，1为组大小1，另测2/4/8/16。尾组通过min处理，不把分组条件冒充合法M白名单。
- Separate的nk/kn布局也能显式调用分组变体，完整调用仍包含分配、转换与融合scale。默认group_m=0，已有runtime策略没有替换。
- 诊断分别测转换、BF16(Q)上的原生MM及scale MM；预转换只作阶段诊断。最终对照计量真实完整调用，不以阶段求和替代。

## 4. 实验与验证

远端独立目录 `/workspace/experiments/tianshu-down4096-fused-20261113-01`，从前轮隔离source/gems复制后只更新本次文件，生产安装包与历史结果不覆盖。开始前GPU无计算进程、70MiB驱动保留，端口9031空闲，前后进程/GPU归档。没有模型服务或其他GPU测试并发。

预设实现误差门槛保持relative L2≤0.005、normalized max≤0.01、输出有限；FP32参考关闭TF32。量化路径参考为FP32 A@Q.T后scale再BF16；原生参考为FP32 A@原BF16权重.T后BF16。量化误差不能用实现误差通过来代替模型精度。

### 正确性与搜索预算

- 2 seed×3 Shape×3算法/布局×4组大小共72条pilot通过：Shape为(1,17,64)、(65,129,256)、(4096,2048,6144)，含N/M尾块、零权重通道，eager及捕获后输入减半graph replay均验证。K65的INT8入口拒绝检查通过。
- M4096直接加载搜索48组：BM/BN={128,256}、BK={32,64,128}、warps={8,16}、stages={1,2}；42成功、6共享内存超限。前4组与group={0,1,2,4,8,16}交叉24组；Separate固定MM两布局×6种组大小共12组。保留旧固定配置和原Separate作为对照。
- 阶段诊断2 seed×2布局×3阶段×5次共60条；最终两个新seed20261123/20261124、每seed5轮随机交错eager/graph，共180条。总324条计时尝试：318成功、6OutOfResources，无数值失败。最大L2（含原生）5.9326644986867905e-5，最大normalized max0.004444444552063942。pilot和dtype审计另计，不将复测条数冒充唯一候选数。
- 初筛seed20261121，pilot另用20261122；初筛do_bench warmup5ms/rep12ms，阶段与复验rep30ms。最终为10次p50中位数。JIT编译可复用持久编译缓存；没有复用计时缓存。显式图捕获包含真实转换/MM节点，图内Python分配不按eager成本解释。

### 阶段诊断（eager，μs，两个seed各五次的中位数）

|阶段|nk|kn|
|---|---:|---:|
|单独转换含目标分配|75.825|80.437|
|BF16(Q)原生MM，不含转换|1103.940|1093.305|
|BF16(Q)融合scale MM，不含转换|1112.304|1099.437|

这些阶段值支持：转换为显著额外成本，融合scale MM本身接近相同BF16(Q)上的原生MM。不能把阶段相加充当完整结果，也不能声称差值全由转换唯一解释。kn转换单独较慢但完整调用较快，进一步说明需联合测量缓存/布局。

### 同场完整调用复验（μs）

|路径|配置/调度|eager|graph|
|---|---|---:|---:|
|原生固定BF16|256/256/32、16warps/2stages|1102.812|1117.648|
|原Separate kn|同MM，转换64×128/16，旧行序|1178.883|1181.100|
|Separate较优分组候选|同MM与转换，group1|1178.389|1181.059|
|直接INT8加载，原固定tile|256/256/32、16/2，旧行序|2224.200|2227.769|
|直接INT8加载较优候选|256/256/64、8/1，group1|1828.723|1831.983|

Separate分组相对原Separate图耗时仅降低约0.0035%，无有效改善；相对原生固定BF16仍慢约5.67%。直接INT8加载较优候选相对原生图慢约63.91%，不采用。所测组大小及配置范围内没有可替换生产策略的收益。

### 同tile dtype编译审计

补充两个新seed20261125/20261126、两个tile×两种B dtype，共8项数值与编译审计。相同tile除B dtype外使用同一scale MM主体，BF16(Q)预转换不计入此诊断，不作为完整量化性能。

|tile BK/warps/stages|B dtype|寄存器|spill|shared bytes|代表两seed耗时μs|
|---|---|---:|---:|---:|---|
|32/16/2|BF16|118|0|65536|1118.38 / 1112.63|
|32/16/2|INT8|118|0|57344|2223.73 / 2222.93|
|64/8/1|BF16|168|0|65536|1678.40 / 1681.25|
|64/8/1|INT8|168|0|65536|1833.78 / 1831.14|

同tile未见寄存器数增加或spill，因此不能把退化归因为寄存器溢出。BK32对照TTGIR两者都有4处async_copy_global_to_local、2处async_wait；INT8增加循环内arith.sitofp，local_alloc/local_load各由2处增至3处。不能说异步拷贝完全失效；证据支持额外转换及shared/layout处理，具体硬件停顿比例仍需进一步证据。IR静态次数不等于动态执行次数。不同tile/group的主实验compiler.jsonl可有不同寄存器数，应以同tile audit对照判断。

## 5. 与已有改动的关系

复用前轮转换/scale/布局实现并保留原native固定对照，未将对未优化默认路径的改善作为量化收益。未更改s2-final排序补丁、top15或其他投影，本次显式配置实验绕过排序，排序开关A/B的结果独立保存。不修改队友原生BF16 GEMM，只增加独立scale变体与诊断入口；没有新增正式模型组合。

## 6. 合规性与风险

无官方服务、Benchmark、数据或checkpoint改动，全部合成数据，无math_500校准。实验代码默认不进入模型路径，不能写成已启用提速。范围仅M4096/N2048/K6144及pilot边界，不推广其他Shape/设备。未验证真实TP、多层工作集、模型覆盖、精度或端到端吞吐。

本轮结论为止损：不替换原Separate策略，不无限扩参数池；直接INT8加载若再尝试，需要针对编译转换/shared布局提出新的具体设计，而非仅减少存储字节。Separate若继续优化应围绕可验证的转换开销/流水线，双表示或跨调用缓存必须另做容量与生命周期设计，不假定量化层仍保留原BF16副本。

## 7. 复现方式

```bash
cd /workspace/experiments/tianshu-down4096-fused-20261113-01
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python source/benchmarks/diagnostics/diagnose_tianshu_down4096_fused.py results
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python source/benchmarks/diagnostics/audit_tianshu_down4096_dtypes.py dtype-audit
```

实际输出重定向run.log、dtype-audit.log，commands.json记录原命令。复现使用新的输出目录，脚本拒绝覆盖已有结果。hashes-final.json绑定全部本次量化/提供者/诊断源码，source与gems实际通过PYTHONPATH隔离；provider-environment.json是前轮同环境版本来源，路径以本次命令为准。results含raw、pilot、leaders、summary、compiler元数据及IR，dtype-audit含同tile审计与IR；完整完成标记、前后资源状态均保留。

本地完整证据 `experiment-artifacts/2026-11-13-tianshu-down4096-fused/evidence.tgz`；没有GPU遗留计算进程。未提交/未推送。
