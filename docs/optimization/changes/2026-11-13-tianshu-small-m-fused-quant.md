# 天数小M融合W8A16与GEMV实验

## 1. 基本信息与摘要
2026-11-13，天数Iluvatar BI-V150、驱动4.5.0。分支perf/tianshu-small-m-fused-quant。新增benchmarks/diagnostics/benchmark_tianshu_small_m.py及tianshu_small_m_kernels.py，后续增加缩短流程。原目标为Gate/Up、QKV、O的M=1/2/4/8/16/32/64；按用户后续优先级调整暂停，完成7/21个Shape，不把其他点写为完成。已完成M64的QKV/O融合路径有正结果，Gate/Up融合路径较BF16慢，全部候选保留。

## 2. 问题与原因分析
上轮Separate在M64需完整BF16权重暂存、转换及再次读取；融合scale不消除这些成本。原生BF16 MM已有同场优化对照，量化路径须在完整调用中胜出。当前天数新版Down M64 Split-K仅有正确性记录，尚缺调优后的同场性能对照；不能把“适合Split-K”的假设写成已验证结论，也不能认为其他投影已证明Split-K不适合。本轮按授权不测试Split-K。

## 3. 具体实现
不修改生产backend与FlagGems主体。复用已有fused_native（原生MM主体内直接读INT8、转换BF16并融合FP32scale）与独立dense；增加寄存器内复用权重的单行/多行GEMV。GEMV对每输出列组加载Q，在K方向FP32乘法及归约，每组复用给RM行，再乘FP32scale输出BF16；该SIMT路径不是Tensor Core dot实现，虽同为W8A16，须单独验证舍入。没有K切分、workspace或持久BF16权重副本。

所有权重为对称per-output-channel INT8、FP32scale，激活及输出BF16。两个GEMM融合实现是INT8转BF16后dot、FP32累加，不是原生INT8×INT8。GEMV适用本次M≤8；全K装入寄存器可能造成压力，RN/RM/warps需要实测，不假定SIMT更快。

原生、fused_native、Separate每布局各288项MM配置：BM16/32/64/128，BN32/64/128/256，BK32/64/128，warps4/8，stages1/2/3；独立Dense限制BM≤64，共216项。Separate每布局前四项交叉15组NK或27组KN转换参数。GEMV RN1/2/4/8/16、RM1/2/4/8且RM≤M、warps4/8。各路径前三项用两独立seed、五轮交错、eager及显式图重放复测。结果仅称所测范围内较优。

## 4. 实验与验证
本地AST和diff空白检查通过。预定Kernel门槛relative L2≤0.005、normalized max≤0.01，须有限值；原生参考为FP32 A×W，量化路径为FP32 A×Q再乘scale。数值检查先于计时。权重及输出量化误差在计时外另记录。GPU正确性、全部候选及最终结果待运行完成后补充。没有全模型精度、吞吐及TTFT新结论。

阶段收尾：原results/完成Gate/Up M1/M2，缩短results-short/导入这些720条确认计时（非新测），另外完成Gate/Up、QKV、O的M64及QKV/O的M1，新增1620条确认计时，合计2340条。QKV M2停在初筛/转换交叉阶段，未写入summary；其他未完成点不作为成绩。全部已完成Shape有效候选均通过原定数值门槛。完整结果和候选配置保存在本地工作区experiment-artifacts/2026-11-13-tianshu-small-m-01/，原包evidence-paused.tgz保留；analyze.py生成comparison.md与selected.json。后续计划以用户最新指令为准：Gate/Up不测Split-K，四投影研究M64较优量化路径到Separate的分界，其他点缩小验证。

## 5. 与已有改动的关系
承接四投影Separate完整调用测试，复用其计时兼容处理和隔离source/gems提供者。不改变上轮结果；已有用户未提交修改保留。本轮不纳入Gate/Up SiLU×Up融合，也不改变Runtime Dispatch。

## 6. 合规性与风险
GPU任务串行；启动前无计算进程，70MiB驱动占用、0%利用率。远端框架提交d799978437998c565b42d369569592d82a3028ac且含已有差异，未修改该仓库。仅同步必要实验文件和隔离overlay。实验结果不代表模型调用覆盖；小M合成输入也不代表完整真实负载。共享内存超限记录为资源失败；设备致命错误停止。未改变官方命令、模型或评测数据。

## 7. 复现方式
远端/workspace/experiments/tianshu-small-m-20261113-01/保存命令、hashes、环境、资源前态、日志及结果：

```bash
cd /workspace/experiments/tianshu-small-m-20261113-01
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python -u source/benchmarks/diagnostics/benchmark_tianshu_small_m.py results
```

结果目录拒绝覆盖。复现须使用新目录并先检查GPU；复用已有Torch2.10.0/Triton3.6.0环境，不重装依赖。manifest含提供者SHA256及候选池，raw.jsonl记录原始计时和失败，summary.json记录复测中位数及范围，全部完成才写COMPLETE。尚未提交或推送。

### 用户批准的缩短流程与M64优先

逐点宽搜索已完成Gate/Up M1、M2；M4仅完成部分初筛。按用户“缩短方案”“先测64热点”的明确指令，核对本任务PID29639后发送TERM停止，原results/及run.log保留，未将其标记为全套完成。GPU释放状态归档在gpu-after-wide-interrupt.txt。

新增benchmark_tianshu_small_m_short.py，在新results-short/中显式导入已完成Shape的summary及确认轮次（带import_source），这些不是新测。先按Gate/Up、QKV、O执行M64，然后补M1，其他M使用相同投影、同算法/布局在已完成端点的领先配置加12个手工小tile控制项，Separate转换复用领先项及NK三组/KN两组控制。GEMV M≤8仍搜索全套上述候选。M1/64仍保留宽搜索，各M保留两seed五轮复测。此流程缩小了非端点候选空间，不能宣称逐点288项搜索或全局最优；manifest显式标记结果复用。后续统计必须区分原results导入轮次与新测。

缩短版实际命令：

```bash
cd /workspace/experiments/tianshu-small-m-20261113-01
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python -u source/benchmarks/diagnostics/benchmark_tianshu_small_m_short.py results-short
```

端点现有结果（GPU完整复测，不是模型验收）：Gate/Up M1原生eager114.245μs、GEMV88.638μs，图119.223/93.075μs；Gate/Up M2原生eager114.241μs、fused_native94.392μs，图118.607/99.349μs。M2 GEMV131.986μs，未超过原生，说明M1收益不能外推。

用户明确要求所有方案均保留。实验Kernel、候选配置、BF16对照、量化各路径及原始结果均保留，不因单点负收益删除候选或收窄算法合法性。Gate/Up M64当前原生BF16较快；融合W8A16仍作为容量与端到端组合候选保留，不据此自动在模型中启用。现有Gate/Up M64复测eager为BF16 123.319μs、fused_native130.397μs、dense131.755μs、Separate252.116μs。纯算子代价与模型容量收益分别记录。
