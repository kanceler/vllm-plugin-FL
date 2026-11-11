# 天数BF16 MM热点完整配置扫描

## 1. 基本信息与摘要

2026-11-06，Iluvatar BI-V150 / CoreX驱动4.5.0，PyTorch 2.10.0、Triton 3.6.0。新增工作区独立诊断脚本experiment-artifacts/2026-11-06-tianshu-bf16-config-scan-01/scan.py；未改生产Kernel、候选YAML、缓存或dispatch。

四个热点Shape各生成518组，2072个唯一组合全部尝试；1644项通过编译、正确性及计时，428项因共享内存超限失败。领先候选及对照共230项五轮交错复测。四组在本次范围内均选择BM/BN/BK=256/256/32、warps=16、stages=2、Split-K=1。M=2048 Down有Kernel级收益，另外三组当前缓存已选中该赢家。无模型精度、官方吞吐或TTFT新结果。

## 2. 问题与原因分析

现有天数MM候选经early prune及性能模型top15后才计时。历史M=2048 Down的完整扫描赢家位于legal_rank=115，预测0.932288579ms，略落后于top组0.931803385ms，未获计时；它不是完全同分截断实例，而是模型对其预测略差。Gate/Up赢家历史rank=7，本已进入top15。

因此本轮证实的是特定Down Shape的性能剪枝遗漏，不能说所有Shape的top15都错。扩展warps/stages、BK及大tile未超过该原池赢家；不能泛化为这些维度永远无价值。

## 3. 具体实现

原池：BM/BN=32/64/128/256，BK=32/64/128，warps=8/16，stages=1/2，共192组。

主扩展：warps补4，stages补3，使主池共432组；BK=256、BM/BN=64/128/256、warps=4/8/16、stages=1/2再加54组；大tile(512,64)/(512,128)/(64,512)/(128,512)、BK=32/64、warps=8/16、stages=1/2再加32组。合计518，Split-K固定1。

脚本解开libentry/libtuner/heuristics包装，直接调用相同JIT Kernel，按原入口重建EVEN_K、AB_DTYPE和UPGRADE条件，绕过top15及计时缓存。使用BF16 A、连续[N,K]权重的转置视图B（stride=[1,K]）、预分配BF16 C、FP32累加。未使用此前失败的JIT包装调用。四Shape pilot验证入口数值与采样量级；M=4096 Down真实缓存为256/256/32、16、2，不能用2048 Down配置充当其基线。

## 4. 实验与验证

seed=42，FP32参考关闭TF32，参考结果转BF16；预设relative L2<0.005且逐元素rtol=0.02、atol=0.125，输出须有限。成功配置最大relative L2约6.14e-5；每Shape411通过、107共享内存超限（设备限额131072 bytes）。没有因数值失败而剔除的配置。

扫描配置随机排序，event计时warmup=10ms、rep=30ms，记录p50/p20/p80。每Shape初筛前10、各候选来源领先项及当前基线去重后，随机交错五轮复测，warmup=30ms、rep=100ms，以五轮p50的中位数排序。全部2072项为本次调用/测量或失败记录，未复用计时缓存；编译可能复用持久JIT缓存，first_call不称冷编译。该计时为预分配输出的Kernel调用，不含公开mm入口分配，不是图重放或模型成绩。

| 投影 | M,N,K | 当前基线 ms | 赢家 ms | Kernel加速 | 赢家来源 |
|---|---|---:|---:|---:|---|
| Down | 2048,2048,6144 | 0.728350 | 0.554154 | 1.3143× | 原池 |
| Gate/Up | 2048,12288,2048 | 1.406335 | 1.406335 | 1.0000× | 原池 |
| Down | 4096,2048,6144 | 1.104802 | 1.104802 | 1.0000× | 原池 |
| Gate/Up | 4096,12288,2048 | 2.829652 | 2.829652 | 1.0000× | 原池 |

所有赢家均为256/256/32、16 warps、2 stages。2048 Down延迟下降23.92%，速度提升31.43%，两种口径不混用。其五轮赢家p50为0.554000/0.554046/0.554154/0.554206/0.554977ms，基线约0.727～0.729ms。另三组保留当前配置，不能宣称新增收益。

结果仅覆盖一个随机种子、四个整除热点Shape；非整除、其他布局、小M、跨层工作集、图重放、公开入口完整开销及模型评测待验证。共同赢家不能直接作为所有Shape默认值。先修复选择机制或固化准确Shape配置，再讨论是否扩大覆盖。

## 5. 与已有改动的关系

承接天数Shape采集及候选遗漏审计。原生BF16底层复用已有实现，未重复实现GEMM，未启用W8A16。四Shape分别确认配置，不使用沐曦tile或性能阈值。插件工作树与FlagGems工作树检查干净；实际测试安装包源码指纹如下。

实际源码：`/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py`，SHA256：`4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090`。完整包装链及环境见metadata.json。

## 6. 合规性与风险

独立合成算子诊断，不访问评测集、不修改官方serve/Benchmark参数，不代表正式收益。开始GPU无计算进程、68MiB；与模型任务串行。进程自然结束、scan.exit=0，结束ixsmi无计算进程、0%利用率、68MiB。只有独立实验脚本新增，生产配置尚未应用；没有模型量化或精度声明。固定配置或候选剪枝的生产改动另行讨论并验证。

## 7. 复现方式

远端目录 `/workspace/experiments/tianshu-bf16-config-scan-20261106-01/`，本地目录 `experiment-artifacts/2026-11-06-tianshu-bf16-config-scan-01/`。

命令：先 `python3 scan.py --pilot`，核对通过后 `python3 scan.py`。scan.jsonl记录2072项，remeasure.jsonl记录230项，summary.json保存逐Shape赢家及复测明细，candidates.json保留518组来源，cache-before.json保存运行前只读缓存记录（含历史计时，不能当新测）。metadata.json保留实际源码哈希、版本、计时协议，gpu-before/after.txt保存资源状态。

本地已检查2072个组合唯一且完整。未提交或推送Git，未生成正式交付包。
