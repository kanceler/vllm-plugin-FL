# K静态per-channel与静态四通道保护：CPU独立校准初筛

## 1. 基本信息与摘要

2026-11-16，用户明确暂缓group32，先推进CPU分析。新增benchmarks/diagnostics/study_k_static_channels_cpu.py，不修改服务、量化Kernel、官方参数或模型权重。工作树feat/tianshu-k-top4，既有HEAD f82fa167及未提交改动保留；未提交/推送。

复用2026-11-13已归档的90份RoPE后Q/K和V，采用按文本类别留一验证。共9种数值变体×90份快照=810条记录。静态per-channel scale取校准最大值的2倍并折进Q时90/90通过初筛；固定四通道保护、继续per-token scale时也90/90通过。1倍/1.25倍静态scale分别仅70/90和84/90通过。结果支持继续验证，不能当作模型精度、GPU速度或全输入保证。

计算全程CUDA_VISIBLE_DEVICES为空，torch线程数2；仅加载CPU张量，没有加载模型或调用CUDA。

## 2. 问题与原因分析

当前K top4每token/head独立scale和动态保护位置增加读取与恢复工作。静态per-channel若成立，可将scale转移到每步Q：q'=BF16(q×s)，K整数转BF16后直接参与dot；固定保护通道则可能省掉动态索引恢复，但必须先证明数值可接受。

历史合成异常值测试及真实张量回放并未证明异常通道固定。本次从跨文本类别迁移、长度增长、Prefill/Decode差异检验此前提；动态top4原值保存作为对照，避免将不同数值方案误认为等价变换。

## 3. 具体实现

### 数据、划分及泄漏控制

来源/workspace/experiments/tianshu-real-qkv-20261113-01/tensors，逐份核对manifest SHA256后再计算。9个案例来自中文、英文、代码、符号数学四段独立自写文本，各512/4096 tokens，另有英文16384 tokens；层0/10/20/30/41，Prefill最后4个query与Decode第4步，共90份快照。

按zh/en/code/math留一验证。每个验证类别只使用其余三个类别的512-token、step0数据校准；按层、KV head分别计算固定scale和保护位置。比如英文全部长度及step0/4均只使用中文/代码/符号数学的512-token Prefill校准。没有使用math_500；这里math指独立符号花园面积文本，非赛事数据。

每折每层有3份校准文件；四折各有一套冻结参数，不能把四折参数声称为一个已经可交付的统一生产配置。校准/验证的文本类别互斥；同类别不同长度和Decode共享轨迹，快照不是90条独立自然文本。

### 方案及数值计算

K维度128，GQA16Q/2KV。V始终per-token/head对称INT8＋FP32 scale，FP32反量化后BF16舍入。量化采用半离零舍入、clamp[-128,127]，scale下限1e-6。

静态scale：s[h,d]=max_calibration_tokens|K[h,t,d]|/127，再乘预先确定的margin∈{1,1.25,2}。验证时参数冻结，不使用本样本最大值更新。分别测：

- static_channel_dequant：BF16(INT8(K/s)×s)，Q保持原BF16。
- static_channel_fold_q：BF16(Q×s)，K整数转BF16；没有K保护。8个共享该KV head的Q heads使用同一通道scale。

两个方案在有限精度下不等价，单列fold_vs_dequant误差。

静态保护：校准token的动态top4索引频率累计，按频率最高选4个固定通道，并列小索引优先。每个验证token保留这4个原BF16值，排除它们计算动态per-token/head scale。不实施实际置换，只模拟固定保护的数值效果；该方案不是动态top4的等价替换，也没有证明可零成本折入权重。

对照为V-only及动态top4。两者在同一CPU精度模型下重新计算，不混用旧FP32-only成绩。

所有QK和PV在CPU用FP32 matmul累加，输入操作数按BF16显式舍入；softmax为完整FP32 softmax，P转BF16后PV。它模拟关键舍入位置，但不是线上分块online-softmax或GPU dot累加次序的精确模拟。另保存相对全FP32 Attention基线的输出误差。门槛固定为输出max abs≤0.20且NRMSE≤0.05，不因结果修改。

## 4. 实验与验证

远端独立目录/workspace/experiments/tianshu-k-static-cpu-20261116-01，现有Python /workspace/tianshu-mixed-env-20261109/bin/python，torch2.10.0；仅CPU、2线程，810条结果，COMPLETE存在。源码与本地逐字节一致，原始张量逐份SHA256检查通过。

| 方案 | 通过/总数 | 最坏输出max abs | 最坏输出NRMSE |
|---|---:|---:|---:|
| V-only | 90/90 | 0.014343 | 0.005526 |
| 动态top4 | 90/90 | 0.035857 | 0.009889 |
| 固定四通道保护＋动态scale | 90/90 | 0.029866 | 0.009401 |
| 静态channel scale×1，K端反量化 | 70/90 | 1.058652 | 0.413804 |
| 静态channel scale×1，折Q | 70/90 | 1.053387 | 0.414078 |
| 静态channel scale×1.25，K端反量化 | 84/90 | 0.559545 | 0.276347 |
| 静态channel scale×1.25，折Q | 84/90 | 0.567009 | 0.273562 |
| 静态channel scale×2，K端反量化 | 90/90 | 0.087682 | 0.042393 |
| 静态channel scale×2，折Q | 90/90 | 0.079243 | 0.042580 |

最大值可以来自不同快照。静态scale×1失败域分布：中文3/20、英文5/30、代码8/20、符号数学4/20；×1.25仍有中文1/20和代码5/20失败。不能因为×2通过而丢弃前两组失败。

三个margin下，单快照INT8范围外元素比例的最大值分别约3.1551%、1.1667%、0.1961%。×2仍有饱和，不能宣称覆盖了所有激活范围或无饱和。范围更宽也会增大量化步长；不预设继续增大margin必然更好。

×2折Q最坏NRMSE来自en-16384-step4-layer30.pt：NRMSE0.04257996，max abs0.07924318，超过INT8范围元素比例0.054918%。其相对全FP32原始Attention输出NRMSE0.04250421，结论未依赖概率BF16基线的宽松差异。折Q与K端反量化两者输出差异的全矩阵最大max abs0.02186370、最大NRMSE0.00607372，证明浮点舍入不可忽略，但本轮没有因此引发门槛失败。

### 固定通道稳定性

下表每层统计所有验证快照×2KV heads，先对每快照/head求token平均，再对这些均值取中位数。不是按全部token或耗时加权。

| 层 | 固定4通道覆盖动态top4元素的比例中位数 | 每token四通道集合完全一致的比例中位数 |
|---|---:|---:|
| 0 | 58.44% | 8.52% |
| 10 | 58.87% | 3.37% |
| 20 | 53.06% | 1.70% |
| 30 | 68.87% | 12.63% |
| 41 | 78.70% | 36.15% |

全快照/head覆盖比例范围约40.43%～87.65%。存在集中趋势，但动态top4不是固定集合。固定保护仍90/90通过是该独立量化方案的数值证据，不构成任意置换与动态top4等价的证据。channel_stability.json另保存按位置四等分的覆盖与固定通道能量占比，供后续研究。

## 5. 与已有改动的关系

复用历史HF BF16轨迹，未重新采集或启动模型。区别于旧replay_real_qkv.py的FP32量化/完整Attention参考，本次显式模拟BF16操作数、BF16概率，并重新测V-only和动态top4，因此误差数值不要求与旧报告逐位相同。

group32候选源码已准备但GPU测试未启动，当前暂缓。此次CPU工具没有接入生产，不改变正在使用的K top4＋V INT8。两轮16K、4K及精度日志不作为本CPU研究的速度依据。

## 6. 合规性与风险

没有评测集校准、正式命令变更或GPU占用。按四类文本交叉验证不等于广泛域覆盖；文本是重复段落，层仅5/42，Decode仅第4步，16K仅英文，未覆盖量化误差反馈后的后续层/生成轨迹。不能声称达到模型精度门槛。

margin×2虽然预先列入候选，但观察验证成绩后挑选继续方向，会使用本验证集作选择；下一阶段需要新的未参与本轮筛选的独立文本做确认，不应把这些90份作为未来候选选择后的最终盲测。

路线A静态fold-Q数值可行性获得初步支持，但最坏NRMSE接近0.05，余量明显小于动态top4；还需更广校准/验证及GPU数值检查。路线B固定保护也值得后续验证，但任意置换不能直接越过RoPE折入Wq/Wk。无需为了CPU结论先修改模型权重或RoPE。

无Kernel计时，不声称K成本降到3%～5%、不声称比V-only快，不用TPOT百分比推算吞吐盈亏线。

## 7. 复现方式与证据

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/workspace/tianshu-mixed-env-20261109/bin/python -u study.py \
/workspace/experiments/tianshu-real-qkv-20261113-01/tensors \
/workspace/experiments/tianshu-k-static-cpu-20261116-01/results
```

输出目录须为新目录，不能覆盖已有结果。归档本地experiment-artifacts/2026-11-16-k-static-cpu-01/evidence.tgz及evidence/，包SHA256 `ccd3962f545c1358b4edead39425823905ac0e6b77fde218b463447ad257766b`，与远端一致。包含script、校准来源/参数、raw.jsonl 810条、summary.json、channel_stability.json、metadata.json及COMPLETE；stability-summary.json为本地从原始记录派生的统计，manifest.json逐文件归档指纹。

本地Python语法检查和git diff --check通过。没有提交、推送、模型接入或模型测评。下一步建议先决定候选方向及新独立验证集，再实施GPU成本消融或静态候选Kernel，不把本报告当GPU性能结论。
