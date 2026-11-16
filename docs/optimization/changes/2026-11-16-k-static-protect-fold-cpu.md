# 固定四通道保护叠加静态 scale 折 Q：CPU 筛查

## 1. 基本信息与摘要

2026-11-16，分支 feat/tianshu-k-top4，HEAD f82fa167，保留已有未提交修改。新增 benchmarks/diagnostics/study_k_static_protect_fold_cpu.py；不修改当前已冻结动态 top4 服务或 Kernel。用户已同意先叠加筛查，再研究完整调用联合优化。本轮仅 CPU，没有 GPU 性能或模型精度结论，未提交或推送。

15 变体 × 90 快照，共 1350 条数值记录。固定保护＋静态 scale×2 折 Q 通过 90/90，最坏输出 NRMSE 0.031240，比无保护静态折 Q 的 0.042580 低，但仍高于动态 top4 的 0.009889。较小 margin 仍失败，不能替换当前服务配置。

## 2. 问题与原因分析

静态 per-channel scale 可从 K 端移到 Q 端，减少逐 token K 反量化乘法。保护通道保存原始 BF16 K，因此其 Q 必须保持原值；若将全部 Q 通道乘 scale，再恢复原始 K，会改变保护通道的点积。

固定四通道不覆盖所有动态异常值。其效果必须和静态 scale 联合评估，不能把此前两个分别通过的方案直接当作叠加通过。

## 3. 具体实现

沿用 study_k_static_channels_cpu.py 的 CPU 模型、独立类别留一校准划分、9 个对照变体，新增每个 margin 的保护后 K 反量化及保护后折 Q，共 6 个变体。

对固定集合 S：d∈S 时 q_eff=q、k_eff=原始 BF16 K；其余通道 q_eff=BF16(q×s)，k_eff=BF16(INT8(K/s))。s 为独立校准 per-channel 最大值/127，再乘预定 margin 1、1.25、2。保护后 K 反量化作为舍入差异对照。V 保持 per-token/head INT8、FP32 scale 反量化后 BF16。

不实施通道置换、不修改 RoPE 或权重，不假设置换可以无条件折入 Wq。当前仅数值原型，没有缓存字节布局、GPU writer 或 Q 处理 Kernel。

## 4. 实验与验证

数据为 /workspace/experiments/tianshu-real-qkv-20261113-01/tensors；90 份已归档 RoPE 后张量，5/42 层，中文/英文/代码/独立符号数学文本，512/4096 及英文 16384 tokens，Prefill 与短 Decode。逐文件校验 SHA256。校准仅取其他三类别的 512-token Prefill，不使用 math_500；四折参数不是一套生产配置。

本轮复用已经筛查过的验证数据，属于探索性叠加验证，不是新的盲测；相同文本的长度/步数快照相关，不能视为90条独立文本。CPU FP32 matmul 累加、BF16 操作数、完整 FP32 softmax 后概率转 BF16，不是 GPU online-softmax 的精确模拟。固定输出门槛 max abs≤0.20、NRMSE≤0.05。

| 方案 | 通过数 | 最坏输出 max abs | 最坏输出 NRMSE |
|---|---:|---:|---:|
| V-only | 90/90 | 0.014343 | 0.005526 |
| 动态 top4 | 90/90 | 0.035857 | 0.009889 |
| 固定保护＋动态 scale | 90/90 | 0.029866 | 0.009401 |
| 固定保护＋静态 scale×1 折 Q | 70/90 | 1.077113 | 0.407949 |
| 固定保护＋静态 scale×1.25 折 Q | 83/90 | 0.575531 | 0.272919 |
| 固定保护＋静态 scale×2 折 Q | 90/90 | 0.074521 | 0.031240 |
| 无保护静态 scale×2 折 Q | 90/90 | 0.079243 | 0.042580 |
| 固定保护＋静态 scale×2，K 端反量化 | 90/90 | 0.073739 | 0.031601 |

最大值可来自不同快照。×1.25 叠加方案比无保护少通过一例，保护不保证误差单调改善。×2 最坏未保护元素饱和比例 0.202400%，分母只含124个未保护通道，与无保护方案128通道比例不可直接比较。

Python 语法检查通过。远端现有 torch2.10.0，CUDA_VISIBLE_DEVICES 为空，CPU2线程，COMPLETE 存在；全部1350条结果及参数已归档。GPU正确性、完整调用耗时、新独立文本验证、官方模型精度均待验证。

## 5. 与已有改动的关系

在 2026-11-16-k-static-channel-cpu-screening.md 基础上增加叠加数值变体。已冻结 page32/separate_meta 动态 top4 联合版保持原样，继续作为 K 对照；主性能对照仍是最初 V-only scale-on-V，而非共同调优后的 V-only。不得把旧 K 官方精度 0.9714 归给本候选。

## 6. 合规性与风险

未改官方命令、数据集、模型或现有服务开关，未使用评测题目校准。未形成42层统一参数，未验证量化反馈、长 Decode 或完整模型行为。CPU通过仅支持进入后续测试，不构成接入或精度验收。后续 GPU 必须计入 Q 处理、实际 KV 写入、Attention、归约及所有新增开销；不能仅报告 Attention 核心耗时。

## 7. 复现方式

远端目录 /workspace/experiments/tianshu-k-static-protect-fold-cpu-20261116-01，源码 study.py，日志 run.log，results 中含 calibration.json、raw.jsonl、summary.json、metadata.json、channel_stability.json、COMPLETE。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/workspace/tianshu-mixed-env-20261109/bin/python \
/workspace/experiments/tianshu-k-static-protect-fold-cpu-20261116-01/study.py \
/workspace/experiments/tianshu-real-qkv-20261113-01/tensors \
/workspace/experiments/tianshu-k-static-protect-fold-cpu-20261116-01/results
```

结果目录要求不存在；复跑使用新目录，禁止覆盖旧结果。本机归档 experiment-artifacts/2026-11-16-k-static-protect-fold-cpu-01/evidence.tgz，SHA256：8c3668f046680e8567919a84886338dcd8602f53ffa6e6164459588ce6488b35，两端一致。
