# 固定四通道＋静态 scale 折 Q：GPU 完整调用候选

## 1. 基本信息与摘要

2026-11-16，Iluvatar BI-V150，分支 feat/tianshu-k-top4，HEAD f82fa167，保留已有未提交改动。用户批准沿已讨论方案继续 GPU 叠加和完整调用比较。本轮只新增诊断候选及测试，不修改已冻结动态 top4 服务，不改官方命令。

新增文件：benchmarks/diagnostics/k_static_protect_fold_candidate.py、benchmark_k_static_protect_fold.py、validate_k_static_protect_fold_real.py、configs/k_static_protect_fold_screen_v1.json。GPU 真实回放90/90通过初筛；独立种子完整调用确认及最终数值见下文。未进行模型接入、官方精度或官方吞吐测试，未提交或推送。

## 2. 问题与原因分析

动态 top4 在每个 K tile 内读取 token scale、索引及保护值，执行反量化乘法和动态位置恢复。CPU 叠加筛查支持测试静态 per-channel scale 与固定四通道保护：只对未保护通道把 scale 折到 Q，保护通道 Q/K 原值保持不变。

本次把固定通道在 RoPE 后同步重排到 Q/K 的前4维，减少 Attention 内的不规则恢复。重排不能无条件折入 Wq/Wk：RoPE 维度配对必须保持正确。本候选不改权重，不省略 Q 处理费用。

## 3. 具体实现

Q 处理 Kernel：每 token、Q head 一个程序，读取所属 KV head 的固定128维置换表，前4维保持原Q，其余维乘静态scale并转BF16，写入预分配工作区。

KV writer：读取相同置换后的K，前4维BF16原值写入元数据固定偏移，其余维按静态scale量化INT8；V仍按原始通道顺序使用per-token/head INT8＋FP32 scale。Attention tile内INT8 K直接转BF16，无token scale乘法，固定0/1/2/3位置恢复BF16保护值，再进入dot；PV及分段归约沿用已有实现。不是原生INT8×INT8 dot。

为控制变量，保留page32、token-major、separate_meta及276B/token/head缓存分配。仅使用元数据的8个保护值字节，其余元数据保留为未使用空间；本轮未压缩缓存容量。置换表与scale为模型层级静态表，初始化/表构造不计稳态，但每次Q变换和真实写入必须计时。

诊断参数表只选少量tile/warps/segments，按同一batch跨长度的等权相对耗时几何均值选择一套配置，确认阶段冻结。缓存布局不随batch或长度切换；图捕获后不依赖Python逐步KV长度选参。没有宣称所测范围外或全局最优。

## 4. 实验与验证

远端 /workspace/experiments/tianshu-k-static-fold-gpu-20261116-01，基于已验证joint-frozen-02源码独立复制，未覆盖主仓库或原冻结目录。Python /workspace/tianshu-mixed-env-20261109/bin/python，torch2.10.0、Triton3.6.0、驱动4.5.0。首次资源检查GPU70MiB、0%、无运行GPU进程；端口检查工具ss缺失，后续以独立socket探测补充。GPU任务串行执行，未启动模型服务。

合成测试使用预定静态scale0.0625，保护集合[[0,3,17,65],[0,9,37,93]]，属于实现单元测试和性能负载参数，不是模型校准。覆盖8个Decode点、3D和2D、混合长度、打乱页表、非连续输入、负slot、缓存字节、Q变换、分段workspace拒绝及长度1捕获后增长重放。完整图计时包含Q变换＋真实增量KV写入＋Attention＋归约，9轮交替顺序，每轮5次重放，报告中位数；阶段相加不替代完整调用。

主V-only对照为最初冻结原始源，SHA256 0f08a4dfe8a04565c5853c407a5b79744296f08b221eba773cdd7f36c8583923；fresh与继承Prefill128状态都保留。原V的2D配置受历史状态影响，两种状态不择优当作模型结果。另保留已冻结联合K和实际旧测评K路径，同一输入/有效长度配对。

真实数据回放复用CPU阶段90份张量及calibration.json（SHA256 26167240f918c477156184000e28989901e93ab2ef713b38fe72d014a4237632），每个类别使用其他三类别512-token Prefill校准参数，margin固定2。GPU总误差参考为原始BF16 Q/K/V的独立FP32 Attention，和CPU概率BF16参考不同，数值不能直接当作逐位一致。真实回放90/90通过max abs≤0.20、NRMSE≤0.05，最坏max abs0.070349、NRMSE0.031740；每例Kernel误差max abs≤0.08、NRMSE≤0.02的检查也全部通过。这是旧筛查数据复用，不是新盲测；只有5/42层，不能当作模型精度。

保留失败：results-01 首个合成样本在head1通道0注入30而原保护集合不含0，静态scale范围约±8，量化误差max abs14.688604、NRMSE0.528022超限，Kernel自身参考检查已通过。这证明未保护通道异常仍可能失败。修正后的合成测试显式保护已知注入通道，不能用于证明真实参数泛化。real-01参考V scale广播维度写错，修正后real-02完整通过；原脚本、异常和日志均保留。

独立种子20261208确认：使用20261207初搜后冻结的每batch配置，未重新择优。8例确认完成；batch56/64的两种长度均使用同一配置，缓存布局全程固定。

| batch / KV长度 | 当前冻结K ms | 旧测评K ms | 最初V fresh ms | 最初V Prefill历史 ms | 静态叠加 ms | 比fresh V耗时降低 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 129 | 0.023734 | 0.021750 | 0.018360 | 0.018364 | 0.033943 | -84.88% |
| 1 / 17408 | 0.135750 | 0.159112 | 0.240313 | 0.240227 | 0.116666 | 51.45% |
| 16 / 4096 | 0.260868 | 0.286003 | 0.235247 | 0.235380 | 0.230292 | 2.11% |
| 56 / 4096 | 0.774202 | 0.908962 | 0.715706 | 0.715110 | 0.689623 | 3.64% |
| 56 / 16384 | 2.859905 | 3.407288 | 2.670475 | 2.674718 | 2.541660 | 4.82% |
| 64 / 4096 | 0.880149 | 0.963107 | 0.824232 | 0.827207 | 0.784805 | 4.78% |
| 64 / 16384 | 3.328834 | 3.545542 | 3.080293 | 3.083314 | 2.938264 | 4.61% |
| 80 / 16384 | 5.982648 | 6.155949 | 4.817442 | 12.559633 | 5.264923 | -9.29% |

56/4K、56/16K、64/4K、64/16K相对最初V fresh的九轮配对耗时降低范围分别为3.34～4.07%、4.50～5.26%、4.16～5.25%、3.81～5.45%，本轮配对均为正值。初搜与独立种子确认方向一致；不是官方吞吐提升率。单轮统计次数有限，未形成跨时段长期稳定性结论。

初搜30条静态候选完整调用路径、确认8条，Kernel参考和总误差门槛均通过。真实90例最坏Kernel max abs 0.00904298、NRMSE 0.00187239。

确认结束GPU70MiB、0%、无GPU进程，9031 socket connect_ex=111（无监听）。ss不可用，未把首次端口检查写成已通过。Python语法检查及git diff --check通过。

## 5. 与已有改动的关系

继承动态top4的separate_meta布局及离线launch框架，依赖study_k_static_protect_fold_cpu.py证明的数学形式。本次是新数值方案，旧K官方精度0.9714和新动态joint服务42层/35图smoke结果不自动适用。当前新服务开关仍运行动态top4，没有热切换或默认启用本候选。

## 6. 合规性与风险

没有使用math_500校准，没有修改官方serve或benchmark参数，没有模型权重重排或改变采样。仅诊断算子；官方精度、42层统一独立校准、新文本盲测、量化反馈、长Decode、多种Prefill、实际全部capture batch及模型端性能尚未验收。短序列与2D batch80存在负收益，不隐藏，不承诺所有batch快于V-only。不能按每步batch切换缓存数值语义回到动态top4。

## 7. 复现方式

运行环境：PATH=/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin，LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib，PYTHONPATH指向实验source；VLLM_PLUGINS=fl、VLLM_FL_INT8_KV_REPAIR=1，K_TOP4_JOINT/K_TOP4/V_ONLY三个优化标志均为0，由诊断直接调用候选。

```bash
# 在实验目录中；每个输出目录必须不存在。
/workspace/tianshu-mixed-env-20261109/bin/python \
 source/benchmarks/diagnostics/benchmark_k_static_protect_fold_v2.py results-02 \
 /workspace/experiments/tianshu-k-joint-20261116-01/original-vonly-source/vllm_fl/attention/v_only.py 20261207
/workspace/tianshu-mixed-env-20261109/bin/python \
 source/benchmarks/diagnostics/validate_k_static_protect_fold_real_v2.py \
 /workspace/experiments/tianshu-real-qkv-20261113-01/tensors \
 /workspace/experiments/tianshu-k-static-protect-fold-cpu-20261116-01/results/calibration.json real-02
/workspace/tianshu-mixed-env-20261109/bin/python \
 source/benchmarks/diagnostics/benchmark_k_static_protect_fold_v4.py confirm-01 \
 /workspace/experiments/tianshu-k-joint-20261116-01/original-vonly-source/vllm_fl/attention/v_only.py \
 20261208 selection.json
```

results-01/02、real-01/02、confirm-01及run*.log保留原始过程。本机归档 experiment-artifacts/2026-11-16-k-static-fold-gpu-01/evidence.tar.xz，SHA256 9da570c4bcd1aaeff8a53c2fac5020ea013aed23752dca1db33bba03d4ebd178，两端一致。归档中最终候选、确认脚本、真实回放脚本、参数表与本地逐字节核对一致；源张量沿用独立数据归档，不重复存储。
