# Down Split-K 预取与 Separate 转换调优

## 1. 基本信息与摘要

2026-11-10，MetaX C500。用户明确批准 Down Split-K 提前加载 INT8、使用时转换实验，以及 Separate 转换与 scale 分块调优。沿用当前 Down-only 组合的提供者，新增 `benchmarks/diagnostics/metax_down_pipeline.py` 和 `benchmark_metax_down_pipeline.py`。最终仅将已验证的 M2048 Down 转换 block4096 接入 `_large_m_separate`，其余形状仍为1024；Split-K和scale不改。未提交或推送。

## 2. 问题与原因分析

最新 Down-only Decode 窗口中 Split-K partial 比 reduce 更重。Gate/Up 原型已证明提前加载 INT8、稍后转换可改善其完整调用，但 Down K 分片长度不同，收益必须重新验证。Separate 大 M 的转换与 scale 分别产生额外访存和启动，先检验现有 kernel 分块配置的改善空间。

## 3. 具体实现

两种 partial 原型均在单 kernel 内提前一块加载 INT8，dot 使用当前块转换后的 BF16、FP32累加；分别沿原 K×N 布局加载、N×K 加载后转置。保留当前归约 kernel，完整调用包含 FP32 workspace 分配、partial、reduce/scale 和 BF16 输出。M56/64、六组配置，Split8为当前对照，Split16作为有限候选。

Separate 在 M2048/N2048/K6144 使用注册 BF16 MM 和已固化 wide 配置，转换/scale 分别测试 block256/512/1024/2048/4096、warps4/8；各选初筛前两项重新组合后比较完整调用。没有持久 BF16 权重缓存，运行时转换留在计时内。

## 4. 实验与验证

远端 `/data/down-pipeline-20261110-03/`。启动前 GPU 无进程、826MiB、9031 bind成功；Python `/opt/conda/bin/python`，MACA `/opt/maca`，复用 `/data/downonly-hotspots-20261110-01/overlay`，其中 BF16 MM 与上轮 Down-only 采样一致。保留源文件、环境、哈希和进程状态。

两随机种子，候选初筛后领先四项与基线五轮交错复测，并做侧流图重放。Kernel相对量化参考L2门槛0.005；Separate对照包含BF16中间舍入，不能混为融合路径参考。第一轮102条初筛均通过预设数值门槛，最大L2约0.00166336。Split-K两种预取均未稳定超过原配置：原M56约62.51/63.84μs，原M64约65.91/65.33μs，保留原Split8。

转换block4096、warps4，两种子约43.70/41.52μs，对照block1024约58.59/56.80μs。第一轮完整Separate候选约461.37/459.58μs，对照480.38/479.04μs。之后对转换和scale单独消融，五轮交错结果：

| 配置（转换block/warps；scale block/warps） | seed20261110 μs | seed20261111 μs |
|---|---:|---:|
| 原1024/4；1024/4 | 477.210 | 479.235 |
| 仅转换4096/4；1024/4 | 461.637 | 462.340 |
| 转换4096/4；scale2048/8 | 460.053 | 459.508 |
| 仅scale1024/4；2048/8 | 475.793 | 475.265 |

只改转换完整调用延迟降低约3.26%/3.52%；单独scale低于1%，组合相对只改转换也低于1%，因此仅接入转换调整。图重放只改转换为476.982/478.339μs，原为490.738/493.647μs，方向一致。图重放每候选一组计时，不夸大稳定性。完整调用改善不能由各阶段耗时相加替代。

接入后 `_large_m_separate` 两组M2048和一组M65回退，eager/图重放均与原Separate舍入参考逐位一致；M2048实际函数完整调用约462.04/461.90μs。验证输出落盘后进程保留GPU分配，核对PID与命令后单独停止，清理状态另行归档；模型精度和官方双场景评测未做。

## 5. 与已有改动的关系

基于已批准 Down-only 三算法路径，不改变其他投影量化范围。复用 Gate/Up 的预取思路，但不沿用其性能结论；不重写原生BF16 MM。旧记录和配置保留对照。

## 6. 合规性与风险

W8A16 per-output-channel INT8权重、FP32 scale、BF16激活与dot输入、FP32累加、BF16输出。不是原生INT8 GEMM。K分片、N整除等能力条件沿用，不提前放宽。未更改官方serve/Benchmark参数和数据集。有限实测范围外继续原配置；没有端到端验证不宣称吞吐提高。

## 7. 复现方式

```bash
PYTHONPATH=/data/downonly-hotspots-20261110-01/overlay \
python /data/down-pipeline-20261110-03/benchmark_metax_down_pipeline.py \
  --output <不存在的新目录>
```

同目录提供 `metax_down_pipeline.py`。`screen.jsonl` 保留失败与初筛，`finalists.jsonl` 保留五轮原始值、median、图重放和数值误差。SSH新连接认证中断后通过已有会话分块传输，传输完成后重新AST核对并记录sha256；没有将连接信息或密码写入源码/报告。

本机原始计时：`artifacts/down-pipeline-20261110-03/screen.jsonl`、`finalists.jsonl`、`verify-results/finalists.jsonl`、`integrated.json`；复现脚本verify_separate.py/check_integrated.py同目录归档。远端完整归档 `/data/down-pipeline-20261110-03-results.tgz`，最终清理追加记录为gpu-final-clean.txt/cleanup.txt。新增生产修改仅FlagGems `_large_m_separate` 的该精确形状转换分块；其余已有未提交改动保留，报告不表示整个工作树均由本次产生。
