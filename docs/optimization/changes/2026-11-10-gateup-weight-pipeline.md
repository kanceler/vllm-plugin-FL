# Gate/Up 权重加载与转换流水线实验

## 1. 基本信息与摘要

2026-11-10，C500；续做用户已批准的 Gate/Up M56 W8A16 优化。沿用 `perf/gemm-w8a16-shape-dispatch`。新增 `benchmarks/diagnostics/metax_gateup_pipeline.py`、`benchmark_metax_gateup_pipeline.py` 与 `validate_metax_gateup_pipeline.py`。保留其他已有修改，没有生产接入、提交或推送。目标是定位并改善 INT8 加载、转换、dot 的重叠，而非将转换移出计时。

## 2. 问题与原因分析

上轮同 tile 诊断显示融合 INT8 约 88 μs，预转换 BF16(Q) 约 64 μs；IR 表明两者权重缓冲不同。进一步检查上轮显式预取 IR：loop iter_args 携带 A/B 寄存器张量，循环内 local_alloc，没有形成权重侧持久双缓冲。因此上轮约 2% 改善不能解释成已恢复双缓冲。

本轮检验两个独立因素：提前多少个 K tile 加载，以及跨迭代携带 INT8 还是已转换 BF16。更深预取可能隐藏延迟，也可能增加活跃寄存器与 shared 转换成本；不能只根据源码中的预取次数推断性能。

## 3. 具体实现

三个实验 Kernel：depth1_raw（提前一块加载 INT8、使用时转换）、depth2_converted（提前两块转换 BF16）、depth2_raw（提前两块加载 INT8、使用时转换）。均在一个 Kernel 内完成 INT8→BF16、BF16 dot、FP32 累加、输出乘 FP32 scale 后 BF16 写回。K 累加顺序保持；首部预加载、循环推进和尾部收尾显式实现。

形状 M56/N12288/K2048，六组 tile；两个种子，每个合法候选五轮交错计时，另做侧流 CUDA Graph replay。每组包含注册原生 BF16、原 Dense 和上轮预取最佳配置对照。生产分派保持原状；不将实验原型当作模型优化已启用。

## 4. 实验与验证

远端 `/data/gateup-pipeline-20261110-02/`；GPU 启动前无进程、826 MiB，9031 bind 成功。Python `/opt/conda/bin/python`，使用前轮已归档的隔离 overlay；记录两仓库 head/status/diff、环境、源文件、命令和日志。测试之间串行，未启动模型服务。

预设相对量化参考 L2<0.005，原 BF16 单独对原始权重参考；固定两种子不是修改官方命令。AST 校验通过。第一轮三个变体共 42 条初筛记录（含对照），优胜为 depth1_raw，配置 64/64/64、8 warps、4 stages。五轮交错 eager 中位数：

| seed | 原生 BF16 μs | 原 Dense μs | 上轮预取 μs | depth1_raw μs |
|---|---:|---:|---:|---:|
| 20261110 | 65.43 | 89.11 | 86.23 | 81.01 |
| 20261111 | 65.14 | 89.04 | 86.26 | 81.13 |

相比原 Dense 延迟降低约 9.1%/8.9%，相比上轮预取降低约 6.1%/5.9%，仍比 BF16 慢约 24%。此表来自同一实验，不用不同轮次最佳值拼接收益。

IR 确认 depth1_raw 跨迭代携带 INT8，sitofp 作用于上一块数据；上轮预取跨迭代携带 BF16。两者均 98 registers、24832 字节 shared、0 spills，因此不能将收益解释成已证明的寄存器减少或持久双缓冲恢复。

进一步新增四个只提前 B、保持 A 常规循环加载的变体，两个种子中领先约 82.03/82.88 μs，未显示优于 depth1_raw 的依据。提前两块 A/B 的某些配置超过 65536 字节 shared 上限，失败原样保留。

对 depth1_raw 局部搜索 24 组配置（BN64/128、BK32/64、warps4/8、stages1/2/4），两种子五轮交错复测后仍是 64/64/64、8/4 较优，约 82.70/81.23 μs。不是全局最优。尾块、非连续输入、零 scale 通道及图重放验证 16/16 通过，最大相对 L2 0.00170357。包括 K=1/63/65/129/193/2048、M=1/55/56/57/64/65/128/2048 的八组明确 shape，各连续与 stride=2 激活两种情况；不能据此声称覆盖全部组合或完整模型精度。

三个性能任务分别记录 42/54/54 条初筛，第一组有 8 条 shared memory 资源失败，后两组无失败；所有合法候选图重放均通过。任务全部结束，最终 GPU 状态已归档。原始归档 `/data/gateup-pipeline-20261110-02.tgz`；本机下载连接中断，尚未确认本机副本，不能称本机归档完成。

本机最终诊断入口移除了从上一轮复制而未使用的 `--prefetch` 和 `--schedule-scan` 选项，修正了旧 metadata 说明；已运行的原始 driver 版本保留在远端 source/ 和 refine.py，未追溯修改日志。

## 5. 与已有改动的关系

继承上一轮 `2026-11-10-gateup-layout-prefetch.md` 的诊断，不覆盖数据或旧 Kernel。与纯 BF16 专属配置分支无修改关系。上轮触发编译器断言的 fullstage+prefetch 不作为当前候选。

## 6. 合规性与风险

仍为 per-output-channel 静态对称 INT8 权重、FP32 scale、BF16 激活/输出、FP32 累加。没有原生 INT8×INT8 dot，也未持有正式 BF16 权重副本。所有运行时转换包含在完整调用内。没有改 serve、Benchmark、模型行为或评测数据。有限候选只能确定所测范围内较优；没有模型精度和官方双场景验证，不宣称全模型收益。寄存器/IR 只作编译证据，不等于硬件占用率实测。

## 7. 复现方式

```bash
PYTHONPATH=/data/metax-m56-pipeline-20261110-01/overlay \
python /data/gateup-pipeline-20261110-02/benchmark_metax_gateup_pipeline.py \
  --output <不存在的新结果目录>
```

同目录放置 `metax_gateup_pipeline.py`、上一轮 `metax_gateup_prefetch.py`。`results/results.jsonl` 保存初筛与失败，`finalists.jsonl` 保存全部合法候选五轮原值、median、图重放和误差；每种 kernel/config 保存资源 metadata 和 IR。

追加运行参数：`--weight-only-lookahead` 测只提前权重的四变体；`--refine gateup_depth1_raw` 测 24 配置。边界脚本 `validate_metax_gateup_pipeline.py --config 64 64 64 8 4 --output <新目录>`。
