# 天数Down大M量化完整调用与原生BF16对照

## 1. 基本信息与摘要

2026-11-13，BI-V150/CoreX4.5.0、Torch2.10.0/Triton3.6.0。新增 `benchmarks/diagnostics/compare_tianshu_down_native.py`，复用同日适配分支与隔离提供者，不改runtime。M2048量化快于当前原生MM；M4096退化。未提交、未推送。

## 2. 问题与原因分析

前次搜索只在量化路径内部比较，无法判断相对原BF16收益。必须使用原BF16权重、同一MM提供者版本比较。当前原生M2048仍走原调优选择，历史001热点固定配置未混入本底座；量化侧使用离线更宽搜索配置，所以不是共同tile的纯量化归因实验。

## 3. 具体实现

原生调用实际 `flag_gems.runtime.backend._iluvatar.ops.mm.mm(a,w.T)`，包含公开入口输出分配。量化调用 `backend.run(a,q,s,'separate',(256,256,32,16,2),(2048,8))`，包含分配、每次INT8转换BF16、融合scale MM。M4096额外比较转换8192/16。权重对称per-channel INT8，scale FP32，激活BF16、累加FP32，输出BF16；不是原生INT8 dot。不缓存转换权重。

## 4. 实验与验证

新seed20261116/20261117，每seed五轮随机交错。eager与完整图重放分别do_bench warmup5ms/rep50ms，汇总十次p50的中位数。编译和原生调优在计时前完成。每个seed每路径校验实现误差：FP32参考关闭TF32，L2≤0.005、normalized max≤0.01，输出有限；图输出另检L2。数值均通过，原始accuracy.jsonl保留。实现误差参考分别是原权重FP32参考和量化后FP32参考，量化误差不混入实现误差。

|M|路径|eager μs|graph μs|相对当前原生图延迟降低率|
|---|---|---:|---:|---:|
|2048|原生BF16|728.754|735.045|—|
|2048|量化转换2048/8|631.657|634.361|+13.6977%，有效改善|
|4096|原生BF16|1105.015|1118.898|—|
|4096|量化转换2048/8|1187.894|1191.975|−6.5312%，有效退化|
|4096|量化转换8192/16|1186.917|1190.671|−6.4146%，有效退化|

单位为微秒，分类依据未先四舍五入的计时比值，指标是独立算子延迟，不是模型吞吐。M2048需要注意当前原生配置遗漏已知更好tile，因此13.7%不能全部归因量化；M4096两个转换参数均无法消除退化。

第一次脚本因Triton单分位数返回float却按列表取值而退出，无完整性能结果；修复后在02新目录重跑，不覆盖01失败记录。

## 5. 与已有改动的关系

承接同日联合搜索。没有将量化内部改善直接当作原生对照。当前final/default基底与旧001固定Down基底分开管理；若要判断相对已优化原生BF16的价值，还需共同配置或原生较优配置对照。

## 6. 合规性与风险

测试前GPU无计算任务、70MiB驱动占用，归档进程；串行运行，无模型/正式Benchmark。未改官方参数、评测数据和checkpoint；仅合成输入。结束gpu-after归档。无模型42层覆盖、math_500精度或官方吞吐/TTFT结论。本轮不据此全局启用量化策略。

## 7. 复现方式

从 `/workspace/experiments/tianshu-down-adapt-20261113-01` 执行：

```bash
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python source/benchmarks/diagnostics/compare_tianshu_down_native.py /workspace/experiments/tianshu-down-native-compare-20261113-02
```

结果在02目录raw.jsonl、accuracy.jsonl、summary-2048/4096.json，日志为相邻02.log。实际代码及hashes-final/runtime已归档，本机完整证据 `experiment-artifacts/2026-11-13-tianshu-native-compare/tianshu-down-native-compare-20261113.tgz`。复现用新输出目录，脚本要求不存在该目录。
