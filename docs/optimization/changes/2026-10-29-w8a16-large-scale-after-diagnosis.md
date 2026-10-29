# W8A16 大 M scale-after-dot 独立路径诊断

## 1. 基本信息与摘要

日期：2026-10-29；目标平台：MetaX C500。修改 `benchmarks/diagnostics/benchmark_w8a16_production_int8pack.py`，新增 `--large-m` 诊断模式；在 FlagGems `_metax/ops/weight_int8pack_mm.py` 接入大 M 独立路径，在 `tests/unit_tests/quantization/test_w8a16_down.py` 新增分派、数值及图重放验证。目标是测清完整开销后统一生产入口。

## 2. 问题与原因分析

改动前生产实现仅小 M 采用优化 Split-K，其他 M 调用通用融合。历史独立反量化将 scale 乘入权重，与当前融合的 scale-after-dot 存在 BF16 舍入位置差异，不能直接移用历史性能结果。新候选额外读取并写回输出，必须测量完整调用。

## 3. 具体实现

保持默认诊断行为；新增 M=128/256/512/1024/2048/4096、N=2048、K=6144 的测试。完整路径包含 BF16 MetaX mm、原独立反量化、通用融合、新独立 scale-after-dot。新候选每次转换 INT8 权重为 BF16，调用现有 MetaX mm，再用独立 Triton kernel 在 FP32 中乘 per-channel scale 并输出 BF16。没有持久 BF16 权重缓存。额外记录转换、GEMM、scale 分量耗时，分量不能直接替代完整调用耗时。

独立 GEMM 先输出 BF16，再做 scale；融合实现先对 FP32 累加器做 scale，再输出 BF16。诊断显式保留并测量此差异，不宣称逐位等价。FP32 参考为 `(X.float() @ Q.float().T) * scale`，最终转换 BF16。

## 4. 实验与验证

本机 Python 语法编译通过（缓存写至 `/tmp/w8a16-pycache`）。GPU 正确性及两轮诊断、统一入口计时均已完成，详见下方执行记录；所有 GPU 任务串行执行。计时沿用三轮交错顺序、warmup=100ms、rep=300ms、轮次中位数，原始数据逐项保存。

## 5. 与已有改动的关系

延续 `2026-10-28-w8a16-int8pack-production.md`。保留小 M 配置，不重扫参数。第二轮补测通过后接入 `weight_int8pack_mm`：原小 M 分支优先；M≥128、N 为 64 倍数、K 为 1024 倍数且输入满足 BF16/INT8/FP32 scale 与连续布局时使用大 M 独立路径，其余保留通用融合。分派不包含评测长度或固定层号。转换与 scale kernel 均 BLOCK=1024，复用 MetaX `mm`，不保留 BF16 权重缓存。上层 opaque custom op 不变。

## 6. 合规性与风险

新增 GEMM 诊断并修改 MetaX 生产分派，不改官方命令、数据、启动参数。权重 INT8、scale FP32、激活 BF16；范围为 Down shape。额外舍入已通过算子误差检查；临时缓冲区及输出读写已计入完整耗时。已确认相对通用融合的算子收益，尚未完成正式验收。按用户指令，当前 scale-after-dot 模型精度合格后，后续组合实现不重复完整精度测评，但须通过算子正确性检查。

## 7. 复现方式

在模型评测结束、归档结果、停止服务并确认 GPU 资源释放后，加载已有环境：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
cd /workspace/vllm-plugin-FL
python benchmarks/diagnostics/benchmark_w8a16_production_int8pack.py \
  --large-m \
  --output-dir /workspace/benchmark_results/w8a16-large-scale-after-20261029-01
```

结果为 `metadata.json`、`raw.json`、`summary.json`。

### 执行进度

模型精度已达到 98.1%（103/105），105 请求均成功、退出码 0。服务停止后 GPU 无进程，826 MiB 驱动保留，端口 9031 已关闭。

第一轮 `w8a16-large-scale-after-20261029-01` 使用 PyTorch `q.to(BF16)`，完整执行退出码 0。M=128/256/512/1024/2048/4096 的新路径耗时分别为 254.976/309.504/369.408/481.280/766.464/1334.016 µs；转换约 186 µs，输出 scale 约 13.824～43.264 µs。M256 新路径略慢于通用融合 295.680 µs，说明不能假设独立路径必然更快。相对 FP32 scale-after-dot 参考，新路径 relative L2 约 0.00282（含 BF16 GEMM 输出舍入），旧独立反量化约 0.00252，通用融合约 0.00002；现有 assert_close 门槛均通过。

依据转换分量瓶颈，第二轮改为独立 Triton INT8→BF16 转换（BLOCK=1024），其他路径、形状及计时方式保持一致，结果目录 `w8a16-large-scale-after-20261029-02`，完整退出码 0。第一轮脚本保留在 `w8a16-large-scale-after-preflight-20261029-01/script.py`，不覆盖原始结果。

第二轮主要完整耗时（µs）：

| M | BF16 | 原独立反量化 | 通用融合 | 新独立 scale-after-dot |
|---|---:|---:|---:|---:|
|128|77.312|122.112|281.344|124.416|
|512|186.880|231.936|517.888|237.312|
|1024|301.568|343.552|759.040|353.792|
|2048|576.256|620.288|1224.192|635.648|
|4096|1136.384|1174.528|2394.112|1202.688|

转换约 55 µs，输出 scale 约 14～42 µs。上述数据是独立算子结果，不代表端到端吞吐。数值检查通过，relative L2 仍约 0.00282；新独立路径存在 BF16 中间结果舍入，未声称与通用融合逐位相同。

组合回归最终 `22 passed in 34.40s`，包含大 M 数值、分派、CUDA Graph 重放与动态 M 编译。首次回归 21 通过、1 失败：旧舍入样例要求所有 M 都区别于 scale-before-dot，与大 M 新的 BF16 中间结果契约不符；修正为验证明确中间结果后通过，原失败日志保留。

实际生产入口完整 microbench 退出码 0：M=64/128/512/2048 分别为 83.200/125.696/237.568/634.368 µs；对应通用融合为 178.944/280.576/518.912/1235.712 µs。生产 custom op 和直接 FlagGems 入口数值及计时一致，误差检查通过。远端目录 `w8a16-unified-microbench-20261029-01`。

新增 `benchmarks/diagnostics/run_w8a16_official_comparison.py` 仅作串行任务控制：依次启动 BF16、候选新服务，原样调用官方命令，完整保留四轮和后三轮平均；两模式之间停止本任务服务并核对资源。记录源码指纹、环境、进程、GPU、健康、覆盖和原始 CSV；失败则不继续下一模式，不修改官方 Benchmark 实现。

重新连接后主机名为 `c1a85143a091`（此前 `09714730460a`），文件与结果仍在；内存 128 GiB、软件版本、仓库提交及官方脚本/量化 kernel 指纹已核对一致。官方任务目录 `/workspace/benchmark_results/w8a16-official-comparison-20261029-01`，控制日志 `/workspace/benchmark_results/w8a16-official-controller-20261029-01/run.log`。

最新核对：该次 BF16 启动被覆盖保护检查阻止，控制脚本报 `Unexpected quantization coverage 0`。其中 0 是 int8pack 模式计数，并不表示未量化：BF16 环境仅清除了变量，但插件 `VLLM_FL_W8A16_DOWN` 默认值为 1，因而触发 `W8A16 enabled` 检查。服务已由 finally 正常停止，官方 benchmark 未开始，无有效 BF16 或候选成绩。需将 BF16 三个开关明确设为 0 后，在新实验目录重新启动，旧失败证据保留。尚未完成正式吞吐/TTFT 验证。

控制脚本本地已修正为显式设置三个开关为 0，候选模式再覆盖启用值；Python 语法检查通过，尚待同步及重新启动。未修改官方服务或 Benchmark 参数。

用户随后明确要求先 W8A16、再 BF16：控制脚本已同步修复版本并调整为 `candidate → bf16`，远端语法检查通过。启动前核对无服务、评测客户端或独立 GPU 测试进程，GPU 利用率 0%、无 GPU 进程、826 MiB 驱动保留。新任务目录为 `/workspace/benchmark_results/w8a16-official-comparison-20261029-02`，控制日志位于 `w8a16-official-controller-20261029-02/run.log`。旧 `-01` 失败目录保留；新任务已启动，完成状态以原始日志及 CSV 为准。
