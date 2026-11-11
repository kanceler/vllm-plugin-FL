# Down 独立 Triton 反量化与融合路径 Microbenchmark 补测

## 1. 基本信息与摘要

日期：2026-10-26（UTC+8）。任务 ID：`w8a16-triton-micro-20261026-03`。用户明确同意补测独立 Triton 反量化完整路径，与 BF16 及现有融合路径对照。

修改文件：`benchmarks/diagnostics/benchmark_w8a16_fused.py`，以及本报告和上一份报告的补测链接。仅修改诊断脚本，不修改生产 Kernel、模型接入、量化算法或官方脚本。沿用 `perf/gemm-w8a16-down-fused`，提交基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，保留已有未提交文件；未提交、未推送。

结论：**九种 M 中，当前融合实现均慢于独立 Triton 反量化＋BF16 GEMM；独立 Triton 路径也均慢于原 BF16。** 此前“小 M 融合快于独立路径”的数据仅针对 PyTorch reference 对照，不能推广到独立 Triton 实现。

## 2. 问题与原因分析

前次动态编译修复任务的临时诊断脚本测量 `F.linear(x,dequantize_reference(q,scale))`，不是历史 Down-only 官方实验使用的独立 Triton 反量化。reference 写法 `(q.float()*scale[:,None]).to(BF16)` 在 eager 下产生 FP32 中间张量及多个操作，因此该数据不能代表已优化的独立反量化成本。

仓库已有诊断脚本包含 `separate_triton`，但测试 M 不完整、输出写到固定历史目录，且没有这次要求的同场三轮结果。本次明确重新执行三路径完整对照，避免混用 reference、旧统计方法或历史不同实验结果。

## 3. 具体实现

固定 N=2048、K=6144，M 为 1/16/32/64/128/256/512/1024/2048。权重量化预处理仅执行一次，不计入每次推理路径耗时：静态对称 per-output-channel INT8 权重、FP32 scale，激活/输出 BF16。

三条计时路径：

```text
bf16:            F.linear(x, original_bf16_weight)
separate_triton: F.linear(x, dequantize_weight(q, scale))
fused:           fused_linear(x, q, scale)
```

独立路径每次调用都重新分配并写出完整 BF16 权重，再调用 GEMM；不存在跨调用 BF16 权重缓存。融合路径沿用自定义算子封装及原 Triton Kernel，不改变内部 BF16 舍入位置。计时包括路径内部输出和临时权重分配的影响；测量工具是 CUDA event GPU 延迟，不能当作完整 CPU/API 端到端时间。

脚本新增必需的 `--output-dir` 参数，拒绝覆盖已有目录；每条路径每个 M 测三轮，轮换三条路径的测试顺序。每轮 `triton.testing.do_bench(warmup=100,rep=300,return_mode='median')`，最终取三轮中位数的中位数。81 条原始计时随测随保存；移除 reference 性能对照，但保留其正确性参照用途。保存 metadata 和 M=2048 的融合 TTIR/TTGIR。

## 4. 实验与验证

环境：容器 `6951ed69dce3`，单卡 MetaX C500 64 GiB、bus `0000:35:00.0`，与前次已停止融合实验相同；Python 3.12.11、PyTorch `2.8.0+metax3.7.0.7`、Triton 3.0.0、MACA 3.7.0.38、驱动 3.8.30。128 GiB RAM，swap 0。实际源码路径 `/workspace/vllm-plugin-FL`。

运行前确认无服务、评测客户端或 GPU 测试残留，GPU 无进程、显存 826 MiB、9031 空闲。显式设置 `PYTHONPATH` 和 FL 插件环境，关闭模型接入相关量化开关；诊断直接调用函数，开关关闭不影响显式三路径计算。生产源码指纹见 `preflight.txt`，诊断脚本 SHA256：`b4fcf30b66adb55fc6a299c8721ea88ad2717a0c818e290011b0bc6f30265512`。

随机 synthetic 数据 seed=42，未使用评测集。所有耗时单位为 ms；“融合/独立”为耗时比，大于 1 表示融合更慢。

| M | BF16 GEMM | 独立 Triton 反量化＋GEMM | 融合反量化＋GEMM | 融合/独立 |
|---|---:|---:|---:|---:|
| 1 | 0.030976 | 0.078080 | 0.140032 | 1.79× |
| 16 | 0.045824 | 0.092416 | 0.145152 | 1.57× |
| 32 | 0.050176 | 0.096000 | 0.144384 | 1.50× |
| 64 | 0.052736 | 0.098048 | 0.156160 | 1.59× |
| 128 | 0.058112 | 0.102656 | 0.197888 | 1.93× |
| 256 | 0.074496 | 0.118528 | 0.481536 | 4.06× |
| 512 | 0.109568 | 0.153856 | 1.096448 | 7.13× |
| 1024 | 0.196864 | 0.240640 | 1.351936 | 5.62× |
| 2048 | 0.291328 | 0.335104 | 2.498048 | 7.45× |

验证结果：

- Triton 反量化权重与 reference 逐元素完全相等。
- 九种 M 的独立路径输出与 `F.linear(x,BF16(q*scale))` 完全相等。
- 融合输出相对同一舍入权重参考的 L2 相对误差范围约 `4.21e-5～2.48e-4`，全部低于 0.005。该误差不代表相对原浮点模型的量化精度。
- 正常退出码 0；本地核对 9 个 Shape、81 条原始计时、每条路径每个 Shape 三轮及最终中位数一致；本机与远端诊断源码 SHA256 一致。
- 测试进程完全退出后，GPU 利用率 0%、无 GPU 进程、显存 826/65536 MiB，9031 可绑定；没有启动模型或官方评测。

本次独立路径相对 BF16 的差值约 0.044～0.047 ms，而前次 reference 路径的差值约 0.27 ms。二者来自不同时间和统计方法，不能直接当作严格配对反量化 Kernel 加速比；本次同场数据足以确认融合在已测九点均无优势。未测试的 M 或布局不外推。

执行异常：最初独立 SCP 会话在传输前断开，导致首次启动提示诊断文件不存在（退出码 2），未执行 GPU 计算。失败日志保留为 `transfer-failed-run.log/transfer-failed-exit.txt`。随后通过现有 SSH 会话完成传输、核对 SHA256 和语法后重新启动；正式结果为新 `run.log/exit.txt`，不覆盖或计入失败尝试。

## 5. 与已有改动的关系

复用 `dequant.py` 的 `_dequant_kernel` 和动态编译修复后的融合接口，不改生产路径。历史 10 月 23 日 Down-only 官方实验的独立 Triton 结果保留原样；前次 reference Microbenchmark 也保留原始数字，但适用结论限于 reference 对照。

这次只补齐比较对象，不进行配置搜索、同分块 BF16 控制 Kernel 实现或生产分派更改。后续调优应以本次 Triton 独立路径为对照之一，不再以 reference 小 M 收益证明融合已经有效。

## 6. 合规性与风险

独立算子诊断在官方 Benchmark 已停止且 GPU 空闲后串行执行；不改正式命令、脚本、数据集或模型行为。没有同时运行模型评测或额外 GPU 负载。`do_bench` 按安装版本内部逻辑进行 cache flush，该行为属于诊断测量，不修改官方评测 warmup 或缓存流程。

本次统计使用三轮 GPU 延迟中位数，前次默认 `do_bench` 为均值；不把两次结果之差归因于源码性能变化。数值正确性、Microbenchmark 和模型精度/吞吐分别报告；本次没有 math_500、TTFT 或完整官方成绩，不宣称通过赛事验收。生产 Kernel 调优仍未开始。

## 7. 复现方式

远端：`/workspace/benchmark_results/w8a16-triton-micro-20261026-03/`。

本地：工作区 `artifacts/w8a16-triton-micro-20261026-03/`。原始归档 `remote-evidence.tar.gz` 已下载并展开；`results/microbenchmark.json` 为汇总，`results/raw_timings.json` 为原始计时，`results/metadata.json` 为测量条件，`audit.json` 为本地完整性核对；另保留修改前脚本。

加载既有 Conda/MACA 并确认 GPU、进程和端口空闲后运行：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=0 VLLM_FL_W8A16_DOWN_FUSED=0 VLLM_FL_W8A16_OFFLINE=0
unset VLLM_FL_W8A16_CHECKPOINT
python /workspace/vllm-plugin-FL/benchmarks/diagnostics/benchmark_w8a16_fused.py \
  --output-dir /workspace/benchmark_results/w8a16-triton-micro-20261026-03/results
```

归档目录现已存在，重跑须使用新的实验 ID/输出目录，不复用上述结果路径。正式启动命令和环境见 `run-environment.txt`、`run.log`；恢复正常退出后的资源状态见 `postflight.txt`。
