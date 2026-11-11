# W8A16 `weight_int8pack_mm` MetaX 接入与小 M 特化

> 最新状态（2026-10-29）：128 GiB 实例已完成模型启动，42/42 Down 启用，完整 Level 3 精度 98.1%（103/105）；统一小 M/大 M 入口回归 22 项通过，生产 microbench 退出码 0。大 M 最新分派及数据见 [后续报告](2026-10-29-w8a16-large-scale-after-diagnosis.md)。本报告前述 21 项测试和通用大 M 分派属于初始版本。

## 1. 基本信息与摘要

- 日期：2026-10-28
- 设备：单卡 MetaX C500 64 GiB，MACA 3.7.0.38，驱动 3.9.35
- FlagGems 基线提交：`a7620cc191a0b42e040194622c5758b22a7a25dc`
- vllm-plugin-FL 基线提交：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`
- 量化范围：MiniCPM5-2B Down-only；加载时静态、对称、per-output-channel INT8 权重，FP32 scale，BF16 激活和输出
- 数值顺序：BF16 激活与 BF16 表示的 INT8 值做 dot，FP32 累加；完成 K/Split-K 归约后乘 FP32 channel scale，最后转 BF16

本次将诊断阶段已验证的小 M Split-K 和整除特化接入 FlagGems 的 MetaX `weight_int8pack_mm` 后端覆盖，并在插件中增加 opaque custom op 调用。`M=1/16/32/64`、`N%64=0`、`K%(128*8)=0` 时使用两阶段 Split-K；其他形状继续使用 FlagGems 通用 scale-after-dot 实现。旧手写 scale-before-dot kernel 保留作历史对照，不作为新路径。

涉及文件：

- `FlagGems/src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py`
- `FlagGems/src/flag_gems/runtime/backend/_metax/ops/__init__.py`
- `vllm_fl/quantization/w8a16/int8pack.py`
- `vllm_fl/quantization/w8a16/linear.py`
- `tests/unit_tests/quantization/test_w8a16_down.py`
- `benchmarks/diagnostics/benchmark_w8a16_production_int8pack.py`

## 2. 问题与原因分析

通用 `weight_int8pack_mm` 已符合 scale-after-dot 数值顺序，但其候选配置以 `BM=32/64/128` 为主，真实 Down shape `N=2048,K=6144` 的小 M 并行度不足。前序诊断已经确认：两阶段 Split-K=8、`BN=64/BK=128/warps=4/stages=2/basic` 在 M=1/16/32 有效；M=64 使用 BM=64 更合适；整除特化进一步减少 K 分片控制及 N/K 边界检查。

模型侧原实现只有两条路径：独立反量化后调用 BF16 GEMM，或者旧手写 scale-before-dot 融合 kernel。二者都不是本轮确定的 FlagGems scale-after-dot 路径，因此需要新的 opaque custom op，避免 Dynamo 将运行时 token 数 M 固化，同时保留 RowParallelLinear 的加载、通信和 TP 语义。

大 M 的问题尚未解决。已有 MetaX NT 诊断表明，INT8 权重加载、BF16 转换及流水线 lowering 仍是主要限制；本轮不接入已实测无收益的 NT 候选。

## 3. 具体实现

MetaX 后端新增同名 `weight_int8pack_mm` 覆盖并由 `_metax.ops.__init__` 导出，使 `SpecOpRegistrar` 将 `flag_gems.weight_int8pack_mm` 替换为后端函数。运行时分派规则只依赖矩阵形状、dtype 和布局，不依赖评测长度、层号或数据集：

```text
M in {1,16,32,64}
and N % 64 == 0
and K % 1024 == 0
and A is contiguous BF16
and B is contiguous INT8
    -> MetaX two-step Split-K=8
otherwise
    -> generic FlagGems weight_int8pack_mm
```

partial kernel 分配 FP32 `[8,M,N]` workspace，每个 K 分片完成 BF16×BF16 dot 和 FP32 累加。reduce kernel 合并 8 份 FP32 partial，再乘 FP32 `[N]` scale 并输出 BF16。INT8 值在 [-128,127] 内可由 BF16 精确表示，因此转换本身不引入权重整数误差。

插件新增 `fl_w8a16_down_int8pack` opaque custom op，将任意 leading dimensions 展平为二维后调用 FlagGems 公共入口，再恢复原形状。新开关为 `VLLM_FL_W8A16_DOWN_INT8PACK=1`；它优先于历史 `VLLM_FL_W8A16_DOWN_FUSED`。当前 FlagGems 算子没有 bias epilogue，因此带 bias 输入明确回退独立反量化，避免在 BF16 GEMM 输出后再加 bias 引入额外舍入。默认仍不开启新开关，等待模型精度与大 M 性能完成。

## 4. 实验与验证

目标环境中 `flag_gems.weight_int8pack_mm.__module__` 实测为 `_metax.ops.weight_int8pack_mm`，证明后端覆盖实际生效。W8A16 单元测试完整运行 21 项，结果 `21 passed in 43.43s`，包括：

- 独立反量化、历史融合、新 int8pack 三种 Down adapter；
- scale-after-dot 单项舍入边界；
- M=16/32/64 特化和 M=128 通用回退；
- 四种 M 共用一个动态模型图；
- 历史 W8A16 正确性和 CUDA Graph 回归。

权威性能数据为远端 `/workspace/benchmark_results/w8a16-int8pack-production-20261028-03`，本地归档在 `artifacts/w8a16-int8pack-production-20261028-09/microbench/`。每项为三轮 GPU latency median 的中位数，warmup 100 ms、rep 300 ms；单位为微秒：

| M | 小 M 特化 | 生产 custom op | 完整独立反量化 | BF16 MetaX | 相对完整独立反量化 |
|---:|:---:|---:|---:|---:|---:|
| 1 | 是 | 40.704 | 87.040 | 41.216 | -53.24% |
| 16 | 是 | 45.312 | 86.528 | 41.472 | -47.63% |
| 32 | 是 | 62.464 | 95.488 | 50.176 | -34.58% |
| 64 | 是 | 83.456 | 116.736 | 72.192 | -28.51% |
| 128 | 否 | 285.696 | 125.184 | 79.872 | +128.22% |
| 512 | 否 | 521.728 | 235.520 | 186.880 | +121.52% |
| 2048 | 否 | 1228.544 | 626.688 | 581.632 | +96.04% |

生产 custom op 相对直接 FlagGems 后端调用只增加约 0.26/0.26/1.54/1.02 微秒（M=1/16/32/64），封装不是主要成本。scale-after-dot 参考的 relative L2 error：M=1 为 0；M=16/32/64 为 `6.22e-5/6.81e-5/7.36e-5`；M=128/512/2048 为 `1.53e-5/1.91e-5/2.54e-5`，全部通过 `rtol=0.02, atol=0.125`。

初始模型启动受 48 GiB 主机内存限制，后续已换用 128 GiB 实例成功启动并完成精度测评（见末节）。旧失败尝试按沐曦官方参数启动，实例 cgroup 主机内存上限为 48 GiB。API 服务收到 SIGKILL，日志中没有 Python/Triton 异常；cgroup 的累计 `memory.events` 同时记录 `oom_kill=2`。由于启动前没有保存该计数，不能把增量逐次归因，但现象与 48 GiB 内存不足高度一致。当次残留的 EngineCore PID 已按归属停止，结束后 GPU 无进程、利用率 0%、显存 860/65536 MiB。日志归档在 `artifacts/w8a16-int8pack-production-20261028-09/model-start/`。因此本轮没有产生模型输出、math_500 分数、TTFT 或官方吞吐结果。

## 5. 与已有改动的关系

本次复用既有 Down-only 加载时量化、FP32 scale、TP local shard 和 custom op 动态图封装。小 M kernel 来自 `2026-10-27-flaggems-w8a16-splitk.md` 与 `2026-10-28-w8a16-small-even.md` 已验证候选，未重新搜索 BM/BN/BK/warps/stages。

历史 `fused.py` 仍保持 scale-before-dot，用于旧实验复现；新路径不调用该 kernel。大 M 继续调用通用 FlagGems scale-after-dot kernel，与 `2026-10-28-w8a16-large-metax-nt.md` 的失败候选分开；本次数据再次确认大 M 尚不能进入正式性能评测。

## 6. 合规性与风险

本次没有修改官方启动参数、Benchmark 实现、评测数据集、模型输出逻辑或调度参数。分派依据为通用矩阵 shape capability，不包含 4K/16K prompt 长度、math_500 题目或固定层号。优化为 FlagGems MetaX GEMM 的实际 kernel 与插件实际调用路径，符合 GEMM 量化任务范围。

最新风险与限制（历史环境阻塞已解除）：

- scale-after-dot 完整 Level 3 已达 103/105；这是大 M 通用融合版本的模型结果，组合版本按用户要求不重复完整精度评测，保留算子验证；
- 128 GiB 实例已完成官方服务启动，旧 48 GiB 启动失败不再是当前阻塞；
- 大 M 已改用独立转换、MetaX GEMM、输出 scale；实际生产入口计时通过，正式吞吐和 TTFT 尚未完成；
- 只验证 MetaX C500、Down `N=2048,K=6144` 及列出的 M；其他 Linear、TP 多卡和天数平台待验证；
- bias 输入会显式回退，正式模型验证需统计实际回退数；
- 新开关默认关闭，尚未宣称比赛验收通过。

## 7. 复现方式

算子测试：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
cd /workspace/vllm-plugin-FL
pytest -q tests/unit_tests/quantization/test_w8a16_down.py -s

python benchmarks/diagnostics/benchmark_w8a16_production_int8pack.py \
  --output-dir /workspace/benchmark_results/w8a16-int8pack-production-20261028-03
```

模型启动使用官方沐曦参数，仅设置自研实现开关：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
export VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=1
export VLLM_FL_W8A16_DOWN_INT8PACK=1
export VLLM_FL_W8A16_DOWN_FUSED=0

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

### 2026-10-29 新实例进度

**版本区分**：以下 `21 passed` 是接入大 M 独立路径之前的历史结果。最新统一入口（包含大 M 独立路径）已通过 `22 passed in 34.40s`，退出码 0；新增一项大 M 分派、数值和 CUDA Graph 重放测试。最终日志已补归档至 `artifacts/w8a16-int8pack-production-20261028-09/w8a16-unified-20261029-02/pytest.log`，生产入口计时退出码也为 0。当前实现与最终验证详见 `2026-10-29-w8a16-large-scale-after-diagnosis.md`。历史日志保留原样，不改写为 22 项。

新实例主机内存 128 GiB，设备仍为 MetaX C500。同步必要代码后，W8A16 回归测试为 `21 passed in 58.42s`。使用上述官方服务参数完成模型加载和图捕获，服务 PID 为 5145，日志显示 `Application startup complete`；42/42 Down 层记录 `mode=int8pack_scale_after_dot`，截至单请求检查未发现错误或回退日志。健康检查 HTTP 200，单请求 `17×23` 正确输出 391；这些检查不替代完整模型精度评测。

北京时间 2026-10-29 12:08 启动官方完整 math_500 Level 3 命令，12:24:39 完成，105 个样本、batch size 8，准确率 **98.1%（103/105）**，退出码 0；报告 execution_summary 显示 requested=105、succeeded=105、errored=0、incomplete=false，达到 0.950 门槛。日志目录为 `/workspace/benchmark_results/w8a16-int8pack-newhost-20261029-01/`（`server.log`、`smoke.json`、`accuracy.log`、`accuracy-report.json`），原始评测输出为 `/workspace/evalscope-datasets/level3/20261029_120823`。启动前确认无其他评测、pytest 或 microbenchmark 进程；未并行启动其他 GPU 测试。本机归档为 `artifacts/w8a16-int8pack-production-20261028-09/w8a16-int8pack-newhost-20261029-01/`。

精度已合格，采纳 scale-after-dot 和现有小 M 路线。随后补测大 M 独立转换、MetaX BF16 GEMM、输出 scale 的完整路径，并统一到 int8pack 分派。按用户指令，后续组合实现不重复完整精度测评，保留算子误差与完整性能验证。最终组合实现及 22 项回归、生产入口计时已完成；正式性能验收尚未完成。模型服务已正常停止，GPU 恢复 826 MiB 驱动保留、利用率 0%、无 GPU 进程，端口 9031 连接拒绝，之后才启动独立 GPU 补测。
