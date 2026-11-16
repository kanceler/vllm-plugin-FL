# 天数大 M 获选配置接入 Runtime Dispatch

## 1. 基本信息与摘要

2026-11-13，Iluvatar BI-V150。用户明确授权“接入”后，将最新四类投影 M2048/4096 共 8 个获选完整 Separate 配置加入 Runtime。沿用 `perf/tianshu-w8a16-boundaries` 分支。

涉及 `vllm_fl/quantization/w8a16/hotspot.py`（新增）、`policy.py`、`cache.py`、`dispatch_manifest.json`，以及诊断脚本 `benchmarks/diagnostics/validate_tianshu_large_m_dispatch.py`。四个独立量化开关均仍默认关闭，不据此宣称模型已启用或取得吞吐收益。

## 2. 问题与原因分析

此前大 M 联合搜索完成，但 Runtime 尚使用早期配置。仅复制 tile 参数无法启用 GROUP_M 与整除快路径，需要将被测 kernel、转换流程和最终配置一起接入。原先“冻结”由本次用户授权明确解除，仅替换指定 8 个点；其余采样点、范围、fallback 和小 M 策略保留。

## 3. 具体实现

- 新增 `hotspot.py`，JIT 函数 AST 与已测 `tianshu_hotspot_kernels.py` 完全一致；新增 `run` 包装保留 INT8→BF16 转换、输出和中间 buffer 分配、MM＋scale 完整调用。
- 新算法标签 `separate_hotspot`，policy 精确匹配 N/K、M、版本后才调用。未修改通用 Dense、GEMV、Split-K 或 Separate。
- `GROUP_M`、`FAST_STORE`、转换布局与配置来自 `2026-11-13-tianshu-hotspot-joint-01/results/summary.json` 的最终量化获选项，未使用初稿或择优拼接计时。
- 新模块加入版本指纹与编译缓存 identity，缓存 namespace 升为 `tianshu_w8a16_four_projection_v3`。
- 物理持久权重仍为 INT8 `[N,K]`，scale FP32 per-output-channel。每次调用临时转换为 BF16；KN 是临时 BF16 布局，没有持久 BF16 副本或加载时 INT8 转置。BF16 激活、BF16 dot、FP32 累加、scale 后 BF16 输出。

## 4. 实验与验证

静态核对：8 个变化均为四类投影的 M2048/4096；其他精确点和全部 ranges 逐项相同。新增 kernel 与被测函数 AST 一致，所有 manifest 内源文件 hash 匹配，Python 语法和 `git diff --check` 通过。

隔离容器验证目录 `/workspace/experiments/tianshu-large-m-runtime-20261113-01`；本地证据 `experiment-artifacts/2026-11-13-tianshu-large-m-runtime-01/`。使用全新源码副本与新 Python 进程，依次测试 0000、0100（Gate/Up-only）、1111 的层选择与既有运行路径，之后测 8 个大 M 的实际 opaque-op runtime graph 和原生 BF16 对照。没有并发 GPU 负载。

运行时结果：全部通过。0000、0100、1111 均有 COMPLETE；全开回归涵盖 67 个实际 N/K 的 M 点及更新输入 graph、非连续输入/bias、动态 compile/空输入/dtype。8 个大 M 均精确选中 separate_hotspot，每点 5 次 graph 配对计时，额外覆盖缩放输入后的 replay 与恢复输入。

| 投影 | M | 实际 runtime graph μs | 原生 BF16 μs | 延迟变化 |
|---|---:|---:|---:|---:|
| down | 2048 | 629.508 | 571.458 | +10.16% |
| down | 4096 | 1175.806 | 1119.108 | +5.07% |
| gate_up | 2048 | 1307.827 | 1428.687 | -8.46% |
| gate_up | 4096 | 2482.519 | 2840.440 | -12.60% |
| qkv | 2048 | 307.583 | 297.567 | +3.37% |
| qkv | 4096 | 523.050 | 510.319 | +2.49% |
| o | 2048 | 229.588 | 215.538 | +6.52% |
| o | 4096 | 421.296 | 404.050 | +4.27% |

数值阈值 L2≤0.005、normalized max≤0.01 保持不变。复测包含 runtime op 包装，不能与独立 kernel 的绝对耗时当作完全相同口径。

## 5. 与已有改动的关系

依赖 `2026-11-13-tianshu-large-m-joint-micro.md` 的最终候选和 `2026-11-13-tianshu-four-projection-boundaries-dispatch.md` 的 Runtime/开关框架。原 `frozen-dispatch.json` 留作历史，不覆盖，不能继续用于要求这 8 点不变。新的接入记录说明本次变化。现有无关用户修改未触碰，未提交或推送。

## 6. 合规性与风险

仅接入算子和离线固化配置，没有修改官方启动参数、评测数据或 Benchmark。没有模型级精度、官方 4K/16K 吞吐/TTFT、实际 TP 通信或全模型层覆盖验证。Gate/Up-only 只是可选运行组合，四个开关默认关闭；真正切换组合需新服务进程。

同优化 BF16 依旧可能比完整量化更快，因此历史量化对原始 BF16 的算子收益不能直接作为端到端或量化独有收益。大 M 以外维持既有性能策略，不将本次结果外推到任意 M。

## 7. 复现方式

远端 `run.sh` 完整记录串行流程。沿用 `/workspace/tianshu-mixed-env-20261109/bin/python`、Torch 2.10.0、Triton 3.6.0、原 MM provider SHA256 `fd120991ed1c5cab988d02cbdb1d97ccde1e9e2eedad55affe90321d875f32ed`。初始化 PATH/LD_LIBRARY_PATH，实际 import 优先 isolated source 与固定 gems。无需改共享安装。

```bash
# 在新的隔离实验目录运行，输出目录不覆盖历史。
PYTHONPATH=source:/workspace/experiments/tianshu-boundaries-20261113-01/gems \
/workspace/tianshu-mixed-env-20261109/bin/python -u \
validate_tianshu_large_m_dispatch.py runtime-new \
/workspace/experiments/tianshu-hotspot-joint-20261113-01/results/summary.json
```

正式部署需要同步这些源码和 manifest 到实际框架 import 路径；本次仅同步至隔离副本。禁止把隔离验证说成共享服务已经启用。

初次隔离副本遗漏旧 Split-K 文件，在策略 hash 检查时退出，尚未执行 GPU 数值测试；补齐完整本地包后重新开始，原日志保留在 setup-incomplete。最终全部本地 Python 源码及 manifest 与远端被测副本逐字节匹配。归档 SHA256 `7f7995b548fc154cc1b37eb315ad6990b8b0d8ffa3a65fb71ca1aa1cd4125089`；结束 GPU 无进程、70MiB 驱动保留。
