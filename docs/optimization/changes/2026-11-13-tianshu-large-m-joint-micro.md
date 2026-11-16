# 天数 M2048/4096 完整 Separate 联合优化测评

## 1. 基本信息与摘要

- 日期：2026-11-13；平台：Iluvatar BI-V150。
- 授权范围：四类投影的 M=2048/4096，完整量化调用与原生 BF16 比较，寻找所测候选内较优配置。
- 分支：`perf/tianshu-w8a16-boundaries`，延续同一测评任务；不修改已冻结 dispatch，不替换公共 MM 提供者，不改官方命令。
- 新增 `benchmarks/diagnostics/tianshu_hotspot_kernels.py`、`benchmark_tianshu_hotspot_joint.py`；分析与原始结果位于工作区 `experiment-artifacts/2026-11-13-tianshu-hotspot-joint-01/`。
- 当前状态：最终版本全部 8 个热点完成，24 个边界检查和 640 条确认计时完成。本报告不构成模型级验收。

## 2. 问题与原因分析

Scale 融合减少输出中间读写，但不能消除 Separate 的 INT8→BF16 转换和临时 BF16 权重写入。已有分组版本尚未在全部四投影、两个大 M 上与布局/配置进行本轮局部联合搜索。

现有大 M 已显式设置 `SPLIT_K=1`，因此本次没有“大 M 去 atomic”的收益。NK 权重的转置视图 stride 为 `(1,K)`，不触发双 stride 大于 1 的 contiguous；KN 转换是显式实验选项，不是 MM 隐式物化。

## 3. 具体实现

诊断 kernel 从固定 Iluvatar `mm_kernel_scale_grouped` 函数体派生，保留累加和 scale 数值顺序：

1. `GROUP_M=0` 保持行优先 CTA 顺序；其他值使用组内重排，包括最后不满的一组。
2. `FAST_STORE` 配合显式 `EVEN_M/EVEN_N: tl.constexpr`，仅在证明整除时去掉对应 store mask，其他情况保留 mask。
3. `APPLY_SCALE=False` 提供同实现的纯 BF16 控制组，区分通用 MM 优化与量化收益；原 provider 的 BF16 kernel 仍是主基线。
4. 所有路径 `SPLIT_K=1`；未改 K 尾块公式。新 kernel 只接受本实验 BF16 MM 输入，不扩展 INT8 尾块能力。

完整量化计时为：分配 BF16 buffer → NK 连续转换或 KN 转换加转置 → 分配输出 → BF16 MM＋FP32 scale → BF16 输出。未持久缓存 BF16 权重，未实施加载时 INT8 KN prepack。图捕获、编译和预热不计入稳态；图重放保留每次转换及实际 MM，分配遵循 CUDA Graph 内存池语义。

候选来自此前宽搜索及最终边界测评；扩展相邻 BK、warps/stages、128/256 tiles。先测 tile/layout，再对各布局前四名交叉 `GROUP_M=1/2/4/8/16`，再对前三名交叉转换配置和 FAST_STORE。保留无分组、无快路径对照。该分阶段搜索不能证明全局最优，可能遗漏初筛排名较后但分组后改善的组合。

## 4. 实验与验证

四类 `(N,K)`：Down `(2048,6144)`、Gate/Up `(12288,2048)`、QKV `(2560,2048)`、O `(2048,2048)`；各 M2048/4096。

- 24 个边界检查覆盖 M/N 整除与非整除、K 尾块、GROUP_M=0/2/8、FAST_STORE 开/关。
- FP32 reference 的 relative L2 ≤0.005、normalized max ≤0.01，阈值未因性能修改。
- 原生 BF16 对照使用原始 BF16 权重；量化对照使用 `A.float() @ Q.float().T * scale`；量化误差另记，不能用内核误差代替模型精度。
- 初筛使用 eager，候选进一步按 graph 筛选；最终两组新数据 seed=20261120/20261121、每组五轮随机顺序 eager/graph 配对确认。
- “frozen”是脚本中的历史 Separate 控制标签，选择历史候选中的较优项；不是对生产 dispatch 的新修改。最终结果单列配置。
- 初稿运行实际上完整结束，未被成功终止。它使用运行时整除判断，归档在 `initial-draft/`；最终版本改用编译期整除常量并全量重测。两版不混合汇总，不择优合并。
- 当前依赖的 libentry 存在 replay benchmarker fallback 警告；本脚本直接解包 JIT 并显式捕获 `torch.cuda.CUDAGraph`，使用 `do_bench(graph.replay)`，不是调用 libentry 自动调优计时。

延迟单位 μs；差值为量化延迟 / BF16 延迟 − 1，负数更快。

| 投影 | M | BF16 graph | 完整量化 graph | 相对 BF16 | 相对历史量化 | 同优化 BF16 | graph 配对胜出 | eager 相对 BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| down | 2048 | 572.693 | 629.742 | +9.96% | -0.78% | 571.712 | 0/10 | +12.92% |
| down | 4096 | 1119.279 | 1175.946 | +5.06% | -0.39% | 1116.235 | 0/10 | +6.34% |
| gate_up | 2048 | 1422.823 | 1308.010 | -8.07% | -11.72% | 1173.565 | 10/10 | -7.28% |
| gate_up | 4096 | 2844.398 | 2479.214 | -12.84% | -12.39% | 2344.301 | 10/10 | -12.46% |
| qkv | 2048 | 297.254 | 307.477 | +3.44% | -0.69% | 292.587 | 0/10 | +3.80% |
| qkv | 4096 | 510.311 | 523.242 | +2.53% | -0.71% | 508.546 | 0/10 | +2.80% |
| o | 2048 | 216.123 | 229.540 | +6.21% | -0.83% | 214.994 | 0/10 | +9.45% |
| o | 4096 | 406.723 | 421.546 | +3.64% | -0.87% | 406.896 | 0/10 | +5.28% |

## 获选完整量化配置

| 投影 | M | BM/BN/BK/warps/stages | BF16 转换布局 | 转换配置 | GROUP_M | FAST_STORE |
|---|---:|---|---|---|---:|---|
| down | 2048 | [256, 256, 32, 16, 2] | nk | [1024, 4] | 2 | True |
| down | 4096 | [256, 256, 32, 16, 2] | kn | [128, 128, 8] | 0 | True |
| gate_up | 2048 | [256, 256, 32, 16, 2] | nk | [4096, 16] | 4 | True |
| gate_up | 4096 | [256, 256, 32, 16, 2] | nk | [4096, 16] | 8 | True |
| qkv | 2048 | [128, 128, 32, 8, 2] | kn | [128, 64, 8] | 0 | True |
| qkv | 4096 | [256, 256, 32, 16, 2] | kn | [128, 64, 8] | 1 | True |
| o | 2048 | [256, 256, 32, 16, 2] | nk | [4096, 16] | 0 | True |
| o | 4096 | [256, 256, 32, 16, 2] | nk | [1024, 4] | 0 | True |

实测记录：{"timing_rows": 2586, "screen_rows": 1870, "confirm_rows": 640, "failed": 36, "boundary_checks": 24, "max_l2": 0.0016737584955990314, "max_normalized_max": 0.0036396675277501345}。36 个失败均为候选共享内存资源超限，记录保留；没有数值检查失败。最终内核最大 relative L2 为 0.001674、normalized max 为 0.003640。

获选量化均启用 FAST_STORE，但不能把获选组合的全部收益归因于去 mask。Gate/Up 的分组版本在初稿无 FAST_STORE 时已明显改善；最终分组、tile、转换和快路径共同选择。除 Gate/Up 外，新量化相对历史量化约降低 0.39%–0.87%，如实记录，不套用官方 1% 分类。

对照同实现优化后的 BF16，完整量化仍全部更慢：

| 投影 | M | 完整量化相对同优化 BF16 |
|---|---:|---:|
| down | 2048 | +10.15% |
| down | 4096 | +5.35% |
| gate_up | 2048 | +11.46% |
| gate_up | 4096 | +5.75% |
| qkv | 2048 | +5.09% |
| qkv | 4096 | +2.89% |
| o | 2048 | +6.77% |
| o | 4096 | +3.60% |

因此 Gate/Up 超过原版 BF16 的主要来源是可供 BF16 复用的 MM 优化，而不是量化独有的算力收益。主基线与 matched 对照均明确保留，未用较弱基线掩盖这一点。

graph 配对差值范围（每对量化/BF16−1）：

| 投影 | M | 中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| down | 2048 | +9.95% | +9.39% | +10.44% |
| down | 4096 | +5.06% | +4.71% | +5.21% |
| gate_up | 2048 | -7.95% | -8.80% | -7.40% |
| gate_up | 4096 | -12.60% | -13.55% | -12.13% |
| qkv | 2048 | +3.43% | +3.33% | +3.58% |
| qkv | 4096 | +2.53% | +2.36% | +2.70% |
| o | 2048 | +6.24% | +5.08% | +8.20% |
| o | 4096 | +3.65% | +2.64% | +3.99% |


## 5. 与已有改动的关系

复用 `2026-11-13-tianshu-four-projection-boundaries-dispatch.md` 的候选和 provider，但本轮新计时均实际执行，不复用历史 timing cache。磁盘编译缓存不清除。原 `backend.py`、`policy.py`、`dispatch_manifest.json` 等保持冻结。已有用户跟踪文件修改不属于本任务，未触碰或提交。

## 6. 合规性与风险

这是独立 microbenchmark，没有模型服务、官方性能或 math_500 测试；不能推导 total tokens/s、TTFT 或容量收益。保持静态 per-output-channel INT8＋FP32 scale、BF16 激活/点积、FP32 累加、BF16 输出，不是 INT8×INT8 点积。覆盖仅为所列真实投影形状和边界样例，不宣称覆盖模型全路径。调用完整性限于量化算子，不含模型层外围的 bias、TP 通信和 Python runtime dispatch 包装。

原生 BF16 底层仍由队友负责；本次只有隔离的诊断副本和量化路径测评，不替换共享实现。性能改善不足 1% 不按官方 ±1% 规则分类，依据配对胜率和测量范围呈现。未提交、未推送、未更新模型启用开关。

## 7. 复现方式

远端目录 `/workspace/experiments/tianshu-hotspot-joint-20261113-01/`：`run.sh` 记录 PATH、LD_LIBRARY_PATH、PYTHONPATH 和四个量化开关（均为 0）。沿用 `/workspace/tianshu-mixed-env-20261109/bin/python`，不重装环境。provider SHA256 为 `fd120991ed1c5cab988d02cbdb1d97ccde1e9e2eedad55affe90321d875f32ed`。

```bash
# 已存在的 results 目录禁止覆盖；复现使用新的输出目录。
PYTHONPATH=/workspace/experiments/tianshu-boundaries-20261113-01/source:/workspace/experiments/tianshu-boundaries-20261113-01/gems \
/workspace/tianshu-mixed-env-20261109/bin/python -u \
benchmark_tianshu_hotspot_joint.py results-new seeds.json
```

资源检查使用 `/usr/local/corex/bin/ixsmi`，开始前无 GPU 进程、70MiB 驱动保留、9031 端口可绑定；本次没有终止其他任务。初始化命令中尝试的 `nvidia-smi`/`ix-smi` 不存在，随后定位并使用实际工具 `ixsmi`，未安装替代工具。

归档 SHA256：`f3ee153d664ad37263d9c2c5e784bf4c0f631d81d7bfd3cc4d0438854d32d965`。本地 `analyze.py` 生成 `comparison.md`、`statistics.json`；`evidence.tgz` 包含最终和初稿原始行、完整脚本、provider/backend/reference 源码、环境和进程/GPU状态。结束后 GPU 无进程、70MiB 驱动保留。8 个冻结源文件指纹逐项匹配，见 `frozen-dispatch-check.json`。
