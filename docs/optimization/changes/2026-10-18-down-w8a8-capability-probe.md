# Down W8A8 最小能力探测：量化码差异与小 M 回退

> 日期：2026-10-18  
> 模块：GEMM  
> 基线提交：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`  
> 结果提交：尚未提交；按用户要求在当前分支开发，提交前再新建相关分支

## 范围与实现

新增 `benchmarks/diagnostics/probe_w8a8_down.py`，只探测 synthetic Down `[M,6144] × [6144,2048]`，沿用动态 per-token / channelwise W8A8 ADR。未修改模型运行代码、量化实现、依赖、官方启动命令、官方 Benchmark 或评测数据。

激活量化使用现有 `CachedOp`，GEMM 使用当前安装的 `flag_gems.scaled_mm`；权重按照现有 Linear 的转置并 contiguous 约定存放。BF16 对照使用 `torch.nn.functional.linear`。权重生成与量化在计时外完成，不代表已实现模型 load-time 接入。

量化码与现有 reference 逐元素比较；scale 使用 `rtol=1e-6, atol=1e-8`。GEMM 使用抽样行/列的 CPU INT64 精确累加、FP32 scale、BF16 输出作为参考，容差 `rtol=0.01, atol=0.02`；另做全输出有限值、shape/dtype 和零输入/零权重检查。抽样验证不代表所有输出均已逐元素验证。

## 环境与复现

| 项目 | 值 |
|---|---|
| 硬件 | MetaX C500，单卡，检查时无其他 GPU 进程 |
| Python | Conda 3.12.11 |
| PyTorch | 2.8.0+metax3.7.0.7 |
| Triton | 3.0.0+metax3.7.0.7 |
| FlagGems 安装包 | 5.3.5 |
| FlagGems 源码仓库 HEAD | a7620cc191a0b42e040194622c5758b22a7a25dc |
| vLLM 安装包 | 0.24.0+empty |
| FL 插件 | 0.3.0rc1.post1+g13eb9be69 |
| MACA / Driver | 3.7.0.38 / 3.8.30 |
| 随机种子 | 20261018 |
| 预热 / 计时 | 每项 5 次预热；3 组，每组 30 次，取 GPU Event 每次耗时的组中位数 |

实际安装的 scaled_mm 源码路径及 SHA256 在结果 JSON 内记录；不能把 `/workspace/FlagGems` HEAD 自动视为安装包构建版本。脚本使用本地框架源码，未安装替换运行库。

在已有 Conda/MACA 环境中运行：

```bash
python /workspace/vllm-plugin-FL/benchmarks/diagnostics/probe_w8a8_down.py \
  --output /workspace/benchmark_results/<new-experiment-id>
```

结果目录必须不存在，避免覆盖；M 是方案中的探测范围，不是模型实际 Shape trace。默认在 M=64、1024 通过正确性后才采集 trace。本轮这两个 Shape 失败，因此另对 M=1 补采：

```bash
python /workspace/vllm-plugin-FL/benchmarks/diagnostics/probe_w8a8_down.py \
  --m 1 --profile-m 1 \
  --output /workspace/benchmark_results/<new-profile-id>
```

原始结果保留在容器 `/workspace/benchmark_results/w8a8-down-20261018-r2/` 和 `w8a8-down-20261018-r3/`，控制台日志位于同一父目录。JSON、r3 trace 和与运行哈希一致的脚本快照已另存本机工作区上一级的 `benchmark_results/`，不自动纳入团队仓库。

r2/r3 执行脚本 SHA256 为 `ecb240c06cfc380614c11aa883c717cf671c514bec2e1c14db004f301495567d`。当前脚本在该版本上另增加启动时自动保存脚本快照；此归档功能不改变测量路径。r1 在 M=8 首次发现差异后停止，r2 补充差异统计并跳过数值失败 Shape 的计时；r3 只补采通过检查的 M=1。

## 正确性结果

零输入、零权重及零 scale 检查通过。动态量化默认 `default.flagos` 执行失败后，实际选择 `default.flagos_triton`。本轮记录了失败实现 ID，未进一步确认首选实现失败的根因。

| M | 量化码不一致数量 | 最大码差 | 计时状态 |
|---:|---:|---:|---|
| 1 | 0 | 0 | 通过本轮检查并计时 |
| 8 | 3 | 1 | 跳过 |
| 16 | 27 | 1 | 跳过 |
| 32 | 21 | 1 | 跳过 |
| 64 | 43 | 1 | 跳过 |
| 128 | 78 | 1 | 跳过 |
| 256 | 224 | 1 | 跳过 |
| 512 | 372 | 1 | 跳过 |
| 1024 | 674 | 1 | 跳过 |
| 2048 | 1473 | 1 | 跳过 |
| 4096 | 2694 | 1 | 跳过 |

例如 M=8，某行 absmax=3.75，输入 1.875，理想缩放值为 63.5；Triton 输出 63，reference 输出 64，记录的 scale 相同。其他样例也存在相反方向差异。scale 最大绝对差约 `3.73e-9`。

这些证据提示除法、乘法精度及舍入边界可能参与差异，但尚未定位编译器或具体运算根因，也不能直接认定 reference 对所有边界都是数学真值。不放宽标准、不修改实现；r2 状态为 `completed_with_failures`，退出码 1 是预期的失败报告。

M=1 抽样 GEMM 输出与 INT64 参考的 BF16 结果最大差为 0。相对于原 BF16 Linear，随机输入的 relative L2 约 0.01372；这不是模型精度，不能与 math_500 的 0.950 门槛比较。

## M=1 Microbenchmark 与 trace

下表为 r3 的稳态测量，单位毫秒：

| 路径 | 中位耗时 | BF16 耗时 / 当前耗时 |
|---|---:|---:|
| BF16 Linear | 0.02112 | 1.000x |
| 已量化 INT8 GEMM + scale 输出 | 0.16529 | 0.128x |
| 动态激活量化 + INT8 GEMM + scale 输出 | 0.18047 | 0.117x |

r2 的完整路径为 0.17984 ms，BF16 为 0.02114 ms，方向一致。仅此 synthetic M=1 的完整路径约慢 8.5 倍；不能推广到所有 Decode、Prefill 或正式 4K/16K 场景。

r3 trace 中实际 GPU Kernel 为：

```text
_dynamic_per_token_quant_int8_kernel
scaled_mm_kernel
```

这确认该次执行使用独立激活量化和 FlagGems 通用 scaled_mm；Kernel 名不能证明底层是否使用原生高效 INT8 指令，需要另看编译产物或更细硬件证据。GPU Event 计时覆盖调用循环，可能包含主机供给造成的间隙，并非纯 Kernel 执行时间。

首调用耗时包含 dispatch/autotune/cache 等影响；零值检查已先使用相同 M=1 Shape，且跨进程存在缓存，因此本轮没有测得可独立归因的冷编译开销。

## 结论与下一步边界

暂不进入 Down 模型接入，不改变 W8A8 主线。尚不能判定 ADR 的全目标 Shape 性能止损条件成立：其余 Shape 没有通过数值契约检查，缺少完整可比性能数据。

后续需先讨论独立计划并经用户同意，再定位激活量化边界差异、首选实现失败原因，以及 C500 原生 INT8 API/布局支持。完成后重跑本探测，再决定 capability gate 和 load-time 量化实现。不把本轮通用路径的慢速结果当作自研 W8A8 必然不可行的结论。

本轮没有模型精度、正式吞吐或 TTFT 测试，也未修改/运行官方端到端评测。
