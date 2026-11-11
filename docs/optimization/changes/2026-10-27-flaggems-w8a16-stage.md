# FlagGems W8A16 两段 Split-K 阶段开销诊断

> 2026-10-27 复核更正：本报告原始 `wall_ms/alloc_ms` 实际为秒；raw_timings 仅保留 M=64。独立阶段计时不可相加，文中的阶段占比和 launch=0、分配开销近似为0的归因撤回。历史数据保留，修正脚本及补测见 `2026-10-27-w8a16-stage-audit.md`。

## 1. 基本信息与摘要

- 日期：2026-10-27（UTC+8，按远端实验日志）。任务 ID：`flaggems-w8a16-stage-20261027-03`。
- 设备：单卡 MetaX C500，65536 MiB。容器 `5e5ec8dfe1a6`、GPU bus `0000:58:00.0`、驱动 3.9.35、MACA 3.7.0.38。**本实例与上轮 splitk 实验（`f3a0dc142860` / bus `d4:00.0`）不同**，绝对值不跨实例比较。
- 新增诊断脚本：`benchmarks/diagnostics/benchmark_flaggems_w8a16_stage.py`，SHA256 `8f6407b41565b3ce9dbf8156bc241016189026e00fb6367c8c1b3df6349d7dc9`。
- 目标：在上轮选定的最优 Split-K 下，把两段结构拆成「partial 单独」「reduce+scale 单独」「完整 fresh 分配」「完整预分配」四项，定位主要成本在哪，为改进完整 `weight_int8pack_mm` 提供依据。
- 原始结论（已撤回，不作为当前依据）：**partial（INT8 加载 + INT8→BF16 转换 + BF16 dot）占完整耗时的 93%～96%，是绝对瓶颈；reduce+scale 只占约 14%～29%；分配开销和 launch 开销均近似为 0。** 因此后续改进应集中在 partial kernel 本身，而不是归约、分配或预分配。

## 2. 问题与原因分析

上轮 `2026-10-27-flaggems-w8a16-splitk.md` 已证明小 M 用两段 Split-K 后接近 BF16（M=1 只慢约 3%），但尚未拆清这「两段 + 分配 + 两次 launch」各自花了多少。原始单 kernel `weight_int8pack_mm` 的权重转换、dot、scale 都在同一 kernel 内，无法单独 profile；而 Split-K 候选已经拆成 `_partial`（算分片积）和 `_reduce_scale`（归约 + 乘 scale）两个独立 kernel，因此可以单独计时。

本轮沿用上轮 tile `(BM,BN,BK)=(16,64,128)`、warps=4、stages=2、partial pipeline=basic、reduce BLOCK_SIZE=256，只在选定最优 S 上拆测各阶段，不重新扫 S。

## 3. 具体实现

- 复用 `_metax/ops/mm.py` 两段 Split-K 的 partial/reduce 结构：partial 在 dot 前把 INT8 权重转 BF16，FP32 累加；reduce 先合并各分片 FP32 partial，再乘 per-channel FP32 scale 并转 BF16，保持 scale-after-dot 契约。
- 新增四个测量函数：`partial_only`（只 launch partial）、`reduce_only`（只 launch reduce）、`splitk_fresh`（内部 fresh 分配 partials/output）、`splitk_prealloc`（复用预分配缓冲）。后两者共用同一段 kernel 逻辑，避免重复。
- 计时两种口径：`do_bench(warmup=100, rep=300, return_mode='median')` 三轮取中位数（GPU 事件时间），以及 `time.perf_counter()+torch.cuda.synchronize()` 五次取中位数（含 CPU 分配与 launch 的 wall time）。
- 派生量：`alloc_ms = wall(full_fresh) - wall(full_prealloc)`；`launch_overhead_ms = max(full_prealloc - (partial + reduce), 0)`；`partial_share = partial/full_prealloc`；`reduce_share = reduce/full_prealloc`。

## 4. 实验与验证

环境：PyTorch `2.8.0+metax3.7.0.7`、Triton `3.0.0`、FlagGems 5.3.5（源码路径 `/workspace/FlagGems/src`）；脚本与辅助文件 SHA256 本地/远端一致。运行前 GPU 利用率 0%、无 GPU 进程、显存 860/65536 MiB、9031 空闲；串行单进程，退出码 0，结束后无残留进程、GPU 释放。

边界正确性：`(M,N,K)=(1,17,33)/(3,129,257)` × S=2/4 共 4 项，相对 L2 误差均为 0.0，并通过 `rtol=0.02, atol=0.125`。真实 Down shape 的四种 M/S 也通过 scale-after-dot 参考校验。

### 历史阶段开销（含已撤回派生列，仅供追溯）

| M | S | partial | reduce | full_fresh | full_prealloc | bf16_metax | alloc_ms | launch_overhead | partial_share | reduce_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.0399 | 0.0120 | 0.0428 | 0.0428 | 0.0422 | 4.8e-5 | 0 | 93.4% | 28.1% |
| 16 | 8 | 0.0422 | 0.0131 | 0.0453 | 0.0451 | 0.0420 | 5.5e-5 | 0 | 93.8% | 29.0% |
| 32 | 8 | 0.0591 | 0.0136 | 0.0625 | 0.0622 | 0.0509 | 1.6e-5 | 0 | 95.1% | 21.8% |
| 64 | 4 | 0.0963 | 0.0141 | 0.1004 | 0.0998 | 0.0727 | 7.1e-5 | 0 | 96.4% | 14.1% |

以下说明均为历史归因，已由页首更正撤回；不能据此认定重叠、阶段占比或零分配/launch 开销。有效分析以 stage-audit 报告为准。

历史说明：

1. `partial_share` 与 `reduce_share` 之和略大于 1，是因为两个 share 都用「单独 kernel 计时」除以「连发完整耗时」。连发时两 kernel 存在 GPU 流水线重叠，`full_prealloc` 小于 `partial + reduce`，故单独计时略有高估。这不影响「partial 占绝对主导」的定性结论。
2. `alloc_ms` 为微秒级（约 1.6e-5～7.1e-5 ms），预分配 vs 每次分配几乎没有可测差异；`launch_overhead_ms` 全为 0。
3. 相对 BF16：M=1 慢约 1.4%、M=16 约 7.3%、M=32 约 22.2%、M=64 约 37.3%（新实例同场，仅供参考，不与旧实例拼接）。

### 关键证据（编译产物）

已保存每个 M/S 的 partial 与 reduce 的 `.ttir/.ttgir/.mcfatbin` 及资源字段。partial 后端资源：M=1 时 `n_regs=120、n_spills=0、shared=12288`，M 增大后寄存器与 shared 上升；reduce 后端资源较小。数据表明 partial 是计算与访存主导段，reduce 为轻量归约段。

## 5. 与已有改动的关系

- 落实 `2026-10-27-flaggems-w8a16-splitk.md` 之后的「先测阶段开销」计划，不新增算法、不改分派、不改生产代码。
- 与上轮 splitk 相比，本轮固定已选最优 S（M=1/16/32→8，M=64→4），只拆测不重扫；tile 与数值契约保持一致。
- 直接服务于后续对完整 `weight_int8pack_mm` 的改进决策：结论指向「优化 partial 的 INT8 加载 + INT8→BF16 转换 + dot」，而非归约、分配或预分配。
- 未修改 FlagGems 生产 Kernel 或 vLLM 接入，未新增生产开关，未提交/推送。

## 6. 合规性与风险

- 全程为独立 synthetic GEMM 诊断，未运行模型、官方 Benchmark 或精度数据集；未改官方参数、数据、缓存或依赖。
- 未把诊断脚本注册为正式算子，未把「partial 主导」直接当作性能收益；分配与 launch 开销的结论仅适用于已测 shape/实例。
- 预分配在生产中的主要价值是 CUDA graph 兼容（静态 workspace），本轮原有“分配不是瓶颈”的证明已撤回，但未做模型级验证。
- 新实例与旧实例 GPU bus 不同，本轮绝对值不与旧实例直接比较；相对阶段占比结论才是本轮核心产出。

## 7. 复现方式与证据

远端目录：`/workspace/benchmark_results/flaggems-w8a16-stage-20261027-03/`；本地镜像：工作区 `artifacts/flaggems-w8a16-stage-20261027-03/`。

```bash
# 加载 Conda/MACA 后，在 exp 目录执行：
cd /workspace/benchmark_results/flaggems-w8a16-stage-20261027-03
source /opt/conda/etc/profile.d/conda.sh && conda activate base
export MACA_PATH=/opt/maca MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
export LIBRARY_PATH=/opt/mxdriver/lib
export PYTHONPATH=/workspace/FlagGems/src
python benchmark_flaggems_w8a16_stage.py \
  --helpers-dir /workspace/benchmark_results/flaggems-w8a16-stage-20261027-03 \
  --output-dir /workspace/benchmark_results/flaggems-w8a16-stage-20261027-03/results
```

证据：`results/stages.json`（汇总）、`results/raw_timings.json`（96 条原始计时）、`results/correctness.json`（边界正确性）、`results/metadata.json`（环境与契约）、每个 M/S 的 partial/reduce 编译产物与资源字段。复现须用新实验 ID，不得覆盖。
