# Dense 启动参数缓存修复重测与三算法分界定稿

## 1. 基本信息与摘要

- 日期：2026-11-05；设备：沐曦 C500（新实例 hostname `477fc22dad8f`，驱动 3.8.30 / MACA 3.7.0.38 / BIOS 1.33.5.0，与 dense-mid-01 指纹逐项一致）。
- 实验目录：`/root/experiments/2026-11-05-w8a16-dense-launchfix-01/`。
- 内容：(a) 将 launch-cache-fix 同构修复应用到 `_configured_dense`（此前仅修 Split-K，Dense 明确遗留），并从原始版 `41397bd` 副本重跑 dense 全部 16 点；(b) 汇总 Split-K launch-fix 60 点、checkfree 60 点与本轮 Dense 16 点，**定稿三算法（Dense / Split-K / Separate）分界**。BF16 原生路径为对照基线，不是候选算法——权重已量化为 INT8，分界只在 W8A16 三算法内部择优。
- 结果：**16/16 完成，`complete=true`；Dense 修复后中段 M 普遍改善 6~36pp，gate_up M80/M96 逼近 BF16（−8.3%/−8.8%）但仍为有效退化；三算法分界定稿为 o/down/qkv 全 M Split-K、gate_up M1-32 Split-K / M48+ Dense、M≥128 Separate**。未提交、未推送。

## 2. 问题与原因分析

`2026-11-05-w8a16-splitk-launch-cache-fix.md` 只将 `_configured_splitk` 的 partial 调用改为 `jit_function.run(...)`，并明确"Dense 仍待其容器有资源时单独处理"。`_configured_dense` 的两个 kernel（`weight_int8pack_mm_dense`、`weight_int8pack_mm_m1_broadcast`）仍经共用 LibEntry 启动，`num_warps/num_stages` 不进缓存键——同一 tile 不同启动参数复用先前编译产物，dense-mid-01 的 16 点名义配置不可认证。

本轮对比证实影响远大于 Split-K：Dense 中段 M 修复后改善 6~36pp（Split-K 修复重测的判性影响最大约 3pp），旧数据的名义赢家大量不可信。

## 3. 具体实现

1. **Dense launch-fix 补丁**（仅容器隔离副本，源码 `41397bd` → `960b42761f3e85d0d0073c6adecad07a3f84e9d21492dea279fd9b53b498cc52`）：`_configured_dense` 两处 LibEntry 调用改为 `jit_function.run(..., grid=..., warmup=False)`，显式传 `num_warps/num_stages/pipeline`，与 Split-K 修复同构；`py_compile` 通过。dense sweep 终测的 Split-K 对照为单一固定配置（S8），无缓存别名风险，不需一并修改。
2. **实验目录**：从 dense-mid-01 硬链接复制 `FlagGems/` 与 `vllm-plugin-FL/`（含 `tune_metax_w8a16_dense.py`、`w8a16_dense_search.py`、`w8a16_splitk_search.py` 全套依赖），断链 backend 文件后打补丁。新目录从头完整重跑（源码变更，不 resume）。
3. **协议**：与 dense-mid-01 完全一致（gate_up/o 两 shape × wide/refined 共 16 点；筛选 warmup20/rep60、终测 2 seed × 5 轮交错 warmup100/rep300；preflight 全套 + metadata 指纹 + O_EXCL 锁 + journal 断点续跑）。

## 4. 实验与验证

### 4.1 完成状态

16/16 点，`complete=true`、`failed_points=[]`，运行后 GPU 无残留进程。

### 4.2 Dense 修复前后对比（相对各自同场 BF16，跨实例只做方向性对比）

| 点 | 未修复 | 修复后 | 变化 | 赢家配置变化 |
|---|---:|---:|---:|---|
| o M56 | −67.2% | −31.0% | +36.1pp | 32-64-128-4-2 → 64-32-128-8-2 |
| o M64 | −65.7% | −33.3% | +32.3pp | 32-32-256-4-2 → 64-32-256-8-1 |
| o M72 | −65.7% | −38.0% | +27.7pp（对照不稳） | → 32-32-128-4-2 |
| o M80 | −70.3% | −42.4% | +27.9pp | → 32-32-128-4-2 |
| o M88 | −61.8% | −39.2% | +22.5pp | → 32-32-128-4-2 |
| o M96 | −64.4% | −43.5% | +20.9pp | → 32-32-128-4-2 |
| o M104 | −69.6% | −65.1% | +4.5pp | → 32-32-128-4-1 |
| o M112 | −68.3% | −68.7% | −0.4pp | 配置不变 |
| gate_up M96 | −35.9% | −8.8% | +27.1pp | 128-64-64-4-1 → 128-128-64-8-2 |
| gate_up M80 | −18.2% | −8.3% | +9.9pp | → 128-128-64-8-2 |
| gate_up M48 | −47.2% | −36.5% | +10.7pp | 64-128-64-8-3 → 64-64-64-8-2 |
| gate_up M24 | −39.5% | −32.3% | +7.2pp | 32-32-128-4-3 → 32-64-128-8-2 |
| gate_up M32/M64 | −33.0/−43.0% | −32.5/−36.2% | ≈+0.6/+6.8pp | M32 配置不变 |

结论：LibEntry 别名对 Dense 的影响远大于 Split-K（判性最大 3pp）；修复后没有任何点翻正（最好 −8.3%），Dense 在 gate_up M48+ 的三算法内部排序中升至第一。

### 4.3 三算法分界定稿（W8A16 内部择优；BF16 仅对照）

| 投影 (N,K) | M 区间 | 采用算法 | 代表配置 | vs BF16 |
|---|---|---|---|---|
| o (2048,2048) | M1-16 | Split-K | [16,64,128,4,2~3,S8-16] | **+3~+9%（翻正）** |
| o | M24-104 | Split-K | [32,32,128,4,1,8] → [64,64,128,4,1~2,S8-16] | −3~−35%（量化开销） |
| gate_up (12288,2048) | M1-32 | Split-K | [16,128,64,4~8,2,2] / [32,128,64,2,2,2] | −7~−27% |
| gate_up | M48-112 | **Dense** | [64,64,64,8,2] / [128,128,64,8,2] | −8~−69%（M80/96 仅 −8.3/−8.8%） |
| down (2048,6144) | M1-112 | Split-K | M56/64: [64,64,128,4,1,8]（**+14% 翻正**）；其余按 winner 表 | −2~−22%（除翻正区） |
| qkv (2560,2048) | M1-96 | Split-K | 按 winner 表 | −6~−39% |
| 全部 | M≥128 | Separate | 现有生产策略（转换+原生MM+scale） | 未做三算法对照 |

分界机制：Split-K 收益来自 K 维切分补并行度——down K=6144 长、o K=2048 且 N 小，小/中 M 并行度缺口大；gate_up N=12288 输出 tile 充足（M64 时 384 tiles ≫ 104 SM），不需要 split，Dense 单 kernel 免去 partial+reduce 开销，在大 N 中大 M 段全面占优。

### 4.4 之前报告的修正

- `2026-11-05-w8a16-splitk-checkfree-top-remeasure.md` 中"gate_up 中段 Dense 出局/BF16 回退"的表述为**错误框架**（当时把 BF16 当作候选）；正确结论是 gate_up M48+ 采用 Dense。该报告其余数据（Split-K 赢家计时、条件检查开销量化）不受影响。
- 三路径对比中 gate_up M48-104 的"Dense 更优但都不如 BF16"排序不变，只是决策输出从"BF16"修正为"Dense"。

## 5. 与已有改动的关系

- 依赖 `2026-11-05-w8a16-splitk-launch-cache-fix.md` 的 Split-K 修复与验证方法；本报告完成其遗留的 Dense 修复。
- dense-launchfix-01 与 dense-mid-01 为不同容器实例，绝对微秒值不拼接；分界结论基于"相对同场 BF16"的归一化口径。
- 生产分派尚未接入任何赢家（`weight_int8pack_mm:558` 仍走 old config `[16/64,64,128,4,2,8]`，实测在 o/qkv/gate_up 多个 M 上为负优化）；接入与收窄方案待用户批准后另行走代码流程。

## 6. 合规性、风险与遗留缺口

- 全部为 synthetic eager 算子诊断，非模型精度、官方吞吐/TTFT；不宣称任何端到端收益。
- 生产分派未改动；dense-fix 仅存在于隔离副本，生产仓库仍为 `41397bd`。
- 遗留缺口：
  1. **down / qkv 的 Dense 从未测过**（本轮 16 点只覆盖 gate_up/o）——"down/qkv 全 M Split-K"目前只与 BF16 对照过，未验证 Dense 是否更优（down K=6144 的 Dense 行为不能从 K2048 外推）；
  2. gate_up M40 的 Dense 缺失（M32→M48 分界处）；
  3. M≥128 的 Separate vs Dense/Split-K 无三算法对照；
  4. o M∈{5,6,7,9-15} 等未逐点实测，区间配置属推测覆盖；
  5. o m2/m8 等点 BF16 对照 spread>2% 判不稳，其 Split-K 正收益待更稳协议复核；
  6. 生产接入需正确性测试 → micro 同机对照 → math_500 精度 → 官方 4K/16K 验证，M≥128 Separate 边界与图捕获路径需确认。

## 7. 复现方式

```bash
# 容器内（登录 shell 加载 Conda/MACA）
cd /root/experiments/2026-11-05-w8a16-dense-launchfix-01
export PYTHONPATH=/root/experiments/2026-11-05-w8a16-dense-launchfix-01/FlagGems/src:/root/experiments/2026-11-05-w8a16-dense-launchfix-01/vllm-plugin-FL
setsid /opt/conda/bin/python vllm-plugin-FL/benchmarks/diagnostics/tune_metax_w8a16_dense.py \
  --output-dir ./results > sweep.log 2>&1 < /dev/null &
```

- 汇总：`results/summary.json`（complete=true, 16/16）、`results/status.json`、`results/details.csv`；逐点 `results/{shape}/m{m}/result.json` 与 `raw_final.jsonl`。
- 指纹：`results/metadata.json`；backend 补丁版 SHA256 `960b4276…98cc52`（原始 `41397bd…`）。
- 本地归档：`experiment-artifacts/dense-launchfix-01-results.tar.gz`，SHA256 `9cf8fa6922010aac272d23655eb73667707d994886d3508cd49c2370709056eb`（两端一致），解压于 `experiment-artifacts/2026-11-05-w8a16-dense-launchfix-01/`（16 个 result.json 已核对）。
- 关联归档：Split-K 60 点 `launch-fix-01-results-final.tar.gz`（SHA256 `13271c69…fba42e`）、checkfree 60 点 `checkfree-01-results.tar.gz`（SHA256 `bff3da87…e87099`）。
