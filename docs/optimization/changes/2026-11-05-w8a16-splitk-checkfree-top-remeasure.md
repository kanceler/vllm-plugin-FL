# Split-K 赢家免筛选直通补测:条件检查开销量化

## 1. 基本信息与摘要

- 日期：2026-11-05；设备：沐曦 C500（hostname `24b02f746c85`，与 launch-fix-01 补测同实例）。
- 实验目录：`/root/experiments/2026-11-05-w8a16-splitk-top-checkfree-01/`。
- 目的：launch-fix-01 60 点 Split-K 搜索的赢家计时走 `run_w8a16_algorithm` → `_configured_splitk`，其首行在**计时窗口内**执行 Python 层 `splitk_rejection_reason` 条件检查。本补测对每点赢家 config 在同机同输入同协议下对比「含检查(checked)」与「免检查(checkfree)」两条路径，量化条件检查对记录计时的污染。
- 结果：**60/60 点完成，`complete=true`；checked 与 checkfree 输出 100% bitwise 一致；条件检查开销中位数 0.000%（最大 +3.00%）；没有任何点的有效/无效分类因排除检查而翻转**。用户「部分点低于基线源于条件调用」的假设被数据否定：负收益点在免检查口径下同样是显著负收益。
- 未修改生产分派逻辑；未提交、未推送。

## 2. 问题与原因分析

原 sweep 的终测计时函数为 `backend.run_w8a16_algorithm(x, q, scale, "splitk", config)`，而 `_configured_splitk` 首行执行：

```python
reason = splitk_rejection_reason(a, b, scales, config)
if reason is not None:
    raise ValueError(...)
```

该检查是纯 Python（N 整除、dtype、config 结构），每次计时调用都执行。小 M 点 GPU kernel 仅约 26–40 µs，若检查的 CPU 时间造成 GPU 提交间隙，do_bench 事件窗口会把空泡计入。因此原 60 点赢家计时是「含筛选」口径，需排除后复测以判断负收益点是否为检查开销所致。

## 3. 具体实现

1. **隔离副本最小补丁**（仅容器实验副本，生产 FlagGems 仓库未动）：`_configured_splitk` 增加默认参数 `check=True`，检查体包进 `if check:`。生产入口 `run_w8a16_algorithm` 仍不带 check 参数调用（默认 True），**生产行为零变化**；免检查仅诊断脚本显式 `check=False` 走直通。补丁后 hash `790393a6d1c58aac1fc5df17527ede4f270f0ee51ed54904bfcd192530e37bd6`（原 launch-fix 版 `bdb0b81b…`）。源码子集以 `cp -al` 硬链接复制并单独断链该文件。
2. **补测脚本** `sweep_top_checkfree.py`（脚本 hash `7c8ed7f8eacac47f13713d63ce08aae4102aa59ab9bfcdb9ab300694af64248d`）：读取 launch-fix-01 `summary.json` 的 60 点 `selected_config`（不重选赢家）；每点独立 subprocess 串行；输入按原 seed 公式（`20261104 + shape_index*100000`、`+10000`、`+m*101`）与同量化方式；preflight 全套（mx-smi 空闲、端口 9031、竞争进程、O_EXCL 锁、metadata 指纹校验）；断点续跑 journal。
3. **计时协议**：与原终测一致（2 独立 seed × 5 轮交错、warmup 100 ms / rep 300 ms、median），三条路径：同场原始 BF16（`mm.mm`）、`checked`（原 `run_w8a16_algorithm`）、`checkfree`（`_configured_splitk(check=False)`，计时前在计时外执行一次检查确保合法）。
4. **正确性门槛**：两路径均过 relative L2 < 0.005 与 `assert_close(rtol=0.02, atol=0.125)`（主输入 + 对抗 edge 输入：整数端点、零通道、scale 跨度、K 分片）；并逐点核对 checked/checkfree 输出 bitwise 一致。

## 4. 实验与验证

### 4.1 完整性与一致性

| 指标 | 值 |
|---|---|
| 完成点 | 60/60（120 行 = 60 点 × 2 seed），`complete=true` |
| checked vs checkfree bitwise 不一致行数 | **0**（两路径为同一 kernel、同一计算） |
| gate overhead（`(checked−checkfree)/checkfree`，逐行） | 中位数 **0.000%**，min −1.676%，max **+3.000%** |

### 4.2 分类翻转分析（每点取 2 seed 中位数，>1% 为有效）

| 类别 | 点数 |
|---|---:|
| 两种口径均有效 | 8 |
| 排除检查后翻转为有效 | **0** |
| 排除检查后翻转为无效 | **0** |
| 两种口径均非有效 | 52 |

**结论：条件检查不是任何点的判性因素。** 用户假设「部分点低于基线是因为条件调用」不成立——down m72–m96（−24%~−40%）、gate_up m40 以后（−62%~−102%）、o m72 以后（−33%~−37%）等负收益在免检查口径下几乎不变（差距 ≤0.5%），是真实的 Split-K 大 M 负收益，而非计时污染。

### 4.3 条件检查开销的分布特征

- 小 kernel 点（BF16 ≈ 26–35 µs）受影响相对最大，如 o m4 +1.50%、qkv m8 +1.34%、o m24/m32 约 +1.3%、down m32 +1.16%；但幅度均不足以改变 ±1% 判性，且部分点为负（CPU 检查与 GPU 执行重叠）。
- 中大 kernel 点（≥50 µs）开销趋近 0，检查时间被 GPU 执行完全覆盖。

### 4.4 两口径均为有效改善的点（免检查口径，同场 BF16 对照）

| 点 | checked | checkfree | 备注 |
|---|---:|---:|---|
| o m1 | +5.93% | +6.38% | |
| o m2 | +7.21% | +7.21% | |
| o m3 | +2.70% | +3.15% | 本轮绝对值低于 launch-fix-01 的 +9.01%，见 4.5 |
| o m4 | +7.95% | +9.27% | |
| o m8 | +7.11% | +7.11% | |
| o m16 | +4.76% | +4.76% | |
| down m56 | +13.87% | +14.05% | |
| down m64 | +7.76% | +7.05% | |

### 4.5 跨实验对照的保留意见

本轮 o m3 的 checked 计时（约 +2.70%）与 launch-fix-01 补测（+9.01%）存在绝对值差异，虽为同容器同源码，两轮相隔数小时、进程/编译状态不同，5 轮 median 存在波动与潜在时钟状态差异。**同轮内 checked vs checkfree 与 checked vs BF16 的对比是严格同场同输入的，结论可靠；跨实验的绝对微秒值不做拼接归因**，与已完成审计的跨实例口径一致。如需确认，可在同进程内做 base-sweep 复刻对照（待验证）。

## 5. 与已有改动的关系

- 直接量化 `2026-11-05-w8a16-splitk-resume-after-rebuild.md`（launch-fix 重测）记录的赢家计时中条件检查的占比，回答其遗留的「负收益是否被检查开销污染」问题。
- 与入口审计（`2026-11-05-w8a16-splitk-entry-audit.md`）互补：入口审计排除了「公共入口结构性变慢」，本补测排除了「条件检查计时污染」，两条证据共同支持原 sweep 计时口径的可信性。
- 隔离副本补丁未进入生产仓库或 FlagGems 提交历史；`run_w8a16_algorithm` 及框架选择逻辑零改动。

## 6. 合规性、风险与后续建议

- 纯 synthetic eager 算子诊断，非模型精度、官方吞吐或 TTFT 结果；不代表任何赛事收益。
- checkfree 直通不进生产分派；生产路径仍含全部条件检查（合规要求的能力验证保留）。
- 未重选赢家：若 checkfree 下排序有变化仅记录不改判（本轮无判性翻转，该风险未触发）。
- 后续建议：Split-K 生产接入的候选范围可依据本轮确认——有效改善集中在 o 极小 M（m1–m16）与 down m56/m64；gate_up m40 以上、down/o m72 以上负收益稳定，应继续由 BF16/Dense 路径覆盖。若需将 down m56/m64 与 o 小 M 差异固化为算法分界，仍需同机 finalist 三算法对照与模型级验证（待验证）。

## 7. 复现方式

```bash
# 容器内（登录 shell 加载 Conda/MACA）
cd /root/experiments/2026-11-05-w8a16-splitk-top-checkfree-01
export PYTHONPATH=/root/experiments/2026-11-05-w8a16-splitk-top-checkfree-01/FlagGems/src:/root/experiments/2026-11-05-w8a16-splitk-top-checkfree-01/vllm-plugin-FL
setsid /opt/conda/bin/python sweep_top_checkfree.py --output-dir ./results > sweep.log 2>&1 < /dev/null &
```

- 汇总：`results/summary_checkfree.json`、`results/details_checkfree.csv`、`results/status.json`（complete=true, 120/120）；逐点 `results/{shape}/m{m}/result.json` 与 `raw_final.jsonl`。
- 指纹：`results/metadata.json`（含补丁后 backend hash `790393a6…` 与脚本 hash `7c8ed7f8…`）；preflight 目录含 GPU/进程证据。
- 本地归档：`experiment-artifacts/checkfree-01-results.tar.gz`（SHA256 `bff3da871b502835772ddfd13884f179564b4842fdcb128591712cafffe87099`，两端一致），解压于 `experiment-artifacts/2026-11-05-w8a16-splitk-top-checkfree-01/`（60 个 result.json 已核对）。
- 基准来源：`/root/experiments/2026-11-05-w8a16-splitk-launch-fix-01/results/summary.json`（`base_summary_sha256` 已记录于 metadata）。
