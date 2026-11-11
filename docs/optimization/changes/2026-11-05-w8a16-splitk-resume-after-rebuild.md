# Split-K 容器中断断点续跑与 60/60 完成归档

## 1. 基本信息与摘要

- 日期：2026-11-05；设备：沐曦 C500（新实例，容器 hostname `24b02f746c85`）。
- 场景：`/root/experiments/2026-11-05-w8a16-splitk-launch-fix-01/` 的完整 Split-K 搜索（launch-fix 修复后重测）因容器中断停止于 51/60 点；新容器实例上完成指纹核对并 `--resume` 续跑剩余 9 点。
- 结果：**60/60 点完成，`summary.json` 与 `status.json` 均 `complete=true`，`failed_points=[]`**。续跑 9 点全部为 `refined o`（M=3,8,24,48,56,72,80,96,104）。运行后 GPU 无残留进程。
- 未修改任何源码、Kernel、官方命令或评测脚本；未提交、未推送。

## 2. 问题与原因分析

容器中断后重建实例，续跑面临三层障碍：

1. **hostname 差异阻断身份校验**。脚本的 `fingerprint()` 将 `socket.gethostname()` 纳入指纹，worker 启动时强制 `fingerprint() == metadata["fingerprint"]`。旧任务 metadata 记录 `6b547934b11d`，新实例为 `24b02f746c85`。除 hostname 外的全部指纹字段——device uuid（`9dde7953-d82f-3392-8f12-832da8498da6`）、驱动（MX-SMI 2.3.1 / Driver 3.8.30 / MACA 3.7.0.38 / BIOS 1.33.5.0）、torch `2.8.0+metax3.7.0.7`、triton `3.0.0`、python `3.12.11`、sm_count 104、warp_size 64、全部 8 个 source_hashes（含 launch-fix 版 `weight_int8pack_mm.py=bdb0b81b…`）——逐项核对完全一致。
2. **平台禁止改 hostname**。原拟方案 A（临时 `hostname 6b547934b11d`）不可行：uid=0 但 `sethostname` 系统调用被容器安全策略拒绝（`hostname: you must be root to change the host name`），无 systemd，`/etc/hostname` 可写但运行时 `gethostname()` 读内核值。误写的 `/etc/hostname` 已恢复原值。
3. **PYTHONPATH 相对/绝对路径差异**。第一次 `--resume` 报 `resume identity mismatch`。逐字段诊断（对 live probe 与文件 metadata 做全 dict 对比）显示逻辑指纹完全一致后，定位到启动环境问题：续跑命令用了相对路径 `PYTHONPATH=./FlagGems/src:./vllm-plugin-FL`，probe 子进程将其原样记入 `fingerprint.flags.PYTHONPATH`，而 metadata.json 记录的是绝对路径。该字段属于指纹，dict 对比不等即拒绝。首次失败发生在身份校验阶段，未执行任何 GPU 点。

## 3. 具体实现（续跑操作，无源码修改）

1. **方案 E（经用户确认）**：备份 `results/metadata.json` → `results/metadata.json.orig`（仅首次，幂等），仅修改 `fingerprint.hostname` 为当前值 `24b02f746c85`，修改记录写入 `results/metadata-hostname-fix.log`（含时间、字段、旧值、新值、原因、备份路径、`no_source_changed: true`）。源码 hash 与其余指纹字段零改动，断点身份的可比性由 device uuid+驱动+源码 hash 一致性保证，hostname 仅作为容器实例标签更新。
2. **Stale lock 清理**：`controller.lock` 内容 1246（与 `sweep.pid` 一致），`kill -0` 确认死亡后删除；符合脚本“Stale locks require PID/ownership verification”要求。
3. **断点核对**：51 个 `result.json` 有效，全部 `active.json` 为 `idle`（中断时无运行中 worker），`screening.jsonl`/`raw_final.jsonl` 逐点 append+fsync 日志完好；缺失 9 点由 `WIDE_MS/REFINED_MS` 与 sweep.log 对账确认为 refined o 全部。
4. **续跑启动**：`setsid` 后台执行 `tune_metax_w8a16_splitk.py --output-dir ./results --resume`，PYTHONPATH 使用与 metadata 一致的绝对路径；登录 shell（`bash -lc`）保证 `LD_LIBRARY_PATH` 与 metadata 的重复式记录一致。控制器自动完成 preflight（GPU 空闲、端口 9031 空闲、无竞争进程）、身份校验、跳过 51 个已完成点、串行补测 9 点。

## 4. 实验与验证

### 4.1 完成状态

| 指标 | 值 |
|---|---|
| 完成点 | 60/60 |
| `complete` / `execution_complete` | true / true |
| `failed_points` | 无 |
| 终测结果行 | 700 |
| 运行后 GPU | `no process found`，显存回到驱动保留水平 |

### 4.2 续跑 refined o 9 点赢家结果（M=2 为中断前已完成，列出作对照）

单位 ms；`reduction_pct` 为相对同场原始 BF16 中位数降低率；`beats_both` 要求两 seed 均 `effective_improvement` 且 BF16 对照 spread ≤2%。

| M | 赢家 config (BM/BN/BK/warps/stages/S) | Split-K (seed A/B) | BF16 (seed A/B) | 降低率 A/B | beats_both |
|---:|---|---|---|---|---|
| 2 | [16,64,128,4,2,16] | 0.025088 / 0.025856 | 0.028416 / 0.026624 | +11.71% / +2.88% | 否（对照 spread≈6.5% 不稳） |
| **3** | **[16,64,128,4,2,16]** | **0.025856 / 0.025856** | **0.028416 / 0.028416** | **+9.01% / +9.01%** | **是** |
| 8 | [16,64,128,4,2,16] | 0.025856 / 0.025856 | 0.028160 / 0.026368 | +8.18% / +1.94% | 否（对照 spread≈6.7% 不稳） |
| 24 | [32,32,128,4,1,8] | 0.029440 / 0.028672 | 0.027648 / 0.027648 | −6.48% / −3.70% | 否 |
| 48 | [64,64,128,4,2,16] | 0.034304 / 0.034048 | 0.033024 / 0.033024 | −3.88% / −3.10% | 否 |
| 56 | [64,64,128,4,1,16] | 0.035072 / 0.035072 | 0.033792 / 0.033280 | −3.79% / −5.38% | 否 |
| 72 | [64,64,128,4,1,8] | 0.044800 / 0.044800 | 0.034048 / 0.033792 | −31.58% / −32.58% | 否 |
| 80 | [64,64,128,4,1,8] | 0.045312 / 0.045312 | 0.033792 / 0.033536 | −34.09% / −35.11% | 否 |
| 96 | [64,64,128,4,1,8] | 0.046592 / 0.045568 | 0.034560 / 0.034816 | −34.81% / −30.88% | 否 |
| 104 | [64,64,128,4,1,8] | 0.046848 / 0.046848 | 0.038656 / 0.039680 | −21.19% / −18.06% | 否 |

要点：

- **O M=3 是唯一两套独立 seed 稳定超过 1% 的有效改善点**（9.01%/9.01%），与 11-05 完成审计中原始 wide-01 的 O 极小 M 正收益方向一致；且本次为 launch-fix 修复后 warps/stages 实际生效的重测。
- O M=2/8 的 Split-K 中位数低于 BF16，但 BF16 对照自身波动 spread>2%，按协议保守判为对照不稳，不计有效改善。
- O M=24 起全部负收益，且随 M 增大恶化至约 −35%：大 M 下输出 tile 数已足以填满 104 SM，Split-K 的 partial 写出 + workspace + reduce/scale 开销成为净负担。这与“Split-K 适用于并行度不足的小 M”的机制一致，为算法分界提供了 launch-fix 后的同机证据。

## 5. 与已有改动的关系

- 本实验是 `2026-11-05-w8a16-splitk-launch-cache-fix.md` 修复后的完整重测收尾；原始 wide-01 归档（`2026-11-05-w8a16-completed-sweeps-analysis.md`）保留不动。
- launch-fix-01 的 metadata 修改仅限 hostname 字段并有备份与日志；不影响 wide-01/dense-mid-01 历史归档的完整性。
- 与入口诊断（entry-audit）结论衔接：大 M 负收益点与“Split-K 不适合并行度充足场景”一致，后续算法分界应以本轮 launch-fix 后数据为准。

## 6. 合规性与风险

- 仅续跑既有诊断搜索，未改生产分派、官方参数、Benchmark、数据集或模型行为；synthetic eager 算子数据，不代表官方吞吐/TTFT/精度。
- **metadata hostname 字段修改是本记录的关键披露项**：修改前逐项核对 device uuid/驱动/源码 hash 等全部一致，修改已备份+日志留痕，原始文件 `metadata.json.orig` 保留。若评审要求严格实例一致，可用 `metadata.json.orig` 配合新目录重跑，代价是 60 点全部重测。
- 大 M 负收益不构成生产接入依据；赢家固化仍需等待同机 finalist 对照与模型级验证。O M3 的改善幅度需在真实 Shape 分组与端到端场景复验后才可计入收益。

## 7. 复现方式

```bash
# 容器内（登录 shell 加载 Conda/MACA 环境）
cd /root/experiments/2026-11-05-w8a16-splitk-launch-fix-01
export PYTHONPATH=/root/experiments/2026-11-05-w8a16-splitk-launch-fix-01/FlagGems/src:/root/experiments/2026-11-05-w8a16-splitk-launch-fix-01/vllm-plugin-FL
setsid /opt/conda/bin/python ./vllm-plugin-FL/benchmarks/diagnostics/tune_metax_w8a16_splitk.py \
  --output-dir ./results --resume > sweep-resume.log 2>&1 < /dev/null &
```

- 汇总：`results/summary.json`（complete=true）、`results/status.json`、`results/details.csv`。
- 指纹修改证据：`results/metadata.json.orig`、`results/metadata-hostname-fix.log`。
- 本地归档：`experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/launch-fix-01-results-final.tar.gz`，SHA256 `13271c69b459d25d29f28e1f808c7284b7ee20aac45ee511999d677499fba42e`（两端一致），解压于同目录 `launch-fix-01-final/`（60 个 result.json 已核对）。
- 上传包历史 SHA256：`fd4bae804702a3225dde79cfb90655e76a89d87aeaf32e1d732927721f957686`；`weight_int8pack_mm.py`（launch-fix 版）`bdb0b81bc9ee77408f4d28e68d5e79cf0b87176bd4407ea054976af2d382e783`。
