# BI-V150 Mixed Attention 居中 Prefill 的 Decode 收集回写

## 基本信息与摘要

2026-11-20，Iluvatar BI-V150。基于 submission 完整组合及首行长 Prefill 直接切片，扩展静态 INT8 KV Mixed 分相，覆盖长 Prefill 两侧的非连续 Decode。

## 问题与原因分析

此前 16K 观察有 91 次满足居中条件。长行前有 1–15 个单 token Decode，长行后有 39–54 个。原前缀分相要求至少 16 条，因此整批进入 2D Attention。首行直接切片两轮诊断的第 2 轮吞吐 +0.85%、Mean TTFT +0.61%，未达到 +3% 吞吐目标。

## 具体实现

`static_kv_compat.py` 在元数据阶段从 CPU Query 前缀识别 1–15 条 Decode 前缀、单个长度至少 1024 的长行、非空连续 Decode 后缀。总 Decode 数 16–80 且满足 scratch 容量时，构建紧凑 Decode 前缀、请求索引和 token 索引。每层收集 Q、block table 和序列长度，Prefill 使用原视图。静态 KV 服务沿用单次 KV 写入，在已有 3D Decode 和 Reduce 后回写输出。物理 INT8 KV 和按 token/head 的 scale 保持原布局。本次同步更新量化集成与静态 KV 源码指纹。

## 实验与验证

同输入三路实模小测覆盖 15 组和 91 个命中 step。相对原完整组合，两侧收集回写每层净省 4.68–6.38 ms，中位 5.68 ms。仅切后缀净省 3.98–5.96 ms，中位 4.79 ms。收集回写在 14/15 组领先，按覆盖和 42 层估算节省 21.90 s GPU 时间。Prefill 输出一致，最差 Decode 单行相对 L2 为 1.46e-3。现有 14 项 CPU 兼容性测试和 7 项 Torch CPU 布局、容量及回写检查通过。集成日志确认实际请求命中收集回写，首个记录是长行位置 1、55 条 Decode。完整 step 单独配对尚未执行。16K 两轮均为 128/128 成功。第 1 轮 2943.06 tok/s、Mean TTFT 90.67 s。第 2 轮 3210.86 tok/s、Mean TTFT 71.02 s。相对归档完整组合的第 2 轮 3095.75 tok/s 和 71.95 s，吞吐 +3.72%，Mean TTFT 降低 1.29%。正式 4K＋16K 四轮任务已按用户调整中止并保留日志。随后用同一叠加候选完成独立 4K 两轮诊断，运行 4096/1024/64/256 且跳过首轮，复用归档两轮基线。两轮均 256/256 成功，服务确认居中收集回写实际命中，首条为长行位置 1、62 条 Decode。

| 场景 | 原完整组合吞吐 | 候选吞吐 | 吞吐变化 | 基线 Mean TTFT | 候选 Mean TTFT | TTFT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 3739.63 tok/s | 3785.16 tok/s | +1.22% | 4207.47 ms | 4162.42 ms | −1.07% |
| 16K | 3095.75 tok/s | 3210.86 tok/s | +3.72% | 71946.17 ms | 71015.98 ms | −1.29% |

两组均用第 2 轮。核对了每场景输入、输出 token 数、成功请求数和汇总一致性。4K 首轮 3435.87 tok/s、Mean TTFT 6504.48 ms，按约定弃用。两轮诊断不充当正式四轮赛事结果。

## 与已有改动的关系

沿用静态 K Folding、INT8 KV Repair、紧凑 Decode 和热点配置。首行连续 Decode 使用直接切片，已有前缀分相继续处理适用布局。基线为同一完整组合的归档两轮诊断。

## 合规性与风险

候选沿用天数官方服务参数。两轮 16K 在独立目录诊断。异常模式、padding、多个长行和容量不足由既有逻辑处理。收集及回写的逐层分配、元数据构建成本和实际 TTFT 尚待端到端核对。完整 MATH Level 3 等用户确认。

## 复现方式

容器目录 `/workspace/experiments/tianshu-mixed-interior-20261120-01`。改前 A/B 小测见 `run_mixed_audit.py` 与 `audit/{pairs,coverage}.jsonl`。集成候选见 `candidate-16k/controller.sh` 和 `serve.sh`，两轮负载 `[16384,1024,64,128]`，第 2 轮汇总。映射检查见 `candidate-16k/mapping-check.log`，耗时、命中、异常看 `benchmark.log`、`serve.log` 和原始 CSV。
