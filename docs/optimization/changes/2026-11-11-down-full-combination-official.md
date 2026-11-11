# Down量化融合与其他三投影专用MM配置组合测评

## 1. 基本信息与摘要

2026-11-11，C500。用户明确要求加入Gate/Up、QKV、O三个既有专用MM配置，并撤销刚刚未完整测完组合的最终测评资格，要求重新跑。新目录 `/data/down-fused-alltiles-official-20261111-01/`，启动脚本 `/data/run_down_fused_alltiles_official_20261111.py`。

## 2. 问题与原因分析

之前只开启Down量化新策略，其他三个投影仍默认MM，不能代表全部已有优化组合。用户要求最终组合纳入三项已验证配置；不得中途切换已有服务或复用其部分轮次。

## 3. 具体实现

复制已验证policy overlay，开启已有worker的C500 NT配置安装钩子。nt_gemm配置表仅保留M2048的Gate/Up、QKV、O：BM128/BN128/BK64、warps4、stages4、cpasync。删除该配置表中的Down旧条目，Down继续由量化内部区间策略选择（193～384 BM64，385～16384 BM128）。MM Kernel、转换BLOCK4096/warps4和scale融合实现均不变。Down-only量化覆盖42层，其余三投影保持BF16。

## 4. 实验与验证

前任务由用户要求中断：`down-fused-policy-official-20261111-02/user-interrupted.txt` 保留原因，已有日志/CSV归档为 `/data/down-fused-policy-interrupted-20261111.tgz`，不计为本组合正式成绩。核对PID和命令后停止本任务客户端及其bench子进程，原控制器退出并清理服务，确认相关PID均退出后才启动新任务。

新服务启动前检查GPU/端口/进程，归档源码与官方benchmark指纹；健康后要求42层Down、三个MM配置安装、新Down M2048策略选择证据，结束后核对三项实际MM命中。官方命令不变，两场景各4轮、跳过首轮后三轮平均。结果待完成。

## 5. 与已有改动的关系

组合前轮已验证Down融合区间策略和此前原生MM三个专用配置；未重新优化MM底层，也不把这三个配置的收益归因为量化本身。本组合与旧默认MM组合区分记录。

## 6. 合规性与风险

新进程重建模型与图状态，不热切换配置，不修改官方命令、数据或统计。中断任务保留原样，不补轮拼接。正式结果及math_500精度尚未完成，不宣称验收通过。未提交或推送。

## 7. 复现方式

实际serve/benchmark及环境见新目录wide/commands.json。独立源overlay `/data/down-fused-alltiles-20261111-source/overlay`；修改仅worker安装钩子和nt_gemm的三条配置选择，源指纹见wide/hashes.json。原始日志server.log/benchmark.log、raw_runs/summary在wide目录；源代码和状态归档不覆盖旧实验。
