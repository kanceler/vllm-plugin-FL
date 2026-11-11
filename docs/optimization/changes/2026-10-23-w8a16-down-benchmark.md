# 2026-10-23：Down-only 与 BF16 性能记录

用户确认执行原始赛事 benchmark，保留现有 `RUNS=4`、`SKIP_FIRST=1`，不改代码或官方参数。本次仅关闭自研开关补测 BF16，两场景各四轮，1536/1536 请求成功。没有重跑 BF16 精度、实现融合或扩展全模型。

BF16 summary：4K total tokens/s 4937.19、Mean TTFT 3309.85 ms；16K total tokens/s 7009.97、Mean TTFT 27241.61 ms。4K 已存 Down-only summary 为 4667.20、3407.75，相对下降 5.47%、TTFT 上升 2.96%。候选 16K 仅首轮，不构成完整成绩；两次容器/GPU bus 不同，不视为同一物理实例的严格 A/B。

权重独立反量化方案精度已验证不代表性能通过。建议后续讨论融合验证，保持 BF16 舍入语义，先测真实 shapes 完整路径，再进行模型评测；没有宣称融合收益。

完整结果、原始日志、异常与实际命令在工作区 `benchmark_results/bf16-baseline-comparison-20261023.md` 与 `artifacts/bf16-baseline-20261023-01/`。本仓库该记录不包含工作区外层成果文件，交付时需一并归档。实验结束已停止本次服务，GPU 无进程。未提交或推送代码。
