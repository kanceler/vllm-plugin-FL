# 天数 GEMM tiling 优化记录

持续开发分支：FlagGems `perf/tianshu-gemm-tiling-optimization`。本地独立工作树 `worktrees/FlagGems-tianshu-down-fixed`，远端 `/workspace/FlagGems-tianshu-down-fixed`。插件基础版本f82fa167，FlagGems远端基础版本4df52d91；本地工作树基础版本不同，具体Iluvatar MM原始文件哈希一致，不能混写仓库提交。

| 编号 | 改动 | 官方4K吞吐 | 官方16K吞吐 | 状态 |
|---|---|---|---|---|
| [001](改动说明-001-天数GEMM-Down-tiling.md) | M=2048 Down固定tiling | +1.21962%，有效提升 | +0.011297%，正常波动 | 完整A/B结束；模型命中覆盖与精度待验证；未提交 |

后续优化沿用此分支，分别新增002、003等说明，不覆盖001事实。每项保留为什么改（基线与热点证据）、代码实现、算子/模型验证、与前项关系、合规风险及复现依据。收益以各自对应可比基线计算，叠加结果独立验证，不把单项收益相加。

原始归档：工作区 `experiment-artifacts/2026-11-06-tianshu-down-fixed-01/`，包含两套官方CSV与日志、命令、环境检查、correctness、独立Trace、代码patch、comparison.json及逐文件manifest。不是赛事最终交付清单。
