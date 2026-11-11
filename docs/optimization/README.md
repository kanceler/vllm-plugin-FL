# MiniCPM5-2B 性能优化文档

本目录记录 MiniCPM5-2B 在 FlagOS/vLLM-FL 上的性能优化背景、实测数据、阶段计划与改动摘要。目标是让团队成员和 AI 在不依赖聊天记录的情况下，快速区分已验证事实、分析判断和待验证假设。

## 文档入口

- [`plans/2026-10-30-w8a16-dispatch-autotuning.md`](plans/2026-10-30-w8a16-dispatch-autotuning.md)：当前执行入口；真实Shape/分支覆盖、三算法共享候选、O1/O2、调优审计、Runtime与完整测评矩阵。原生BF16 MM由队友负责。
- [`plans/2026-10-29-global-w8a16-followup.md`](plans/2026-10-29-global-w8a16-followup.md)：全局投影、容量、历史证据与测评流程；分工及搜索实施以10-30计划为准。
- [`CONTEXT.md`](CONTEXT.md)：稳定的项目背景、基线、约束和当前状态；开始工作前优先阅读。
- [`competition-requirements.md`](competition-requirements.md)：精度、吞吐、TTFT、合规边界和成绩无效条件。
- [`decisions/0002-w8a16-mainline.md`](decisions/0002-w8a16-mainline.md)：GEMM 量化主路线（weight-only W8A16，当前有效）。
- [`decisions/0001-w8a8-mainline.md`](decisions/0001-w8a8-mainline.md)：已取代的 W8A8 主线决策（保留历史）。
- [`profiling/metax-c500.md`](profiling/metax-c500.md)：MetaX C500 的 Baseline、采样方法和六组 Kernel Profiling 结果。
- [`research/w8a16-plan.md`](research/w8a16-plan.md)：早期 W8A16 实施计划和 Down MVP 历史；后续流程以 2026-10-30 三算法计划为准。
- [`research/gptq-feasibility.md`](research/gptq-feasibility.md)：GPTQ/W4A16 可行性分析（当前搁置）。
- [`plans/phase-1.md`](plans/phase-1.md)：第一阶段的三个模块、边界与协作依赖。
- [`changes/TEMPLATE.md`](changes/TEMPLATE.md)：阶段性推送或 PR 的改动摘要模板。

## 维护规则

1. 实测数据必须记录硬件、代码版本、模型、dtype、启动参数和负载参数。
2. 使用以下状态标签，避免把推测写成事实：
   - **已验证**：有代码、日志或可复现测试支持；
   - **分析判断**：由已有数据推导，但尚未做针对性实验；
   - **待验证**：尚无足够证据的候选方向。
3. `CONTEXT.md` 只保存稳定且当前有效的信息；详细数据放入 `profiling/`。
4. 每个**可独立评审的阶段性改动**在 `changes/` 下新增一份文档，不要求每次 push 或每个小 commit 都写摘要，也不维护所有人频繁修改的单一流水账。
5. 改动摘要应说明“为什么改、改了哪里、如何验证、结果如何、还有什么风险”，不要复制 Git diff。
6. 若框架层改动影响 Batch、Shape、Kernel 调用频率或 KV Cache 布局，相关算子结论应标记为需要重新 Profiling。

## 推荐工作流

```text
阅读 CONTEXT、当前计划与既有实验
→ 离线核对调用链、配置、形状与理论成本
→ 最小GPU补测、分组完整调用micro和有依据的内部优化
→ 选择独立开关组合、统一接入检查
→ 全局精度及官方4K/16K吞吐、TTFT
→ 如有失败，必要的定向消融
→ 每项代码改动同步更新changes/<日期>-<主题>.md
```

截至2026-10-30，Down W8A16已有实现、算子实验和官方负结果；见[阶段总结](research/2026-10-29-w8a16-stage-summary.md)。三算法联合调优与Runtime计划已归档，尚未实施新框架或GPU补测。原生BF16 MM底层及大M配置优化由队友负责，旧LM Head专项不自动纳入当前任务。各投影先micro，不逐组做完整模型评测；最终测选定的全局组合。

### 何时新增改动摘要

满足以下任一条件时新增一份 `changes/<日期>-<主题>.md`：

- 形成了可独立评审的代码或配置改动；
- 完成了一轮有明确结论的实验，即使结论是不采用该方案；
- 准备提交 PR、合并分支或向其他模块交接；
- 基线、关键假设或后续工作方向发生变化。

仅包含格式调整、拼写修正或同一阶段内的中间小提交时，不单独新增摘要。
