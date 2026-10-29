# MiniCPM5-2B 性能优化文档

本目录记录 MiniCPM5-2B 在 FlagOS/vLLM-FL 上的性能优化背景、实测数据、阶段计划与改动摘要。目标是让团队成员和 AI 在不依赖聊天记录的情况下，快速区分已验证事实、分析判断和待验证假设。

## 文档入口

- [`CONTEXT.md`](CONTEXT.md)：稳定的项目背景、基线、约束和当前状态；开始工作前优先阅读。
- [`competition-requirements.md`](competition-requirements.md)：精度、吞吐、TTFT、合规边界和成绩无效条件。
- [`decisions/0002-w8a16-mainline.md`](decisions/0002-w8a16-mainline.md)：GEMM 量化主路线（weight-only W8A16，当前有效）。
- [`decisions/0001-w8a8-mainline.md`](decisions/0001-w8a8-mainline.md)：已取代的 W8A8 主线决策（保留历史）。
- [`profiling/metax-c500.md`](profiling/metax-c500.md)：MetaX C500 的 Baseline、采样方法和六组 Kernel Profiling 结果。
- [`research/w8a16-plan.md`](research/w8a16-plan.md)：W8A16 weight-only 实施计划（阶段0、MVP、优化阶梯、验证策略）。
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
阅读 CONTEXT 与对应 profiling
→ 明确一个可证伪的假设
→ 单变量修改
→ 正确性与 Micro Benchmark
→ 4K/16K 端到端验证
→ 精度与 TTFT 回归
→ 重新 Profiling
→ 新增 changes/<日期>-<主题>.md
```

当前阶段只完成资料归档与方案设计，尚未开始新的实验或性能代码修改。

### 何时新增改动摘要

满足以下任一条件时新增一份 `changes/<日期>-<主题>.md`：

- 形成了可独立评审的代码或配置改动；
- 完成了一轮有明确结论的实验，即使结论是不采用该方案；
- 准备提交 PR、合并分支或向其他模块交接；
- 基线、关键假设或后续工作方向发生变化。

仅包含格式调整、拼写修正或同一阶段内的中间小提交时，不单独新增摘要。
