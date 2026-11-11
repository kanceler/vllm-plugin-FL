# 将 GEMM 主线确定为 W8A8 量化

> 日期：2026-10-12  
> 模块：GEMM  
> 基线提交：`13eb9be`  
> 结果提交：尚未提交

## 目标

基于 C500 能力、赛题合规边界和仓库现有量化代码，确定 GEMM 量化的具体格式、实施顺序、代码落点与止损条件。

## 结论

- 第一版采用静态 per-output-channel weight + 动态 per-token activation 的对称 W8A8；
- INT8×INT8 使用 INT32 accumulate，scale 在 epilogue 中恢复为 BF16；
- 不优先 FP8，因为 C500无原生 FP8；W4A16 作为后续高风险方向；
- 复用仓库现有 W8A8 Linear registry、reference、packed scheme 和动态量化代码；
- 新开发重点是 C500 capability gate、原生 MetaX scaled-MM、BF16 load-time量化，以及 RMSNorm/SiluAndMul 与 activation quant融合；
- MVP 先覆盖 Down，再扩展 Packed Gate/Up；
- 若完整 W8A8路径不快于 BF16或精度无法达到 0.950，则及时停止。
- 路线以 ADR 固化；模型、数据集或 checkpoint变化通过 capability和可再生 manifest处理，不反复修改量化核心代码。
- MVP之后依次推进 Shape特化、前驱量化融合、scaled epilogue融合、权重预打包、Prefill/Decode双Kernel和自动混合精度。

## 修改范围

- 新增 GEMM量化主线实施文档；
- 更新优化入口、稳定上下文和第一阶段优先级；
- 未修改运行代码、依赖、配置或 Benchmark；
- 未执行硬件实验。
