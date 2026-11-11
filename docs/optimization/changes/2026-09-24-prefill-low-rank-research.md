# MiniCPM5-2B Prefill 低秩 GEMM 方案调研

> 日期：2026-09-24  
> 模块：GEMM  
> 基线提交：`13eb9be`  
> 结果提交：尚未提交

## 目标

结合 MiniCPM5-2B 投影形状、MetaX C500 既有 Profiling、vLLM-FL 执行方式和 CUDA 低秩研究，从截断 SVD、rSVD、Block-LU/QR、CUR、Tucker-2 与 Cholesky 中筛选值得继续的方案。

## 结论

- 主方案：activation-aware、逐层逐投影异构 rank 的截断 SVD；
- 首选对象：少量非敏感层的 Down，其次是 Packed Gate/Up；
- 普通截断 SVD只作谱筛选与基线；
- rSVD 只作离线候选因子生成加速，不视为独立推理方案；
- Block-LU/QR、CUR、Tucker-2、Cholesky 暂不进入第一阶段；
- 低秩路径采用 Prefill-first，Decode 与 mixed batch 默认回退原 GEMM；
- 双份权重会增加显存并挤压长上下文 KV Cache，是主要系统风险。

## 修改范围

- 新增完整方案调研文档；
- 统一分类精确分解、低秩近似和离线求解算法，并记录全部候选的淘汰理由；
- 补充方案评分矩阵、Prefill/Decode 适用性、mixed batch、精度校准、CUDA 证据边界和最终推荐形态；
- 在优化文档入口和稳定上下文中加入调研结论；
- 未修改代码、配置、依赖或模型权重；
- 未执行新实验。

## 后续验证入口

进入实现前依次补齐：真实权重与 vLLM Linear 布局、投影 Shape trace、逐层奇异值谱、activation-aware 输出误差，以及 C500 两段 GEMM 的实测 break-even rank。
