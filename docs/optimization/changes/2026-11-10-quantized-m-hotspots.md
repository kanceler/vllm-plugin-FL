# 量化后按M聚合的热点（有限窗口）

2026-11-10，MetaX C500。数据来自`/data/quantized-hotspots-20261110-01/observer/`，诊断随机token负载，不是正式Benchmark。按CPU launch correlation关联forward_context_num_tokens_padded，匹配weight_int8pack_mm_kernel的M耗时覆盖100%。只覆盖该Kernel，不称全部GEMM覆盖。

|场景窗口|窗口内排名|M|累计ms|该窗口匹配GEMM耗时占比|全部GPU Kernel耗时占比|Kernel次数|平均us|
|---|---:|---:|---:|---:|---:|---:|---:|
|4K起始Prefill/Mixed|1|2048|12251.61|100%|86.90%|10752|1139.47|
|16K起始Prefill/Mixed|1|2048|12313.50|100%|72.73%|10752|1145.23|
|16K Decode|1|56|528.71|100%|35.39%|5376|98.35|

4K Trace `trace-5bc39916a6c344c5918f239836712a61-pid59078.json`；16K起始`trace-4dd2eac102724901b89cd013c6c7a624-pid59078.json`；16K Decode `trace-74e745b4d231472fbf091ba614172d94-pid59078.json`。分组明细为同目录`.m-summary.json`。

不同窗口未按模型全程采样权重加权，禁止直接拼成全程M排名；每窗口仅一个M不能声称不存在其他热点，也不能说M56取代M64。4K Decode缺失；Prefill/Mixed标记可能有嵌套重复，step数需要去重后再报。此次均出现通用Kernel，尚未验证MetaX专用三算法接入，因此micro配置优劣不能外推实际模型。应先完成真实调用路径审计，再在同一明确采样协议下补齐M分布。本文为只读归档，无源码修改、未提交或推送。
