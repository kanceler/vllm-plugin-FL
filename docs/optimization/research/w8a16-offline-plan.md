# 全目标 Linear 离线 W8A16 实施计划

日期：2026-10-22（UTC+8）。用户已确认计划，并要求新增独立优化开关。本文件是计划，不表示离线转换或全量精度已完成。

## 顺序与边界

先完成当前 C500 原 Triton Down-only 的官方 math_500 Level 3；不重跑 BF16 Baseline。之后实施全目标离线量化、加载与严格覆盖审计，再执行全量 Level 3。精度目标 >0.95，需同时核对有效样本、请求错误、异常输出；数据集 Level 3 为 105 题。历史 BF16 0.962 仅作用户指定参考。

复用当前 INT8 per-output-channel 对称量化函数、FP32 scale、独立反量化和原 BF16 GEMM。不做融合，不改变量化算法、官方服务或评测命令，不修改评测数据，不根据评测题目/答案校准或选层。Embedding、LM Head、Norm 和 Attention 主体不属于目标；Q/K/V/O 投影属于本次目标。

Down-only 的 C500 数值路径使用归档的 `dequantize_weight`。保留本机用户对开关及 reference 路径的改动，不把它们静默当作已经在 C500 验证的版本；每次运行单独保存源码及指纹。

## 新增开关

| 模式 | VLLM_FL_W8A16_DOWN | VLLM_FL_W8A16_OFFLINE | VLLM_FL_W8A16_CHECKPOINT |
|---|---|---|---|
| 原 BF16 | 0 | 0 | 不使用 |
| Down 加载时量化 | 1 | 0 | 不使用 |
| 全量离线 | 0 | 1 | 离线 checkpoint 目录 |

新开关 `VLLM_FL_W8A16_OFFLINE` 默认 0；保留现有 Down 开关。两种量化模式同时启用时报错，避免重复量化。环境变量须在进程启动前设置，不用于热切换已加载的模型。新开关仅控制自研实现，不启用框架已有量化功能。

## 拟修改文件

- 新增 `vllm_fl/quantization/w8a16/checkpoint.py`：manifest、离线导出、文件指纹和验证。
- 新增 `vllm_fl/quantization/w8a16/offline.py`：选择目标 Linear、直接创建 INT8 参数、加载 FP32 scale、独立反量化、严格覆盖审计。
- 新增 `tools/quantize_w8a16.py`：离线转换入口，禁止覆盖原始 BF16 模型或已有输出。
- 修改 `vllm_fl/__init__.py`：先检查新离线开关，否则保留现有 Down 接入。
- 新增相应测试和 `docs/optimization/changes/` 记录。

## Checkpoint 与加载契约

按原始 checkpoint 的每个逻辑投影完整行计算 scale，输出 INT8 权重和 FP32 scale；TP 分片时不再量化，因此 checkpoint 不依赖 TP 数量。RowParallel 的 scale 沿 K 不分片，复制到各 rank；ColumnParallel 按输出通道分片 scale。packed QKV 使用框架原有 Q/K/V 分片及 GQA KV 复制语义；Gate/Up 保持两段边界。原 Down-only 在 TP>1 时是本地分片 absmax，不能声称与此全局离线 scale 完全相同；当前官方单卡不存在该差异。

离线输出新目录，保留原权重键名及配置/分词器；目标 weight 为 INT8，非目标保持原精度。Scale 和 manifest 放在独立子目录，避免被默认 root safetensors iterator 错读。Manifest 包含数值契约、源模型配置/文件指纹、逻辑到 packed 模块映射、shape、dtype 及文件校验。

启用自研离线 loader 时，仅将模型权重来源重定向到经校验的离线目录，官方 `vllm serve /workspace/MiniCPM5-2B ...` 参数不变，原 BF16 文件不移动、不覆盖。创建目标参数时直接分配 INT8，使用原权重加载器分片与拼接，不先载成 BF16 再量化。缺失/错误 manifest、weight、scale 或不支持的目标类型明确失败，不静默回退。

## 覆盖与验证

真实 checkpoint 为 42 层，每层 7 个逻辑目标（Q/K/V/O/Gate/Up/Down），预期 294 个逻辑投影；框架合并后每层 4 个模块，预期 168 个。实现以可再生 manifest 为准，不把固定层号或评测长度硬编码进 kernel。覆盖报告同时给出目标模块数、逻辑权重数、参数量、dtype、scale dtype、缺失和回退，不能把保留 BF16 的 Embedding/LM Head 算作意外回退。

验证顺序：转换往返及源文件不变 → 权重直接 INT8 加载 → Row/Column/packed QKV/Gate-Up 分片测试（含 GQA KV 复制）→ 模型启动与 100% 目标覆盖 → 独立输出检查 → 官方 Level 3。GPU 算子测试与服务串行，不重现已观察到的多进程图测试停顿。真实多卡条件不足时明确区分分片单元测试和多卡模型实测。

各版本分别归档源码、开关、checkpoint 指纹、完整命令、覆盖报告、原始评测日志和结果。Down-only 与全量结果分开报告；无正式吞吐/TTFT 数据时不声明性能验收。
