# W8A16 Down 第一阶段：独立反量化原型与 C500 接入验证

日期：2026-10-22（UTC+8）
状态：实现、单元测试及真实模型接入验证完成；用户要求暂不做精度评测，math_500 Level 3 未启动。未完成精度、吞吐或 TTFT 验收。

## 范围与代码

按用户批准的 `research/w8a16-plan.md`，采用独立反量化 + 现有 BF16 GEMM，未实施融合 GEMM。分支 `feat/gemm-w8a16-down`，起点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，本次未提交、未推送。原有未跟踪文档及诊断文件不默认纳入提交。

- 新增 `vllm_fl/quantization/w8a16/`：加载时使用 FP32 absmax/scale、round-to-nearest-even、对称 INT8 [-127,127]；全零通道 scale=1，非有限权重拒绝量化。
- 持久保存本地 TP 分片的 INT8 权重 [N,K] 和 FP32 scale [N]；每次 forward 独立计算 `BF16(FP32(Q) * scale[:,None])`，不长期缓存完整 BF16 权重。
- 插件初始化时安装幂等 post-load 适配。仅选择未量化的标准 RowParallelLinear、语义名 down_proj、CUDA-alike BF16 二维权重；不按固定层号或评测长度选择，不修改官方服务参数或启用现成量化配置。
- 成功量化后只在该 Down 上安装专用 Linear method；原 UnquantizedLinearMethod.apply 保持不变。反量化后的权重显式传给 vLLM 原 `dispatch_unquantized_gemm`，同时保留 batch-invariant 分支。上层通信和 bias 语义保持不变。
- 加载时不支持的候选记录回退原因并调用原逻辑。量化后无原始 BF16 副本，不兼容的激活 dtype/device 明确报错，不把有损恢复称为原 BF16 回退；kernel 编译或执行错误不会被静默吞掉。
- 新增 `tests/unit_tests/quantization/test_w8a16_down.py`，覆盖数值边界、分片契约、GPU 反量化/GEMM 对照、图重放和接入行为。FlagGems 未修改。

## 环境与复现依据

| 项目 | 实际值 |
|---|---|
| GPU | MetaX C500，64 GiB |
| 驱动 / MACA | 3.8.30 / 3.7.0.38 |
| Python | 3.12.11 |
| PyTorch | 2.8.0+metax3.7.0.7 |
| Triton | 3.0.0+metax3.7.0.7 |
| vLLM | 0.24.0+empty |
| evalscope / pytest | 1.11.1 / 8.4.1 |
| FlagGems 提交 | a7620cc191a0b42e040194622c5758b22a7a25dc |
| 模型 | MiniCPM5-2B，BF16，LlamaForCausalLM，42 层 |

远端起点提交与本地一致，初始工作区干净；只同步本次必要文件和工作区规则。容器原默认导入已安装插件副本，验证时设置 `PYTHONPATH=/workspace/vllm-plugin-FL`，并确认源码入口 `/workspace/vllm-plugin-FL/vllm_fl/__init__.py`，环境传给所有 worker；没有重装依赖。模型配置、数据集和源码指纹见实验归档。

## 已执行验证

在源码目录执行：

```bash
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL
python -m pytest tests/unit_tests/quantization/test_w8a16_down.py -q
```

| 轮次 | 实际结果 | 说明 |
|---|---|---|
| 初次 | 9 passed，33.99 秒 | 包含真实 Down N=2048/K=6144 及非整齐尺寸的 GPU 反量化/BF16 GEMM 对照 |
| 接入修复后 | 9 passed，33.45 秒 | Down 专用 method，非目标 apply 保持原路径 |
| 最终回归 | 9 passed，309.12 秒 | 增加非目标 apply 不得被替换的断言；与服务并存时停在图测试，停止服务后完成，不能将该耗时当作性能数据 |

测试覆盖零通道、对称端点、舍入、非有限值、非连续输入处理、本地 TP scale、GPU reference 对照、CUDAGraph 重放、安装幂等与非目标层原路径。隔离的 TP 分片测试不等同真实多卡通信验证。

服务使用官方沐曦原命令：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

- attempt01：编译阶段失败。全局 apply 包装器对非目标 MergedColumnParallelLinear 的缺失 `_fl_w8a16_ready` 属性构建 TorchDynamo guard 时触发 InternalTorchDynamoError。修复为仅在已量化 Down 上安装专用 method，没有修改编译参数。
- attempt02：修复后的启动尝试，为补齐量化覆盖日志而主动停止；不作为精度或性能结果。
- attempt03：服务正常启动，默认 FULL_AND_PIECEWISE 配置未更改。启动日志确认 42 个唯一 Down 前缀启用，加载回退 0；详见 coverage.json。
- 独立输出检查：请求“请计算17乘以23，给出最终结果。”返回 391，finish_reason=stop，生成 100 tokens，HTTP 请求耗时约 1.33 秒。它仅作输出与接入检查，不是正式时延或模型精度结果；请求及响应均归档。
- 模型服务保持默认 seed=0；GPU 对照测试内 seed=42。未为服务或正式评测修改随机性参数。

## 未执行与限制

- 用户明确要求暂不做精度评测：仅核对数据集等级计数（Level 3 为 105 题），没有读取题目用于调参，没有启动 evalscope 任务，未产生精度得分。
- 尚未执行正式 4K/16K 吞吐、TTFT 或完整路径性能评测；历史精度 0.962 及历史吞吐不作为本次新环境的实测结果。
- 独立反量化额外读取 INT8/scale，并写回完整 BF16 权重；不宣称加速或端到端显存收益。最终回归的多进程停顿提示后续 GPU 测试与服务应串行执行，具体底层原因尚未定位。
- 实测只覆盖 C500 单卡。天数平台、真实多卡 TP、混合精度扩展及其他投影尚未验证。
- 后续恢复评测时，应先核对源码指纹、重新按官方命令启动并确认覆盖，再使用官方 Level 3 命令；独立反量化版本通过也不能替代融合版或其他投影的重新验证。

## 归档与结束状态

实验目录：`/workspace/benchmark_results/w8a16-down-dequant-20261022-01/`。

归档包括三轮单元测试、各次服务日志、服务参数与源码导入路径、硬件/版本/输入指纹、候选源码快照、量化覆盖、独立请求/响应和结束状态。本地副本位于工作区 `artifacts/w8a16-down-dequant-20261022-01/`，不纳入框架仓库默认提交。

按用户要求停止后，检查确认 9031 端口不再接受连接，无本次 pytest 进程，且不存在 eval.pid；没有留下本次候选服务或精度评测任务。
