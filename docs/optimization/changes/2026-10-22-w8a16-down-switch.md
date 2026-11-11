# W8A16 Down 独立反量化（weight-only）：完整记录与启停开关

日期：2026-10-22（UTC+8）｜设备平台：MetaX C500（沐曦）已验证，天数 Iluvatar BI-V150 待验证
涉及文件：`vllm_fl/quantization/w8a16/`（`__init__.py`、`reference.py`、`dequant.py`、`linear.py`）、`vllm_fl/__init__.py`、`tests/unit_tests/quantization/test_w8a16_down.py`

## 1. 基本信息与摘要

采用 weight-only W8A16：权重静态、对称、per-output-channel INT8 + FP32 scale，激活保持 BF16，计算仍走 BF16 tensor core。实现「加载时量化 + 每次 forward 独立反量化 + 原 BF16 GEMM」的 Down 路径，并新增显式开关 `VLLM_FL_W8A16_DOWN` 支持与 BF16 基线 A/B 对照与安全回退。解决 Decode 小 M 场景权重带宽主导下，W8A8（量化写显存 + 无原生 INT8 收益证据，synthetic M=1 慢约 7.8×）不可行、纯 BF16 权重读取量偏大的问题。

## 2. 问题与原因分析

- 原实现：Down 投影用 BF16 权重 + BF16 GEMM，权重 `[N,K]`（N=2048、K=6144）。
- Decode 小 M 时权重读取主导，BF16 权重带宽是主要开销；W8A16 把权重减半为 INT8，但激活不量化，绕开 A8 的「量化/反量化写显存往返 + 无 INT8 收益证据」。
- 计算仍是 BF16 tensor core，不依赖「原生 INT8 是否高效」这一未验证前提，风险低。

## 3. 具体实现

量化契约（`reference.py`）：FP32 求 `amax[n]=max_k(abs(W[n,k]))`，`scale[n]=amax/127`（全零通道 scale=1），`Q=round_to_nearest_even(FP32(W)/scale)` 后 `clamp(-127,127).to(int8)`；非有限权重拒绝量化。持久保存 `Q=[N,K] INT8` + `scale=[N] FP32`，无 group、无 zero point、无位解包。

- `dequant.py`：Triton kernel `INT8→FP32、×scale、→BF16`，`BLOCK=1024`。
- `linear.py`：插件注册时 `install_w8a16_down()` 幂等安装 post-load 适配器；仅选语义名 `down_proj` 的标准 `RowParallelLinear`、`UnquantizedLinearMethod`、BF16 CUDA-alike 二维权重；加载/TP 分片完成后量化本地分片，`layer.weight=INT8`、`register_buffer("w8a16_scale")`，每次 forward 独立反量化后走原 BF16 `dispatch_unquantized_gemm`（输出不再乘 scale）。非目标层保持原 `apply`，不支持项记录 fallback 原因。
- 启停开关（参考注意力 `VLLM_FL_ATTN_TILE_POLICY`）：

```python
def install_w8a16_down() -> bool:
    if os.environ.get("VLLM_FL_W8A16_DOWN", "1").lower() in ("0", "false", "no"):
        print("[FL] w8a16 down disabled by env; using upstream defaults.")
        return False
    ...
```

  默认 `1` 启用，`0`/`false`/`no` 关闭，回退到纯 BF16。

## 4. 实验与验证

| 项 | 结果 |
|---|---|
| 单元测试（C500） | 三轮各 9 项通过（含 GPU 反量化/BF16 GEMM 对照、CUDAGraph 重放、安装幂等、非目标层不受影响） |
| 服务启动（C500 官方命令） | 成功，42 层 Down 全部启用、加载回退 0 |
| 独立请求 | 17×23=391，finish_reason=stop，100 tokens |
| 量化范围/覆盖率 | 仅 Down（`down_proj`），覆盖率 42/42；Gate/Up、QKV、O、Embedding、LM Head、Norm、Attention、KV cache 保持 BF16 |
| 权重与 Scale 格式 | 权重 INT8 `[N,K]`，scale FP32 `[N]`，per-output-channel，对称 |
| 反量化方式 | 每次 forward Triton 独立反量化 `BF16(FP32(Q)*scale)`，不长期缓存完整 BF16 副本 |
| 精度损失 | math_500 Level 3 待验证（用户此前要求暂缓）；dequant 数值逐位一致（`rtol=0,atol=0`） |
| 开关行为 | 待验证（关断日志 `[FL] w8a16 down disabled by env; using upstream defaults.`、开启日志 `W8A16 enabled prefix=...`） |

**待验证**：math_500 Level 3 精度、正式 4K/16K 吞吐与 TTFT、天数平台、真实多卡 TP、开关运行时行为。

## 5. 与已有改动的关系

- 依赖前置 W8A16 Down 独立反量化原型（`2026-10-22-w8a16-down-dequant.md`）。
- 不修改 FlagGems、TP 通信、bias 语义或官方服务参数。
- 与 ADR-0002（W8A16 主线）一致；W8A8 已放弃、W4A16/GPTQ 搁置，不并行推进。

## 6. 合规性与风险

- 合规：自研量化的代码级开关，非「用启动参数开启现有量化」；量化有实质实现，开关仅用于 A/B 与回退。
- 风险：默认启用；精度不达标时可 `VLLM_FL_W8A16_DOWN=0` 回退（跳过安装即纯 BF16，代价为零）。
- 未验证：独立反量化会增加 INT8/scale 读取 + 完整 BF16 临时权重写回，不宣称性能收益；性能收益留待第二阶段融合版。

## 7. 复现方式

沐曦 C500（已验证，官方命令不含 `--compilation-config`）：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

天数 BI-V150（端到端官方命令）：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072

python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

开关对照（与注意力开关 `VLLM_FL_ATTN_TILE_POLICY` 同一模式）：

```bash
# 关闭本改动做对照（回到纯 BF16）
export VLLM_FL_W8A16_DOWN=0
# 开启（默认即开）
export VLLM_FL_W8A16_DOWN=1   # 或省略，默认 1
```

启动日志应出现：`W8A16 enabled prefix=... weight=int8 scale=float32 mode=separate_dequant`；关断时出现 `[FL] w8a16 down disabled by env; using upstream defaults.`。日志位置 `/workspace/vllm_serve.log`。
