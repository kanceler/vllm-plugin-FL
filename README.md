# vllm-plugin-FL

## 本分支：MiniCPM5-2B GEMM 量化优化

面向 **MetaX C500 / MiniCPM5-2B**，实现 Down 投影的 W8A16 weight-only 量化及 FlagGems 算子接入，覆盖 42/42 个 Down 层。权重在加载时按输出通道对称量化为 INT8，scale 为 FP32；激活和输出保持 BF16，GEMM 使用 BF16 dot、FP32 累加。INT8 用于权重存储，不代表原生 INT8 GEMM。

**当前结论：实现和官方同环境对照已完成，但 4K/16K 吞吐、TTFT 均未达到赛事门槛，暂不作为有效加速方案。** Down 权重数据约减半不等于整体显存减半，也不等于推理加速。

### 相对默认分支的改动

| 改动 | 实现与作用 |
|---|---|
| 加载时权重量化 | 新增 [`w8a16/linear.py`](vllm_fl/quantization/w8a16/linear.py) 与 [`reference.py`](vllm_fl/quantization/w8a16/reference.py)，在本地 TP 权重分片加载完成后量化 Down，保留 RowParallelLinear 的通信与 bias 处理逻辑。 |
| 编译图接入 | 新增 [`int8pack.py`](vllm_fl/quantization/w8a16/int8pack.py)，通过 custom op 调用 `flag_gems.weight_int8pack_mm`，支持动态图与 CUDA Graph。 |
| 小 M 后端优化 | 配套 FlagGems MetaX 后端对满足能力条件的 M=1/16/32/64 使用 Split-K=8 与整除特化，提高并行度、减少边界判断。 |
| 大 M 后端路径 | 满足能力条件的 M≥128 使用 INT8→临时 BF16 权重、MetaX BF16 GEMM、独立输出 scale；其他受支持 shape 保留通用融合。 |
| 回退与历史对照 | 保留独立反量化和旧手写融合用于对照；不支持的布局/bias 等回退到独立反量化后执行 BF16 GEMM。恢复原始 BF16 权重模式需关闭量化并重启服务。 |
| 验证工具 | 新增 [`单元测试`](tests/unit_tests/quantization/test_w8a16_down.py) 与 [`诊断脚本`](benchmarks/diagnostics/)，覆盖数值、分派、图重放及官方串行对照。 |

后端优化在**独立的 FlagGems 仓库**中，本插件分支不包含其源码；本地配套提交为 `f6a4257`（`feat(metax): add weight_int8pack_mm backend`，基点 `a7620cc`）。复现需要安装该配套版本，不能仅拉取本插件就假定后端优化已生效。

### 官方测评结果

2026-10-29，同环境、相同官方命令；每场景 4 轮，跳过首轮、取后三轮平均。吞吐单位 total tokens/s，TTFT 为 Mean、单位 ms。

| 场景 | BF16 吞吐 | W8A16 吞吐 | 吞吐变化 | BF16 TTFT | W8A16 TTFT | TTFT 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 4K `[4096,1024,64,256]` | 4989.25 | 4730.11 | −5.19% | 3180.14 | 3428.53 | +7.81% |
| 16K `[16384,1024,64,128]` | 7004.75 | 6821.13 | −2.62% | 27258.78 | 28161.24 | +3.31% |

两个场景均未满足吞吐≥基线×0.99、TTFT≤基线×1.01。最终组合回归 **22 项通过**；此前“小 M Split-K＋大 M 通用融合”版本 math_500 Level 3 为 **103/105（98.1%）**。最终组合改用大 M 独立 scale-after-dot 后仅补算子验证，未重新完整测精度，不能将 98.1% 视为最终组合的再次实测结果。

### 模式选择

在已有 Conda/MACA 环境及配套 FlagGems 中，启动新服务前显式选择模式，服务与 Benchmark 继续使用赛事原命令：

```bash
# 本次被测 W8A16 组合
export VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=1
export VLLM_FL_W8A16_DOWN_INT8PACK=1
export VLLM_FL_W8A16_DOWN_FUSED=0
```

```bash
# 原始 BF16 对照；切换后须重启服务
export VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=0
export VLLM_FL_W8A16_DOWN_INT8PACK=0
export VLLM_FL_W8A16_DOWN_FUSED=0
```

当前代码 `DOWN` 默认值为 `1`、`INT8PACK` 为 `0`，因此不设置变量会进入独立反量化模式，并非上述被测组合。评测前需按工作区规则归档、停止本任务残留进程、确认资源释放并核对版本与实际 import 路径。

详细依据：[阶段总结](docs/optimization/research/2026-10-29-w8a16-stage-summary.md) · [接入报告](docs/optimization/changes/2026-10-28-w8a16-int8pack-production.md) · [大 M 路径与算子数据](docs/optimization/changes/2026-10-29-w8a16-large-scale-after-diagnosis.md) · [官方测评报告及日志位置](docs/optimization/changes/2026-10-29-w8a16-official-comparison-negative.md)。以下保留默认分支的通用项目与安装说明。

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments. Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

## Version Compatibility

| vllm-plugin-FL Branch | Community vLLM Version |
|-----------------------|------------------------|
| `release/0.2` | [v0.20.2](https://github.com/vllm-project/vllm/tree/v0.20.2) |
| `main` | [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) |

## Supported Models and Chips

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved. The tables below summarize the current support status of end-to-end verified models and chips, including both fully supported and in-progress ("Merging") entries.

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| Qwen3.5-397B-A17B | Supported | [example](./examples/qwen3_5_offline_inference.py) |
| Qwen3-Next-80B-A3B | Supported | [example](./examples/qwen3_next_offline_inference.py) |
| Qwen3-4B | Supported | [example](./examples/offline_inference.py) |
| MiniCPM-o 4.5 | Supported | [example](./examples/minicpm/) |
| GLM-5 | Supported | [example](./examples/glm_5_offline_inference.py) |
| Qwen3.5-35B-A3B | Supported | [example](./examples/qwen3_5_offline_inference.py)  |
| BAAI/bge-m3 | Supported | [implementation](./vllm_fl/models/bge_m3.py) |
| MiniMax-M2.7 | Supported | [implementation](./examples/minimax_m27_offline_inference.py) |

### Supported Chips

| Chip Vendor | Status | Reference |
|-------------|--------|-----------|
| NVIDIA | Supported | - |
| Ascend | Supported | - |
| MetaX | Supported | - |
| T-Head | Supported | - |
| Iluvatar | Supported | - |
| Tsingmicro | Supported | - |
| Moore Threads | Supported | - |
| Hygon | Supported | - |
| Sunrise | Supported | - |

## Quick Start

### Setup

1. Install vLLM

    For **NVIDIA** GPUs, install vLLM from the official [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) release (optional if the correct version is already installed):
    ```sh
    pip install vllm==0.24.0
    ```

    For **non-NVIDIA** chips, install vLLM from source with the `empty` device target:
    ```sh
    git clone -b v0.24.0 https://github.com/vllm-project/vllm.git
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .
    ```

2. Install vllm-plugin-FL

    2.1 Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

    2.2 Install
    ```sh
    cd vllm-plugin-FL
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

    For CUDA-like devices, including CUDA and HIP/ROCm environments that use
    PyTorch's CUDA dispatch key, build the plugin native extension by setting
    `VLLM_VENDOR=cuda` during installation:
    ```sh
    cd vllm-plugin-FL
    VLLM_VENDOR=cuda pip install --no-build-isolation .
    # or editable install
    VLLM_VENDOR=cuda pip install --no-build-isolation -e .
    ```

    This builds and installs `vllm_fl._C`, which provides native C++ support
    required by some graph/custom-op paths, especially when vLLM is installed
    with `VLLM_TARGET_DEVICE=empty`.

    If `VLLM_VENDOR` is not set, vllm-plugin-FL is installed as a Python-only
    plugin and the native extension is skipped.

3. Install [FlagGems](https://flagos-ai.github.io/FlagGems/getting-started/install/)

    3.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.2 Install FlagGems

    ```sh
    git clone -b v5.3.4 https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

### Runtime compatibility hooks

The plugin installs runtime compatibility hooks through vLLM's plugin entry
points without modifying the installed vLLM package. Model-specific config and
model registrations are loaded only for their corresponding architectures.

Operator adapters use the plugin dispatch manager, so backend selection,
fallback, per-op policy, operator-list recording, and I/O diagnostics continue
to follow the common FlagOS controls.

4. (Optional) Install [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

    4.1 Clone the repository:
    ```sh
    git clone -b v0.13.0 https://github.com/flagos-ai/FlagCX.git
    cd FlagCX
    git submodule update --init --recursive
    ```

    4.2 Build the library with different flags targeting to different platforms:
    ```sh
    make USE_NVIDIA=1
    ```

    4.3 Set environment
    ```sh
    export FLAGCX_PATH="$PWD"
    ```

    4.4 Installation FlagCX
    ```sh
    cd plugin/torch/
    FLAGCX_ADAPTOR=[xxx] pip install . --no-build-isolation
    # or editable install
    FLAGCX_ADAPTOR=[xxx] pip install -e . --no-build-isolation
    ```
    Note: [xxx] should be selected according to the current platform, e.g., nvidia, ascend, etc.


If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.

### Additional Steps for Ascend

1. Install [FlagTree](https://github.com/flagos-ai/flagtree/)

    ```sh
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
    python3 -m pip install flagtree==0.6.1rc1+ascend3.5 $RES
    ```

    For other chips, please refer to [FlagTree](https://github.com/flagos-ai/flagtree/) for the corresponding version (e.g., `flagtree==0.6.1+iluvatar3.6`, `flagtree==0.6.1+metax3.6`, etc.).

2. Set required environment variable

    ```sh
    export TRITON_ALL_BLOCKS_PARALLEL=1
    ```

3. Enable eager execution

    Ascend requires eager execution. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.


### Run a Task

#### Offline Batched Inference
With vLLM and vLLM-fl installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [offline_inference](./examples/offline_inference.py). Or use blow python script directly.
```python
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Advanced use

For dispatch environment variable usage, see [environment variables usage](./vllm_fl/dispatch/README.md#environment-variables).

### Using Cuda Communication library
If you want to use the original Cuda Communication, you can unset the following environment variables.
```sh
unset FLAGCX_PATH
```

### Using native CUDA operators
If you want to use the original CUDA operators, you can set the following environment variables.
```sh
export USE_FLAGGEMS=0
```
