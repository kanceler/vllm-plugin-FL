# MiniCPM5-2B 优化方案评审复现说明

本说明以赛事预装环境为起点，分别复现沐曦 C500 **完整 F** 与天数 **静态KV＋#011–#013**，不要求重跑消融。F包括Down W8A16、配套融合GEMM及专用配置、QKV/O/GateUp BF16热点配置、Top-p、GEMM分桶、自适应Prefill chunk与逐层同步消除。W8A16使用BF16激活、INT8权重在计算时转换为BF16、FP32累加及scale输出，不是原生INT8×INT8 GEMM。

天数与沐曦的复现可以按任意顺序进行。每个平台在对应的赛事容器中新开终端，先初始化该平台的软件栈，再安装本包的两份配套源码。安装后完整执行对应的环境开关配置，再按官方命令启动和评测。沐曦按第3节初始化，安装后执行第4～8节。天数按第9节初始化，执行第3节的共同安装和第4节的数据准备，再回到第9节配置并启动。

## 1. 交付与固定目录

```text
参赛队伍名/
├── FlagGems/
├── vllm-plugin-FL/
├── readme.md
└── report.pdf
```

评审将对应源码部署至 `/workspace/FlagGems`、`/workspace/vllm-plugin-FL`。模型和数据由赛方预置：

| 内容 | 路径 |
|---|---|
| 原始模型 | `/workspace/MiniCPM5-2B` |
| math_500 | `/workspace/evalscope-datasets/math_500` |
| 精度输出 | `/workspace/evalscope-datasets/level3` |
| 性能CSV | `/workspace/benchmark_results/` |

保留源码中的manifest、运行配置和冻结Kernel依赖。部分运行路径引用 `benchmarks/diagnostics/` 下的Python模块及JSON，这些属于运行所需源码，不能因为位于diagnostics目录而全部删除。无需分发模型权重、数据集、历史日志、编译缓存或Git内部目录。

## 2. 沐曦实测软硬件与依赖

| 项目 | 实测环境 |
|---|---|
| GPU | 单卡MetaX C500，65536 MiB |
| 驱动 / MACA | 3.8.30 / 3.7.0.38 |
| 系统 | Ubuntu 22.04.3 LTS，x86_64，glibc 2.35 |
| 容器可见内核 | Linux 5.15.0-58-generic |
| CPU | 容器可见128逻辑CPU，型号字符串i7-8550U为虚拟化呈现值 |
| Python | 3.12.11，Conda `/opt/conda` |
| PyTorch | `2.8.0+metax3.7.0.7` |
| Triton发行版 | `3.0.0+metax3.7.0.7`，模块版本3.0.0 |
| vLLM | `0.24.0+empty` |
| evalscope | 1.11.1 |
| transformers / tokenizers | 5.9.0 / 0.22.2 |
| numpy / safetensors | 1.26.4 / 0.7.0 |
| huggingface-hub / sentencepiece | 1.17.0 / 0.2.1 |
| einops | 0.8.1 |
| pip / setuptools | 24.3.1 / 80.10.2 |
| gcc / cmake / ninja | 11.4.0 / 4.4.0 / 1.11.1.4 |
| 已安装插件 / FlagGems元数据 | `0.3.0rc1.post1+g13eb9be69` / 5.3.5；实际代码由源码路径和manifest决定 |

复用赛事镜像内的厂商依赖，不用通用CUDA/CPU wheel替换MACA PyTorch或Triton，也不拉取最新上游作为本方案。赛事预装环境标识由平台提供，本说明不编造镜像tag。

## 3. 环境初始化、源码部署与编译

安装和测评前确认GPU及9031端口无冲突。精度与性能串行执行，切换源码或优化开关必须新服务；不删除磁盘Triton/JIT/FlagGems调优缓存。不得停止其他人的任务。

下列初始化仅适用于沐曦。天数使用第9节CoreX初始化，不执行本段Conda/MACA命令；随后两平台均执行本节共同安装步骤。赛事预装环境已初始化时核对实际Python和路径即可：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/mxdriver/lib:${LIBRARY_PATH:-}
export PATH=/opt/conda/bin:/opt/conda/condabin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
which python
python --version
mx-smi
```

将两份配套源码分别放到 `/workspace/FlagGems` 和 `/workspace/vllm-plugin-FL`，然后执行下列安装命令。

按赛方方案安装源码，不重装厂商软件栈：

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.3.5
cd /workspace/FlagGems
python -m pip install --no-build-isolation .
cd /workspace/vllm-plugin-FL
env -u VLLM_VENDOR python -m pip install --no-build-isolation .
```

源码交付包不含Git元数据；上述 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.3.5` 为源码包构建明确指定包版本，避免setuptools-scm无法推导版本而安装失败。它不改变Kernel、提供者绑定或优化开关，运行时使用下文配置的源码路径。

不设置不受当前setup支持的 `VLLM_VENDOR=metax`。本方案自研Kernel由Triton在服务初始化和实际调用时JIT编译，无需额外编译自研C++扩展。

## 4. 模型与数据准备

使用赛方原始MiniCPM5-2B及完整tokenizer、配置、权重索引和权重分片，不另行下载预量化模型。量化在模型加载时完成，不需用math_500校准。检查文件与数据条数：

```bash
python - <<'PY'
from pathlib import Path
import json
model = Path('/workspace/MiniCPM5-2B')
cfg = json.loads((model/'config.json').read_text())
assert (cfg['num_hidden_layers'], cfg['hidden_size'], cfg['intermediate_size']) == (42, 2048, 6144)
index = json.loads((model/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (model/name).is_file(), name
data = Path('/workspace/evalscope-datasets/math_500/test.jsonl')
rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
assert len(rows) == 500
assert sum(str(row['level']) in ('3', 'Level 3') for row in rows) == 105
print('MODEL_FILES_AND_DATASET_OK')
PY
```

性能脚本使用其原有random负载，无需准备额外prompt文件。

## 5. 设置沐曦完整F环境

**官方启动和评测命令不加自定义参数；完整F由源码和下列环境变量选择。** 仅设置 `VLLM_PLUGINS=fl` 不能保证完整F。以下在启动服务的终端执行。

```bash
unset VLLM_CACHE_ROOT VLLM_VENDOR FLAGGEMS_ENABLE_OPLIST_PATH FLAGGEMS_W8A8_LINEAR_OP FLAGGEMS_W8A8_MOE_OP
for name in $(env | cut -d= -f1 | grep '^VLLM_FL_'); do unset "$name"; done
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=1
export VLLM_FL_W8A16_DOWN_INT8PACK=1
export VLLM_FL_W8A16_SHAPE_DISPATCH=1
export VLLM_FL_W8A16_FORCE_SEPARATE=0
export VLLM_FL_C500_GEMM_BUCKET=1
export VLLM_FL_C500_TOP_P_SELECTION=1
export VLLM_FL_ADAPTIVE_CHUNK_ENABLED=1
export VLLM_FL_C500_NT_TILING=1
export VLLM_FL_C500_ATTN_NO_HOST_SYNC=1
```

最后两项 C500 优化开关分别启用 BF16 专用分块配置和 Attention 无主机同步路径。

## 6. 官方启动与简单测试

在上述终端原样运行官方沐曦启动命令：

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

服务就绪后核对Down量化42层、`C500_W8A16_SCALE_POLICY`命中、`C500 NT GEMM buckets ready:`、BF16热点安装/命中和自适应scheduler；实际采样后可见 `C500 FP64 top-p mass selection active`。预热命中点数量会随负载变化，不要求精度负载一定命中性能负载的全部六点。发现provider不匹配、量化数量错误或异常回退时先排查，不把错误模式成绩计入全F。

另一终端可按赛方示例简单测试；它只检查接口，不替代正式精度或性能。正式复现时在新服务初始化后让官方流程自行warmup，不在benchmark内部插入额外请求。

```python
import requests
import json

response = requests.post(
    'http://127.0.0.1:9031/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'minicpm',
        'messages': [{'role': 'user', 'content': 'introduce llm'}],
        'max_tokens': 256
    }
)
print(response.json())
```

## 7. 官方精度命令及校验

在客户端终端原样执行；输出目录保留历史，不添加limit、seed或更改采样参数：

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

根据本次日志打印的时间戳目录查看评测结果。`--ignore-errors`不代表出错也算完整：必须检查本次reports、predictions、reviews均对应105题，requested=105、succeeded=105、errored=0、incomplete=false，并复核逐题得分。工具自动保存结果；查看本次输出，不从历史目录择优取分。

检查本次报告中的准确率达到 0.950，并检查输出是否存在明显异常。生成使用 temperature=1.0，分数可能波动。

## 8. 官方性能命令及校验

性能与精度分别使用新的对应平台主方案服务，确认前任务退出、资源释放后再开始。性能命令原样执行：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

保留脚本原有RUNS=4、SKIP_FIRST=1，每场景四轮，正式summary取第2/3/4轮平均；raw_runs共8行、summary共2行。每轮成功请求数分别256/128、状态SUCCESS；检查输入/输出token总数、失败请求和TTFT，不只看吞吐。

查看本次日志打印的 `/workspace/benchmark_results/summary_*.csv`，以 `Total tok/s` 为吞吐指标，同时查看Mean TTFT；`raw_runs_*.csv`由脚本自动生成，用于核对原始四轮与后三轮平均。无需额外复制或归档CSV，也不要求重跑消融。按赛事要求提交测评工具生成的结果，排名以组委会统一复现为准。

4K 和 16K 分别用同环境、同命令的可比基线计算吞吐及 TTFT 变化。吞吐提升严格超过 1% 才算有效提升，TTFT 增加超过 1% 为有效退化。

### 沐曦完整 F 复现参考结果

使用提交 `ceca4b1` 在沐曦 C500 上复现，Math 500 Level 3 共评测 105 题，报告准确率为 **97.1%**。

| 轮次 | 4K 吞吐（tok/s） | 4K 平均 TTFT（ms） | 16K 吞吐（tok/s） | 16K 平均 TTFT（ms） |
|---|---:|---:|---:|---:|
| 1（不计入均值） | 8964.59 | 2274.60 | 10368.92 | 16854.62 |
| 2 | 9094.18 | 2024.10 | 10327.03 | 16980.08 |
| 3 | 9090.04 | 2014.32 | 10332.46 | 16954.02 |
| 4 | 9063.12 | 2041.42 | 10312.79 | 16964.26 |
| **正式汇总（第 2～4 轮）** | **9082.45** | **2026.61** | **10324.09** | **16966.12** |

八轮均成功。复现时以本次执行生成的官方报告和汇总为准。

## 9. 天数 BI-V150 完整复现

2026-11-18现场环境：Ubuntu24.04.2、glibc2.39、Python3.12.11，单卡Iluvatar BI-V150 32768MiB；驱动4.5.0、IX-ML4.4.0；Torch发行版2.10.0+corex.4.5.0（模块2.10.0），Triton模块3.6.0，vLLM0.24.0+empty，evalscope1.11.1、transformers5.14.1、tokenizers0.22.2、numpy1.26.4、safetensors0.8.0。

先在天数容器的独立终端初始化 CoreX：

```bash
export PATH=/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
which python
python --version
```

在同一终端完成第3节的共同安装和第4节的数据准备，再完整执行以下环境配置：

```bash
unset VLLM_CACHE_ROOT VLLM_VENDOR FLAGGEMS_ENABLE_OPLIST_PATH FLAGGEMS_W8A8_LINEAR_OP FLAGGEMS_W8A8_MOE_OP
for name in $(env | cut -d= -f1 | grep '^VLLM_FL_'); do unset "$name"; done
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=fl
export VLLM_FL_K_STATIC_FOLD=1
export VLLM_FL_INT8_KV_REPAIR=1
export VLLM_FL_K_TOP4=0
export VLLM_FL_K_TOP4_JOINT=0
export VLLM_FL_V_ONLY=0
export VLLM_FL_MIXED_ATTN_SPLIT=1
export VLLM_FL_BIV150_MIXED_EXTENDED_SPLIT=1
export VLLM_FL_BIV150_DECODE_KV_PREFETCH=0
export VLLM_FL_BIV150_DECODE_DIRECT_INDEX=1
export VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING=1
export FLAGGEMS_ILUVATAR_BF16_HOTSPOTS=1
export VLLM_FL_TIANSHU_W8A16_DOWN=0
export VLLM_FL_TIANSHU_W8A16_GATE_UP=0
export VLLM_FL_TIANSHU_W8A16_QKV=0
export VLLM_FL_TIANSHU_W8A16_O=0
```

静态KV要求原模型42层校准、page32/276B布局以及绑定的vLLM Attention提供者，源码会自动适配缓存分配；不增加serve参数。与其他KV量化模式互斥，切换必须重启服务。

官方启动命令原样如下：

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```


服务就绪后核对静态KV初始化42层、workspace和FULL_DECODE_ONLY图捕获成功；简单测试使用第6节的 Python 请求。

`VLLM_FL_BIV150_MIXED_EXTENDED_SPLIT=1` 在现有 Mixed 分相上同时启用两条路径。长 Prefill 在首时，连续 Decode 直接切片。长 Prefill 居中且满足容量和布局条件时，两侧 Decode 合并走 3D Attention 和 Reduce，再按原位置回写。实际命中居中布局时，服务日志会出现 `Static KV Mixed gather active`。旧版完整组合可将此开关设为 `0` 复现。天数最终组合保持 BF16 Decode KV 页预取关闭。

### 天数精度测评

天数服务就绪后，在同一容器的客户端终端运行：

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

查看本次输出目录，按第7节核对 Math 500 Level 3 的完成数、错误数和准确率。

### 天数 4K / 16K 性能测评

天数服务就绪后，在同一容器的客户端终端运行：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

结果写入 `/workspace/benchmark_results/summary_*.csv`，按第8节核对吞吐和 TTFT。

#### 完整热点组合诊断记录

以下为天数完整热点组合（全部大 M 热点及 M64 O、QKV）的两轮端到端诊断。保留第 1 轮供查阅，第 2 轮用于该次诊断对照。

| 场景 | 轮次 | 总吞吐（tokens/s） | Mean TTFT（ms） | 成功请求 |
| --- | ---: | ---: | ---: | ---: |
| 4K | 1 | 3175.56 | 13904.71 | 256/256 |
| 4K | 2 | 3734.23 | 4209.22 | 256/256 |
| 16K | 1 | 2879.74 | 91889.49 | 128/128 |
| 16K | 2 | 3118.40 | 72323.39 | 128/128 |

数据来自接入总开关前的隔离诊断。正式复现按上面的命令完整运行，以官方脚本输出的后三轮平均为准。

## 10. 可选：为本次运行指定独立缓存目录

通常使用赛事容器的默认缓存目录即可。两平台如需单独保存本次运行的 vLLM 缓存，可在完成第5节或第9节的环境配置后、启动服务前执行：

```bash
export VLLM_CACHE_ROOT="$(mktemp -d /tmp/fl-review.XXXXXXXX)"
```

`mktemp -d` 会创建目录。切换方案时仍需重启服务。

## 11. W8A16 跨平台可选适配

天数提供四类投影的独立 W8A16 开关。每项设为 `1` 启用对应投影的量化，设为 `0` 关闭。天数主复现方案按第9节执行，这四项均为 `0`。

| 投影类型 | 天数 W8A16 开关 | 主复现设置 |
|---|---|---|
| Down | `VLLM_FL_TIANSHU_W8A16_DOWN` | `0` |
| Gate/Up | `VLLM_FL_TIANSHU_W8A16_GATE_UP` | `0` |
| QKV | `VLLM_FL_TIANSHU_W8A16_QKV` | `0` |
| O | `VLLM_FL_TIANSHU_W8A16_O` | `0` |

需要在天数独立选择 Down 量化时，仅将表中的 Down 开关设为 `1`，其余三项保持 `0`，并将静态 KV 开关 `VLLM_FL_K_STATIC_FOLD` 和 `VLLM_FL_INT8_KV_REPAIR` 设为 `0`。切换后启动新服务。以上开关用于跨平台可选适配，不作为天数主复现方案。

沐曦完整 F 的 Down 量化按第5节配置，使用沐曦的 `VLLM_FL_W8A16_DOWN` 及配套开关。
