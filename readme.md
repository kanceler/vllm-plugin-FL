# MiniCPM5-2B 优化方案评审复现说明

本说明以赛事预装环境为起点，复现沐曦 C500 **完整 F**，不要求重跑消融。F包括Down W8A16、配套融合GEMM及专用配置、QKV/O/GateUp BF16热点配置、Top-p、GEMM分桶、自适应Prefill chunk与逐层同步消除。W8A16使用BF16激活、INT8权重在计算时转换为BF16、FP32累加及scale输出，不是原生INT8×INT8 GEMM。

本文件可作为最终交付根目录的 `readme.md`。当前待提交源码仍在整合，沐曦数据绑定已冻结F版本；加入后续天数改动后须复查身份与路径，不将其他源码版本自动视为已验证。天数官方启动命令列在末尾，整合版天数结果待设备验证。

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

每次评测先归档上一任务，核对并停止本任务残留服务和客户端，确认GPU及9031端口无冲突。精度与性能串行执行，切换源码或优化开关必须新服务；不删除磁盘Triton/JIT/FlagGems调优缓存。不得停止其他人的任务。

在本次实测容器中初始化如下；赛事环境若已经初始化，核对路径即可：

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

将两份配套源码放到固定目录后，**先在尚未安装的干净交付快照核验**：

```bash
cd /workspace/vllm-plugin-FL
python3 benchmarks/diagnostics/audit_quantization_integration.py
python3 benchmarks/diagnostics/audit_integrated_flaggems.py
```

当前整合清单同时核验插件内 `FlagGems/` 快照，所以暂须保留该副本，且与同级 `/workspace/FlagGems` 的源文件一致。最终拆分清单若改变，需要同步更新本步骤；不能删掉绑定文件再绕过审计。安装会生成build/egg-info，完整快照审计应在安装前执行，安装后不将构建产物纳入原始manifest。

按赛方方案安装源码，不重装厂商软件栈：

```bash
cd /workspace/FlagGems
python -m pip install --no-build-isolation .
cd /workspace/vllm-plugin-FL
env -u VLLM_VENDOR python -m pip install --no-build-isolation .
```

不设置不受当前setup支持的 `VLLM_VENDOR=metax`。本方案自研Kernel由Triton在服务初始化和实际调用时JIT编译，无需额外编译自研C++扩展。实测F复用预装entry point并以PYTHONPATH加载冻结源码；上述安装步骤是评审源码更新流程，不能将安装元数据版本当作实际运行源码身份。

## 4. 模型与数据准备

使用赛方原始MiniCPM5-2B及完整tokenizer、配置、权重索引和权重分片，不另行下载预量化模型。量化在模型加载时完成，不需用math_500校准。检查文件与数据条数：

```bash
python - <<'PY'
from pathlib import Path
import json, hashlib
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
assert hashlib.sha256(data.read_bytes()).hexdigest() == '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'
print('MODEL_FILES_AND_DATASET_OK')
PY
sha256sum /workspace/MiniCPM5-2B/model-00000-of-00001.safetensors
```

实测单分片权重SHA256为 `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`。校验应在评测前完成，勿与GPU评测并发读取完整权重。性能脚本使用其原有random负载，无需准备额外prompt文件。

## 5. 设置沐曦完整F环境

**官方启动和评测命令不加自定义参数；完整F由源码和下列环境变量选择。** 仅设置 `VLLM_PLUGINS=fl` 不能保证完整F。以下在启动服务的终端执行；每次使用新的实验目录，不覆盖历史结果。

```bash
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
export F_REPRO_DIR=/workspace/benchmark_results/full-F-$(date +%Y%m%d-%H%M%S)
mkdir "$F_REPRO_DIR"
export VLLM_CACHE_ROOT="$F_REPRO_DIR/vllm-cache"
env | sort | grep -E '^(VLLM_[^=]*|PYTHONPATH|MACA_PATH)=' > "$F_REPRO_DIR/run-environment.txt"
python - <<'PY'
from pathlib import Path
import importlib, hashlib
import vllm_fl, flag_gems
mm = importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
print('PLUGIN', vllm_fl.__file__)
print('FLAGGEMS', flag_gems.__file__)
print('MM', mm.__file__)
assert Path(vllm_fl.__file__).resolve().is_relative_to('/workspace/vllm-plugin-FL')
assert Path(mm.__file__).resolve().is_relative_to('/workspace/FlagGems')
assert hashlib.sha256(Path(mm.__file__).read_bytes()).hexdigest() == '3abfeba5c64c159370543602a2afc329efb6c7fc09e46659c11faa6b64485213'
assert hasattr(mm, 'mm_kernel_nt_scale_direct')
PY
python -m pip freeze > "$F_REPRO_DIR/pip-freeze.txt"
mx-smi > "$F_REPRO_DIR/gpu-before.txt"
```

后两个C500开关是消融接口，在实测原F中相应功能已启用；在当前含接口的源码中显式设置1，避免继承F−T/F−S配置。不同版本不能仅更新provider哈希冒称相同路径。

## 6. 官方启动与简单测试

在上述终端原样运行官方沐曦启动命令，保留完整服务日志：

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

记录命令退出码和日志中的本次时间戳目录。`--ignore-errors`不代表出错也算完整：必须检查本次reports、predictions、reviews均对应105题，requested=105、succeeded=105、errored=0、incomplete=false，并复核逐题得分。保存完整时间戳目录及服务、客户端日志；不要从历史目录择优取分。

以下以本次目录为例；重新运行时把EVAL_RUN_DIR改为客户端日志打印的本次输出目录，再执行校验：

```bash
export EVAL_RUN_DIR=/workspace/evalscope-datasets/level3/20261118_102223
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['EVAL_RUN_DIR'])
report = json.loads((root/'reports/minicpm/math_500.json').read_text())
reviews = [json.loads(x) for x in (root/'reviews/minicpm/math_500_Level 3.jsonl').read_text().splitlines()]
predictions = [json.loads(x) for x in (root/'predictions/minicpm/math_500_Level 3.jsonl').read_text().splitlines()]
summary = report['execution_summary']
assert summary['requested'] == summary['succeeded'] == 105
assert summary['errored'] == 0 and not summary['incomplete']
assert len(reviews) == len(predictions) == len({r['index'] for r in reviews}) == 105
assert all(r['sample_score']['score']['status'] == 'success' for r in reviews)
correct = sum(r['sample_score']['score']['value']['accuracy'] for r in reviews)
accuracy = correct / 105
print('CORRECT', correct, 'TOTAL', 105, 'ACCURACY', accuracy)
assert accuracy >= 0.950, 'accuracy below competition threshold'
PY
```

本次实测目录为 `20261118_102223`，报告 `reports/minicpm/math_500.json`。完整F结果为 **102/105=0.97142857（报告0.9714，97.14%）**，105题全部成功、0错误，达到最低准确率0.950。生成参数含temperature=1.0，复现分数可能波动；还需检查输出有无明显异常，不能只凭阈值宣称所有能力均已验证。

## 8. 官方性能命令及校验

性能与精度分别使用新的全F服务，确认前任务退出、资源释放后再开始。性能命令原样执行：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

保留脚本原有RUNS=4、SKIP_FIRST=1，每场景四轮，正式summary取第2/3/4轮平均；raw_runs共8行、summary共2行。每轮成功请求数分别256/128、状态SUCCESS；检查输入/输出token总数、失败请求和TTFT，不只看吞吐。实测官方脚本SHA256为 `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`。

按本次脚本输出的文件名设置RAW_CSV和SUMMARY_CSV（下列为已归档F的时间戳示例，重跑后需替换），执行：

```bash
export RAW_CSV=/workspace/benchmark_results/raw_runs_20261117_181414.csv
export SUMMARY_CSV=/workspace/benchmark_results/summary_20261117_181414.csv
python - <<'PY'
import csv, os
with open(os.environ['RAW_CSV']) as f:
    raw = list(csv.DictReader(f))
with open(os.environ['SUMMARY_CSV']) as f:
    summary = list(csv.DictReader(f))
assert len(raw) == 8 and len(summary) == 2
for length, requests in [(4096, 256), (16384, 128)]:
    runs = [r for r in raw if int(r['Prefill']) == length]
    assert len(runs) == 4
    for r in runs:
        assert r['Run Status'] == 'SUCCESS'
        assert float(r['Successful Requests']) == requests
        assert float(r['Total Input Tokens']) == length * requests
        assert float(r['Total Output Tokens']) == 1024 * requests
    row = next(r for r in summary if int(r['Prefill']) == length)
    for metric in ['Total tok/s', 'Mean TTFT (ms)']:
        expected = sum(float(r[metric]) for r in runs[1:]) / 3
        assert abs(float(row[metric]) - expected) <= 0.011
    print(length, row['Total tok/s'], row['Mean TTFT (ms)'])
print('RAW_AND_SUMMARY_OK')
PY
```

| 完整F | total tokens/s | Mean TTFT/ms |
|---|---:|---:|
| 4K | 8995.27 | 2039.00 |
| 16K | 10381.41 | 16819.88 |

这些数值归属冻结F和上述环境，不保证其他环境得到完全相同数值。官方分类以可比基线计算：吞吐提升严格超过1%才算有效提升；TTFT增加超过1%为有效退化，恰好1%仍为正常波动。4K/16K分别报告，不自行换算赛事分值。

## 9. 归档与验证范围

每次保留源码manifest、实际import路径、环境与依赖、开关、启动/评测命令、GPU与进程状态、健康与量化覆盖、服务和客户端日志、完整精度时间戳目录、性能raw及summary。停止本任务服务后确认9031和GPU释放。旧日志和持久缓存不删除。

已测F源码包SHA256：`b880572a2984027c2bf631463163a3f3d50eb848fce1c040b30d535e8c0ddbf4`，插件365项和FlagGems4782项逐文件校验通过。当前待提交分支包含后加消融接口，默认全开，但不能把整个工作树逐字节等同于冻结F。提交前应依据最终清单核验沐曦路径并补齐变更后的验证记录。

## 10. 天数官方启动命令（独立平台）

天数在其赛事预装环境中执行，不沿用上面的MACA初始化或C500开关；配套优化环境与整合版验证结果待天数源码对齐后补充。官方命令保持如下，不作为当前整合版天数已验收声明：

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```
