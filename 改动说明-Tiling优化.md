# 改动说明：MetaX C500 GEMM Tiling 优化

本文说明基于 `flagos-2026-s2` 的四组 prefill GEMM 分块配置优化，包括问题、实现、实测结果和复现步骤。

## 1. 为什么做这次改动

### 1.1 优化对象

MiniCPM5-2B 在沐曦 C500 上执行官方 4k、16k benchmark 时，QKV、O、Gate/Up、Down 四类主干投影主要通过 FlagGems MetaX 的 `mm_kernel_nt` 完成矩阵乘法。两种场景均使用 chunked prefill，其中一组主要计算形状的 M=2048。

此前完整分支诊断还发现 O、Down 在尾部小 batch 中会使用 Split-K。因此本次优化限定在已经进入 NT 分支的四组大 M 形状，保留原有分支选择。

### 1.2 发现的问题

基线运行中，四组形状命中了已有调优缓存。实际配置能够正常执行，但独立 GEMM 测量发现，它们在本轮测试范围内并非最快。

FlagGems 的默认 NT 候选集较小；随包提供的 MetaX 扩展调优配置描述了更大的参数范围，但本次基线处于默认调优模式，没有遍历该扩展空间。

我们从默认候选和扩展范围中选取配置进行实测，每个 prefill 形状尝试了 54 组去重后的完整配置，对领先候选进行 5 轮复测，找到四组较优配置。扩展候选采用有限抽样，且偏向 stages=2；这些结果代表本轮搜索范围内的较优组合，不代表全局最优。

### 1.3 为什么需要落实到插件源码

先前通过修改独立调优数据库副本，已经验证四组配置组合能够改善端到端吞吐。但只保留实验数据库，无法让其他人克隆插件分支后自动获得相同配置，也容易受到缓存标识和版本变化影响。

本次将配置选择落实到插件源码：安装优化分支后，符合条件的 GEMM 自动使用已验证配置。部署不依赖分发实验数据库或人工替换缓存记录。

## 2. 具体改动与原理

### 2.1 改动范围

开发基线为：

```text
仓库：https://github.com/kanceler/vllm-plugin-FL
基线分支：flagos-2026-s2
基线提交：13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0
优化分支：perf/metax-c500-prefill-nt
```

生产代码涉及两个文件，共新增 102 行；另新增 95 行、7 项回归测试。

| 文件 | 改动内容 |
| --- | --- |
| [vllm_fl/dispatch/backends/vendor/metax/nt_gemm.py](vllm_fl/dispatch/backends/vendor/metax/nt_gemm.py) | 新增 C500 BF16 NT 配置表、条件判断及固定配置启动入口 |
| [vllm_fl/worker/worker.py](vllm_fl/worker/worker.py) | 新增 5 行，在设备设置完成后、模型构建和编译前安装配置特化 |
| [tests/unit_tests/patches/test_metax_nt_gemm.py](tests/unit_tests/patches/test_metax_nt_gemm.py) | 新增回退条件、输出语义、模块定位和重复安装的回归测试 |

FlagGems 安装包、GEMM kernel 计算代码、全局候选 YAML、模型权重和官方 benchmark 脚本均未修改。

### 2.2 参数变化

下表的形状顺序为 `(M, N, K)`，对应 `A[M,K] × B[K,N] → C[M,N]`。

| 投影 | 形状 | 原 BM×BN×BK | 优化 BM×BN×BK | 原 warps/stages | 优化 warps/stages |
| --- | --- | --- | --- | --- | --- |
| QKV | 2048,2560,2048 | 128×128×32 | 256×256×32 | 8/2 | 8/2 |
| O | 2048,2048,2048 | 128×128×32 | 128×128×64 | 8/2 | 8/2 |
| Gate/Up | 2048,12288,2048 | 128×128×32 | 256×256×32 | 8/2 | 8/2 |
| Down | 2048,2048,6144 | 64×64×64 | 128×128×64 | 4/4 | 8/2 |

其余相关参数保持：`pipeline="basic"`、`scenario=""`、`GROUP_M=8`、`num_ctas=1`。

这些参数联合决定单个 program 的输出分块、K 方向分段、warp 协作规模和软件流水线深度，影响数据复用、访存、寄存器及共享内存压力。实测验证的是整套配置的收益，尚未通过单独消融拆分每个参数的贡献。

### 2.3 接入位置与执行过程

```text
原有 mm / mm_out
    ↓ 保留 GEMV、Split-K、NN / NT 的原有选择
进入 MetaX NT helper：general_mm_nt
    ├─ 匹配条件：使用固定配置执行原 mm_kernel_nt
    └─ 其他情况：回到原 general_mm_nt
```

接入代码沿 `flag_gems.mm.__module__` 定位实际注册的后端模块。在本次 FlagGems 中，它是 `_metax.ops.mm`。这样可以避免按完整包路径导入同一文件的另一个模块副本，导致修改没有作用于真正注册的 aten 调用。

四组目标形状分别创建独立的 LibEntry，并包装只有一个候选配置的 LibTuner，复用原有 Heuristics 和 JIT kernel。当前版本的 LibTuner 在单候选路径中直接使用该配置，不再查询旧的最佳配置或执行候选竞速；独立启动缓存也避免不同配置之间错误复用编译结果。

安装过程具有幂等性，同一设备重复调用不会重复包装。

### 2.4 生效条件与回退

仅在以下条件同时满足时使用新配置：

- worker 平台为 MetaX，启用 FlagGems，设备名称为 `MetaX C500`；
- 形状精确匹配上表四组 `(M,N,K)`；
- A、B、C 均为 BF16，且位于同一目标设备；
- A、C 为预期连续布局，B 为连续 `[N,K]` 权重的转置布局，stride 分别为 `(K,1)`、`(N,1)`、`(1,K)`。

其他形状、数据类型、布局和设备沿用原路径；不扩大到其他 MetaX 型号。Decode、小 M、尾部 Split-K、Attention 和 LM head 没有作为本次优化目标。

## 3. 实验与验证结果

### 3.1 测试环境与方法

测试于 2026-11-03 完成，两版在同一台机器上顺序运行：

| 项目 | 环境 |
| --- | --- |
| GPU | 单张 MetaX C500 64 GiB，104 SM，8 MiB L2 |
| 主机内存上限 | 128 GiB |
| 驱动 / MACA | 3.8.30 / 3.7.0.38 |
| PyTorch | 2.8.0+metax3.7.0.7 |
| Triton | 3.0.0+metax3.7.0.7 |
| FlagGems | 5.3.5 |
| vLLM / EvalScope | 0.24.0+empty / 1.11.1 |
| 模型 | MiniCPM5-2B，BF16，单卡 |

使用未修改的官方 HTTP benchmark：每版、每场景 1 次预热＋3 次正式测试，共 16 次完整运行。首轮丢弃，后三轮指标取算术平均，全部请求数和输入、输出 token 数校验通过。

两版从相同的原始调优数据库快照出发，隔离 vLLM / Inductor 编译缓存；优化版不继承原版运行新产生的调优结果。两版均没有人工替换数据库配置行，优化参数仅来自插件源码。

服务端启用 `VLLM_PLUGINS=fl`，纯 HTTP 客户端禁用无用推理插件；两版共同设置 `TORCHINDUCTOR_COMPILE_THREADS=1`。冷分派选择记录方式相同，计时期间没有启用 profiler。

### 3.2 性能对比

场景字段为 `[输入长度, 输出长度, 并发数, 请求数]`：4k 为 `[4096,1024,64,256]`，16k 为 `[16384,1024,64,128]`。

| 场景 | 指标 | 原版 | 优化版 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 4k | Total tokens/s ↑ | 4841.77 | 5030.66 | **+3.90%** |
| 4k | Output tokens/s ↑ | 968.35 | 1006.13 | +3.90% |
| 4k | Mean TTFT ms ↓ | 3230.34 | 2805.48 | **-13.15%** |
| 4k | Mean TPOT ms ↓ | 62.72 | 60.66 | — |
| 4k | 整轮耗时 s ↓ | 271.06 | 260.92 | — |
| 16k | Total tokens/s ↑ | 6996.49 | 7409.34 | **+5.90%** |
| 16k | Output tokens/s ↑ | 411.56 | 435.84 | +5.90% |
| 16k | Mean TTFT ms ↓ | 27286.68 | 24881.90 | **-8.81%** |
| 16k | Mean TPOT ms ↓ | 125.06 | 118.73 | — |
| 16k | 整轮耗时 s ↓ | 318.48 | 300.73 | — |

吞吐变化按“优化版三轮吞吐均值 / 原版三轮吞吐均值 - 1”计算，TTFT 同理。整轮耗时是三轮耗时的均值，不使用耗时均值重新计算吞吐成绩。

三轮正式 total tokens/s 原始值：

| 场景 | 版本 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | --- | ---: | ---: | ---: |
| 4k | 原版 | 4715.00 | 4721.66 | 5088.66 |
| 4k | 优化版 | 4875.35 | 5300.16 | 4916.48 |
| 16k | 原版 | 6997.50 | 6997.91 | 6994.06 |
| 16k | 优化版 | 7402.79 | 7415.35 | 7409.87 |

4k 正式轮仍有较大波动，保留全部样本和运行时自动调优等真实开销，没有扣除开销或挑选轮次。以上是本次同机标准对照，不代表换机或其他时间一定得到相同提升幅度。

### 3.3 精度对比

| 版本 | 子集 | 成功请求 / 总数 | 正确题数 | Accuracy | 请求错误 |
| --- | --- | --- | --- | ---: | ---: |
| 原版 | MATH-500 Level 3 | 105/105 | 102 | **97.14%** | 0 |
| 优化版 | MATH-500 Level 3 | 105/105 | 102 | **97.14%** | 0 |

两版均达到准确率 >=95% 的要求，报告均为完整运行、无请求错误。总体准确率相同，不表示生成文本逐字一致。评测参数为 batch size=8、temperature=1.0、top_p=0.95、max_tokens=32768、timeout=3600。

### 3.4 其他验证

- 7 项回归测试全部通过。
- 四组完整形状经真实注册的 `torch.mm`、`torch.mm(out=...)` 执行，并与注册前生成的 native Torch 参考值比较，均通过。
- 四组均通过改变输入后的 CUDA Graph 重放检查，实际选型全部符合目标配置。
- 短验证使用全新调优数据库路径，结束后数据库文件未被创建，证明固定配置不依赖预写缓存。
- 最终审计确认原安装数据库、FlagGems mm / 候选 YAML 和官方 benchmark 源码未变；优化版数据库中原有 24 条相关目标记录保持原值，实际配置来自源码。

## 4. 复现步骤

核心流程：**用服务器 `/workspace` 中的原版跑两场景 benchmark → 停止服务 → 安装优化分支 → 用相同命令再跑两场景 → 比较结果。** 不需要手工配置分块参数或复制实验数据库。

使用沐曦 C500 比赛镜像，模型和数据集路径分别为 `/workspace/MiniCPM5-2B`、`/workspace/evalscope-datasets/math_500`。以下命令在镜像正常的 MACA / Conda 环境下运行。

### 4.1 使用 `/workspace` 原版测试

终端 A：核对原版提交，明确安装原版插件，再启动服务：

```bash
mkdir -p /data/flagos
cd /data/flagos

git -C /workspace/vllm-plugin-FL rev-parse HEAD
# 本报告原版提交：13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0

python -m pip install --no-deps --no-build-isolation -e /workspace/vllm-plugin-FL
python -c 'import vllm_fl; print(vllm_fl.__file__)'
# 应指向 /workspace/vllm-plugin-FL 下的代码

export VLLM_PLUGINS=fl
export TORCHINDUCTOR_COMPILE_THREADS=1
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

如果镜像原版提交与上面不同，应先确认基线版本，再进行对照。

服务就绪后，终端 B 执行官方两场景 benchmark，结果保存在原版独立目录：

```bash
mkdir -p /data/flagos/results-baseline
cd /data/flagos/results-baseline
VLLM_PLUGINS= python /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

脚本自身执行每场景 1 次预热＋3 次正式测试，不需要再手工重复运行四遍。

### 4.2 安装优化分支，再测试一次

先在终端 A 停止原版服务，确认服务退出、GPU 资源释放。

克隆并安装优化插件：

```bash
cd /data/flagos
git clone --single-branch --branch perf/metax-c500-prefill-nt \
  https://github.com/kanceler/vllm-plugin-FL.git \
  /data/flagos/vllm-plugin-FL-tiling

python -m pip install --no-deps --no-build-isolation -e /data/flagos/vllm-plugin-FL-tiling
python -c 'import vllm_fl; print(vllm_fl.__file__)'
# 应指向 /data/flagos/vllm-plugin-FL-tiling 下的代码
```

安装后，在终端 A **重新执行 4.1 中相同的服务启动命令**，无需增加优化开关。优化版在 C500 上会记录 `Enabled four C500 BF16 NT tiling overrides`。

服务就绪后，终端 B 使用同一份官方脚本测试，结果换一个目录保存：

```bash
mkdir -p /data/flagos/results-tiling
cd /data/flagos/results-tiling
VLLM_PLUGINS= python /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

两次 benchmark 使用同一脚本、同一模型和同一启动参数，区别在于服务实际加载的插件代码。

### 4.3 比较结果

分别查看两个目录中 `benchmark_results/summary_*.csv` 的 `Total tok/s` 和 `Mean TTFT (ms)`。同时检查 `raw_runs_*.csv` 中各轮请求全部成功，4k 为 256 个请求，16k 为 128 个请求，每请求输出 1024 token。

每次更换插件后都需要重启服务，仅克隆代码或修改安装路径不会改变已经运行的进程。两版服务顺序运行，不同时占用 GPU。

以上是便捷复测流程。本报告实测额外使用了相同初始调优快照和独立编译缓存，以控制对照条件；这些属于实验控制措施，不是优化生效的部署前提。普通复测的缓存和冷调优状态可能不同，不能保证重现完全相同的提升百分比。

### 4.4 如需复核精度

在各版本服务仍运行时执行官方精度评测。以下示例保存原版结果，测试优化版时将输出目录改为 `/data/flagos/accuracy-tiling`：

```bash
VLLM_PLUGINS= evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
  --work-dir /data/flagos/accuracy-baseline \
  --ignore-errors
```

确认实际完成 105 题、请求错误为 0、报告不标记 incomplete，且准确率 >=0.95。
