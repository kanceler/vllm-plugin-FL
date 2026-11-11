# muxifinal：002＋003与W8A16完整组合

## 1. 平台数据与整体摘要

2026-11-12。本分支整合 **002自适应prefill预算、003消除Attention逐层host同步、Down-only W8A16完整路径、M2048及M4096专用MM配置**。目标是在保持数值和赛事限制的前提下提高total tokens/s。本轮官方性能已完成，结果见第4节；新原始默认BF16基线正在补跑；用户明确本轮不重跑精度。

| 项目 | 本轮环境 |
|---|---|
| GPU | 单张MetaX C500，64GiB |
| 驱动 / MACA | 3.8.30 / 3.7.0.38 |
| vLLM | 0.24.0+empty |
| PyTorch | 2.8.0+metax3.7.0.7 |
| Triton distribution / Python版本字符串 | 3.0.0+metax3.7.0.7 / 3.0.0 |
| EvalScope | 1.11.1 |
| 模型 | MiniCPM5-2B，BF16激活，单设备 |
| 代码 | 插件与FlagGems分别为muxifinal分支；插件仓库同时包含完整FlagGems/源码 |

FlagGems基底为13d3db1，量化Kernel SHA256 `826ee3b73b01f4cc329286f9e268c87a87f7b069464afe7b121e682f1a9abf03`，MM提供者SHA256 `3abfeba5c64c159370543602a2afc329efb6c7fc09e46659c11faa6b64485213`。Down策略在设备、Torch/Triton版本及提供者指纹匹配时启用；不匹配沿原合法选择。

**量化仅覆盖42层Down**。Gate/Up、QKV、O使用原生BF16计算和专用配置；Embedding、LM Head等不扩大量化。INT8权重转换为BF16后dot、FP32累加，不是原生INT8×INT8 GEMM。

## 2. 为什么改：002前后算子采集与瓶颈

已有M2048专用配置只在该精确M生效。002可能提高长prompt的单步token预算，因此先做off/on对照，保持同一量化和MM组合，仅切换002，未混入003的新改动。

采用原官方serve参数，诊断请求使用固定synthetic token IDs，4K/16K分别为输入4096/16384、输出1024、并发64、请求256/128。每模式分别采prefill16个匹配step和decode128个匹配step，利用torch.profiler、forward上下文、capture/replay marker关联实际M；这是诊断负载，不是官方性能计分结果。

| 采样窗口 | 002关闭 | 002开启 |
|---|---:|---:|
| 4K prefill/mixed | M2048 | M2048 |
| 16K prefill/mixed | M2048 | **M4096** |
| 4K decode | M64 | M64 |
| 16K decode | M64 | M64 |

8个窗口全部保存，两个模式各4K256/256、16K128/128请求均HTTP200。匹配GEMM集合的精确M耗时关联覆盖率为100%；这不是模型全部算子的覆盖率，也不表示未来所有Shape均已覆盖。

002开启后的16K prefill窗口中，M4096匹配GEMM占全部GPU Kernel时长约52.58%：普通BF16 MM约43.51%、Down MM＋融合scale约8.59%、权重转换约0.47%。普通MM内三类投影共用Kernel名，不按层顺序猜测各自占比。两个预算下16个step处理的token数量不同，不能用窗口总GPU时间直接推导加速。

结论：Down已有区间策略能覆盖4096；其余三个投影的旧M2048特判不命中4096，值得专门寻找配置。不能仅提高BM并按M比例外推：此前BM256精确同比例候选需96KiB shared，超过64KiB限制，其他可运行BM256候选也未超过BM128。

## 3. 怎么改：配置搜索与最终运行路径

### 配置搜索与接入

复用现有MM，不重写其底层。固定M4096，分别搜索四个真实N/K：

| 路径 | N / K | 完整测量口径 |
|---|---|---|
| Gate/Up | 12288 / 2048 | 输出分配＋原生BF16 MM |
| QKV | 2560 / 2048 | 输出分配＋原生BF16 MM |
| O | 2048 / 2048 | 输出分配＋原生BF16 MM |
| Down | 2048 / 6144 | 权重暂存/输出分配＋INT8转BF16＋MM＋融合scale |

每组BM/BN64、128、256；BK32、64、128；warps4/8；stages2/4；basic/cpasync，共216项，四组共864项。每组137项成功、79项失败，均保留原因；失败数不算成功计时数。seed20261205初筛，保留前六与控制项，再用20261206/20261207独立种子做7轮交错完整调用、5轮交错图replay。相对FP32对应参考L2<0.005。只称所测范围内较优，不称全局最优。

| 路径 | 领先BM/BN/BK | 完整调用μs（两个独立seed） | 图replayμs |
|---|---|---:|---:|
| Gate/Up | 128/128/64 | 1213.44 / 1212.48 | 1228.97 / 1230.12 |
| QKV | 128/128/64 | 298.05 / 298.77 | 313.79 / 314.36 |
| O | 128/128/128 | 220.41 / 220.93 | 236.28 / 236.64 |
| Down | 128/128/128 | 564.33 / 564.43 | 580.43 / 579.69 |

均warps4、stages4、cpasync、GROUP_M=8。Down的完整调用包含BLOCK4096/warps4转换，转换与scale融合固定，并未在本轮重搜转换参数；其结果与当前实际dispatch约564μs一致，所以不重复新增Down特判。

在公开注册提供者接入六条原生MM映射后，两个额外seed20261208/20261209对M2048/4096四投影共16项做eager和图正确性检查，最大L2=0.001661356，全部通过。接入后的M4096原生MM完整调用复测：Gate/Up约1290.93～1367.04μs，对应原选择1850.34～1971.12μs；QKV约299.24～299.90μs，对应427.12～427.87μs；O约219.78～220.85μs，对应317.61～317.71μs。不同进程/测量轮次的绝对时间有变化，不把跨运行差异归因为代码收益。

### 当前生效配置

| 组件 | Shape或M范围 | 参数 |
|---|---|---|
| Down Split-K | M56/M64，N2048/K6144 | BM64/BN64/BK128、warps4、stages1、Split8 |
| Down大M融合 | M193～384 | BM64/BN128/BK128、warps4、stages4、cpasync |
| Down大M融合 | M385～16384（含2048、4096） | BM128/BN128/BK128、warps4、stages4、cpasync |
| Gate/Up、QKV、O原生MM | M2048 | BM128/BN128/BK64、warps4、stages4、cpasync |
| Gate/Up、QKV原生MM | M4096 | BM128/BN128/BK64、warps4、stages4、cpasync |
| O原生MM | M4096 | BM128/BN128/BK128、warps4、stages4、cpasync |
| 转换 | eligible Down大M融合 | BLOCK4096、warps4 |

Down较小M仍按精确配置/区间候选池选Split-K，范围外保留合法回退；不得把性能分界当合法性条件。193～255已逐整数补测，更大区间采用代表点及留出验证，不宣称全范围性能最优。M113～192沿原提供者配置。

### 002和003定向接入

002取a09c3c7已归档版本，仅加入`adaptive_sched.py`、4项env定义及platform钩子。上限4096、decode_gate=-1、long_prompt=0，长prompt阈值自动取4×基线预算（2048时为8192）；缓冲区上界与运行预算解耦，并保留基线编译区间。003取2f76394精确改动，将prefill_seq_lens的逐层device→host `.tolist()`同步改为device上的zeros＋cat再cumsum。整数前缀构造的独立检查通过，模型级验证以本轮新服务为准。

002/003为既有上游优化的定向接入，明确归属，不算作本次量化原创。没有合并整个最新上游分支。正式控制器的启动检查随预热M调整为要求M4096策略命中；这不改变MM选择，也不删除M2048配置。

### 源码位置

插件：`vllm_fl/quantization/w8a16/`、`dispatch/backends/vendor/metax/nt_gemm.py`、`worker/adaptive_sched.py`、`platform.py`、`envs.py`以及`dispatch/backends/vendor/metax/impl/attention/flash_attn.py`。完整算子源码随仓库放在`FlagGems/`，独立FlagGems仓库也有muxifinal分支。

测试与搜索：`benchmarks/diagnostics/search_metax_m4096.py`及`verify_m4096_integration.py`。原始BF16搜索记录的kind字段继承早期工具写为quant，仅作内部计时分组；projection与实际执行分支明确前三投影为BF16，Down才使用量化，参考与算法未混用；原始数据不追改。

## 4. 模型测评结果与对照

本轮是 **002＋003＋三投影M2048/M4096专用配置＋Down量化完整组合**。官方4K/16K顺序执行，各4轮，跳过首轮取后三轮平均。最终填写生成的summary，不择优或拼接旧轮次。

| 场景 | total tokens/s | Mean TTFT ms | 完整轮次/成功请求 |
|---|---:|---:|---|
| 4K | **5668.66** | **2111.42** | 4轮，各256/256成功，失败0 |
| 16K | **8675.61** | **17018.61** | 4轮，各128/128成功，失败0 |

两组历史参照（不是本轮新测基线）：

| 场景 | 默认BF16 tok/s / TTFT ms | 专用配置纯BF16 tok/s / TTFT ms |
|---|---:|---:|
| 4K | 4729.59 / 3250.42 | 5127.00 / 2639.22 |
| 16K | 7022.20 / 27133.58 | 7651.48 / 23658.99 |

01:10:08控制器COMPLETE，退出及清理正常。官方脚本指纹与启动前一致，命令参数未改，raw八轮全部SUCCESS。summary为第2/3/4轮平均。

| 历史对照 | 4K吞吐提升 | 4K TTFT降低 | 16K吞吐提升 | 16K TTFT降低 |
|---|---:|---:|---:|---:|
| 默认BF16 | 19.86% | 35.04% | 23.55% | 37.28% |
| 专用配置纯BF16 | 10.56% | 20.00% | 13.38% | 28.07% |
本轮相对两组历史参照的四项指标均有效改善；这里是组合总效果，不隔离量化贡献。用户另要求补跑原始默认BF16（量化、002/003和专用MM均关闭），任务目录 `/data/default-bf16-baseline-20261112-01/`；尚未完成，不预填结果。性能计分指标为total tokens/s；吞吐不得低于指定baseline×0.99、TTFT不得高于baseline×1.01；±1%含边界归正常波动，有效改善须严格超过1%。本文不自行扩展TTFT计分权重，正式baseline由组委会口径决定。参照来自同设备不同时间，且没有完整新增002/003和M4096配置，不能将所有组合收益归因量化；若隔离量化贡献需另做相同底座纯BF16对照。

用户要求本轮不再重跑精度。上一版不含新002/003/M4096映射的量化完整组合math_500 Level3报告accuracy=0.9714、105成功/0错误，达到0.950数值门槛。该结果保留为前版本证据，**不写成本轮组合重新测得的精度**。

## 5. 复现命令与版本核对

先在匹配赛事镜像中按既有安装方式使用本分支插件与所附FlagGems源码，不升级vLLM/Torch/Triton。检查实际import位置和`muxifinal-source-manifest.json`中的关键文件指纹；不能只看仓库HEAD。两个仓库的独立源码组合与插件内完整FlagGems副本一致。完整镜像已有的native扩展和依赖仍需保留，Git不存放编译缓存或二进制权重。

每任务开始前归档上一任务，核对并停止本任务残留服务/客户端，检查GPU、显存和9031，确认没有其他GPU负载。基线/候选切换必须新服务，不在两场景之间额外重启或清缓存。

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=1
export VLLM_FL_W8A16_DOWN_INT8PACK=1
export VLLM_FL_W8A16_DOWN_FUSED=0
export VLLM_FL_W8A16_SHAPE_DISPATCH=1
export VLLM_FL_W8A16_FORCE_SEPARATE=0
export VLLM_FL_ADAPTIVE_CHUNK_ENABLED=1
export VLLM_FL_ADAPTIVE_CHUNK_MAX=4096
export VLLM_FL_ADAPTIVE_CHUNK_DECODE_GATE=-1
export VLLM_FL_ADAPTIVE_CHUNK_LONG_PROMPT=0

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

新服务健康后核对42层Down启用、`Enabled 6 C500 BF16 NT tiling overrides`、M4096三个原生MM命中与`C500_W8A16_SCALE_POLICY M=4096`。选择日志只证明Python选择，不是图replay计数。

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

保留原脚本RUNS=4/SKIP_FIRST=1，记录脚本SHA256，summary每场景一行、raw八行。独立micro复现：在相同导入环境下运行`python benchmarks/diagnostics/search_metax_m4096.py --projection gate_up --output <新目录>`，其他projection为qkv/o/down；`verify_m4096_integration.py --output <新目录>`验证实际注册路径。GPU任务串行。

## 6. 与已有改动的关系、合规与限制

001～008原报告保持原样，对应历史版本；本篇是muxifinal当前组合入口。量化主线、候选池、通用合法回退与完整FlagGems保留，不恢复前次已删除的旧路线或无关实验。官方serve、Benchmark、模型和数据均不修改；新增002源码改变内部调度行为，系用户明确要求接入的既有优化，参数与来源单列。创新归属区分MM配置、量化实现、002/003作者贡献。

范围外M、不同软件/设备需复验；更大区间仅代表点覆盖。003可能影响执行时序、002影响分块，不能借旧精度报告宣称新组合已重新验收。官方性能流程已完整完成，但本轮精度按用户要求未重测，不宣称全面验收通过或全局最优。

容器与分支的关键文件做过字节核对：计算Kernel、int8pack接口、新配置、002与003一致。linear.py差异是清理永远关闭的其他投影选择；worker差异仅为不在C500执行的GCU缓存及昆仑芯warmup分支，活动C500路径经差异审计一致。实际运行overlay文件指纹单独保留，不用Git提交标识替代实验版本。

## 7. 实验与源码归档

- 002采样：`/data/adaptive-chunk-m-sampling-20261111-01/`，off/on共8份trace；本机摘要`artifacts/adaptive-chunk-m-sampling-20261111/`。大型raw trace在容器保留，不塞入Git。
- M4096搜索：`/data/m4096-config-search-20261111-01/`，原始包`artifacts/m4096-config-search-20261111/m4096-config-search-20261111-results.tgz`。
- 实际接入：`/data/m4096-002003-integration-20261112-01/`，本机归档SHA256 `a62f9fa17a7d7c71be643dbba0a7bf8f4f175b58a3bf93bef90b1b3c7d9dbddc`。
- 本轮官方：`/data/m4096-002003-official-20261112-02/`，控制器`/data/run_m4096_002003_official_20261112_v2.py`。01目录因旧预热检查提前停止，未启动Benchmark，不与02拼接。
- 仓库小型证据：`docs/optimization/evidence/2026-11-12-muxifinal/`；完整日志、逐轮计时及归档保留artifacts，源码与关键指纹由Git管理。最终summary和八轮raw已收录；完整归档SHA256为 `f77dede4dc0a42cbe3742413b9d8140a183cd0af4438d3e0d4a13b92c0714cbc`。新的同期原始基线待补。
