# W8A16 真实负载 Shape 观察器兼容修复与首段采集

## 1. 基本信息与摘要

- 日期：2026-11-03（UTC+8）
- 平台：MetaX C500 64 GiB，MACA 3.7.0.38
- 容器基线：vllm-plugin-FL `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc`
- 本机开发基线：vllm-plugin-FL `f93489fa1afb74969b76701c8b551552a55ae3bf`
- 修改文件：
  - `benchmarks/diagnostics/collect_gemm_shapes.py`
  - `benchmarks/diagnostics/summarize_gemm_shapes.py`
  - `benchmarks/diagnostics/test_summarize_gemm_shapes.py`

本次修复 MetaX profiler trace 的 JSON 兼容问题，并为编译及 CUDA Graph 路径补充模型实际 padded M 的关联。修复后完成 BF16 4K 起始、4K 中段及限量 16K 的首段 Shape 采集。没有修改 Kernel、量化格式、生产分派、官方服务参数或 Benchmark 实现。

## 2. 问题与原因分析

观察器原先把 JSON 文本直接拼入 `torch.profiler.record_function` 名称。MetaX 的 Chrome trace exporter 没有转义名称内的双引号，生成如下非法 JSON：

```text
"name": "GEMM_DIAG_STEP {"label": "pilot", ...}"
```

因此原始 trace 虽已落盘，`json.load` 会在 marker 行失败。

修复 JSON 后，试采集仍显示 exact M 覆盖率为 0。原因是 vLLM 编译及图重放路径不会为每个 GEMM 暴露独立 `aten::mm` input shape；仅依靠 operator event 无法取得 M。模型 runner 在 `set_forward_context` 中仍传入实际 `num_tokens_padded`，该上下文严格包围模型 forward，可作为该次模型 GEMM 的执行 M。它与调度器的逻辑 active tokens 分开记录，避免把请求数或未 padding token 数冒充 Kernel M。

## 3. 具体实现

1. Step、graph 和 forward marker 改用 URL-safe Base64 编码，去除 padding 字符，保证 marker 名称只含 JSON 安全字符。
2. 汇总器同时兼容历史明文 marker 与新 Base64 marker。
3. 在 `vllm_fl.worker.model_runner.set_forward_context` 外增加仅在 profiler 已启用时生效的 `record_function`，记录：
   - `num_tokens`：模型实际 padded token extent；
   - `graph_tokens`：batch descriptor token extent；
   - `cudagraph_runtime_mode`。
4. 汇总器把合法的 `num_tokens` 标为 `M_source=forward_context_num_tokens_padded`。若存在更具体的 operator input shape，仍以 operator 事件补齐 N/K 和投影候选。
5. 新增 Base64 JSON 安全性和 forward-context M 关联测试；保留异步 launch correlation、图 extent 不冒充 M、冲突关联 unknown、投影歧义及 capture 排除测试。
6. 同日续段新增 phase、active-token 上下界和跳过匹配次数的诊断窗口条件。条件未命中时 profiler 不启动；状态写为 pending，匹配后才开始有限 step 采集。

观察器仍为进程级 opt-in 工具，未修改已安装包和服务参数。未开启采集时只保留低频 control 文件轮询。

## 4. 实验与验证

### 4.1 测试

本机和容器均执行：

```bash
cd benchmarks/diagnostics
python3 -m unittest test_summarize_gemm_shapes.py
```

结果均为 `Ran 8 tests ... OK`。容器工具 SHA256：

```text
cc2dc1419878401e47d32e780f2c4f1a1c35e906316afe121b84292786c4322a  collect_gemm_shapes.py
7f8c2857ad116e56662ba2e533d299f0500dd8320be970288fd92cedb7039be0  summarize_gemm_shapes.py
6bdecbbe6582c8fa9bfd01f79a39b1215c6e1f82b4b1f595ad54453783bdb6bd  test_summarize_gemm_shapes.py
```

上述为首段版本。条件触发续段在本机与容器均为 `Ran 10 tests ... OK`，容器最终 `collect_gemm_shapes.py` SHA256 为 `62ef8273e8d48ebe15c5af1007e058d85cc629f6ffeab6fc56f7be97ad3d3587`，测试文件 SHA256 为 `dc88aec44b860401343a0ff9d287442727da5ec3b55e558cd6ee18185f059729`；汇总器哈希不变。

### 4.2 采集完整性

| 窗口 | steps | matched kernels | matched GPU time | exact M coverage | unknown correlation |
|---|---:|---:|---:|---:|---:|
| 4K start | 64 | 10,752 | 5,079,984.64 µs | 100% | 0 |
| 4K mid | 64 | 10,752 | 1,006,156.34 µs | 100% | 0 |
| 16K limited | 32 | 5,376 | 2,564,238.34 µs | 100% | 0 |

4K start 观察到四类投影的 M=2048。4K mid 观察到 M=2048、1554、320、64，其中 decode padded M=64 对应 active tokens 62～64，M=320 对应 active tokens 312。限量 16K 窗口观察到 prefill/mixed M=2048。

模型 inventory 共 170 个二维权重模块：四类投影各 42 层，形状分别为 QKV `[2560,2048]`、O `[2048,2048]`、Gate/Up `[12288,2048]`、Down `[2048,6144]`；另有 Embedding 和 LM Head。该 inventory 描述层和布局覆盖，不等于每个动态图 replay 的逐投影调用覆盖。

### 4.3 条件触发 16K Mixed 与尾部补采

补采使用 16K 输入、并发 64，单轮诊断客户端。128 prompts、output 256 的首次运行 128/128 请求成功，但 `phase=decode && active_tokens>=48` 没有命中；64 prompts、output 128 的复验同样没有命中。高并发生成期在调度器中主要标为 Mixed，纯 Decode 只在后段出现。此两次为未触发的诊断尝试，没有 trace，也不作为性能成绩。

改按观测状态触发后，64 prompts、output 128 的 Mixed 窗口完成 32 step，观测 padded M 为 40/48/56/64/72/80/88/96/104/112，并有 M=64 的 Decode。尾部在 `active_tokens=32` 时触发，客户端结束前取得 14 step，保存的纯 Decode M 为 32/24/16/8/4，最低 active tokens 为 3。两组 matched GEMM GPU 时间的精确 M 关联率均为 100%，correlation unknown 为 0；分别匹配 5,376 和 2,016 个 kernel。尾部实际为 14 step，不能写成预设的 32 step。

这些窗口仍无法证明完整官方负载的 M 频率，也没有 M=1/2 的样本。图重放 kernel 的 N/K 与投影归属继续标为 unknown。

### 4.4 中止运行边界

一次官方双场景 wrapper 曾用于首轮 Shape 采集。用户指出本阶段不应执行完整官方测评后，已停止外层 PID 及其遗留客户端。只完成两轮 4K，第三轮中止，16K 未开始，没有 summary。该运行不属于完整正式评测，归档的 raw CSV 和日志不得作为性能成绩。

随后 16K 只执行限量诊断：requested input length 16384、output length 64、concurrency 64、64 prompts，64/64 请求成功。其吞吐和 TTFT不用于优化判断。

## 5. 与已有改动的关系

本次延续 P0-A 计划中的 `collect_gemm_shapes.py` 和 `summarize_gemm_shapes.py`，没有修改 11 月 1 日三算法候选、W8A16 生产入口或队友负责的 BF16 MM。首段结果为后续选择 O1/O2 验证 M 和 Runtime manifest 提供输入，但尚未证明任何 W8A16 算法在真实模型中有收益。

## 6. 合规性与风险

- 官方服务参数未修改；正式 Benchmark 脚本源码未修改。
- profiler 只用于诊断，相关性能数据不申报。
- 限量 16K 明确改变请求数和输出长度，只能用于 Shape 采样。
- 原始 trace 的 kernel-name matcher 尚未证明覆盖所有 GEMM。
- eager/operator 可取得 N/K 和投影候选；部分编译及图重放 kernel 只能关联 M，N/K 和投影保留 unknown。
- 当前没有旧 W8A16 分支覆盖、模型精度、正式吞吐或 TTFT结论。
- 16K 补采的输出长度为 128 或 256，请求数为 64 或 128；其 M 集合只代表对应诊断窗口，不代表官方 1024 输出的完整分布。

## 7. 复现方式

启动观察器时，服务参数保持沐曦官方命令：

```bash
export VLLM_PLUGINS=fl
python3 benchmarks/diagnostics/collect_gemm_shapes.py serve \
  --output /root/experiments/2026-11-03-w8a16-shape-coverage/results/<run> -- \
  vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

限量窗口通过 control 文件启停，汇总命令为：

```bash
python3 benchmarks/diagnostics/collect_gemm_shapes.py control \
  --output <run> --label <window> --steps <bounded_steps>

python3 benchmarks/diagnostics/summarize_gemm_shapes.py <trace.json> \
  --inventory <worker-inventory.json> --output <summary_dir>
```

本地原始证据索引见：

```text
/Users/tianfr/muxi_2b/experiment-artifacts/2026-11-03-w8a16-shape-coverage/DELIVERY.md
```

