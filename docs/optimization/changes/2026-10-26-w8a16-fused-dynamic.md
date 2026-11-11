# W8A16 Down 融合调用的动态形状编译兼容修复

> 后续补测：[独立 Triton 反量化完整路径对照](2026-10-26-w8a16-triton-microbench.md)。下文历史 Microbenchmark 的独立路径为 PyTorch reference；补测九种 M 中，融合均慢于独立 Triton，不能将此前 reference 对照的小 M 收益推广到 Triton 独立实现。

## 1. 基本信息与摘要

日期：2026-10-26（UTC+8）。设备：MetaX C500 64 GiB。用户已明确批准上一任务提出的编译兼容修复和后续官方性能对比。

涉及源码：`vllm_fl/quantization/w8a16/fused.py`、`linear.py`、`tests/unit_tests/quantization/test_w8a16_down.py`。沿用分支 `perf/gemm-w8a16-down-fused`，基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，保留已有未提交工作。本次未提交、未推送。

目标：修复默认官方模型启动中，直接跟踪 Triton constexpr M 导致动态 token 维度被固定的问题；不改 Kernel 算法、量化精度契约、分块配置或官方命令。单元测试、独立性能诊断及模型启动已完成；用户于 11:37 明确要求停止官方评测，当前只保留不完整诊断结果。

实验 ID：`fused-dynamic-20261026-02`。容器 `6951ed69dce3`、GPU bus `0000:35:00.0`，与上次失败实验实例不同，因此重新建立同实例 BF16 基线，不复用旧实例成绩计算收益。

## 2. 问题与原因分析

上次 `fused-vs-bf16-20261026-01/fused/serve.log` 报告 `ConstraintViolationError`：标记为 dynamic 的 `input_ids/positions` 第 0 维被特化为 2048。用户调用栈指向 `fused.py:63` 的 Triton launch。既有 `fused_linear` 从 `x.numel()/K` 得到 M，在 Python 中选择 BM，再将 M 作为 `tl.constexpr` 传入 Kernel。将整段直接纳入 Dynamo 跟踪会对动态模型图施加具体 M 约束。

独立 eager 测试和固定形状 CUDAGraph 通过不能覆盖此问题。vLLM 默认需要 `(1,2048)` 范围的动态图；不应通过更改官方 compilation 参数绕过。

## 3. 具体实现

将原函数执行体保持为 `_fused_linear_impl`，通过 vLLM 现有 `direct_register_custom_op` 注册 `vllm::fl_w8a16_down_fused`，声明无输入修改；fake 实现只根据 `x.shape[:-1]` 和权重 N 创建输出元信息，保留符号维度。公共 `fused_linear` 调用该算子。

```text
编译时：x[..., K] → 自定义算子 fake → y[..., N]，保留动态 leading dimensions
运行时：自定义算子 → 原 M 分块选择 → 原 Triton Kernel → BF16 输出
```

`install_w8a16_down` 在开启融合时提前导入并完成注册，避免模型 forward 跟踪期间注册算子。原 capability guard、batch-invariant 分支和不支持形状的原回退逻辑保留。BF16 关闭路径不导入或调用新算子。

数学与 Kernel 保持不变：

```text
w_tile = BF16(FP32(q_int8) * FP32(channel_scale))
acc_fp32 += dot(x_bf16, w_tile_bf16)
y = BF16(acc_fp32 + optional_bias)
```

权重连续 INT8 `[N,K]`、scale 连续 FP32 `[N]`、激活 BF16；M≤32 使用 `BM/BN/BK=16/64/64`，其余 `32/64/64`，warps=4、stages=2，`enable_fp_fusion=False`。不增加完整持久 BF16 权重副本；不移动 scale 到累加之后。

新增有/无 bias 两项严格动态 M 回归：标记 `min=2,max=2048`，依次输入 M=2048/16/33/64，要求单一 fullgraph 编译图复用，输出对照已舍入权重的 BF16 Linear。通过 strict mark_dynamic 暴露被迫特化问题，避免 dynamic=True 自动降为具体形状而漏检。

## 4. 实验与验证

运行环境：单卡 C500，128 GiB RAM 限额，swap 0；Python 3.12.11、PyTorch `2.8.0+metax3.7.0.7`、vLLM 0.24.0、MACA 3.7.0.38、驱动 3.8.30。实际源码路径 `/workspace/vllm-plugin-FL`，各文件 SHA256 记录于 `candidate.sha256`。运行前 GPU 空闲显存 826 MiB，无 GPU 进程、9031 空闲。

| 验证 | 结果 |
|---|---|
| 本地 AST 语法及 diff 空白检查 | 通过 |
| C500 单元测试 | 19 passed，51.82s；退出码 0 |
| 严格动态 M 编译 | 有/无 bias 均通过，四种 M 复用同一图 |
| BF16 舍入边界及 CUDAGraph 重放 | 包含在通过的单元测试中 |
| 独立完整路径 Microbenchmark | 完成，退出码 0，见下表 |
| 模型启动及融合编译图 | 成功；健康检查 200，42 层加载启用；编译图包含 42 个融合调用，token 维为符号维 |
| 官方 4K/16K 吞吐及 TTFT | 用户要求停止；4K 完成 3 轮、第 4 轮中断，16K 未开始，无正式 summary |
| math_500 Level 3 精度 | 本次未执行，待验证 |

诊断为随机 synthetic 真实 Down N=2048、K=6144，seed=42，BF16 输入/输出，`triton.testing.do_bench(warmup=100,rep=300)`；值为该工具默认统计输出，不作为官方模型成绩。

| M | BF16 (ms) | 独立 PyTorch reference 反量化+GEMM (ms) | 原直接融合调用 (ms) | 自定义算子融合调用 (ms) | 相对 L2 误差 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.040864 | 0.309263 | 0.141814 | 0.141779 | 0.000247678 |
| 64 | 0.074330 | 0.339049 | 0.168527 | 0.169927 | 0.000120670 |
| 2048 | 0.297650 | 0.567206 | 3.122921 | 3.152767 | 0.000046371 |

两种融合调用均执行同一 Kernel，用于区分集成封装与原 Kernel 成本。自定义算子融合路径相对 BF16 的耗时约 3.47×/2.29×/10.59×，本诊断未观察到融合优于 BF16，不能承诺端到端加速。相对误差参照 `F.linear(x,BF16(q*scale))`，不是相对原浮点模型的量化精度结论。

串行执行异常记录：pytest 输出通过后仍处于进程退出阶段，性能诊断曾被过早启动；立即按核对后的 PID 停止，日志只有 `nohup` 提示、无性能数据，保留为 `micro-aborted-waiting-unit-exit.log`。正式诊断等待 pytest 完全退出、退出码 0 且 GPU 无进程后重新执行；该提前启动不作为正式结果。

执行顺序调整：用户在 BF16 首次尝试开始后明确要求优先测融合。已按核对后的任务进程组停止 BF16 服务/客户端及编排进程，将其归档为 `bf16-interrupted/`，不计入正式基线；确认 GPU 无进程、显存 826 MiB、9031 空闲后，新建融合服务。后续顺序为融合完整官方命令，再重新建立完整 BF16 对照。没有截断融合版官方流程；此前 BF16 中断属于用户授权的诊断例外。

模型初始化已通过动态形状编译及默认 FULL_AND_PIECEWISE 图捕获。`fused/computation_graph.py` 中恰有 42 个 `torch.ops.vllm.fl_w8a16_down_fused` 调用，输出形状为 `bf16[s72,2048]`，证实符号 token 维保留且融合进入模型图。启动覆盖日志 42/42、回退 0；实际推理已返回 HTTP 200，停止后全日志核对无融合回退，记录于 `fused/coverage-final.json`。不完整第四轮不计入成功请求统计。

### 用户停止后的实际状态（11:37，UTC+8）

按用户明确指令停止编排进程，取消其后自动 BF16 流程，再停止经 PID/进程组核对的服务和客户端。保留原始 CSV、stdout、启动/覆盖日志、资源采样及停止前后快照，不补跑、不择优、不将不完整结果当成正式成绩。

| 4K 轮次 | total tokens/s | Mean TTFT (ms) | 成功请求数 | 状态 |
|---|---:|---:|---:|---|
| 1 | 3618.57 | 6881.54 | 256/256 | 完成；按官方逻辑应跳过 |
| 2 | 2835.74 | 8830.00 | 256/256 | 完成 |
| 3 | 3577.74 | 5828.05 | 256/256 | 完成 |
| 4 | — | — | 未形成完整统计 | 用户中断 |

原始文件为 `raw_runs_20261026_110911.csv`，共 3 个完成轮次；无对应完整 `summary`，不人工用第 2/3 轮替代第 2/3/4 轮平均。16K 未开始；当前实例 BF16 首次尝试已中断，后续完整对照未启动。因此没有同实例完整优化收益结论。此次停止后只归档和释放资源，尚未开展新的 Kernel 调优。

11:38 最终核对：编排、服务及客户端进程组均已退出，GPU 无进程、利用率 0%、显存 826/65536 MiB，9031 可绑定。证据见 `user-stop-before.txt`、`user-stop-after-initial.txt`、`user-stop-final.txt`；完整停止归档为本地 `remote-user-stopped.tar.gz`，已下载并展开保存。

## 5. 与已有改动的关系

依赖 `2026-10-23-w8a16-down-fused.md` 的已有加载时量化和 tile 内反量化 Kernel。与 `2026-10-26-fused-vs-bf16-benchmark.md` 的失败定位对应；不覆盖原报告或原 CSV。没有调整 W8A16 路线、精度算法或其他投影；FlagGems 无修改。

本次仅恢复融合实现与模型动态编译的接口兼容性，不是 Kernel 性能调优。独立诊断提示后续可能仍需 Kernel 优化，但该工作不混入本次修复。

## 6. 合规性与风险

不更改官方服务/评测参数、脚本实现、RUNS/SKIP_FIRST、数据集或采样。保留框架 GEMM 选择与不支持输入的回退逻辑。未使用 `torch.compiler.disable`、禁用编译或修改 CUDAGraph 模式绕过启动问题。

仍需模型启动和实际推理验证 opaque custom op 与 vLLM 图捕获的集成；动态编译回归采用记录 backend 验证图不特化，不能替代真实模型 Inductor 验证。性能诊断显示当前 Kernel 明显慢于 BF16，完整模型指标以官方汇总为准。缺少融合模型精度，不宣称完成赛事验收。

## 7. 复现方式

远端证据：`/workspace/benchmark_results/fused-dynamic-20261026-02/`；本地证据：工作区 `artifacts/fused-dynamic-20261026-02/`。包含修改前/候选源码归档、`candidate.sha256`、`fix.diff`、`unit.log/unit.exit`、`micro.log/microbenchmark.json/micro.exit`、编译产物以及两种模式的服务/评测日志和资源采样。

先加载已有 Conda/MACA，确认资源空闲，串行执行：

```bash
cd /workspace/vllm-plugin-FL
export PYTHONPATH=/workspace/vllm-plugin-FL VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=1 VLLM_FL_W8A16_DOWN_FUSED=0 VLLM_FL_W8A16_OFFLINE=0
unset VLLM_FL_W8A16_CHECKPOINT
python -m pytest tests/unit_tests/quantization/test_w8a16_down.py -q
# 等待测试进程完全退出，核对 GPU 空闲后：
python /workspace/benchmark_results/fused-dynamic-20261026-02/microbenchmark.py
```

GPU 诊断退出并确认释放后，新建模型服务。BF16 为 `DOWN=0,FUSED=0`；候选为 `DOWN=1,FUSED=1`，每次切换都重启服务：

```bash
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

服务健康和覆盖核对后原样执行：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

赛事脚本 SHA256 保持 `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`，每场景 4 轮、跳过首轮取后三轮平均；保留所有 raw 行和原始 stdout。运行期间不额外重启服务或清缓存，结果不择优。
