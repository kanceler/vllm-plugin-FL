# 改动 #011：BI-V150 混合 Attention 分相启动

> 环境：天数 BI-V150（16 SM / 128 KB SMEM / warp 64），MiniCPM5-2B，BF16
> 基线：`s2-track2-final` 的 `f82fa1670509f081aa13b58464cb6e5627fc10c4`
> 状态：官方 4k / 16k 四轮脚本已完成，丢弃首轮；官方 MATH Level 3 达到门槛。新路径的附加 MATH 覆盖未完成，见 §3.3。

---

## 1. 为什么这样改

#006 / #009 / #010 已分别优化 prefill 和 decode 的启动参数，但上游
`unified_attention` 仍以整个批次的 `max_seqlen_q` 选择执行形状：

```python
if max_seqlen_q > 1:
    # 整批使用 2D attention，包括其中 query_len=1 的 decode 请求
    ...
else:
    # 请求数和暂存空间满足条件时，可以使用 3D KV 分段和归约
    ...
```

因此，只要混合批次包含 prefill，decode 请求也会使用 prefill 的大分块，
失去原有 decode 路径的 KV 分段并行。对本设备，原 prefill 配置为
`BM=256 / warps=16 / stages=2 / TILE=128`，decode 则为
`BM=16 / warps=2 / stages=1 / TILE=16`；把单 token 的 decode 行放进前者并不划算。

本改动处理的是**同一混合批次内两类请求如何启动**，没有重新调优 #009 / #010 的参数。

## 2. 改动内容

新增 `vllm_fl/attention/mixed_policy.py`，由现有 `tile_policy.py` 接入。

1. 在 metadata 构造阶段读取已有的 CPU `query_start_loc_cpu`，识别连续的单 token
   decode 前缀。每步生成一次拆分计划，各 attention 层复用，不增加逐层 D2H 同步。
2. 把该前缀交给现有 decode 启动器；将其余 query、输出、序列长度和 block table
   作为视图交给现有 prefill 启动器。prefill 的 query 前缀和减去 decode 行数，
   输出仍写回原位置，KV cache 不复制。
3. 通过 `ContextVar` 传递当前层的计划，`finally` 中恢复上下文；只有当前调用的
   query 前缀和与计划相同时才拆分。

### 2.1 保守的适用条件

- 只处理混合批次中**连续的 decode 前缀 D≥16**，同时要求后面确有 prefill。
- D 不超过现有 3D 阈值和暂存空间容量。
- 纯 prefill、纯 decode、decode 太少、cascade、非普通 causal、PrefixLM、特殊量化、
  输出量化、QQ bias 和 tensor descriptor 等不支持的路径保留原实现。
- D=4 / 8 的首轮 GPU 小实验存在额外 kernel 启动开销大于收益的情况，因此收紧到 16；
  该门槛是本设备的实测选择，不宣称适用于所有设备。

### 2.2 只在已验证设备启用

共用的 tile policy 也会被其他平台调用，因此安装新 hook 前同时检查：

```text
current_platform.vendor_name == "iluvatar"
实际 GPU 型号包含独立的 BI-V150 型号标识
```

实际型号通过 `torch.cuda.get_device_name()` 读取，不能使用这里的
`FlagOSPlatform.get_device_name()`，后者返回的是逻辑设备类型。
其他厂商、其他天数型号、设备识别失败时都不安装新 hook，原 tile policy 继续工作。

`VLLM_FL_MIXED_ATTN_SPLIT=0` 可在启动前关闭本策略；设置为 1 也不会绕过设备范围保护。
新增测试覆盖设备范围、禁止探测其他厂商的 CUDA、识别失败回退、重复安装及原有拆分/回退行为。

提交前 **22 项回归全部通过**；真实 BI-V150 的安装检查也通过，确认识别到
`vendor=iluvatar, device=Iluvatar BI-V150` 并安装新 hook。该检查不加载模型，
结束后 GPU 空闲、原始调优库未变。Ruff 0.14.0 的格式和测试文件 lint 检查通过。

## 3. 收益验证

### 3.1 正式端到端对照

同实例先测干净 final，再测候选。原版 `benchmark_throughput_serve.py`，每场景 4 轮，
丢弃首轮，后 3 轮取均值；两版服务参数一致，均限制 CPU 编译并发为 1。
两组使用同一 FlagGems 调优库种子的独立副本，并隔离 vLLM / Inductor / Triton 缓存。

| 场景 `[输入,输出,并发,请求数]` | final total tokens/s | 候选 total tokens/s | 吞吐变化 |
| --- | ---: | ---: | ---: |
| `[4096,1024,64,256]` | 3174.15 | **3330.59** | **+4.93%** |
| `[16384,1024,64,128]` | 2707.02 | **2781.10** | **+2.74%** |

| 场景 | final Mean TTFT ms | 候选 Mean TTFT ms | 变化 |
| --- | ---: | ---: | ---: |
| 4k | 5003.88 | **4471.86** | **-10.63%** |
| 16k | 181833.28 | **176468.06** | **-2.95%** |

| 版本 | 场景 | 三轮计分 total tokens/s | CV |
| --- | --- | --- | ---: |
| final | 4k | 3135.48 / 3178.73 / 3208.23 | 1.15% |
| 候选 | 4k | 3315.60 / 3337.51 / 3338.67 | 0.39% |
| final | 16k | 2706.71 / 2704.90 / 2709.44 | 0.08% |
| 候选 | 16k | 2784.12 / 2779.44 / 2779.74 | 0.09% |

共 16 轮、3072 个请求全部成功，输入/输出 token 数符合设定。候选日志确认拆分实际触发。
没有用比赛最初的历史基线计算本次增量。

**口径限制**：这是一组顺序 A/B，不能排除全部跨时段漂移。另一次同提交 final 的历史
16k 结果为 2790.28 tokens/s，接近本次候选；因此 +2.74% 是本次对照的实测增量，
不等于已经证明跨次运行都能稳定复现这一比例。

上述长测发生在提交收尾的设备范围保护之前。收尾仅增加安装范围保护和格式整理，
未改变已测的拆分门槛、张量切分、kernel 或启动参数；不会把这些历史数据写成收尾后重新长测的成绩。

### 3.2 小实验与数值检查

真实 42 层模型的同输入交替前向，4k 的 D=16 / 24 / 32 / 48 样本耗时分别减少约
6.3% / 9.4% / 13.5% / 17.2%；16k 的 D=16 / 24 样本分别减少约 10.0% / 14.8%。
这些是前向窗口结果，不能替代 §3.1 的端到端 tokens/s。

首版 GPU 矩阵覆盖分页、不等长序列、NHD/HND、回退和 decode 图重放；随后门槛收紧为 D≥16。
抽查第 1 / 21 / 42 层，候选 decode 输出更接近独立 CPU float64 参考，prefill 逐位一致。
decode 隐藏状态、logits 和部分贪心结果会改变，不以“与旧混合路径逐位相同”作为精度证明。

### 3.3 官方 MATH 精度

EvalScope 1.11.1，MATH-500 **Level 3 全部 105 题**，并发 8，
`temperature=1.0, top_p=0.95, max_tokens=32768`：

- **100/105（95.24%）**，达到 ≥95% 门槛。
- 105 条预测、105 条评审、0 个请求错误；逐题合计与官方汇总一致。
- 103 题正常停止，2 题达到 token 上限，原样纳入评分，未剔除。

**覆盖限制**：官方并发 8 不足以触发 D≥16，因此这一轮只证明官方配置过线。
附加并发 32 的覆盖轮因实例到期中断，仅保留 1 条预测；用户决定不再追加精度轮次。
不把它记为通过，也不声称新路径在所有并发下均已完成 MATH 验收。
温度 1.0 存在采样波动，95.24% 余量较小，记录本轮成绩而不承诺每次重跑相同。

## 4. 与其它改动的关系

- **#006 / #009 / #010**：继续使用其 prefill / decode 参数推导，不修改公式或 kernel。
- **#002**：不改变调度器、请求排序或 token 预算，只识别调度器已有的请求布局。
- **#003**：复用已有 CPU metadata，不增加逐层 GPU→CPU 长度读取。
- **#007**：GEMM 路径、推导模型和全局 FlagGems 安装均未改动。

## 5. 复现与开关

使用本分支对应环境的 `vllm`，沿用天数固定服务参数：

```bash
VLLM_PLUGINS=fl TORCHINDUCTOR_COMPILE_THREADS=1 \
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072

python benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

启动日志包含 `FL mixed attention split installed`；首次满足拆分条件时输出
`FL mixed attention split active`。两条日志只用于确认安装和命中，不逐步打印。
设备不匹配或设置 `VLLM_FL_MIXED_ATTN_SPLIT=0` 时不安装新策略。

## 6. 合规性

- 保留 vLLM 的 Triton unified attention 及框架算子选择流程，只拆分同一批次的启动。
- 不改模型权重、精度、KV 格式、采样参数或 `vllm serve` 参数。
- 不启用量化或投机采样，不修改 benchmark、评测题集或评分器。
- 不依赖题目内容、请求编号或 benchmark 场景识别；适用条件来自实际批次布局。
- 当前默认范围只覆盖已验证的 BI-V150；扩展到其他设备需要独立验证，不将本设备收益泛化。
