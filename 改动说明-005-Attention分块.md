# 改动 #005：按设备属性重调 Triton Attention 的分块参数

> 平台：天数 BI-V150（Iluvatar），官方启动命令
> 提交物：`vllm-plugin-FL`（`vllm_fl/attention/tile_policy.py`）
> 日期：2026-10-21

---

## 摘要

天数上跑的 attention 是 **vLLM 自带的 `TritonAttentionBackend`**。它的启动器把
**为解码设计的小分块**（`BLOCK_M=16`，每个 CTA 只处理 2 个 query token）也套用到了 prefill 上。

在 SM 数很多的 NVIDIA 卡上，这个参数靠并行度就能把带宽喂饱；
在**只有 16 个 SM** 的天数上，它让 prefill 阶段的 K/V 被重复读取 8 倍。

本改动在**不修改 vLLM 安装包**的前提下，按设备属性（SM 数、共享内存、warp 大小）
为 prefill 重新选择分块；decode 保持原样。

| 场景 | 改动前 | 改动后 | 提升 |
|---|---|---|---|
| **4k** Total tok/s | 1969.80 | **2571.23** | **+30.5%** |
| **16k** Total tok/s | 886.55 | **1962.14** | **+121.3%** |
| 4k Mean TTFT | 13,601 ms | **5,392 ms** | **−60%** |
| 16k Mean TTFT | 613,162 ms | **251,349 ms** | **−59%** |
| 精度（MATH-500 L3）| 0.962（官方基线）| **0.971** | 无损（门槛 0.95）|

---

## 1. 问题

### 1.1 上游的默认值

`vllm/v1/attention/ops/triton_unified_attention.py`：

```python
BLOCK_M = (
    16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
)
BLOCK_Q = BLOCK_M // num_queries_per_kv
```

MiniCPM5-2B 的 `num_queries_per_kv = 16 / 2 = 8`，于是 `BLOCK_M=16`、`BLOCK_Q=2`——
**每个 CTA 只处理 2 个 query token**。这组参数同时被 prefill 和 decode 使用。

### 1.2 为什么它是瓶颈

一次注意力迭代：

```
运算量 = 4 · BLOCK_M · BLOCK_N · d
搬运量 = 4 · BLOCK_N · d            （K + V，bf16）
算术强度 = 运算量 / 搬运量 = BLOCK_M
```

**算术强度恰好等于 `BLOCK_M`。**

| | BLOCK_M | 算术强度 | 含义 |
|---|---|---|---|
| 默认 | 16 | 16 FLOP/字节 | 读一次 K/V 只服务 2 个 query |
| 本改动 | 128 | 128 FLOP/字节 | 读一次服务 16 个 query |

加速器的"平衡点"（峰值算力 ÷ 峰值带宽）在几百的量级，**16 远低于它 ⇒ 纯带宽受限**。

在 104 SM 的卡上，CTA 数量足够多，带宽靠并行度就能喂饱，问题被掩盖；
在 16 SM 的卡上，必须靠"每个 CTA 多干活"来提高复用。

### 1.3 上游自己也在做同样的事

同一个文件里已有按平台重调的先例：

```python
tuned_large_head = (head_size == 256 and max_seqlen_q > 1
                    and current_platform.is_device_capability_family(100))
if tuned_large_head:
    BLOCK_M = 32; launch_num_warps = 8; launch_num_stages = 2
    # 注释原文："is ~2x faster on B200"
```

本改动只是把触发条件从"NVIDIA 的大 head"换成"SM 数少的加速器"。

---

## 2. 实现

### 2.1 分块策略（`vllm_fl/attention/tile_policy.py`）

```python
def compute_prefill_tiles(max_seqlen_q, num_queries_per_kv, head_size):
    default_bm = 16 if num_queries_per_kv <= 16 else next_pow2(num_queries_per_kv)

    if max_seqlen_q <= 1:          # decode 形状：3D 路径已由上游调优，不动
        return default_bm, None, None

    sms, smem, warp = device_limits()
    if sms >= 64:                  # 大 SM 数设备：默认值不是瓶颈，不动
        return default_bm, None, None

    warps, stages = 8, 2
    threads = warps * warp

    bm_reg  = (128 * threads) // head_size                 # 寄存器约束
    bm_smem = smem_budget // (head_size * 2)               # 共享内存约束
    bm_q    = 2 * max_seqlen_q                             # 尾块浪费约束

    bm = floor_pow2(min(bm_reg, bm_smem, bm_q))
    return max(default_bm, min(bm, 128)), warps, stages
```

**不写死任何平台名**：换一张 SM 更多的卡，公式会自然退回默认；
换一张共享内存更小的卡，会得到更小的 `BLOCK_M`。

天数上的实际取值：

```
prefill q=2048 → (BLOCK_M=128, warps=8, stages=2)
prefill q=4096 → (BLOCK_M=128, warps=8, stages=2)
decode  q=1    → (BLOCK_M=16,  None,    None)   ← 与上游完全一致
```

### 2.2 如何接上去（不改 vLLM 安装包）

`BLOCK_M` / `launch_num_warps` / `launch_num_stages` 都是启动器里的**局部变量**，
无法从外部注入。因此采用**精确文本替换 + 重新绑定符号**：

1. 读取上游启动器源码，对计算分块的那 4 行做**整块精确匹配**
2. 替换为对 `_fl_prefill_tiles()` 的调用
3. 用 `ast` 取出 `unified_attention` 这一个函数，编译并绑定到
   `vllm.v1.attention.backends.triton_attn.unified_attention`

**Triton 内核 `kernel_unified_attention` 一行不动。**

安全性：

- 匹配失败（vLLM 升级改了这段）⇒ 记 warning 并**放弃安装**，退回上游默认行为
- 任何异常都只记日志，不影响启动
- 环境变量 `VLLM_FL_ATTN_TILE_POLICY=0` 可整体关闭

### 2.3 接入点

`vllm_fl/dispatch/backends/flaggems/flaggems.py::FlagGemsBackend.attention_backend()`：
在返回 `TRITON_ATTN` 类路径之前调用 `install_tile_policy()`。
这是引擎初始化时必经的一处，且此时 vLLM 的 attention 模块可以安全导入。

---

## 3. 实验结果

### 3.1 离线扫描（同一份实现，只改常量）

**prefill**

| 形状 | 默认 | BM=128 w8 s2 | 加速 |
|---|---|---|---|
| 4k：b=4, q=2048, kv=4096 | 56.11 ms | **9.64 ms** | **5.8×** |
| 16k：b=2, q=4096, kv=16384 | 260.48 ms | **43.78 ms** | **5.95×** |
| 小：b=8, q=512, kv=2048 | 16.58 ms | **3.14 ms** | **5.3×** |

**decode**（3D 路径；真实 decode 走的就是这条）

| 形状 | 默认 | 其他配置 |
|---|---|---|
| 4k：b=32, kv=4096 | **0.820 ms** | 全部 +10%~+49% |
| 16k：b=31, kv=16384 | **2.934 ms** | 全部更慢 |
| 满：b=64, kv=4096 | **1.611 ms** | 全部更慢 |

**⇒ decode 的默认配置已是最优，这正是本改动不碰它的原因。**

数值：默认 / BM=64 / BM=128 三个配置的输出**逐元素一致**。

### 3.2 端到端（官方 benchmark 命令）

**4k**（`[[4096,1024,64,256]]`）

| | stock | 改动后 | 变化 |
|---|---|---|---|
| R1 Total tok/s | 1756.92 | **2208.60** | +25.7% |
| R2 Total tok/s | 1969.80 | **2571.23** | **+30.5%** |
| Mean TTFT | 13,601 ms | **5,392 ms** | −60% |
| 成功请求 | 256/256 | 256/256 | ✓ |

**16k**（`[[16384,1024,64,128]]`）

| | stock | 改动后 | 变化 |
|---|---|---|---|
| Total tok/s | 886.55 | **1962.14** | **+121.3%** |
| Benchmark 时长 | 2513 s | **1136 s** | **2.21×** |
| Mean TTFT | 613,162 ms | **251,349 ms** | −59% |
| 成功请求 | 128/128 | 128/128 | ✓ |

> 16k 的 stock 基线取自先前会话。同一会话内对"含 002/003、无本改动"的版本
> 实测为 896.60 tok/s，与 886.55 一致，故差异全部来自本改动。

### 3.3 为什么 16k 提升远大于 4k

| | prefill 占比 | 上下文长度 | 提升 |
|---|---|---|---|
| 4k | 43% | 4096 | +30.5% |
| 16k | 51% | 16384 | **+121.3%** |

两个因素叠加：prefill 占比更高（decode 不受影响），且上下文长 4 倍 ⇒
`BLOCK_Q=2` 造成的带宽浪费在大上下文下被放大得更多。

### 3.4 精度

evalscope `math_500` Level 3（105 题，官方命令逐字执行）：

| | 精度 |
|---|---|
| 官方基线 | 0.962 |
| **改动后** | **0.971** |
| 门槛 | ≥ 0.95 |

分块只改变计算切分顺序，数学等价；离线逐元素比对也确认不同配置输出一致。
0.971 与 0.962 的差异属于采样噪声。

---

## 4. 与已有改动的关系

| 改动 | 作用面 | 天数 | 沐曦 |
|---|---|---|---|
| #002 自适应 chunk 预算 | 调度 | +1.1%（噪声内）| +15.79%（16k）|
| #003 消除 host 同步 | 沐曦 attention 封装 | 不适用 | 含在 +4.90% 内 |
| **#005 本改动** | 天数 attention 内核 | **+30.5% / +121.3%** | 不适用 |

三者互不冲突：沐曦走 `FLASH_ATTN`（厂商 `.so`），根本不进入
`unified_attention` 这段代码，**对沐曦完全无影响**。

---

## 5. 合规性

| 规则 | 对应 |
|---|---|
| 允许「计算逻辑优化（算子融合、**内核重写**）」| 重写内核的分块策略与启动参数 |
| 禁止「**没有算子优化的情况下**切换算子」| 不切换算子：同一 kernel、同一路由、同一选择逻辑 |
| 禁止「删掉框架主要算子选择逻辑」| 未改动 `attention_backend` 的选择逻辑 |
| 禁止「直接 merge 最新分支」| 非版本升级，按设备属性重新推导参数 |
| serve 参数与基线一致 | 未改任何 serve 参数 |
| 禁止 benchmark 脚本调优 | 脚本未改 |
| 策略体现在代码并最终使用 | 默认生效，环境变量可关闭 |

---

## 6. 复现

```bash
# 端到端（官方命令）
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

# 启动日志里应出现：
#   [FL] attention tile policy installed (device: sms=16, smem=128 KB, warp=64).

# 关闭本改动做对照
export VLLM_FL_ATTN_TILE_POLICY=0
```
