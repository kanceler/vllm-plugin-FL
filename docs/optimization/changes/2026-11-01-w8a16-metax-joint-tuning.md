# MetaX W8A16 三算法显式候选与首轮联合调优

## 1. 基本信息与摘要

- 日期：2026-11-01
- 平台：沐曦 MetaX C500 64 GiB
- 软件：PyTorch `2.8.0+metax3.7.0.7`、Triton `3.0.0`、FlagGems `5.3.5`
- 本机 FlagGems 基线：`f6a4257e3d042acd26a7d3c496848025a4b32c07`
- 本机 vllm-plugin-FL 基线：`f93489fa1afb74969b76701c8b551552a55ae3bf`
- 变更文件：
  - FlagGems `src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py`
  - vllm-plugin-FL `benchmarks/diagnostics/tune_metax_w8a16_algorithms.py`

本次实现了可独立强制调用的 BF16 Dense、M=1 广播 Dense、可配置
Split-K 和 Separate 四个执行入口，其中 M=1 广播属于 Dense 内部候选，顶层仍为
Dense、Split-K、Separate 三算法。新增入口用于离线正确性和完整调用联合调优，未修改
现有生产分派策略。

在 Down、Packed Gate/Up、Packed QKV、O 四组本地 N/K 及 12 个代表或交界 M
上共筛选 584 个候选，全部满足预先固定的正确性门槛。官方式 BF16 Dense 和 M=1
广播候选在所测真实投影 Shape 上没有形成可用收益。唯一经独立复验的正收益窗口是
O 形状 `N=K=2048` 的极小 M Split-K：M=1/2/8/16 相对同版本原始 BF16 MM
快约 5.2%～6.4%；M=24 起反转为明显退化。该结论仅为 synthetic 算子级结果，
当前没有接入 O 投影，也不构成端到端性能收益。

## 2. 问题与原因分析

原 MetaX W8A16 生产实现包含三种执行方式，但接口和策略耦合：

1. Split-K 只允许固定 M 白名单 `(1,16,32,64)`，把已测性能策略写进能力判断。
2. Separate 的入口以 `M>=128` 等策略条件代表算法能力，无法在交界点强制比较。
3. 旧联合诊断中的 `separate` 实际是先将 `q*scale` 反量化为 BF16 再做 MM，
   scale-before-dot；它不是当前主线要求的转换 INT8→BF16、BF16 MM、输出后乘
   FP32 scale 的 Separate 完整调用。
4. 官方新增 Dense 使用手工 tile、整除无 mask 路径和 M=1 广播路径，但其 BF16
   输入仍强制转 FP16 dot。本项目已验证 BF16 dot 可用，因此只能借鉴 tile、边界
   专门化和广播候选思想，不能直接复制其数值路径。

需要把算法合法性、性能策略和显式配置入口拆开，才能回答每个真实 N/K 与 M 下究竟
是哪条完整算法更合适，并避免将固定白名单误写成算法限制。

## 3. 具体实现

### 3.1 显式算法接口

FlagGems MetaX 后端新增：

- `dense_rejection_reason`、`splitk_rejection_reason`、
  `separate_rejection_reason`：返回明确拒绝原因。
- `algorithm_workspace_bytes`：记录 Split-K FP32 partial workspace，以及 Separate
  的 BF16 转换权重和未缩放输出临时量。
- `run_w8a16_algorithm`：按算法和显式配置强制完整调用，不经过生产策略选择。
- `DENSE_CONFIGS`、`SPLITK_CONFIGS`、`M1_DENSE_CONFIGS`：共享候选池，保存
  官方 BF16 tile、FP16 大 M 迁移候选、现有生产控制项和历史控制项。

生产 `weight_int8pack_mm` 的算法顺序和阈值保持不变。`_can_use_*` 仍表示生产策略，
但内部复用独立合法性检查，避免继续复制 dtype、布局和整除条件。

### 3.2 BF16 Dense 与边界路径

新增 Dense kernel 使用 BF16 activation 和 INT8→BF16 tile-local conversion，
FP32 累加后乘 FP32 per-output-channel scale，最后输出 BF16。M、N、K 分别生成
`EVEN_M/EVEN_N/EVEN_K` 编译常量：整除维度直接加载，尾块维度保留准确 mask，
没有沿用官方“任一维不整除就给三维全部加 mask”的粗粒度分支。

配置包含 `BM/BN/BK/warps/stages`。GROUP_M 固定为 8，与现有通用 Dense 的
分组顺序一致。官方选择条件仅作为候选来源，不作为新的生产阈值。

### 3.3 M=1 广播候选

新增 BF16 版 M=1 广播候选：A 的单行广播为 `[16,BK]`，与权重 tile 做 dot，
只写累加器第 0 行。保留官方 `BN=64、BK=256、warps=4、stages=3` 作为显式
候选，但移除 `N>=16384` 的硬策略限制，以便在 Gate/Up 等真实宽 N 上独立验证。
该候选没有接入生产分派。

### 3.4 联合调优脚本

新增离线脚本按 Shape 和 M：

1. 计算输出 program 数、K 循环数、M/N tile 效率和 K tile 效率。
2. 在固定预算内保留公式排序候选、官方 tile 和生产／历史控制项。
3. 先做正确性，再计时；记录非法剪枝、编译或资源失败、workspace、来源和误差。
4. 每算法选出 screening 赢家后，与同版本原始 BF16 MM、量化后 BF16 权重 MM、
   现有生产分派和原 FlagGems 通用 Dense 做交错终测。

计时始终覆盖完整调用：Split-K 包含 workspace 分配、partial 和 reduce/scale；
Separate 包含 INT8→BF16 转换、实际 MetaX BF16 MM 和输出 scale。预转换权重只作为
量化误差诊断，不参与 W8A16 赢家选择。

## 4. 实验与验证

### 4.1 正确性与回归

- 独立门禁覆盖 Dense 的 M/N/K 尾块 `(3,129,257)`、M=1 尾块、Split-K
  非整除 M、Separate 中间 BF16 舍入，全部通过。
- vllm-plugin-FL `test_w8a16_down.py`：`22 passed in 35.80s`。
- 完整矩阵：584/584 候选通过，无编译或资源失败。
- 门槛在筛选前固定：relative L2 `<0.005`，`rtol=0.02`，`atol=0.125`。
- 完整矩阵赢家最大 relative L2 为 `0.0049017845`；生产／通用控制项最大
  relative L2 为 `0.0028518457`。Separate 的误差包含其定义中的 BF16 中间舍入。

### 4.2 O1/O2 完整矩阵

测试 Shape：Down `(N=2048,K=6144)`、Gate/Up `(12288,2048)`、QKV
`(2560,2048)`、O `(2048,2048)`；M 为
`1/16/32/64/80/96/112/128/256/512/1024/2048`。

下表列出各组较有代表性的最优 W8A16 完整调用；百分比为
`W8A16/BF16-1`，正数表示退化。

| Shape | M | 所测最佳 W8A16 | W8A16 ms | BF16 ms | 相对变化 |
|---|---:|---|---:|---:|---:|
| Down | 1 | Split-K | 0.043776 | 0.041472 | +5.56% |
| Down | 64 | Split-K | 0.082944 | 0.071936 | +15.30% |
| Down | 2048 | Separate | 0.642560 | 0.581120 | +10.57% |
| Gate/Up | 1 | Split-K | 0.059648 | 0.050176 | +18.88% |
| Gate/Up | 64 | Dense | 0.095488 | 0.068096 | +40.23% |
| Gate/Up | 2048 | Separate | 1.060864 | 0.900352 | +17.83% |
| QKV | 1 | Split-K | 0.028672 | 0.026368 | +8.74% |
| QKV | 2048 | Separate | 0.315136 | 0.252928 | +24.60% |
| O | 1 | Split-K | 0.026368 | 0.028160 | -6.36% |
| O | 16 | Split-K/生产分派 | 0.027904 | 0.029440 | -5.22% |
| O | 32 | Split-K | 0.034816 | 0.027648 | +25.93% |
| O | 2048 | 生产分派 | 0.228608 | 0.195840 | +16.73% |

结论：官方式 Dense 在 Gate/Up 的 M=32/64 虽是三算法内部赢家，仍分别比 BF16
慢 38.46%/40.23%；M=1 广播在 Gate/Up 为 `0.082688 ms`，BF16 为
`0.049920 ms`，退化约 65.64%。因此不应照搬官方 Dense 分派或 `N>=16384`
广播阈值。

### 4.3 O 投影极小 M 独立复验

使用独立实验，screening 为 50/150 ms，终测为 100/300 ms、五轮交错：

| M | 最佳路径 | W8A16 ms | BF16 ms | 相对变化 | 五轮稳定性 |
|---:|---|---:|---:|---:|---|
| 1 | Split-K S=8 | 0.026368 | 0.028160 | -6.36% | 五轮 W8A16 相同 |
| 2 | Split-K S=8 | 0.026624 | 0.028160 | -5.45% | 0.026368～0.026624 |
| 8 | Split-K S=8 | 0.026624 | 0.028160 | -5.45% | 五轮 W8A16 相同 |
| 16 | 生产 Split-K S=8 | 0.027904 | 0.029440 | -5.22% | 五轮两侧均相同 |
| 24 | Split-K S=4 | 0.034048 | 0.027648 | +23.15% | 明确反转 |
| 31 | Split-K S=4 | 0.034816 | 0.027648 | +25.93% | 明确退化 |
| 32 | Split-K S=4 | 0.034560 | 0.027392 | +26.17% | 明确退化 |

收益严格超过 1%，但仅在此 synthetic O Shape 和极小 M 范围内成立。当前没有真实
运行 M 频次和 O 投影接入证据，不能据此计算模型收益。

## 5. 与已有改动的关系

- 复用已有静态对称 per-output-channel INT8 权重和 FP32 scale，不改变 checkpoint
  格式、量化器或上层 Down 适配器。
- 复用已有 MetaX Split-K partial/reduce、Separate 转换和 MetaX BF16 MM 提供者。
- 原 FlagGems 通用 Dense、现有生产分派和历史固定配置均作为控制项保留。
- 本次没有直接合并官方最新分支；官方 tile、整除路径和 M=1 广播只作为独立实现的
  候选来源。官方 FP16 强制转换没有进入 BF16 主线。
- 历史 Down-only 官方 4K/16K 负结果保持不变；本次 synthetic 数据不能覆盖或改写
  该端到端结论。

## 6. 合规性与风险

- 没有修改官方服务参数、Benchmark、模型或数据集，没有启用现有量化启动参数替代
  算子实现，也没有合并上游性能提交。
- 新算法入口和调优脚本不改变生产 dispatch；不会在推理或图重放期间启动搜索。
- 未修改队友负责的 MetaX BF16 MM，只记录其实际提供者文件和版本。
- 尚未验证真实 O/Packed 投影接入、TP/packed 行顺序、真实动态 M 分布、CUDA Graph
  workspace 生命周期、模型精度、4K/16K 吞吐和 TTFT。
- O 极小 M 的算子收益可能因模型接入、调用频次、图捕获和 BF16 提供者版本变化而
  消失。下一阶段需先采集真实 Shape/分支覆盖，再讨论 Runtime manifest 与 O 接入。
- M=1 广播会计算 16 行累加器而只写一行；当前实测已显示无收益，按止损项保留为
  研究候选，不进入生产策略。

## 7. 复现方式

容器使用隔离源码：

```bash
export PYTHONPATH=/workspace/experiments/w8a16-20261101/FlagGems/src:/workspace/experiments/w8a16-20261101/vllm-plugin-FL

python3 -m pytest -q \
  /workspace/experiments/w8a16-20261101/vllm-plugin-FL/tests/unit_tests/quantization/test_w8a16_down.py

python3 /workspace/experiments/w8a16-20261101/vllm-plugin-FL/benchmarks/diagnostics/tune_metax_w8a16_algorithms.py \
  --output-dir /workspace/experiments/w8a16-20261101/results/joint-o1o2-all-04 \
  --shapes down gate_up qkv o \
  --ms 1 16 32 64 80 96 112 128 256 512 1024 2048

python3 /workspace/experiments/w8a16-20261101/vllm-plugin-FL/benchmarks/diagnostics/tune_metax_w8a16_algorithms.py \
  --output-dir /workspace/experiments/w8a16-20261101/results/joint-o2-o-small-01 \
  --shapes o --ms 1 2 8 16 24 31 32 --dense-budget 6 \
  --screen-warmup 50 --screen-rep 150 \
  --final-warmup 100 --final-rep 300 --rounds 5
```

本地原始证据归档在：

```text
/Users/tianfr/muxi_2b/experiment-artifacts/2026-11-01-w8a16-joint/
├── joint-o1o2-all-04/{metadata,screening,raw_final,summary}.json
├── joint-o2-o-small-01/{metadata,screening,raw_final,summary}.json
└── unit-joint-01/pytest.log
```

关键 SHA256：

- 完整矩阵 `summary.json`：
  `53a34b9ceceaad07cdd34db35b7291893d52fff907f3e52735706aba31809ad9`
- O 极小 M 复验 `summary.json`：
  `e8baaa8e063e174cafe3e5743f6d106a3195d030eb42673da52913290133d2e6`
- 单元测试日志：
  `58cdda968d810ca6b40c2da77d3abda842e1d78ccbff52980449843d48aba510`

`joint-o1o2-all-02` 是一次被交互中断的不完整诊断；`joint-o1o2-all-03` 因重连后
未设置隔离 `PYTHONPATH` 在导入阶段失败。两者保留原样，不参与上述结论。
