> 2026-11-05 完成后审计：发现显式候选的LibEntry启动参数缓存别名。原始计时保留，名义warps/stages与完整联合搜索结论不能直接认证；完整归档、入口实测及修正见[完成后审计](2026-11-05-w8a16-completed-sweeps-analysis.md)。

# W8A16 小 M Split-K 宽搜索与精简搜索工具

## 1. 基本信息与摘要

- 日期：2026-11-04，UTC+8。
- 目标设备：MetaX C500；本次仅完成本地实现及 CPU 验证，未连接当前 GPU 实例。
- 插件开发基点：`f93489fa1afb74969b76701c8b551552a55ae3bf`。
- FlagGems 本地基点：`f6a4257e3d042acd26a7d3c496848025a4b32c07`，已有未提交 W8A16 显式后端；本次不修改 FlagGems。
- 新分支：`perf/gemm-w8a16-splitk-wide-search`，保留原分支的既有未提交内容，未提交或推送。
- 新增文件：
  - `benchmarks/diagnostics/w8a16_splitk_search.py`：纯 CPU 候选生成、合法性、理论筛选、精简池及性能分类。
  - `benchmarks/diagnostics/tune_metax_w8a16_splitk.py`：串行批量执行、完整调用计时、独立输入复验、证据与续跑。
  - `benchmarks/diagnostics/test_w8a16_splitk_search.py`：CPU 逻辑验证。

用户已确认本轮方案：先利用现有三算法粗分区，重点补齐小 M Split-K；20 个宽搜索点、40 个精简搜索点，不搜索 M≥160，不测试 S=1，不改变 basic 流水线，不调 Reduce。该工具尚无实际 GPU 性能数据，不能称为完成调优或生产优化。

## 2. 问题与原因分析

11 月 1 日及 11 月 3 日的显式 Split-K 池只有六组：BM={16,32,64}、S={4,8}，而 BN=64、BK=128、warps=4、stages=2 固定。最新 76 点的 456 条 Split-K 筛选记录并不能证明更宽参数空间没有收益。

特别是 O 小 M，历史和最新 Split-K 本身耗时接近，原始 BF16 对照耗时不同，导致相对收益反转。因此本轮既要扩大配置覆盖，也要记录同场 BF16 的实际子路径、输入布局、提供者和配置状态。

当前整除型实现的结构为：

```text
P_output = ceil(M/BM) * ceil(N/BN)
P_partial = P_output * S
K_iterations_per_split = K/(BK*S)
workspace_bytes = 4*S*M*N
```

S、BM、BN、BK共同改变并行度、每块循环和临时数据成本。并行块数不是实测占用率，workspace 也不等于完整执行峰值。保留当前 FP32 partial→FP32 reduce→FP32 scale→BF16 输出契约。

## 3. 具体实现

### 候选域与 M 清单

配置顺序为 `(BM,BN,BK,warps,stages,S)`：

| 参数 | 范围 |
|---|---|
| BM | 16、32、64 |
| BN | 32、64、128 |
| BK | 64、128、256 |
| S | 2、4、8、16、32 |
| warps | 2、4、8 |
| stages | 1、2、3 |
| pipeline / Reduce | basic / 原 BLOCK256、warps4 |

| 投影 `(N,K)` | 宽搜索 M | 精简搜索 M |
|---|---|---|
| Down `(2048,6144)` | 1、4、16、32、64 | 2、3、8、24、48、56、72、80、96、104、112 |
| Gate/Up `(12288,2048)` | 1、4、16、24、32 | 2、3、8、40、48、64、88、96、104、112 |
| QKV `(2560,2048)` | 1、4、16、32、64 | 2、3、8、24、48、56、72、80、96 |
| O `(2048,2048)` | 1、4、16、32、64 | 2、3、8、24、48、56、72、80、96、104 |

每点域内有 1,215 组组合；对本次四组 N/K，`N%BN==0` 且 `K%(BK*S)==0` 后剩 972 组。BK64 支持 S2/4/8/16/32；BK128 支持 S2/4/8/16；BK256 支持 S2/4/8。线程布局、寄存器与实际 shared 资源仍待编译验证。

### 理论筛选与可追溯性

- 硬条件：当前代码的 N/K 整除约束；声明的 FP32 workspace 预算默认 256 MiB。
- 软条件：`BM > max(16,2*M)` 的行浪费；以及每片至多两次 K 循环、partial 块数超过实际 SM 数 16 倍的组合。均明确标为启发式，不能写成算法不合法。
- 原六组配置保留。每个被软筛选的 BM/BN/BK/S 结构，保留 warps4/stages2 的审计候选，包含合法的 S32 探针。
- staged operand 字节、每线程 accumulator 元素仅为规划代理值，不用作实际资源超限的证明。
- 宽搜索完整执行所有保留候选，逐条记录剔除原因；不再设每点 128 组上限。
- 每个宽搜索点保留 Top5 和 1% 范围内的其他 tile/S 结构；按投影合并，并加入历史控制项，作为该组精简池。独立正确性失败的配置不作为新优胜项传播。
- 选择来源明确标为有限池离线调优；直接显式调用每次重新计时。续跑复用旧计时单独计数；首次调用含编译/缓存等，不能等同新编译数。

### 正确性与最终计时

- 每个配置先对真实投影 synthetic 输入验证，固定 relative L2 `<0.005`、rtol=0.02、atol=0.125。未通过者不进入计时赢家。
- Finalist 增加 M=3、N=128、实际 K 的 INT8端点、零通道、scale范围检查，验证 M mask 和不同分片覆盖；不放宽 N/K 尾部能力。
- 终测为两套独立输入，各五轮交错；warmup100ms/rep300ms，逐种子取中位数。初筛20ms/60ms。均为独立算子诊断协议，不是官方模型评测。
- 计时覆盖每次输出/workspace分配对应的完整调用、partial、Reduce/scale；未预转换权重或预分配 workspace。
- 同场原始 BF16 为正式算子对照；BF16(Q)仅为诊断。保存调用子路径、原始BF16与Q权重的布局/地址对齐、调优器状态与已编译资源快照。
- 调优器 best_config 快照可能被 LibEntry 快路径绕过，明确不是逐launch配置证据；未知项不能写成已确认缓存命中或已确认配置。
- 终测从初筛保留项中按两个独立输入的归一化延迟中位数选择配置；每个输入的延迟及分类分别保留，只有两套输入都降低超过1%才标为两套均有有效收益。

### 隔离、续跑和结果

控制器不持有 GPU 上下文，每个 Shape 用独立子进程串行运行；不按配置反复导入整套框架。任务开始检查 GPU 空闲、9031端口及进程，绑定实际设备/软件、驱动、Kernel、BF16提供者、配置YAML、调优器和工具指纹。缺少实际SM/warp信息时拒绝套用默认硬件值。

普通编译、资源和正确性失败逐条保存。进程崩溃或设备错误保留 active 配置、退出状态和日志并停止；核查后显式 `--resume`，不无依据反复重试。续跑须指纹一致，重用样本不冒充新测；残留锁文件先核对PID，不能自动删除。撕裂JSONL拒绝静默丢弃。

输出包含 `plan.json`、`metadata.json`、每点候选及筛选JSONL、独立数值结果、完整调用原始五轮、编译资源、逐种子结果和全局 `summary.json/details.csv`。未完成60点时 summary.complete 为 false。

## 4. 实验与验证

| 项目 | 本次结果 |
|---|---|
| CPU 搜索/计数/日志测试 | 11项通过，详见本地验证日志 |
| Python 语法检查 | 通过 |
| CPU 规划样例 | 20宽点＋40精简点；不是GPU实验 |
| GPU 编译及新配置数值验证 | 待验证 |
| Split-K 对原生 BF16 的性能 | 待验证 |
| 图重放、多层工作集、模型精度、官方吞吐/TTFT | 本轮未执行，不宣称收益 |

CPU 样例显式使用 SM=64、warp=64，用于验证工具输入输出，**不是当前 C500 硬件实测值，也不是正式搜索清单**。该样例宽点累计 24,300 条组合：4,860 条整除失败，7,776 条启发式筛选，11,664 条计划保留。实际GPU任务按真实属性重新规划，编译/资源通过数尚未知。

本地证据目录：`experiment-artifacts/2026-11-04-w8a16-splitk-wide-prep-01/`。旧临时连接文件已不存在，工作区残留的远程脚本属于更早实例；未使用旧连接启动实验。当前连接信息已向用户询问，等待有效实例后执行。

## 5. 与已有改动的关系

直接复用已有 `run_w8a16_algorithm(...,"splitk",config)`，因此本次不修改 FlagGems Kernel、Reduce或生产分派。11月1日/3日的六组实验和旧三算法工具保留原样，本次新增专用工具，避免改写历史复现入口。

原生 BF16底层归队友负责，本次只调用并记录其选择；不优化或覆盖其实现。旧Down-only官方负结果和103/105精度不作为本轮新配置的结果。算法粗分区用于限定本次搜索范围，最终三路径Dispatch留到后续。

## 6. 合规性与风险

- 无模型量化层接入，层覆盖率、真实算法调用覆盖、耗时覆盖均未测试；当前数据来源为synthetic算子输入。
- 权重INT8 `[N,K]`、静态对称每输出通道量化、FP32 scale；激活BF16、INT8转BF16后dot、FP32累加/归约、输出BF16。不是原生INT8×INT8 GEMM。
- 官方启动参数、Benchmark、数据集、依赖和持久编译缓存均不改动；不并发GPU实验，不清理其他任务。
- 软剪枝存在遗漏风险，审计候选只能评价部分遗漏；最终只称所测范围内较优。
- 固定basic和Reduce，新配置GPU编译、数值及真实性能未验证。CPU测试不能替代上卡门禁。
- 本轮20＋40点均用于调优，不能作为独立Shape留出集；后续Dispatch另设留出验证。
- BF16配置状态快照不是逐Kernel replay证据；GPU事件完整调用不等于CPU壁钟或图重放性能。

## 7. 复现方式

本地CPU检查（不需要Torch/Triton）：

```bash
cd /Users/tianfr/muxi_2b/vllm-plugin-FL
PYTHONPYCACHEPREFIX=/private/tmp/w8a16-splitk-pycache \
  python3 -m unittest discover -s benchmarks/diagnostics \
  -p test_w8a16_splitk_search.py -v
```

已执行CPU规划样例命令：

```bash
python3 benchmarks/diagnostics/tune_metax_w8a16_splitk.py \
  --plan-only --sm-count 64 --warp-size 64 \
  --output-dir /Users/tianfr/muxi_2b/experiment-artifacts/2026-11-04-w8a16-splitk-wide-prep-01/cpu-plan-example
```

GPU命令模板如下，**尚未执行**。先按根AGENTS归档和核查目标仓库、已有环境、实际import、进程/显存/端口；使用隔离工作目录同步必要文件，并清除与本模式冲突的旧W8A16开关。路径需绑定实际新实验目录，不能覆盖旧结果。

```bash
python benchmarks/diagnostics/tune_metax_w8a16_splitk.py \
  --output-dir /root/experiments/<new-experiment>/results
```

只在同一实验指纹及已核对资源条件下显式续跑：

```bash
python benchmarks/diagnostics/tune_metax_w8a16_splitk.py \
  --output-dir /root/experiments/<same-experiment>/results --resume
```

实际GPU命令、环境指纹、配置结果、数值误差、性能与异常须在执行后补充；本报告不以模板充当执行证据。
