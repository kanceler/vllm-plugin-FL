# 天数 Down-only W8A16 框架适配与算子正确性

## 1. 基本信息与摘要

日期：2026-11-13。设备：Iluvatar BI-V150，驱动/CoreX 4.5.0，PyTorch 2.10.0、Triton 3.6.0。第一阶段完成框架适配及独立算子正确性，不包含性能选优或模型验收。

插件分支 `feat/tianshu-down-w8a16-adapt`，基于 final `f82fa1670509f081aa13b58464cb6e5627fc10c4`，复用容器 `tianshu-default-enable-20261111-01/source` 的既有 default 优化。新增 `vllm_fl/quantization/w8a16/`、两个 `benchmarks/diagnostics/validate_tianshu_down*.py`，在 `vllm_fl/__init__.py` 注册。FlagGems 分支 `feat/tianshu-down-scale` 修改 `_iluvatar/ops/mm.py`，独立新增 scale 输出变体。

本机两仓库位于 `worktrees/vllm-plugin-FL-tianshu-down-adapt` 和 `worktrees/FlagGems-tianshu-down-adapt`。均未提交、未推送。

## 2. 问题与原因分析

沐曦最终量化设计不能直接导入 `_metax` 后端作为天数实现。天数原始MM候选 Split-K=1，已有 Separate 采用每次INT8转换、BF16 MM中间输出、独立FP32 scale；转换和后处理均构成完整调用的真实成本。大M融合需要天数自己的MM输出变体，小M Split-K需要明确的partial与归约，而不能仅把原生atomic分支的Split-K改大。

先前口头将天数基底统一描述为001不准确：当前容器已有11-11 default-enable。此次按用户要求保留它，实际MM对照为安装包原始 `mm.py`，SHA256 `4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090`。没有把001固定Down分支混入本次对照。

## 3. 具体实现

- 权重静态、对称、per-local-output-channel INT8 `[N,K]`，scale FP32 `[N]`；零通道scale=1。激活和输出BF16，dot FP32累加，不是INT8×INT8 GEMM。
- 只选择 `down_proj`、原始UnquantizedLinearMethod及受支持Linear类；替换加载后的本地分片及apply，不修改TP通信、上层forward和其他投影。开关 `VLLM_FL_TIANSHU_W8A16_DOWN=1`，默认关闭。实际模型42层覆盖未测试；独立合成层选择检查通过。
- `backend.run`提供 dense、splitk、separate、separate_rounded及auto。Dense融合转换/scale并处理尾块；Split-K在partial内把INT8转换BF16，写FP32 workspace，reduce融合scale。Separate每次分配BF16暂存并转换，复用原Iluvatar MM计算主体的独立 `mm_kernel_scale` 变体，FP32累加后乘scale再写BF16。rounded对照保留中间BF16舍入。
- 初始配置BM32/BN64/BK64、warps8/stages1，Split-K默认4，转换BLOCK1024/warps8；仅为已测正确性的保守起点，不是性能赢家。auto对Down M≥128用Separate，其余用Dense；这是暂定策略，**没有宣称M64已自动选择Split-K**。Split-K可显式调用，后续实测再接入性能策略。
- 转换BLOCK/warps、MM tile/warps/stages和Split-K可通过独立接口指定，供后续2048/4096联合搜索；本轮未搜索布局或候选池。
- opaque op提供fake实现；模型编译身份绑定全部量化源码及MM提供者源码。输出、权重暂存、partial均在完整调用中分配，没有持久BF16副本。
- 非连续/偏移激活由wrapper在调用内整理，bias保持额外后处理，尚无bias融合性能声明。输入不支持时明确拒绝；量化前不支持的层保持BF16，不把INT8权重回退称作原始BF16。

## 4. 实验与验证

独立目录 `/workspace/experiments/tianshu-down-adapt-20261113-01`；既有Python环境 `/workspace/tianshu-mixed-env-20261109/bin/python`。插件与FlagGems均从隔离副本导入，生产安装包未覆盖。测试前GPU无计算进程、70MiB驱动保留、端口9031未监听；串行执行算子与框架检查，无模型服务/Benchmark并行。

预设实现误差门槛：输出有限、relative L2≤0.005、normalized max≤0.01；normalized max定义为最大绝对误差/参考最大绝对值。参考FP32 matmul关闭TF32后，按各算法契约舍入BF16。量化误差单独报告，不以实现误差门槛判断模型精度。

| 项目 | 结果 |
|---|---|
| 两种子42/20261113，9组Shape，5路径eager+图重放及wrapper检查 | 198项全部通过 |
| 最大实现relative L2 | 0.000037100249755894765 |
| 最大normalized max | 0.003816793905571103 |
| 合成权重相对原BF16参考的量化L2范围 | 0.0060588657～0.0092478050；仅诊断，不是math_500精度 |
| Down-only选择、apply、安装幂等及重复加载保护 | 通过；合成RowParallelLinear对象，checkpoint后处理使用stub |
| opaque op fullgraph/Inductor编译及动态M=2/5 | 通过 |
| 非连续、三维输入、bias、空输入、不支持dtype拒绝 | 通过 |
| 模型加载/42层覆盖、真实TP、官方精度/性能 | 待验证 |

Shape为 `(1,17,33)、(3,129,257)、(63/64/65,2048,6144)、(127/128,129,257)、(2048/4096,2048,6144)`；每条图测试捕获后将输入减半并重放，重算参考，非仅比较固定输入。包括零权重通道。原始198条在 `results-v2.jsonl`。

首次validation-v1在custom op注册时因函数缺类型标注退出，未进行GPU数值验证；修复后v2全部通过，失败日志保留。integration-v1退出码0。日志中的FlagGems replay benchmarker不可用警告是调优计时能力提示；本次显式CUDA Graph捕获/replay成功，不宣称获得replay性能。vllm._C缺失警告未阻断本轮验证，真实模型启动仍需独立核对。

## 5. 与已有改动的关系

复用天数旧量化加载/缓存/fake参考结构及沐曦三算法数学契约，不复用C500 M分界、cpasync、tile或provider manifest。default-enable的mixed/decode/sampling源码是已有前置，不作为本次新增优化。FlagGems本地分支从含001的ea34f2e派生，但本次恢复其MM基础部分到实际final安装包版本再追加独立scale变体，故diff含001热点分支删除；不修改旧001工作树或历史结果。

## 6. 合规性与风险

未改官方命令、数据、Benchmark或模型checkpoint，未启动模型评测，未进行评测集校准。没有实测吞吐收益。当前Auto策略/转换参数/配置属于正确性起点；M64 Split-K与2048/4096联合布局和配置优化仍为下一阶段。量化在框架synthetic层测试启用，不代表模型实际42层启用。真实TP、更多Shape/留出集、workspace峰值、提供者失效策略及模型精度仍需验证。direct scale与rounded舍入契约不同，不能沿用历史模型精度。

## 7. 复现方式

进入远端独立实验目录，先完成资源检查再串行执行：

```bash
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python source/benchmarks/diagnostics/validate_tianshu_down.py results-new.jsonl
PYTHONPATH=$PWD/source:$PWD/gems /workspace/tianshu-mixed-env-20261109/bin/python source/benchmarks/diagnostics/validate_tianshu_down_integration.py
```

实际命令记录于 `test-commands.json`，原日志为validation-v1/v2.log、integration-v1.log；源码指纹以hashes-final.json为准，hashes.json是修复前快照，不替代最终指纹。runtime.json记录实际版本及import。本地证据目录为工作区 `experiment-artifacts/2026-11-13-tianshu-down-adapt-01/`。本轮没有性能表或模型准确率，均待后续实测。
