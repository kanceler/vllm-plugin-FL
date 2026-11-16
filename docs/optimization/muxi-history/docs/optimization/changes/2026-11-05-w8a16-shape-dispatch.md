# MetaX C500 W8A16 Shape-aware 分发接入

## 1. 基本信息与摘要

- 日期：2026-11-05；平台：MetaX C500。
- 分支：插件与 FlagGems 均为 `perf/gemm-w8a16-shape-dispatch`。
- 涉及：FlagGems `weight_int8pack_mm.py`、`w8a16_dispatch.json`、`pyproject.toml`；插件 `w8a16/linear.py`。
- 目标：增加默认关闭的 `VLLM_FL_W8A16_SHAPE_DISPATCH=1`，按本地 N/K 和 M 选择 Split-K、Dense、Separate，四类投影在模型加载后量化。原服务参数、Benchmark 实现不变。

## 2. 问题与原因分析

旧生产路径只量化 Down，FlagGems 固定 M `{1,16,32,64}` 选择固定 Split-K、M≥128 选择 Separate，其余转通用路径。此前修复后实验显示配置与算法依 N/K/M 明显变化，固定配置无法代表全部 Shape。Dense 和 Split-K 显式调用的 LibEntry 缓存键未包含 warps/stages，本次均改用 Triton JIT 的完整启动配置缓存。

## 3. 具体实现

模型加载端的新开关启用 INT8PACK 并选择 Down、GateUp、QKV、O 四类 `UnquantizedLinearMethod` 权重，对每个本地 TP shard 做对称 per-output-channel INT8 量化、FP32 scale。旧开关关闭时保持原有 Down-only 行为。启动时预加载并校验 manifest，不在 Forward 读文件。

FlagGems 分发先识别 Down `(64,2048,6144)` 固定 `(64,64,128,4,1,8)`；其余根据 `(N,K,M)` 连续区间选择算法，精确已测赢家优先，再按池内固定顺序选首个通过能力和轻量填充/资源剪枝的完整配置。Dense 只作为 GateUp M48～112 首选。未命中或无候选时使用 Separate；共同输入契约不满足时保留原有通用正确性路径。候选池不在运行时计时搜索。Separate 仍先转换权重、由原生 BF16 MM 计算、最后独立乘 scale；Dense/Split-K 保持 scale-after-dot 数值语义。BF16 原始权重在量化后不保留。

GateUp M33～47 采用 **Dense 候选区间**：从修复后邻点 M24/32/48 的完整赢家构成池 `(32,64,128,4,2)`、`(32,64,128,8,2)`、`(64,64,64,8,2)`，按能力与轻量剪枝取首个。若无合法 Dense 候选，再从 Split-K 池 `(32,64,128,4,1,4)`、`(32,128,64,2,2,2)`、`(32,64,128,4,1,2)` 取首个；两池均不可用才走 Separate。M40 修复后 Split-K 约 118.5～118.8 µs，早期同场 Separate 约 154 µs；旧同场 Dense 约 96 µs，但其 warps/stages 曾受 LibEntry 缓存别名影响。修复后 Dense 尚缺 M40 实测，故该区间为算法策略而非已证三算法赢家。原设计的 GateUp M33～47 Separate 区间由此修正。

Manifest 记录数值契约、设备、Torch/Triton 版本、61 个精确已测映射及按 N/K/M 区间去重的完整候选池；实验来源与状态写入条目。版本不符会拒绝启用新分发，避免静默使用失效配置。生产前仍须核对实际 FlagGems 导入路径与源码指纹。

## 4. 实验与验证

- 静态语法与 diff 检查：通过。
- 历史模型 inventory：四类投影各 42 层，均为 `UnquantizedLinearMethod`；Down/O 为 RowParallel，GateUp 为 MergedColumnParallel，QKV 为 QKVParallel。
- `.81` 隔离容器 `/root/experiments/2026-11-05-w8a16-shape-dispatch-4k-01/` 中，实际导入 FlagGems 路径与上传源码一致；Torch `2.8.0+metax3.7.0.7`、Triton `3.0.0`，manifest 61 个精确映射加载成功。首次四投影最小正确性检查中 Down M64 相对 L2 `1.231406e-4` 通过，随后 GateUp 因候选池键 `gateup_dense_mid` 与 manifest `gate_up_dense_mid` 不一致而中止。修正键名并重新上传后，四组 M64 全部通过：Down `7.9775e-5`、GateUp `1.2809e-5`、QKV `6.3170e-5`、O `4.0594e-5`，均低于预设 `0.005`。这些是随机张量 FP32 参考的算子正确性，不是模型精度。
- 新分发算子正确性、真实路径命中、模型精度、4K 诊断性能、16K 正式性能与 TTFT：**待验证**。本报告初稿不声称收益。
- 旧修复后 micro 数据：Split-K 60/60、Dense 16/16、入口对照 60/60 完成；不同实例的绝对微秒不用于直接固化三算法赢家。

## 5. 与已有改动的关系

沿用 W8A16 数值契约、原生 BF16 MM 提供者、修复后的显式启动方式和已归档候选。新开关隔离旧 Down-only 路径；不合并原生 NT tile 分支。GateUp/QKV/O 是本次新增量化覆盖，必须另做模型精度验收。

## 6. 合规性与风险

本次有实质性量化与 GEMM 分发代码改动，不修改官方 serve 参数、Benchmark 或数据集。主要风险是更多层量化可能使 math_500 精度低于 0.950；区间内未测 M 采用已测候选池但不能称已证最优；Dense/Separate 中间 BF16 舍入差异需分别验证；动态分发应在 CUDA Graph 捕获/重放下核对实际命中。只跑 4K 属诊断，不能宣称双场景正式验收。异常 GPU 编译/资源失败时不可误报为性能收益。

## 7. 复现方式

使用本分支插件和 FlagGems 源码优先导入，在已有 Conda/MACA 环境中设置 `VLLM_FL_W8A16_SHAPE_DISPATCH=1`，保持沐曦官方 `vllm serve` 参数不变。先保存服务启动、量化覆盖和源码指纹，再执行用户指定的 4K 诊断负载；命令、日志、结果及退出状态在实际运行后补充。正式双场景仍须按工作区 AGENTS.md 的官方原命令执行并归档。
