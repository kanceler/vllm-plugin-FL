# Down partial shared布局与同步单因素实验

## 1. 基本信息与摘要

2026-11-11，MetaX C500。用户在编译/计数器审计后批准开展shared布局与同步实验。本轮仅针对Down M56/M64 partial，新增 `benchmarks/diagnostics/benchmark_metax_down_shared.py` 和 `benchmark_metax_down_swizzle.py`；生产实现未改。第一阶段实验目录 `/data/down-shared-layout-20261111-02/`，第二阶段隔离复测目录 `/data/down-shared-swizzle-20261111-02/`。

## 2. 问题与原因分析

上一轮确认当前Kernel无spill、累加已被编译器合并，但存在INT8→FP32→BF16、shared重排及同步链，bank-conflict事件有随启动次数增长的增量。尚不能据此断言总时延主要来自哪一项，因此固定tile和Split数，逐项改变编译布局/调度策略，不同时扩大tile搜索。

## 3. 具体实现

保留实际 `_shape_dispatch` 与相同手工完整调用包装下的原partial两组对照。Down逻辑Shape为M×6144乘6144×2048，INT8权重逻辑[N,K]、FP32 per-output-channel scale，转换为BF16后dot并FP32累加。固定BM/BN/BK=64/64/128、4 warps、1 stage、Split8，原reduce/scale不变。复制实际partial函数体生成独立名称的实验函数，只改变明确的编译选项，避免环境变量未进入JIT内存缓存键而复用错误产物。

候选逐项为：`cvt_shm_no_pad`、`dot_operands_out_loop`、`smemIndexOpt`、`smemOffsetCache`、开启merge_convert_layout、关闭merge_equal_shared_layout、关闭constancy_load_layout优化、在原basic编译参数上开启INT8 SLP向量化，以及仅将INT8权重一次性重排成K×N连续物理布局。选项依据容器安装的 `triton/backends/metax/compiler.py`，不修改安装编译器。

重排仍保留逻辑[N,K]，stride由[K,1]变为[1,N]，不产生BF16权重副本。仅作为一次性加载布局诊断；生产加载、容量和生命周期尚未实现。它同时改变global加载布局，不能将其收益单独归因于shared。若未来采用，须另验证初始化开销及推理调用完整成本，不能把每次推理所需转换移出计时。

## 4. 实验与验证

两随机种子20261116/20261117，M56/M64；相对FP32量化参考L2<0.005。每候选保存TTIR/TTGIR、fatbin、寄存器/spill/shared metadata。partial独立计时与完整Split-K调用分列，完整调用包括workspace和输出分配、partial、原reduce/scale；所有合法候选七轮交错完整调用计时、五轮交错图重放，避免只复测初筛最小值。

第一阶段已完成，40组成功结果，4组INT8向量化编译失败；成功候选及图重放的最大相对L2为0.00166234，低于预设0.005。原始运行脚本SHA256为 `40bda763dab45a7a91909002d28b56a8ae6d5f4726ce1822473c7d2d035c5944`。启动前GPU无进程、826MiB驱动保留，控制器重新核对GPU与9031端口；源码、仓库和命令指纹归档。

下表为完整Split-K/图重放中位数（微秒）；每格分别对应两个随机种子，不能跨种子直接比较。

| 候选 | M56 seed16 | M56 seed17 | M64 seed16 | M64 seed17 |
|---|---:|---:|---:|---:|
| 实际dispatch | 66.60/80.26 | 64.15/77.90 | 71.01/87.09 | 65.83/79.34 |
| 手工原实现 | 66.53/81.03 | 64.13/77.58 | 71.56/89.38 | 65.62/78.71 |
| shared no pad | 66.88/80.73 | 64.77/78.10 | 71.41/86.40 | 65.34/78.86 |
| dot operands out loop | 66.48/80.49 | 64.05/79.36 | 71.78/86.86 | 65.31/78.99 |
| shared index | 66.78/80.59 | 64.31/77.92 | 71.79/86.02 | 65.66/78.97 |
| shared offset cache | 66.66/80.13 | 64.85/78.30 | 71.38/85.68 | 65.16/78.39 |
| merge convert | 66.55/80.34 | 64.13/77.76 | 71.89/85.90 | 65.39/78.99 |
| separate shared layouts | 66.66/83.35 | 64.25/77.77 | 70.65/85.32 | 65.93/79.60 |
| disable constancy layout | 66.75/81.31 | 64.70/78.33 | 72.05/86.55 | 65.33/78.77 |
| packed KN INT8 | 257.57/274.46 | 257.80/272.01 | 269.37/289.20 | 269.19/285.75 |

没有候选在两种子、完整调用和图重放下均相对两组对照稳定改善超过1%。此处1%只作为micro实验的保守筛选参考，不代表通过官方模型性能门槛。

产物检查：多数选项去除符号和调试位置后的TTGIR相同，不代表所有后端机器码都相同。`no_pad`只改变输出accumulator的convert_layout，不改变权重shared布局，寄存器增加4。`packed_kn`达到256寄存器并出现M56/M64分别73/69个编译器报告spill，完整调用显著退化。首轮向量化编译失败与extra_options拼接空格有关，不能据此判定后端不支持向量化；修正后在第二阶段单独验证，原失败记录保留。

第二阶段新增 `benchmark_metax_down_swizzle.py`，直接以原始TTGIR为输入，只改变权重shared属性，保留原IR重编译和实际dispatch对照。原属性为vec8/perPhase2/maxPhase8；单因素尝试phase1、phase4、maxPhase4、vec4、无swizzle，以及修正参数的INT8向量化。此方式仅为离线消融，不注入生产编译流程。使用新种子20261118/20261119，验证标准和计时轮次不变。目录 `/data/down-shared-swizzle-20261111-01`，脚本SHA256 `15dd3c9a45040083b87400d681d98f76506f4b71e2457f0ac5d11dddf4dfd3a0`。初次运行失败及隔离复测结果如下。

第二阶段初次运行 `/data/down-shared-swizzle-20261111-01` 在 `partial_swizzle_vec4` 发生非法访存（日志精确定位），进程退出且GPU恢复空闲。之后同进程其他候选的异常属于污染结果，不计为独立失败。修订脚本为每个进程仅一个候选加dispatch及原IR对照，非法访存立即退出；排除已失败vec4，五个其余候选在新的 `...-02` 目录串行运行，每个进程前后核对GPU与端口。修订脚本SHA256为 `7a2f78f2ca81b3daf5d639115b0ccdc287f79417cdd00aa794fd4aaf68606947`。不会通过删除必要同步来换取不正确的结果。

隔离复测五个候选全部完成，共60组结果（包括每个进程内的两组对照），没有正确性或图重放失败，最大L2为0.001662824。以下是相对同进程原IR对照的耗时变化范围，覆盖M56/M64及两个随机种子；正数表示变慢，不能将范围内最优值当作最终收益。

| 单因素 | 完整Split-K耗时变化 | 图重放耗时变化 | 结论 |
|---|---:|---:|---|
| perPhase=1 | -0.59%～+2.21% | -0.55%～+1.72% | 无稳定改善 |
| perPhase=4 | +1.57%～+3.30% | +1.62%～+2.58% | 退化 |
| maxPhase=4 | +1.60%～+3.69% | +1.43%～+3.01% | 退化 |
| 无swizzle | +29.76%～+34.77% | +24.83%～+28.36% | 明显退化 |
| INT8向量化参数修正 | -0.71%～+1.07% | -1.87%～+1.05% | 无稳定改善 |

这些候选M56/M64均为140/136寄存器、0 spill、16KiB shared，因此无swizzle变慢不能归因于新增spill。实验支持当前swizzle对性能有实际作用；没有重新采集逐候选硬件计数器，不能将全部差异精确归因于bank conflict，也不能声称同步或转换已达最优。

**最终决定：不接入任何候选，保持生产partial与dispatch不变。** 在当前固定tile、已测布局/编译选项内没有发现稳定收益；这不证明partial已经没有优化空间。未开展模型精度及官方4K/16K新测评，不声明端到端收益。隔离实验结束后GPU无进程，9031端口可绑定；本地两个诊断脚本AST与空白检查通过。此次只交付诊断实现和实验记录。

## 5. 与已有改动的关系

复用最新官方融合组合的隔离overlay与当前配置；与11-10预取、11-11有限tile搜索不同，本轮只审计编译控制项及一项INT8物理布局。Gate/Up历史同类选项结果不作为Down结论。既有未提交文件不覆盖、不暂存。

## 6. 合规性与风险

只改诊断工具，不更改官方参数、量化覆盖或BF16 MM。测试为独立Down Shape覆盖，不代表模型层调用覆盖或端到端耗时覆盖；L2相对量化后的FP32参考，只验证Kernel舍入误差，不代表原始模型量化精度。所有失败保留；编译选项名称不等于实际生效，须看IR/产物后判断。只在完整调用及图重放有稳定收益时考虑接入；无模型验证不得宣称吞吐收益。

## 7. 复现方式

```bash
PYTHONPATH=/data/down-shared-layout-20261111-02/overlay \
VLLM_FL_W8A16_SHAPE_DISPATCH=1 VLLM_FL_W8A16_FORCE_SEPARATE=0 \
python /data/down-shared-layout-20261111-02/benchmark_metax_down_shared.py \
  --output <新的不存在的目录>
```

原始命令见commands.json，进度见status.txt/driver.log；数据在results/screen.jsonl、finalists.jsonl及编译产物。`...-01`在文件同步失败后准备中止，未启动GPU测评，保留原controller日志；有效运行使用新目录`...-02`。通过已有SSH会话分块传输并核验哈希，未写入连接凭据。未提交或推送。

第二阶段每个候选使用新进程；下面以phase1为例，其他候选分别为phase4、maxphase4、noswizzle和vector_int8：

```bash
PYTHONPATH=/data/down-shared-swizzle-20261111-02/overlay \
VLLM_FL_W8A16_SHAPE_DISPATCH=1 VLLM_FL_W8A16_FORCE_SEPARATE=0 \
python /data/down-shared-swizzle-20261111-02/benchmark_metax_down_swizzle.py \
  --source-ir /data/down-shared-layout-20261111-02/results \
  --variant phase1 --output <新的不存在的目录>
```

第一阶段原始归档已下载至本机 `artifacts/down-shared-layout-20261111/down-shared-layout-20261111-02.tgz`，两端SHA256一致：`1374aeea2d093a9d6ad10808215836e085bf46c22b7563e2d38693f274db0280`。归档不包含重复overlay，容器保留overlay及源码指纹。归档脚本保留原始向量化参数失败版本；本地诊断脚本修复其空格，后续有效向量化结果以第二阶段为准。

第二阶段完整归档（含首次非法访存失败及五组独立复测）也已下载：`artifacts/down-shared-layout-20261111/down-shared-swizzle-20261111.tgz`，两端SHA256一致为 `8104e397b3e877dd2aba1a5081c3c119958490f40c14b53371040f5e13589bc4`。每组原始数据、七轮完整调用和五轮图重放的明细见 `<候选>/results/finalists.jsonl`，独立partial初筛计时见 `screen.jsonl`，不把阶段计时相加替代完整调用。
