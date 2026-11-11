# 天数 GEMM 提供者与配置缓存追踪

## 1. 基本信息与摘要

2026-11-04，只读追踪天数已运行的MM路径，未启动新GPU测试或修改源码。结合上一轮GPU Trace、上层注册源码、Triton源码位置及Shape配置缓存，证据指向FlagGems Iluvatar MM实现；撤回此前仅凭enabled-ops和extern调用将其直接归类vendor BLAS的判断。

## 2. 问题与原因分析

`extern_kernels.mm`在Inductor中注册自 `ExternKernelChoice(torch.mm, "at::mm_out", op_overload=aten.mm.out)`，因此“extern”仅表示编译器外部调用，并不排除ATen dispatch被FlagGems接管。FlagGems注册了mm和mm.out，worker根据白/黑名单启用对应实现。此前enabled-ops缺少名称不足以排除接管。

## 3. 具体只读发现

### 提供者证据

实际Prefill Trace中168个MM事件均为mm_kernel。与实测8-warps/49152 shared bytes、16-warps/65536 shared bytes相容的Triton缓存包含源码位置：

`/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py:78`。

同名缓存有历史 `/tmp/t5.py` 等版本，故单凭缓存名仍不足以唯一绑定本次二进制；本报告采用实现路径、Shape选择缓存与实际launch相互吻合的综合判断，不声称已做运行二进制哈希绑定。

实际源码SHA256：mm.py为 `4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090`；Iluvatar tune_configs.yaml为 `dd1c7432a84b72dde9fe7d48527f4b1744383fa188a6feb69adb1d8fe5748bc0`。

### Prefill配置交叉核验

只读查询 `/root/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db` 的选择表 `mm_kernel_7ea0a6df3012d3bc5475d21d5eb156a7-45eed7266d4cc0f7be44aaaa091bc3b1`，dtype均BF16：

| 投影 | M,N,K | BM,BN,BK | Split-K | warps | stages | 推算grid / 实测grid |
|---|---|---|---:|---:|---:|---|
| QKV | 2048,2560,2048 | 256,128,32 | 1 | 8 | 2 | 160 / 160 |
| O | 2048,2048,2048 | 256,128,32 | 1 | 8 | 2 | 128 / 128 |
| Gate/Up | 2048,12288,2048 | 256,256,32 | 1 | 16 | 2 | 384 / 384 |
| Down | 2048,2048,6144 | 256,128,32 | 1 | 8 | 2 | 128 / 128 |

推算grid为ceil(M/BM)×ceil(N/BN)，实测block线程数分别512和1024，与warp64一致。共享内存也与两阶段BF16 A/B tile容量相符。这些一致性增强实现与配置归属证据，不代表配置全局最优。

### Decode缓存：仅缓存证据

同表M=8：QKV=(8,64,128,warps8,stages1)，O=(8,64,128,8,2)，Gate/Up=(8,128,128,8,1)，Down=(8,32,128,8,2)，Split-K均1。因Decode图内事件未展开，这些只标为缓存选择，不当作本次图重放配置实测。

### 候选与剪枝

Iluvatar YAML的MM候选：BM/BN∈{32,64,128,256}，BK∈{32,64,128}，Split-K={1}，warps∈{8,16}，stages∈{1,2}。原始笛卡尔积192组，生成/剪枝后实际数量待审计；M=8缓存中的BM=8不直接出现在该YAML，需继续核查early_config_prune对小M候选的变换。

MM装饰器调用 `triton.ops.matmul_perf_model.early_config_prune` 和 `estimate_matmul_time`，top_k=15，再warmup5/rep10计时。该模型存在NVIDIA风格假设：每SM四subcore、min(num_warps,4)、mma估算；时钟查询已有Iluvatar缺省1.5e6 kHz补丁。不是未适配的Intel固定tile，但仍可能存在跨架构性能排序偏差。

候选已按天数提供，性能模型却可能在实测前排掉好候选，因此“已autotune”不等于充分搜索。性能影响尚未验证，不能把它直接称为低级错误或收益来源。

## 4. 实验与验证

本轮仅读取SQLite mode=ro、源码及编译缓存，没有新计时。四个Prefill Shape的缓存配置与现有Trace grid/block吻合。原始选择记录已导出 `mm-selected-cache.json`；不把缓存p50当作新测，也不宣称本次出现cache miss。

Decode仍未获得图内Kernel逐节点耗时。现有FULL marker证明执行上下文，不能证明每个MM配置。没有为观察而关闭图或更改官方参数。

## 5. 与已有改动的关系

补充[短时采样报告](2026-11-04-tianshu-gemm-bounded-profile.md)，进一步收窄vendor归属争议。既有“FlagGems MM配置不一定影响外部MM”的提醒保留为一般规则，但当前证据已支持本平台该路径由FlagGems接管，应继续审计其实际配置。

沐曦的配置与结果不合并；共享仅限两端验证后的实现。W8A16与BF16融合候选暂不实施。

## 6. 风险与下一步计划

建议下一项为**天数MM候选剪枝小范围审计**：优先Gate/Up和Down的真实M=2048，保留当前配置对照，比较当前top15策略与更宽合法候选集合；另取一个小M以检查变换。先统计候选来源、能力剪枝与性能剪枝，后决定少量计时，不能直接穷举全模型。

该实验涉及独立诊断工具及GPU计时，需先明确脚本、候选预算、正确性门槛与计时协议并按工作区要求讨论同意；本报告不是已实施实验。原生BF16底层由队友负责，本任务先给出配置证据和遗漏审计，避免重复实现通用GEMM。

Decode可见性需进一步查询CoreX profiler对graph节点追踪支持，无法获得时保留unknown，不用eager代替正式路径。当前没有必须立即修改的已确认错误。

## 7. 复现与证据

本地目录 `experiment-artifacts/2026-11-04-tianshu-gemm-path-02/`：

- `prefill-mm-by-shape.csv`：实际GPU事件的Shape/launch聚合。
- `remote/mm-selected-cache.json`：只读选择缓存导出。
- `remote/observer/trace-*.json`：原始运行Trace。
- `remote/matmul_perf_model.py`：本次读取的性能剪枝模型。

源码配置基线快照另见 `experiment-artifacts/2026-11-04-tianshu-final-01/flag_gems/runtime/backend/_iluvatar/`，具体结论以上述当前哈希为准。未修改源码、未提交或推送Git。
