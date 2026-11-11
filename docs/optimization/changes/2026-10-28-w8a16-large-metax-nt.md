# W8A16 大 M 复用 MetaX NT GEMM 诊断

> 后续恢复状态：主实验原始 105 条计时已恢复本地归档；NT-only 对照已完整执行，见 [恢复报告](2026-10-28-w8a16-nt-dtype-recovery.md)。下文连接中断描述为当次历史状态。generic BF16 编译断言仍保留为失败，未被 NT-only 成功替代。

## 1. 基本信息与摘要

日期2026-10-28，任务 `flaggems-w8a16-large-nt-20261028-06`。用户明确要求优先执行需要GPU的大M验证，已授权本轮诊断候选。新增 `benchmarks/diagnostics/benchmark_flaggems_w8a16_large_nt.py`，从现有MetaX `mm_kernel_nt` 提取主体，增加FP32每通道scale；不改生产注册、路由、运行开关或官方命令。保持任务分支 `perf/gemm-w8a16-down-fused`，插件基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`、FlagGems基点 `a7620cc191a0b42e040194622c5758b22a7a25dc`。用户已有工作保留，未提交未推送。

## 2. 问题与原因分析

上一轮M128～2048普通融合和有限Split-K候选未超过独立反量化＋MetaX GEMM。因此不重复旧Split-K；本轮验证将现有MetaX BF16 NT实现组织用于W8A16是否有价值。

原weight_int8pack_mm使用通用strides、64位offset、每轮从索引构建加载地址及K mask；MetaX NT对连续布局使用32位元素offset、递增指针、EVEN_M/N/K编译期特化，整除shape可移除mask/取模。两者都有GROUP_M8、FP32 dot累加、软件流水线，并非原融合缺乏这些基础结构。旧IR已将 `acc += dot` 合并成带acc的dot，不能仅凭源码表达式推断多一次矩阵加法。

MetaX mm主体已有混合dtype分支：INT8 B可转成输出BF16再dot。本轮直接保留该分支，新增scale-after-dot输出，避免重新手写整套GEMM。需验证INT8共享内存加载/转换、布局及pipeline编译差异；源代码上的简化不等于已获得性能收益。

## 3. 具体实现

`nt_scaled`来源为 `FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py::mm_kernel_nt`：修改函数名，增加Scales参数，在FP32累加结果转BF16前乘FP32 scale，其余主体保留。只允许连续A[M,K]、Q[N,K]及scale[N]，与原通用算子能力不同，仅用于诊断，不替代生产接口。

固定BM64/BN64/BK64/warps4/GROUP_M8，比较两种历史MetaX BF16实际选中过的配置：stages4/pipeline basic及stages3/pipeline cpasync。分别运行原generic JIT和NT候选，形成匹配配置对照；另外保留同场MetaX BF16、Triton独立反量化＋同MetaX mm、原autotune融合。每个M最多7条完整路径，M128/256/512/1024/2048。没有新Split-K或预分配快路径。

## 4. 实验与验证

远端容器 `d7a568648975`，MetaX C500 64GiB，bus `0000:58:00.0`，driver3.9.35、MACA3.7.0.38，Python3.12.11、48GiB RAM、无swap。运行前GPU无进程、util0、860MiB，两生产仓库clean；显式加载已有Conda/MACA环境，PYTHONPATH指向FlagGems源码，清除历史W8A16开关，用socket核对9031空闲。容器与旧实验不同，绝对值不跨轮拼接。

候选先验证两个非整除shape `(3,129,257)` 和 `(65,67,33)`，含零权重通道；真实Down shape再次验证。参考FP32 dot后scale转BF16，relative L2<0.005，rtol0.02、atol0.125。每配置保存编译产物与资源信息。完整调用包含输出分配、单kernel与scale输出；反量化对照包含完整BF16权重生成和GEMM。权重量化在计时外。

计时seed20261028，所有路径先编译/调优，再轮换顺序3轮do_bench(warmup100ms,rep300ms)，取3轮中位数。不能解释为CPU精确wall time或模型图捕获性能。

主实验远端退出0，终端已返回五个M全部七条路径的汇总值。完整raw、正确性和编译产物仍在远端，取回前连接被服务器断开，SCP密码认证两次失败，已向用户询问连接状态。本轮尚不能宣称完成本地原始数据审计，也未取得最终postflight；断开前GPU检查无进程、util0、860MiB。以下为终端返回summary的暂存记录，待远端归档逐项核验。

| M | BF16 | 独立反量化 | 原融合 | generic basic4 | generic cpasync3 | NT basic4 | NT cpasync3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 80.384 | 125.696 | 284.672 | 313.856 | 250.112 | 272.384 | 235.264 |
| 256 | 130.304 | 177.152 | 300.032 | 517.376 | 262.912 | 467.200 | 241.152 |
| 512 | 187.904 | 237.056 | 522.240 | 751.104 | 371.712 | 678.656 | 365.312 |
| 1024 | 304.640 | 348.160 | 763.136 | 1204.224 | 1035.776 | 1095.680 | 1018.368 |
| 2048 | 582.656 | 626.944 | 1229.568 | 2335.232 | 1944.320 | 2134.016 | 1728.768 |

单位微秒。NT在同配置下比generic更快，说明现有MetaX主体组织有局部价值；但两个NT配置在全部五个M均慢于独立反量化，M1024/2048还慢于原autotune融合，不接入生产。cpasync3/basic4同时改变pipeline与stage，不能把差异单独归因于pipeline。结果没有支持“照搬BF16配置即可解决W8A16大M问题”。主实验各输出的relative L2约6.7e-5～8.5e-5，未测模型精度。

### dtype归因对照与失败

增加 `benchmark_w8a16_nt_dtype_control.py`：预先把Q的整数值等值转为BF16（不乘scale），对同一generic/NT kernel测INT8和BF16输入，以区分存储dtype/转换编译路径的影响。转换在计时外，BF16权重占用翻倍，仅是归因控制，不作为部署候选或量化收益。

首次运行退出134（SIGABRT）。已生成m256 NT INT8/BF16以及generic INT8资源文件，随后generic BF16 cpasync编译触发MACACommon.cpp:1243 `genSwiSubMask` 的 `validMaskSwizzle` 断言。没有产生完整对照计时，不能归因说INT8转换已被证实为主要瓶颈；该失败不影响此前主实验退出0的事实。远端保留失败脚本副本 `dtype-control-before-abort.py` 和dtype.log。修正诊断脚本增加 `--nt-only` 可避开失败分支，但同步时SSH断开，本次未执行成功，不写作完成。

## 5. 与已有改动的关系

复用baseline的reference、dequant与结果保存helper。沿用scale-after-dot契约，与历史scale-before-dot手写kernel不同，不将二者误差口径混为一谈。小M配置和partial分析保持原状，本轮只研究大M普通融合的MetaX适配。

## 6. 合规性与风险

仅GEMM诊断，不修改模型、官方参数、评测脚本或数据。不是原生INT8×INT8路径：BF16激活、INT8权重转BF16、FP32 dot、FP32 scale、BF16输出。不推导模型精度、吞吐或TTFT。NT候选有连续布局限制，32位offset仅在当前元素offset可表示的测试shape使用，不能直接泛化到所有大矩阵。cpasync参数是否实际生成对应流水线必须看编译产物，不凭字符串宣称已使用。失效候选归档，不新增生产开关。

## 7. 复现方式

远端 `/workspace/benchmark_results/flaggems-w8a16-large-nt-20261028-06/`，新目录保留脚本、helpers、preflight/environment/run.pid/run.log/exit、postflight、原始及汇总JSON、编译IR/资源/二进制。

```bash
export PYTHONPATH=/workspace/FlagGems/src
unset VLLM_FL_W8A16_DOWN VLLM_FL_W8A16_DOWN_FUSED VLLM_FL_W8A16_DOWN_OFFLINE VLLM_FL_W8A16_DOWN_CHECKPOINT
python benchmark_flaggems_w8a16_large_nt.py --helpers-dir . --output-dir results
```

### 2026-10-28 归档恢复

资源恢复后取回 `artifacts/flaggems-w8a16-large-nt-20261028-06/`。本地核验主实验退出0，105条原始计时、5个shape的所有中位数与summary一致，failures为空。此前终端暂存表已得到raw支持。dtype退出134及断言日志也已归档，NT-only对照仍未完成，不因此改写为原因诊断已完成。恢复时GPU空闲证据单独保存在recovery-postflight.txt，不能替代历史结束瞬间的记录。

### NT-only归因已补齐

后续独立实例的NT-only实验已完成，详见 [dtype恢复与编译证据](2026-10-28-w8a16-nt-dtype-recovery.md)。发现同配置下INT8与BF16(Q)实际流水线不同；generic BF16失败未修复。不能继续笼统写作全部dtype对照未完成，亦不能把归因控制当部署方案。
