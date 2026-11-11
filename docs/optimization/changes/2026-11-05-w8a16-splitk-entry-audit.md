> 2026-11-05 完成后审计：发现显式候选的LibEntry启动参数缓存别名。原始计时保留，名义warps/stages与完整联合搜索结论不能直接认证；完整归档、入口实测及修正见[完成后审计](2026-11-05-w8a16-completed-sweeps-analysis.md)。

# Split-K 历史配置与当前入口差异诊断

## 1. 基本信息与摘要

2026-11-05，MetaX C500。用户授权在不干扰现有测试的前提下检查Down小M历史收益未复现的原因。新分支diag/gemm-splitk-entry-audit，插件基点f93489fa1afb74969b76701c8b551552a55ae3bf。新增benchmarks/diagnostics/benchmark_w8a16_entry_audit.py；本地证据与排队控制器在experiment-artifacts/2026-11-05-w8a16-entry-audit-prep/。未修改正在测试的Split-K/Dense脚本、Kernel或BF16提供者，未提交或推送。

## 2. 问题与原因分析

历史Down M1生产入口40.704us，对照BF16 41.216us，单次任务耗时降低约1.24%。本轮相同历史配置(16,64,128,4,2,8)终测55.296/57.344us，当前赢家50.176/51.456us。跨实例绝对值不足以归因，因此先核对静态计算与入口，再准备同机对照。

参考FlagGems已提交生产版本f6a4257e3d042acd26a7d3c496848025a4b32c07，对比当前工作树：weight_int8pack_mm_splitk_partial、weight_int8pack_mm_splitk_reduce_scale两函数AST一致；旧_small_m_splitk直接分配并启动两段，当前转入_configured_splitk并新增能力检查。只能确认该提交参考与当前的计算函数一致，不等价于证明10月28日所有实际编译二进制相同。

现有Down M1编译快照包含与历史整除诊断相符的资源特征：partial96寄存器/0spill/24576B shared/warps4/stages2/basic，reduce22寄存器/0spill/0shared/warps4/stages3/basic。但快照未完整绑定配置及cache键，不能将这些条目唯一归属历史配置，亦不能宣称IR相同。Python入口增加提交间隙是待验证假设，不将全部差异归因于检查逻辑。

## 3. 具体实现

从上述参考提交提取单个旧_small_m_splitk函数，在当前后端globals的独立副本中加载。复用同一对实际Kernel对象，比较旧式完整入口、当前_configured_splitk与公共run_w8a16_algorithm，再加同场原始BF16。

所有W8A16路径均按相同历史配置BM16/BN64/BK128/warps4/stages2/S8，包含输出和FP32 workspace分配、partial、reduce/scale。M1/16，两套seed20261028/20261104，数值门槛relative L2<0.005、rtol0.02、atol0.125；五轮交错终测warmup100ms/rep300ms，保留每个原始样本。INT8权重、FP32 per-output-channel scale、BF16激活、权重转BF16后dot，FP32累加与归约后scale，输出BF16。无模型接入或层覆盖统计。

只诊断入口结构，不假装复现历史硬件。脚本记录源码/环境指纹、helper SHA256、布局、误差、原始与汇总计时。旧函数不会安装进生产模块，不改变已加载对象。

排队控制器只用标准库，无GPU上下文。等待当前Split-K控制进程PID1118退出，并要求原任务exit-code=0、status.complete=true后才运行。否则记queue-error并停止，不擅自重跑。诊断自身先检查GPU空闲及端口/竞争任务，运行后退出并归档资源状态。

## 4. 实验与验证

已完成CPU AST比较、历史helper加载、namespace隔离及语法检查；git diff --check通过。当前静态比较JSON与提取源码已保存。

GPU入口对照尚未执行，排队等待当前60点Split-K任务正常结束；不写成已确认CPU开销或已确认Kernel退化。历史与本轮数据来自此前原始报告/计时，本次没有新GPU性能结果。

## 5. 与已有改动的关系

诊断依赖现有Split-K工具的日志、fingerprint与preflight辅助函数，不修改这些文件。旧生产参考为f6a4257，保留当前更宽候选及所有历史实验。Dense实例独立运行，不使用跨实例时间拼接分界。

## 6. 合规性与风险

不并发GPU测试，不改生产分派、参数、数据、依赖或持久缓存。GPU事件完整调用包含可能的CPU提交间隙，不是单kernel纯执行时间，也不是精确CPU壁钟或图重放性能。同机入口实验可检测入口差异，不能单独排除驱动、时钟、编译器等历史环境因素。

## 7. 复现方式

远端新目录/root/experiments/2026-11-05-w8a16-entry-audit-01/，排队日志queue.log，完成后exit-code.txt、gpu-after.txt及results/{metadata,raw,summary}保存证据。上传包两端SHA256应核对为a98c60add248aea04d185c2bd2d0722cb443eb456bd7b53b4746ccd4a0ad8612。

实际排队入口：

```bash
nohup /opt/conda/bin/python /root/experiments/2026-11-05-w8a16-entry-audit-01/queue_entry_audit.py > /root/experiments/2026-11-05-w8a16-entry-audit-01/queue.log 2>&1 &
```

控制器等待原实验正常完成并通过资源核查后，设置隔离源码PYTHONPATH，再调用benchmark_w8a16_entry_audit.py；完整命令会记录command.json。GPU结果待排队执行完成后归档。
