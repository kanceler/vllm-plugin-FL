# M56 Dense流水线实现实验（进行中）

## 1. 基本信息与摘要
2026-11-10，MetaX C500。用户授权优化M56量化实现，目标GateUp/QKV/O完整调用优于BF16，不重搜Down。涉及FlagGems weight_int8pack_mm.py及诊断benchmark_metax_m56_pipelines.py。

## 2. 问题与原因分析
固定basic广搜索耗时长，已测GateUp Dense约90.64µs仍慢于BF16 65.71µs。BF16扩展搜索中cpasync有收益，提示值得验证INT8转BF16融合路径的流水线，而非继续扩大相同basic候选。该提示是待验证假设，不能把BF16收益转为量化结论。

## 3. 具体实现
_configured_dense新增显式pipeline参数，仅允许basic/cpasync，默认basic不变，不改变生产分发。通过原Triton JIT调用完整配置缓存，不改全局LibEntry。流水线改变编译调度而非数学契约，INT8转BF16 dot、FP32累加、输出scale后BF16保持。固定12组tile/启动配置，两pipeline，同场BF16、当前dispatch、Separate及历史Split-K作对照。

## 4. 实验与验证
本地AST通过。计划三投影、两seed、完整调用初筛、领先候选及对照五轮交错，图重放正确性和耗时单列；relative L2<0.005，BF16对原始权重参考，量化对量化参考。不支持的编译/图捕获逐项记录。GPU结果待验证。广搜索原数据保留并标为中断，不宣称完成。

## 5. 与已有改动的关系
复用历史M56控制项、修正后专用后端和统一BF16提供者。M2048已验证纯BF16分支不受本修改影响。量化代码留在原量化工作树，不混入新建纯BF16分支。

## 6. 合规性与风险
独立GPU诊断与模型任务串行，官方服务和Benchmark未修改。cpasync可能无法支持转换流水线或无收益，失败如实记录；未验证前不固化，不宣称超过BF16。有限12组不称全局最优。

## 7. 复现
远端/data/metax-m56-pipeline-20261110-01/，源码隔离overlay与source，原搜索/data/metax-m56-search-20261110-01/保留。原进程按PID171315停止并记录中断；新任务在资源释放后启动。命令使用对应PYTHONPATH，运行source/vllm-plugin-FL/benchmarks/diagnostics/benchmark_metax_m56_pipelines.py --output新目录。未提交、未推送。

## 实测与后续归约实验
Dense实验168条初筛（3投影×2seed×28路径）执行结束。eager：GateUp basic领先约90.21µs，BF16约65.76µs；QKV basic约45.38µs，cpasync约52.20µs，历史SplitK约40.70µs，BF16约35.97µs；O basic约44.84µs，cpasync约51.78µs，历史SplitK约36.84µs，BF16约33.95µs。仅列seed20261110，另一seed及完整原始数据保留；不声称cpasync有效。
首次图捕获因默认stream被MACA拒绝，属于诊断脚本错误而非Kernel错误，全部graph_failure保留。改用独立侧流后，单独补跑原finalists得到42条记录；不覆盖旧文件，图计时口径为graph.replay完整捕获调用。性能结论待逐seed整理。
新增实验性FP32向量归约：将Split-K平面一起加载，tl.sum沿split轴归约并乘scale；原序列归约保留。_configured_splitk新增默认关闭vector_reduce和默认256的reduce_block选项，允许128/256/512。向量归约改变浮点加法顺序，必须数值验证，不因性能放宽门槛。QKV/O M56两seed、split4/8/16、两种归约、三block的完整调用及图重放对照已启动；生产默认未变。该实验不假定归约是主要瓶颈，实际结果可能无收益。
