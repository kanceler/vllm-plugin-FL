# 002自适应chunk对GEMM热点M的影响采样

## 1. 基本信息与摘要

2026-11-11，MetaX C500。用户批准采样002对热点M的影响。隔离目录 `/data/adaptive-chunk-m-sampling-20261111-01/`，off→on串行启动新服务。量化和三个原生BF16专用配置复用最终完整组合。本任务是诊断采样，不是正式性能成绩。

## 2. 问题与原因分析

002可能抬高prefill预算，使实际GEMM M从2048转为4096，影响既有配置命中与耗时覆盖。预算上限不等于每步实际M；也不能把请求并发或Python图捕获次数当成GEMM M或replay频次。需区分初始化、capture与实际负载。

## 3. 具体实现

002源码取已归档a09c3c7，adaptive_sched与本地s2-track2-final版本一致；本地origin默认flagos-2026-s2尚无此文件，故不声称测试了刚获取的远端最新默认分支。仅增加adaptive_sched.py、四个对应env映射与平台调用钩子，不合并其他上游优化，不额外修改003。off/on使用同一overlay，仅ENABLED=0/1不同；MAX4096、DECODE_GATE=-1、LONG_PROMPT=0固定。自动长prompt阈值为4×基线预算，预计4K与16K可能不同，实际以采样为准。

## 4. 实验与验证

沿用既有collect_gemm_shapes及rank工具。官方serve参数原样；诊断负载为确定性synthetic token IDs，4K/16K长度、输出1024、并发64、请求256/128，每模式各采prefill16个匹配step与decode128个匹配step。通过torch.profiler与上下文/图replay marker关联M，分别报告unknown、精确关联覆盖、kernel耗时及配置选择；不拼接窗口推断模型全程比例。负载不替代官方Benchmark，不报告计分吞吐。

各任务启动检查GPU空闲、9031可绑定，记录前后进程、源码指纹、命令、量化覆盖；两模式之间退出服务并确认GPU释放。最终模型量化覆盖必须42层Down。两模式共8个窗口已全部完成，各模式4K256/256、16K128/128请求均HTTP200。GPU已释放。

## 5. 与已有改动的关系

不改变最终Down区间策略、转换或scale融合，也不改变三个其他投影的M2048专用配置。新M若出现，优先核对现有区间命中，而非立即重调Kernel。保留先前正式性能及精度结果，不把本任务的数据覆盖到正式summary。

## 6. 合规性与风险

用户明确要求002影响诊断，源码在隔离overlay中接入，官方服务参数不变。Profiler有开销、有限窗口存在采样偏差；未关联项单列，不能宣称采样覆盖未来全部Shape。若002自身与当前vLLM不兼容，保留错误而不静默换参数。未更改共享安装或量化分支的运行代码，未提交/推送新实验。

## 7. 复现与归档

原始runner位于off/run.py、on/run.py；run_all.py负责串行与失败停止。源码补丁来源保留在 `experiment-artifacts/2026-11-11-adaptive-chunk-m-sampling/patch-source.json`。实际env/命令见各模式commands.json，源哈希为source-hashes.json。每模式observer目录保留worker配置/状态和原始trace，ranking.json为派生统计；完成后再归档并生成off/on对照。

### 最终采样结论

|窗口|002关闭|002开启|
|---|---:|---:|
|4K prefill（含mixed）|M2048|M2048|
|16K prefill（含mixed）|M2048|M4096|
|4K decode|M64|M64|
|16K decode|M64|M64|

每个窗口匹配GEMM集合的精确M耗时关联覆盖率均100%，不等于覆盖模型全部算子或未来全部输入。每prefill窗口16个匹配step、decode128个step；两个prefill窗口在不同预算下处理token工作量不同，禁止直接比较窗口GPU总耗时推导加速。decode仍有其他原生BF16投影的mm_kernel_nt，不能误认为Down回退。

开启组日志明确确认Down M4096命中 `(128,128,128,4,4,cpasync)`。当前量化区间策略已覆盖该热点，无需仅因M变化修改Down配置。但Gate/Up、QKV、O的既有专用配置仅针对M2048，切换到M4096后不再命中这些三条特判；这是后续需要队友提供者配置复验的范围，不应把其性能变化全归于量化。

摘要、ranking、命令及状态已取回 `artifacts/adaptive-chunk-m-sampling-20261111/adaptive-chunk-m-sampling-20261111-summary.tgz`；完整8份原始trace和overlay保留容器原目录，摘要包排除重复overlay和大型trace。未开展002开启后的新官方性能/精度验收，未提交或推送本次实验。
