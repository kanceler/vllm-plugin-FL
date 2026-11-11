# 纯BF16两套M2048 tiling官方对照（进行中）

## 1. 基本信息与摘要
2026-11-10，MetaX C500。用户授权比较d61ee2f tiling分支配置与扩大搜索赢家，不开启量化。远端/data/bf16-tiling-ab-20261110-01/。不修改本地生产代码，本次采用隔离安装包overlay。

## 2. 问题与原因分析
micro新配置较快，但预分配输出与原入口有差异，不能外推官方收益。需要相同完整接入方式、原生JIT主体与软件环境下端到端对照。

## 3. 具体实现
两套使用同一nt_gemm.py接入，worker在编译捕获前安装，沿flag_gems.mm.__module__定位真实后端；每个Shape独立LibEntry与单候选LibTuner。只改变四个M2048 Shape的tile、warps/stages/pipeline，其余路径保留。A：QKV/GateUp 256/256/32，O/Down 128/128/64，均8/2/basic；B：Down64/128/128，其余128/128/64，均4/4/cpasync。按首次实际命中记录Shape。两套复制同一个镜像安装插件，量化相关开关全部显式0。

## 4. 实验与验证
先每套四Shape eager与CUDA Graph replay正确性relative L2<0.005，再启动新模型服务和官方完整4K/16K，RUNS=4、SKIP_FIRST=1不变。当前正在执行，最终吞吐、TTFT、四Shape模型实际命中及模型精度待验证。没有完成前不宣称性能收益。

## 5. 与已有改动的关系
A依据origin/perf/metax-c500-prefill-nt:d61ee2f，B依据扩展搜索provisional赢家。两套均不包含W8A16，不混入此前量化结果。不合并最新上游性能代码。

## 6. 合规性与风险
官方serve参数、Benchmark与负载不改，保留全部八轮明细、异常和summary。串行运行，退出本模式进程组后确认GPU空闲才进入下一模式。任何模式失败中止后续并保存failure.txt。启用日志不代替实际Shape命中证据。精度未测，不代表完整赛事验收。

## 7. 复现
远端/data/run_bf16_tiling_ab.py编排，/data/run_bf16_tiling_ab.log记录进度；各模式commands.json保存实际命令和显式环境，hashes.json保存接入、worker、官方Benchmark指纹，probe.log保存算子与图重放验证，server.log保存真实命中，summary/raw_runs原样归档。当前未提交、未推送。

## 完成结果

16:49:51两套官方双场景完成，均关闭量化。各场景四轮，跳过首轮取后三轮平均。

|场景|A吞吐|B吞吐|吞吐提升|A Mean TTFT ms|B Mean TTFT ms|TTFT降低|
|---|---:|---:|---:|---:|---:|---:|
|4K|5044.20|5277.76|4.6303%|2808.70|2574.93|8.3231%|
|16K|7425.94|7653.26|3.0612%|24781.69|23606.25|4.7432%|

两个场景吞吐及TTFT均超过1%改善，属有效改善。基于官方summary已输出精度计算，远离边界。四个Shape首次实际命中日志均保存；它证明入口命中，不等于全程调用或CUDA Graph replay次数统计。模型精度尚未测，不宣称完整赛事验收。远端归档/data/bf16-tiling-ab-results-20261110.tgz已生成；首次本地下载连接中断，须后续取回，不能误报本地归档完成。未提交、未推送。
