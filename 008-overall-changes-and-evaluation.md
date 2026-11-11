# W8A16整体改动与最终测评

## 1. 基本信息与摘要

2026-11-11，MetaX C500、MACA3.7.0.38、驱动3.8.30，Torch2.8.0+metax3.7.0.7、Triton3.0.0。量化分支muxi-quantization，FlagGems组合提交13d3db1。当前量化Kernel SHA256为826ee3b73b01f4cc329286f9e268c87a87f7b069464afe7b121e682f1a9abf03。只量化42层Down；LM Head与其余投影保持BF16。性能和精度任务已完成，数据与原始命令随文归档。

## 2. 问题与原因分析

权重INT8节省容量及读取量，但额外转换、partial/workspace/reduce及scale可能抵消收益。小M与prefill不能共用一种固定算法；原BF16 MM的独立较优配置也不等于转换和scale融合后的完整路径较优配置。历史未完整组合及中断任务不能代替最终结果。

## 3. 整体实现

1. 静态对称per-output-channel INT8权重，FP32 scale；激活BF16，INT8转换BF16后dot、FP32累加、BF16输出。不是原生INT8×INT8。
2. 保留三算法dispatch能力与合法回退，精确Shape与区间候选池优先级及启动缓存修复见002/004。当前Down热点小M使用Split-K，M56/M64配置64/64/128/4/1/Split8；一般M1～112采用Split-K区间，M>=113进入Separate。保留通用能力逻辑。
3. Separate的转换BLOCK4096/warps4，转换在完整调用计时内，不保留持久BF16权重副本。scale融合到MM的FP32 epilogue，减少中间张量及一次舍入，融合路径单独验证。
4. Down新路径配置M193～384为64/128/128/4/4/cpasync，M385～16384为128/128/128/4/4/cpasync。M113～192及范围外复用原策略；版本/设备/提供者指纹不符时回退。193～255已逐整数补测；更大区间为代表点与留出验证，不能宣称全覆盖最优。
5. Gate/Up、QKV、O的M2048原生BF16专用配置均128/128/64/4/4/cpasync，同时开启。Down由量化内部新配置管理；这些其他投影的MM配置收益不归为量化自身收益。

## 4. 正式实验与验证

原样官方命令，4K[4096,1024,64,256]、16K[16384,1024,64,128]；各4轮，跳过首轮取后三轮平均。最终8轮均成功，无失败请求。新服务隔离状态，确认Down42层、另外三个MM专用配置实际命中以及Down M2048新配置命中；完成后资源清理正常。

|场景|指标|默认BF16|专用配置纯BF16|最终量化组合|
|---|---|---:|---:|---:|
|4K|total tokens/s|4729.59|5127.00|5357.92|
|4K|Mean TTFT ms|3250.42|2639.22|2710.88|
|16K|total tokens/s|7022.20|7651.48|7540.58|
|16K|Mean TTFT ms|27133.58|23658.99|24205.68|

|对照|4K吞吐|4K TTFT|16K吞吐|16K TTFT|
|---|---:|---:|---:|---:|
|默认BF16|+13.28%|降低16.60%|+7.38%|降低10.79%|
|专用配置纯BF16|+4.50%|增加2.72%|-1.45%|增加2.31%|

均为同一C500不同时间运行，非同期交错对照。相对专用配置纯BF16，仅4K吞吐有效提升；16K吞吐和两场景TTFT有效退化（严格超过1%），不能以默认基线收益掩盖此结果，不自行推导赛事评分。

官方math_500 Level3：报告accuracy=0.9714，105题请求/成功、0错误、incomplete=false，等价102/105正确；高于0.950门槛，报告分数比历史基线0.962高0.0094，但单次随机生成不能证明精度提升。使用temperature1.0/top_p0.95/max_tokens32768、batch8，未修改官方参数。控制器及evalscope退出码0。逐题输出保留在完整本机归档供复查；当前仅确认准确率及执行完整性，未对所有输出行为作全面人工审计。

## 5. 与已有改动的关系

001～007保留热点、配置、候选池、Down选择、dispatch、partial负结果、转换/融合及联合配置依据。BM256和新partial/shared候选未稳定胜出，不启用。未对默认分支#002/#003等框架/Attention新优化作统一合入验证；#002接入文件在本次overlay缺失，#003具体状态未逐项确认。本报告只代表实际记录的overlay。用户要求中断的down-fused-policy-official-20261111-02不计入本结果，日志保留。

## 6. 合规性与限制

没有更改官方启动/Benchmark、评测数据或随机生成设置。无投机、答案硬编码或原生MM底层重写。本组合达到精度数值门槛，但对专用配置纯BF16仍有16K/TTFT退化，不能宣称全面性能验收通过。配置指纹绑定来源版本，跨设备或提供者更新后须重新验证；更大M区间有未知点退化风险。

## 7. 复现与归档

小型可提交证据位于docs/optimization/evidence/2026-11-11-final/：三个版本的summary/raw、可取得的命令及指纹、精度JSON、官方精度命令和checksums。详细实现见001～007。

完整本机归档：artifacts/down-fused-alltiles-official-20261111/down-fused-alltiles-official-20261111-results.tgz；artifacts/down-full-accuracy-20261111/down-full-accuracy-20261111-results.tgz。容器最终性能目录/data/down-fused-alltiles-official-20261111-01/，精度目录/data/down-full-accuracy-20261111-01/，evalscope输出/workspace/evalscope-datasets/level3/20261111_223157/。完整日志/逐题输出不纳入Git；不包含密码或私钥。
