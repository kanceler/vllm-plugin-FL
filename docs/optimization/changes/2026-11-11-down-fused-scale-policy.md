# Down融合路径区间配置接入

## 1. 基本信息与摘要

2026-11-11，MetaX C500。用户批准接入前轮区间策略，要求第一步验证归档后再推进后续评测。修改FlagGems独立工作树 `worktrees/FlagGems-scale-epilogue/src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py`，分支 `perf/metax-w8a16-scale-epilogue`。新增两个诊断脚本 `verify_metax_down_transition.py`、`verify_metax_down_fused_policy.py`。既有未提交修改保留，本次修改前文件备份在 `artifacts/down-fused-policy-20261111/source-before/`。

## 2. 问题与原因分析

普通BF16 MM配置并非转换与scale融合完整路径的最佳配置。前轮已验证不同M偏好BM64或128，但193～255存在采样空档。113是Separate算法起点，不是配置性能分界。接入前补测M192～256全部整数，两种子，并检查相邻边界。

## 3. 具体实现

新增纯策略函数 `_down_fused_scale_tile`：M193～384取64/128/128/4/4/cpasync，M385～16384取128/128/128/4/4/cpasync，其余返回None，复用原配置选择。策略仅作用Down N2048/K6144、现有合法连续布局、scale direct融合路径。BLOCK4096/warps4转换和FP32累加后scale、BF16输出保持不变，未改变小M算法和量化覆盖。

绑定MetaX C500、Torch2.8.0+metax3.7.0.7、Triton3.0.0，以及实际MM提供者SHA256 `3abfeba5c64c159370543602a2afc329efb6c7fc09e46659c11faa6b64485213`。指纹校验按路径缓存，首次warmup读取一次；不匹配时保留原选择。仍保留原M2048配置作为旧路径回退。新增一次性选择日志 `C500_W8A16_SCALE_POLICY`；日志只证明Python执行时配置选择，不当作图replay频次。

## 4. 实验与验证

边界实验 `/data/down-transition-20261111-01/` 已COMPLETE，M192～256逐整数、seed20261202/20261203、各点三轮交错完整调用和三轮图replay，共260条结果。193～255所有点的两种子都改善，含M256时最小完整调用降低19.2938%、图降低19.6334%；最大L2为0.001663114，低于0.005。M192有一组eager异常抖动，但另一seed和图均不支持改用专用BM64，保持原配置，不依据单次异常扩大下界。

实际接入验证目录 `/data/down-fused-policy-20261111-01/`。接入源码SHA256 `6164d1c4844231b97c79ab47ab7e4f4a6a9335903ea038057b542cfb72642b02`。检查实际策略命中、192/193、384/385、16384/16385、113及Split-K边界、非整除M、数值和图replay；含代表点完整调用复测。23个边界/代表M均通过eager和图数值检查，最大L2为0.001664218；实际策略命中符合预期，16385正确回退。


实际接入完整调用复测（μs），控制为同路径原BM64专用配置：

| M | 接入完整 | 原专用完整 | 接入图 | 原专用图 |
|---|---:|---:|---:|---:|
| 193 | 123.51 | 123.41 | 140.91 | 140.82 |
| 384 | 124.46 | 124.41 | 138.19 | 138.16 |
| 385 | 146.62 | 201.33 | 160.29 | 216.04 |
| 2048 | 352.90 | 445.27 | 367.65 | 460.61 |
| 2049 | 356.39 | 525.37 | 371.06 | 540.98 |
| 4096 | 564.01 | 850.65 | 578.98 | 866.42 |

## 5. 与已有改动的关系

沿用转换与scale direct融合，原专用M2048分支是修改前既有实现。本轮不修改MM内核或纯BF16提供者。BM256补测已失败，不采用；区间范围来自有限性能证据，不作为能力限制。已有本机修改和容器隔离overlay的源文件分别归档，不假设两个文件系统自动同步。

## 6. 合规性与风险

官方参数、数据、模型与依赖不变。没有运行时搜索、持久BF16副本或转换移出计时。区间内未逐整数覆盖全部385～16384，仅有代表点和留出验证，存在未知点退化风险；超范围或版本变化沿原合法回退。模型精度及官方4K/16K尚未复测，不能由micro声称TTFT或吞吐改善。未提交或推送。

## 7. 复现方式

边界脚本 `verify_metax_down_transition.py --output <新目录>`；接入脚本 `verify_metax_down_fused_policy.py --output <新目录>`，均在对应overlay的PYTHONPATH下，设置SHAPE_DISPATCH=1、FORCE_SEPARATE=0。外围run.py负责GPU/端口核对、命令和前后状态归档。源文件、哈希及实际import路径以对应实验目录为准。性能测试保留完整原始轮次，图正确性和eager分别记录。

第一步已完成并归档：`artifacts/down-fused-policy-20261111/down-fused-policy-step1-20261111.tgz`，两端SHA256一致 `79b5dbae0a96ad976dbee96dad10c060b1010ceedefc953364d27bfac1cce735`。包含边界原始轮次、接入验证、实际源文件及指纹；容器原overlay保留。第一步结束GPU无进程、9031可绑定。下一步为相同小M及其他MM设置下的官方4K/16K性能验证；math_500仍待验证。

### 下一步启动记录

第一步归档校验完成后启动官方性能任务：`/data/down-fused-policy-official-20261111-01/`，外围控制器 `/data/run_down_fused_policy_official_20261111.py`。使用已验证policy overlay，保留Down-only、小M配置及其他MM默认模式；官方serve与benchmark原命令、RUNS4/SKIP_FIRST1不变。新服务隔离状态，要求42层Down覆盖和M2048策略选择日志；结束后归档原始8轮及summary。端口资源检查采用SO_REUSEADDR并同时核对进程/GPU，避免TIME_WAIT导致清理误报，不改变benchmark实现。任务刚启动，结果待完成；math_500精度仍待验证。

### 启动日志检查与修复

首次官方启动 `down-fused-policy-official-20261111-01` 在健康后检查策略选择日志失败，未启动Benchmark；服务已清理退出。原因是模块INFO日志未被服务日志配置输出，不将该次启动记为正式性能结果。为策略专用logger增加独立StreamHandler和INFO级别（仅一次性选择日志，不改计算和配置），新源码SHA256 `826ee3b73b01f4cc329286f9e268c87a87f7b069464afe7b121e682f1a9abf03`。03目录补做M193/M2048数值、分发、完整调用和图验证；02目录在同步失败后未启动GPU任务，保留失败记录。修复版本需重新归档后再启动新的官方实验目录。

03目录的日志修复复验已COMPLETE：M193/M2048数值及图检查通过，实际策略日志可见。M2048新完整调用352.482μs、原专用445.245μs；图367.188/460.414μs。修复源码及复验已归档 `/data/down-fused-policy-logging-20261111.tgz`，SHA256 `db901585a5d659e83dbdf084d630cf146b42671cc63897472711b8100b985d25`。归档后启动新官方目录 `/data/down-fused-policy-official-20261111-02/`，控制器v2；当前仅启动服务，正式Benchmark状态以该目录status/health/client.pid为准，不混用01失败启动。
