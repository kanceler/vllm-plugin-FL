# #013完整天数底座与静态KV兼容接入

## 1. 基本信息与摘要

2026-11-16，目标Iluvatar BI-V150。用户明确批准完整天数#013后叠加静态KV，并按CPU→GPU算子/图→完整micro→组合官方精度→组合官方四轮性能的顺序推进；不重跑#013底座，W8A16全部关闭。

新工作树分支feat/biv150-static-kv-compat，保留原KV最终分支a28788f。天数新增代码取自210c775（#011/#012/#013及默认启用修正），不纳入之后C500代码。当前实现和CPU结果如下，后续GPU/模型结果追加，不把待验证写成完成。

## 2. 问题与原因分析

两支共同修改tile_policy.py安装入口；#011包装unified_attention后，其globals不再含Kernel，旧INT8 repair直接取globals会KeyError。反向安装又导致#013追踪__wrapped__失败。只靠调换顺序不能保证两条数值路径互不干扰。

静态KV覆写forward，原#011包裹基类forward不会自动让量化路径分相；其通用分相还主动排除量化KV。#013又限定BF16/page16/未量化，不能直接放宽成uint8/page32而不适配。直接合并后“无异常”不代表两个优化实际生效。

## 3. 具体实现

### 原BF16与修复launcher分离

tile_policy保存原始调优launcher，在#013包装BF16 head之前生成独立INT8修复launcher；之后安装原#011/#013，最后发布量化路由。v_only从明确的修复句柄构建静态数值Kernel，不再依赖包装函数globals。KV关闭时不构建repair，保留完整原BF16分发。

### 静态KV的Mixed分相

static_kv_compat.py在MetadataBuilder使用已有CPU query前缀分类，拒绝内部洞，仅支持合法密集Decode前缀/Prefill后缀及尾部padding。复用#011已有GPU前缀或每步生成一次，无逐层GPU→CPU读取。forward先对全部增量token执行一次融合Q变换和真实KV写入，再拆Attention，不重复writer。只切Q/out、seq_lens、block_table及前缀；物理缓存、保护元数据和scale不随请求行切片。Decode前缀按实际行数选现有静态配置、完整归约，Prefill后缀保持对应静态格式。

### 静态KV的直接索引与紧凑网格

对已经完成静态K/V处理的最终JIT源码，精确替换序列二分定位；不从原BF16源码重建而丢失INT8恢复。仅对元数据证明的单Query行使用(num_seqs,2,segments)紧凑grid，保护page32、INT8 cache/BF16 Q/out、D128及8/16/32分段；特殊功能、越界或未验证配置保留原静态head。数值、softmax和Reduce不改。Python命中日志仅为eager/capture证据，不冒充图replay次数。

### 接入、模式及版本

#012原Sampling模块保留。原42层校准文件及数值策略不变；新增k_static_fold_biv150_compat_v1.json绑定22个源码与原provider，新ABI另外绑定Mixed/Direct/Sampling开关，避免误用旧图。旧manifest保留。VLLM_FL_K_STATIC_FOLD=1、INT8_KV_REPAIR=1；三个天数优化开关均1；旧动态KV/V-only及四个天数W8A16开关均0。服务启动参数与Benchmark不改。

## 4. 实验与验证

本地及远端14项CPU unittest均通过：前缀/尾padding/内部洞、分相缓存与scale身份、沿用#011前缀、真实forward写一次再分相、raw launcher隔离、包装后路由及紧凑grid功能边界。源码绑定匹配，AST及diff检查通过。CPU mock不代表GPU数值或性能。

远端使用新目录/workspace/experiments/tianshu-static-kv-013-compat-20261116-01，复用原静态精度04隔离source并应用必要文件，不覆盖/workspace/vllm-plugin-FL。环境核对：原服务manifest 517e652e…，GPU空闲70MiB/0%、9031无监听；远端主仓库d799978有既有改动，FlagGems4df52d91干净，状态及差异已归档，不作清理或覆盖。

GPU原底座回归、静态算子/字节/图、micro、模型精度和官方性能待完成；没有新增模型成绩。用户授权交互SSH连接；初次BatchMode认证失败，交互连接成功，不保存凭据。

## 5. 与已有优化点的关系

#011的分相思想和#013直接索引扩展到静态缓存数值路径，而不是让它们被覆写或仅安全回退。#012位于Sampling，保留实现并回归。KV算法、42层参数和276B/page32布局不变。W8A16及Down负结果研究不纳入本组合，原分支不改。

## 6. 合规性与风险

官方serve、精度/性能命令和负载不变。已知图资格依赖runner密集前缀/尾padding不变量，新增调度/布局须重新验证。Mixed可能改变浮点归约顺序，必须按预定算子门槛及模型精度检验，不能从CPU输出推出逐位等价。提供者/模型/设备指纹不匹配拒绝启动。

完整micro包括Q处理、KV实际写入、Attention和Reduce；正负都保留。不重跑#013底座意味着后续模型百分比仅为跨时段/不同轮次的历史对照，不能严格归因KV收益。任何关卡失败不进入后一关。

## 7. 复现方式

CPU命令（本地不需Torch）：

```bash
python3 -m unittest discover -s tests/compat_cpu -v
```

隔离容器GPU关卡控制器（新实验先完成CPU_COMPLETE、source/环境归档及资源检查）：

```bash
/workspace/tianshu-mixed-env-20261109/bin/python \
 /workspace/experiments/tianshu-static-kv-013-compat-20261116-01/source/benchmarks/diagnostics/run_static_kv_013_gates.py \
 /workspace/experiments/tianshu-static-kv-013-compat-20261116-01
```

它先执行原#011/#012/#013回归，再执行新静态算子、缓存字节及图重放检查；成功生成GPU_COMPLETE后才允许同控制器追加--micro。控制器为每项任务归档命令、开关、独立DB、进程及GPU状态，拒绝占用冲突；日志和失败输出不覆盖。


### CPU与GPU算子关卡完成

原#011/#012/#013回归83/83通过；静态KV新验证49例全部通过，覆盖35个标准Decode batch尺寸（含范围外合法回退）、6个4K/16K Decode点、3个Mixed点、小前缀回退、尾部padding及3类Prefill。新分相3例、紧凑grid23例确认命中；最大输出max abs=3.0517578125e-5、NRMSE=0.00208302，低于预设0.08/0.02；K/V缓存字节和Q变换一致，更新Q/K/V及序列长度后的图重放通过。fixture使用合成scale，不冒充42层校准或模型精度。

首轮验证脚本以位置参数读取lru_cache，而实际调用使用关键字，误拿到另一个未调用的launcher对象；首例数值通过且真实compact日志已出现，仍被脚本断言拦截。只修正检查方式，生产实现/阈值不变，保留validation-initial及operators-01失败；operators-02完成49例。

### 完整micro首轮出现退化，先做有限复核

两组9轮配对连续图重放：B64/KV4096旧静态0.839687ms、新0.858440ms（约慢2.23%）；B32/KV16384为1.609624→1.645763ms（约慢2.25%）；Mixed16 Decode＋17/5-token Prefill为0.496375→0.683267ms（约慢37.65%）；纯Prefill对照基本不变。数值均过门槛，不因micro完成就宣称性能通过。

暂未进入模型服务，追加仅6个Mixed完整调用点（128/512-token后缀、Decode16/32/64）核对分相开销，不扩大搜索、不改数值配置或按单点择优拼策略。全部负结果保留。后续精度和正式性能仍待验证。


### 有限Mixed复核与模型阶段启动

6个追加Mixed点均正确、图及缓存字节通过，完整调用耗时分别降低40.49%、52.52%、20.73%、33.65%、52.71%、15.85%（Decode16/32/64，Prefill128/512或17）。不改变原门槛/配置，不把这些点拼为新策略；纯Decode约2.2%退化及小Prefill Mixed约37.65%退化均保留。此审查支持继续用户要求的完整组合模型验证，不预判模型收益。

原始micro完整轮次与汇总已收录docs/optimization/evidence/2026-11-16-static-kv-013-compat/。阶段包SHA256 01620c4dd3c80099b2a85f107dec7a4dccbb6dc9a9617180ead29f7c1dfbba49。

模型控制器已按顺序启动新服务smoke→新服务官方精度→精度过线后新服务官方四轮性能。启动时状态starting，不把它写成完成；#013历史底座未重跑。

模型启动前已对照远端source-before-model.json，所有本地vllm_fl/*.py与实际隔离运行副本SHA256一致（无差异），不仅核对局部manifest。运行仍使用原镜像site-packages FlagGems4df52d91及绑定的vLLM提供者；本地内附FlagGems4ae85e2不是本次新环境替换安装，不以它冒充已测整个镜像。


### 用户取消精度，改为仅跑正式性能

模型smoke新服务完成42层、35图、page32及compact路径检查，chat并发1/2/7共10请求非空返回，均达到smoke设置的128token长度上限；不是官方精度，无能力通过结论。随后精度新服务已启动并开始客户端后，用户明确取消精度。只停止本任务控制器、精度客户端和服务，GPU恢复70MiB/0%/无进程，原日志和accuracy-user-cancelled.json保留；没有有效组合精度成绩。

按最新指令，不再以精度完成为性能前置条件。复用同源码已完成smoke，另启新服务执行官方性能完整命令，RUNS4/SKIP_FIRST1不变，不重跑#013底座。控制器--performance-only使用独立状态文件，明确记录accuracy=cancelled_incomplete_by_user，不伪造ACCURACY_COMPLETE。

性能新服务已观察KV容量969856tokens、Qcapacity4096、page32/threshold80/segments32；这是实际启动容量，不从历史971168差值归因单项优化。16K×64=1048576输入token已超过该容量，量化后仍可能吃满KV，不以理论字节压缩承诺64条长请求同时常驻。两轮诊断已完成，结果见下文；未完成官方四轮验收。


### 用户改为两轮诊断

15:57用户要求每场景两轮。原完整四轮命令已在4K首轮运行，保留当前服务与原始轮次，不改官方脚本RUNS4/SKIP_FIRST1。新的外部限轮控制器仅暂停等待中的本任务外层管理进程，待4K两轮成功写入raw后停止其四轮CLI，再沿用同一服务调用原run_once完成16K两轮；结束后由原管理进程清理自己的服务。原四轮状态及不完整输出保留，若第三轮CLI已开始则单列中断情况，不算有效轮次。

最终仅形成two-round-diagnostic-raw.csv，四条原始结果逐轮展示，不生成或冒充官方四轮summary，不选择较快一轮代替完整结果。精度仍按用户要求取消，#013底座不重跑。运行时间：15:50开始Benchmark，截至15:57约7分钟，仍在4K第一轮。


### 4K两轮已完成（诊断，不是四轮summary）

| 轮次 | 请求 | tok/s | Mean TTFT ms | 对历史#011～#013吞吐变化 | TTFT增加 |
|---|---|---:|---:|---:|---:|
| 1 | 256/256 | 3334.04 | 6960.32 | -3.54% | +55.31% |
| 2 | 256/256 | 3575.78 | 4657.29 | +3.45% | +3.92% |

历史#011～#013为3456.39tok/s、TTFT4481.68ms，来自原简化预热一轮/正式一轮结果，不是同场重测，不能把百分比当作严格KV单因素收益。两轮均保留，不择优。4K第二轮后四轮CLI已按用户要求中止（已输出第三轮banner，没有第三轮有效结果）；同服务继续16K两轮，原不完整四轮CSV另存，官方源码RUNS4/SKIP_FIRST1未改。

实际模型中42层Mixed分相均有一次启用日志，compact有capture/eager命中；这些不是replay计数。16K首轮运行时KV使用率观察到99.8%，并存在56运行/8等待的容量压力，未见服务异常堆栈；不通过清缓存或改参数绕过。


### 两轮诊断最终结果（16:30完成）

四轮请求全部成功：4K各256/256，16K各128/128。原始结果为实验目录内`two-round-diagnostic-raw.csv`，控制器`two-round-state.json`为complete。未生成官方四轮summary，精度按用户要求取消。

| 场景/轮次 | total tok/s | Mean TTFT ms | 对历史单独静态KV吞吐变化 | TTFT降低率 |
|---|---:|---:|---:|---:|
| 4K/1 | 3334.04 | 6960.32 | +11.63% | -29.19% |
| 4K/2 | 3575.78 | 4657.29 | +10.26% | +8.77% |
| 16K/1 | 2816.32 | 77364.31 | +8.42% | +12.19% |
| 16K/2 | 3062.64 | 74217.25 | +7.74% | +8.44% |

历史静态KV对应轮次来自report.md第4.3节：4K吞吐2986.77/3243.05、TTFT5387.51/5105.03ms；16K吞吐2597.72/2842.64、TTFT88101.33/81062.23ms。跨时间、非配对诊断，仅展示差值，不归因严格单因素收益。四轮不择优，4K首轮TTFT负结果保留。

历史#011～#013的16K为2865.16tok/s、TTFT172305.75ms，对应本次两轮吞吐变化-1.70%/+6.89%，TTFT降低55.10%/56.93%。同样不是本次同场基线；不据此宣称正式性能验收。

16:33确认服务已退出，GPU无运行进程、70MiB驱动保留、利用率0%。关服日志存在ZMQ context及semaphore清理警告，不影响四轮已成功完成的请求统计。当前实现通过算子与模型运行兼容检查，但本组合没有精度验收，也没有完整官方四轮性能summary。
