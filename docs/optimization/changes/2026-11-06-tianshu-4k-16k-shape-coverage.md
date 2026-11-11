# 天数4K/16K GEMM Shape分阶段采集

## 1. 基本信息与摘要

2026-11-06，Iluvatar BI-V150 / CoreX 4.5.0。复用沐曦现有collect_gemm_shapes.py观察器，新增实验目录内run_windows.py和analyze.py；未改生产源码或官方参数。服务插件f82fa1670509f081aa13b58464cb6e5627fc10c4，实际PYTHONPATH指向/workspace/tianshu-final-20261104，FlagGems使用site-packages。量化关闭，BF16诊断。

两个输入长度均完成起始、Mixed、Decode、尾部及追加低并发尾部采集。可见MM共13,440个，External id唯一关联率100%；图内Decode MM仍不可见，不能宣称所有阶段逐算子Shape与耗时采集完成。

## 2. 问题与原因分析

历史仅有M=2048的一个Prefill窗口及并发8的Decode上下文，无法支撑实际热点调优。本轮沿用沐曦条件窗口逻辑：在相同官方服务上，按phase和active tokens触发有限步profiler。天数图内活动未展开，需要区分真实aten::mm Shape与forward上下文中的图输入尺寸。

## 3. 具体实现与负载

服务保持天数官方参数：模型/workspace/MiniCPM5-2B、9031、FULL_DECODE_ONLY、served-model-name=minicpm、gpu-memory-utilization=0.85、max-model-len=131072。观察器外部bootstrap注入，不修改模型计算。

诊断客户端为vllm bench serve：random-input-len分别4096/16384，random-output-len=256、num-prompts=128、max-concurrency=64、random-range-ratio=0.0、ignore-eos、temperature=0。客户端含初始单请求测试；窗口可能包含初始测试，不能将所有窗口都认作主请求段。采样日志显示实际random输入长度4095/16383。4K另补64请求、128输出的低并发尾部；这些属于诊断负载，不是官方1024输出的正式成绩或完整M频率。

分析以GPU mm_kernel和CPU aten::mm的External id唯一关联，输出Input Dims/Strides/type、调用数、累计及平均GPU事件时间。step标记按(label,step)去重，避免重复标记翻倍；相同N/K归为投影类别，不恢复具体层号。图上下文尺寸仅作为上下文证据。

## 4. 实验与验证

### 窗口及实际覆盖

| 场景/窗口 | 实际step | phase组成 | active tokens | forward输入尺寸 | 可见MM数 |
|---|---:|---|---|---|---:|
| 4K start | 16 | Prefill 2 + Mixed 14 | 2048 | 2048 | 2688 |
| 4K mixed | 24 | Mixed | 2048 | 2048 | 4032 |
| 4K decode | 24 | Decode | 64 | 64 | 0，图内不可见 |
| 4K tail | 32 | Decode | 17～32 | 24/32 | 0，图内不可见 |
| 4K tail-low | 33 | Decode 32 + empty 1 | 0～16 | 1/2/4/8/16 | 0，图内不可见 |
| 16K start | 16 | Prefill 4 + Mixed 12 | 4096 | 4096 | 2688 |
| 16K mixed | 24 | Mixed | 4096 | 4096 | 4032 |
| 16K decode | 24 | Decode | 31 | 32 | 0，图内不可见 |
| 16K tail | 32 | Decode | 31 | 32 | 0，图内不可见 |
| 16K tail-low | 64 | Decode | 4～9 | 4/8/16 | 0，图内不可见 |

总计289个唯一step，其中288个非empty。4K追加尾部预设64但仅取得33（包含empty），通过stop控制及诊断请求触发落盘，reason=replaced_by_control；不写成满64步。16K的两个常规Decode窗口均停在31个active，覆盖较窄；未观察到16K主负载decode=64，不表示该尺寸不会出现。

### 可见MM形状与耗时

各start窗口每类672次（16×42），各mixed窗口每类1008次（24×42）。下表为start均值，单位μs；mixed均值另见CSV。

| 投影 | N,K | 4K M=2048 | 16K M=4096 |
|---|---|---:|---:|
| QKV | 2560,2048 | 302.888 | 489.408 |
| O | 2048,2048 | 244.916 | 388.052 |
| Gate/Up | 12288,2048 | 1403.272 | 2821.085 |
| Down | 2048,6144 | 725.731 | 1100.713 |

全部可见MM输入/输出BF16。A stride=[K,1]，参与MM的转置权重视图stride=[1,K]，输出stride=[N,1]。四类共8组精确M/N/K（M=2048和4096）；不把Decode forward尺寸扩写为已关联的逐投影Shape。

MM累计GPU事件时间/该窗口全部已记录GPU Kernel事件时间：4K start约60.99%、mixed约54.01%；16K start约34.00%、mixed约31.89%。这是采样窗口事件dur之比，包含观测开销，不是墙钟占比或完整正式负载占比。Decode图内活动缺失时其分母也不完整，不计算Decode GEMM占比。

### 当前可用于后续测试的集合

- 精确MM热点：M=2048/4096，N/K按四类投影；先Gate/Up与Down，再QKV/O。
- 图上下文验证集合：4K为1/2/4/8/16/24/32/64，16K为4/8/16/32；仅forward BF16模型输入上下文尺寸，逐投影M及Kernel耗时unknown。
- 范围外与留出点仍须合法回退及边界验证；本轮未覆盖未来全部M，不形成固定白名单。

## 5. 与已有改动的关系

补充11-04天数短时采样和11-06执行计划T1，保留此前历史结果。16K这次已观测M=4096，不能继续只测2048。通用Dense先前仅测到2048，4096正确性与完整调用性能待追加。沐曦采样方法可复用，但其图内可见性和耗时结论不能迁移到天数。

## 6. 合规性、环境和限制

测试前GPU无计算进程、68MiB、9031空闲；只启动本任务服务。4K和16K及追加尾部串行执行；未改官方参数/Benchmark/数据，未清持久缓存。三个诊断客户端退出码均0；日志确认4K与16K分别128/128成功、补采64/64成功，失败请求均为0。

采集结束核对PID900为本任务API服务、1043为其EngineCore后，TERM 900。结束ixsmi确认无计算进程、0%利用率、68MiB；不终止其他任务。不将现有采样当正式吞吐或精度结果。图内MM、16K更广Decode尺寸、边界Mixed M与正式1024输出全程频率仍有缺口。

## 7. 复现及证据

远端目录 `/workspace/experiments/tianshu-shapes-20261106-01/`；本地 `experiment-artifacts/2026-11-06-tianshu-shapes-01/`。完整归档及manifest内全部文件已在本地逐一核对哈希。capture.tar.gz的SHA256：41c8ba9b4425961653a70e4775a31fed07b559b78fb070a0666bd82a852d32fe。manifest逐文件保存哈希。gpu-after.txt在打包后单独保存。

归档含观察器源码、launcher/worker配置及层清单、控制脚本、命令JSON、客户端/服务/窗口日志、10份Trace、分析脚本及analysis/windows.json、analysis/mm-shapes.csv。运行顺序为官方参数观察器服务 → python3 run_windows.py → 4K低并发补采 → stop触发落盘 → python3 analyze.py → 停服务/检查资源。完整客户端命令见run_windows.py及*-command.json，补采命令参数见第三节。

后续首先利用这些Shape验证三算法完整调用；要取得Decode逐Kernel耗时，应另查CoreX图节点追踪能力，不用eager替代正式路径。未提交或推送Git。
