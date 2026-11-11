# Down-only：默认 MM 与 M2048 wide MM 官方双场景对照

## 1. 基本信息与摘要

2026-11-11，C500。用户指定只运行两组Down-only，不新增纯BF16基线。A：官方默认MM；B：M2048四类投影使用当前固化wide配置，其他M维持相同默认提供者和Down三算法分派。两边均启用相同转换优化。两组于02:54:04完整完成，最终GPU无进程。

## 2. 问题与原因分析

该对照直接衡量MM配置覆盖在当前Down-only组合下的效果。两边量化相同，不能独立证明量化相对纯BF16的收益；用户明确要求不补跑基线，已有BF16数据仅按其环境/版本可比范围引用。

## 3. 具体实现

独立目录 `/data/downonly-mm-official-ab-20261111-01/`。两套overlay从同一已验证候选复制，唯一预期源码差异为worker是否安装M2048 wide覆盖，controller自动核对哈希差异。Down-only选择、专用MetaX入口、manifest、Split-K配置、Separate和转换均相同。Down M56/64为64/64/128、4warps/1stage、Split8；Down N2048/K6144且113≤M≤4096转换block4096、warps4，其他转换1024。Gate/Up、QKV、O保留原始BF16。

## 4. 实验与验证

原样执行沐曦官方serve与benchmark命令。脚本已核对RUNS4/SKIP_FIRST1并记录sha256；每组4K/16K各四轮，summary为后三轮平均，raw保留8条。两组串行、切换服务重启。启动前归档GPU、进程、端口、源文件、仓库head/status/diff、环境和命令；健康后要求恰好42个Down量化启用记录且MM覆盖状态符合模式。B结束要求四个实际M2048命中记录。控制器自动验证退出码、请求成功数、summary两行/raw八行，再停止本任务服务并检查资源释放。当前结果待汇总。

### 正式结果

每组4K/16K各四轮全部请求成功，正式summary使用第2/3/4轮平均。未重算或择优剔除第三轮4K低值。

| 场景 | 默认MM total tok/s | wide total tok/s | 吞吐提升 | 默认Mean TTFT ms | wide Mean TTFT ms | TTFT降低 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 4524.09 | 4924.46 | 8.850% | 3517.03 | 2968.41 | 15.599% |
| 16K | 6839.53 | 7454.43 | 8.990% | 28309.09 | 24706.73 | 12.725% |

两场景吞吐、TTFT均超过1%改善门槛。该A/B两边量化相同，只证明M2048 MM配置覆盖在当前组合下有收益，不能独立归因为量化收益；也没有新的精度验收。4K原始轮次存在明显波动，完整保留，未作因果归因。

wide四个M2048命中记录与42个Down覆盖已核对。两组关键哈希仅worker开关不同，量化入口、Down算法实现、MM配置表、转换及官方脚本相同。default先跑、wide后跑，期间有端口检查中断及时间间隔，完整记录，不称交错端到端实验。

## 5. 与已有改动的关系

承接Down-only热点补采、转换4096验证及wide BF16配置，不混入Gate/Up量化预取原型。本次未新增算法或更改官方参数。正式性能结果与诊断micro分开。

## 6. 合规性与风险

量化仍为BF16激活×转换后的BF16权重、FP32累加、输出BF16，scale FP32。Separate保留BF16中间舍入。未新增精度评测，不能将本次性能任务称完整精度验收。吞吐/TTFT逐场景按严格超过1%分类，不能只报平均。服务/客户端失败时归档异常，不截断或改参数重算成完整成绩。

## 7. 复现方式

远端控制器 `/data/run_downonly_mm_ab_20261111.py`；日志 `/data/downonly-mm-official-ab-20261111-controller.log`；状态在实验目录status.txt，异常failure.txt。每组commands.json保存原始命令，hashes.json保存源文件与官方脚本指纹，server.log、benchmark.log及summary/raw保留原样。复现须使用新目录，不能覆盖既有结果。

### 已核对的启动状态

2026-11-11 00:19:21，default已健康并进入官方4K+16K命令，日志42个启用记录均为Down；未启动新的纯BF16基线。控制器PID445282；归档等待进程PID446070，只处理结果文件，不产生GPU负载。default结束后自动清理和重启wide，执行完整官方命令。

自动归档脚本 `/data/archive_downonly_mm_ab_20261111.py` 完成后生成实验目录中的comparison.json/comparison.md，并保留全量源文件overlay、CSV和原始日志至 `/data/downonly-mm-official-ab-20261111-01-results.tgz`。启动状态不是完成结果；尚无正式汇总数据。

### 默认组完成与续跑记录

默认组官方命令完整完成，8轮全部请求成功。正式summary：4K total tokens/s 4524.09、Mean TTFT 3517.03ms；16K total tokens/s 6839.53、Mean TTFT 28309.09ms。

01:09:34控制器在服务退出后的普通socket bind端口检查报EADDRINUSE，未进入wide。该异常发生在默认组正式结果生成及完整性检查后，不将测评本身标为失败，也不隐瞒自动流程中断。原failure.txt与首次归档保留。02:04核对GPU826MiB无进程、无服务进程、9031连接拒绝且/proc/net/tcp/tcp6无该端口监听，SO_REUSEADDR bind成功；先前报错可能与关闭连接的临时状态有关，未取得当时TCP状态，不作确定归因。

使用新控制器 `/data/resume_downonly_mm_ab_20261111.py` 仅续跑wide，增加SO_REUSEADDR端口检查，不改正式serve/benchmark、不重跑default、不覆盖default数据。新异常写resume-failure.txt。续跑控制器PID458052，归档等待PID458053；成功后生成对照并归档至 `/data/downonly-mm-official-ab-20261111-01-resumed-results.tgz`。

### 最终归档

wide于02:53:50完成正式命令，02:54:04完成清理与完整性检查。远端完整归档已生成（约8.1MiB），保留源码overlay、全部日志、CSV及历史中断记录。本机已取回 `artifacts/downonly-mm-official-ab-20261111-01/` 下的两组summary/raw、覆盖、实际命中、命令、关键哈希、健康/GPU退出状态和comparison.json/md，并重新验证每组8条SUCCESS及请求数一致。完整大包仍在远端，不宣称已下载本机。未提交、未推送。
