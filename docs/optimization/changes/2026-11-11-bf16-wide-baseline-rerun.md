# 相同优化 MM 的纯 BF16 官方基线补跑

## 1. 基本信息与摘要

2026-11-11，C500。用户要求补跑纯BF16基线，使用刚完成Down-only wide组的相同M2048 MM配置，单独关闭量化，原样运行官方4K/16K。02续跑于04:00:31完整完成，8轮请求全部成功，最终GPU无进程。不新增源码优化、未提交或推送。

## 2. 问题与原因分析

刚完成的两组Down-only A/B只能衡量MM覆盖效果，历史纯BF16参考存在时间/环境差异。此轮以相同overlay、MM配置及官方脚本建立新的纯BF16对照，检验Down-only端到端收益。两次顺序运行并非交错重复，时间漂移与4K轮次波动仍须记录。

## 3. 具体实现

从 `/data/downonly-mm-official-ab-20261111-01/wide/overlay` 复制完全相同源文件，仅将DOWN、DOWN_INT8PACK、SHAPE_DISPATCH等量化环境开关显式置0，保留M2048四类wide MM覆盖。新服务重建模型、缓存和图状态，不在旧模型上热切换。环境检查要求没有W8A16启用记录且MM覆盖启用，结束后要求四个M2048命中。

## 4. 实验与验证

实验目录 `/data/bf16-wide-baseline-20261111-01/`；控制器 `/data/run_bf16_wide_baseline_20261111.py`。启动前GPU无进程、826MiB；归档进程、端口、源码hash、仓库head/status/diff、命令和环境。官方脚本RUNS4/SKIP_FIRST1不改，4K/16K各四轮，raw八行，summary两行，正式成绩为后三轮平均。当前结果待完成。

## 5. 与已有改动的关系

对比对象为刚完成Down-only wide：4K4924.46tok/s、MeanTTFT2968.41ms；16K7454.43tok/s、MeanTTFT24706.73ms。该轮补测不沿用历史5277.76/7653.26作为新基线，不覆盖既有结果。

## 6. 合规性与风险

保留官方serve/Benchmark原参数与数据；外围控制器只管检查、启动、归档。无量化，模型保持原BF16。没有新的精度评测，不能称完整精度验收。吞吐与TTFT逐场景按严格超过1%分类，不因预期结论择优取轮。端口检查使用SO_REUSEADDR，避免先前关闭连接后的普通bind误阻塞；仍核对服务进程和GPU释放。

## 7. 复现方式

远端控制器及本次bf16-wide/commands.json记录原始命令。状态status.txt，异常failure.txt，server.log与benchmark.log保留原样。完成后 `/data/archive_bf16_wide_baseline_20261111.py` 自动生成与Down-only wide的comparison.json/md并归档 `/data/bf16-wide-baseline-20261111-01-results.tgz`。复现使用新实验目录，禁止覆盖本轮。

### 中断与恢复说明

03:10曾误将用户对旧基线的解释当作切换任务，停止了01目录下尚未完成的优化MM纯BF16任务；原日志、源文件和interrupted.txt保留，不算正式结果。用户明确“补测不变”后，03:12恢复原方案，使用新目录 `/data/bf16-wide-baseline-20261111-02/` 从官方完整命令重新开始，不拼接或复用中断轮次。控制器 `/data/run_bf16_wide_baseline_20261111_retry.py`，PID468131；归档等待PID468132。未启动完全默认MM的新基线任务。用户索要的数据对照使用此前完整默认BF16基线4729.59/7022.20及对应TTFT3250.42/27133.58，不与本补测配置混淆。

### 最终正式汇总

| 场景 | 新纯BF16 tok/s | Down-only tok/s | 量化吞吐变化 | 新BF16 TTFT ms | Down-only TTFT ms | 量化TTFT增加 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 5127.00 | 4924.46 | -3.950% | 2639.22 | 2968.41 | 12.473% |
| 16K | 7651.48 | 7454.43 | -2.575% | 23658.99 | 24706.73 | 4.429% |

相同MM配置下，Down-only两场景吞吐与TTFT均有效退化；量化性能优势未得到支持。顺序运行存在时间差，4K轮次波动原样保留，不择优。新纯BF16四个M2048配置命中、无量化启用记录已核对。原始CSV及健康/退出状态已取回 `artifacts/bf16-wide-baseline-20261111-02/`；完整包保留远端 `/data/bf16-wide-baseline-20261111-02-results.tgz`。精度未重新评测。
