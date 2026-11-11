> 2026-11-05 完成后审计：发现显式候选的LibEntry启动参数缓存别名。原始计时保留，名义warps/stages与完整联合搜索结论不能直接认证；完整归档、入口实测及修正见[完成后审计](2026-11-05-w8a16-completed-sweeps-analysis.md)。

# Gate/Up 与 O 中段 Dense 搜索准备

## 1. 基本信息与摘要

2026-11-05，目标MetaX C500。沿用插件基点f93489fa1afb74969b76701c8b551552a55ae3bf，新分支perf/gemm-w8a16-dense-mid-search。新增诊断文件tune_metax_w8a16_dense.py、w8a16_dense_search.py、test_w8a16_dense_search.py，保留已有未提交修改。本次用户授权实现并上卡测试；工具已部署，因资源占用未解释暂未启动GPU搜索。

## 2. 问题与原因分析

旧三算法有限池中，Gate/Up和O部分中段Dense有竞争力，但不足以确定分界或快于原始BF16。当前另一实例进行Split-K宽搜索，本实例补齐Dense，避免重复搜索。两实例结果不直接拼接成最终dispatch。

## 3. 具体实现

6个代表点宽搜：Gate/Up M24/64/96、O M64/80/96。10个精简点：Gate/Up M16/32/48/80/104，O M56/72/88/104/112。O M56为交界探针，其余按本轮已讨论的观测点范围标注；全部16点参与调优，不作为独立留出集。

BM16/32/64/128、BN32/64/128/256、BK32/64/128/256、warps2/4/8、stages1/2/3构成576组组合，固定basic。BM过大为软筛选，保留warps4/stages2审计项和历史控制项。按投影汇总代表点Top5及1%内其他tile结构形成精简池，失败配置不作为优胜项传播。

复用Split-K工具的串行子进程、指纹、日志、完整调用计时、数值门槛和两套独立输入机制，独立文件保留原工具。终测同时记录原始BF16、BF16(Q)诊断、历史Split-K和Separate完整调用；控制项原始计时保存raw_final.jsonl。数值先行，relative L2<0.005、rtol0.02、atol0.125。输出为静态per-output-channel INT8权重、FP32 scale、BF16激活，权重在dot前转BF16，FP32累加后scale并写BF16。层覆盖率不适用，模型未接入。

## 4. 实验与验证

本地4项CPU回归通过：16点矩阵与576组合、保留控制项、拒绝失败配置传播、五字段Dense shortlist、无赢家不能误报完成（部分内容在同一测试中）。Python编译检查及git diff --check通过。GPU编译、数值与性能均待验证。

远端独立目录/root/experiments/2026-11-05-w8a16-dense-mid-01/已准备，上传包SHA256两端一致：f55d24efb64e26f0da513bec7e66e8c6a7219e7176f37b32a06ee32856954b94。GPU UUID为8048c377-071e-0e6e-8098-1d617152e242，104 SM；与Split-K实例UUID不同。驱动3.8.30、MACA3.7.0.38，已有Conda Python正常。

启动前GPU利用率0%，显存2272MiB；mx-smi与--show-all-process均无进程，/proc可见设备句柄扫描无使用者。该占用尚不能归类为已确认驱动保留，按根AGENTS第六节暂不启动GPU任务。证据已归档gpu-before.txt、gpu-processes.txt、processes-before.txt、environment.json、pending-resource-check.txt及两仓库提交/status/diff。

## 5. 与已有改动的关系

复用已有显式W8A16后端，不改生产Kernel和原生BF16提供者，不接入tile分支。当前Split-K实验不受影响。后续三算法分界须同机同提供者重测。

## 6. 合规性与风险

无模型精度、官方吞吐或TTFT结果，无收益声明；未启动不写成完成。编译资源通过数未知，启发式有遗漏风险，只能称有限池搜索。新资源占用解释前不得启动；不清理其他任务、不重置GPU、不删除编译缓存。

## 7. 复现方式

本地已执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s vllm-plugin-FL/benchmarks/diagnostics -p test_w8a16_dense_search.py -v
```

远端后续拟执行（尚未执行，资源核对通过后使用新的results目录）：

```bash
export PYTHONPATH=/root/experiments/2026-11-05-w8a16-dense-mid-01/FlagGems/src:/root/experiments/2026-11-05-w8a16-dense-mid-01/vllm-plugin-FL
python /root/experiments/2026-11-05-w8a16-dense-mid-01/vllm-plugin-FL/benchmarks/diagnostics/tune_metax_w8a16_dense.py --output-dir /root/experiments/2026-11-05-w8a16-dense-mid-01/results
```

本次未提交、未推送。资源问题解决后补充实际执行状态和结果。

### 用户要求复核旧测试占用后的追加检查

2026-11-05再次检查：容器内无测试进程或GPU设备句柄，9031连接返回111（未监听）。只读驱动节点 `/sys/class/drm/renderD128/device/xpu_vm_va_used` 显示XPID28/29/30/31，地址空间字节分别为486801408、486801408、405012480、572817408。该字段表示GPU虚拟地址空间记录，不等于各进程实际物理显存量，也不能直接把XPID当作Linux PID。mx-smi仍显示2272MiB且无可见进程，尚不能确认上述记录归属本任务或已结束的旧测试。

已在远端resource-audit时间戳目录保存GPU状态、全进程列表、驱动地址空间与显存节点、端口及结论。未执行kill、GPU reset或缓存清理，Dense搜索仍未启动。需要宿主机/平台侧核对这些上下文的归属并定向释放，不能仅凭无可见PID认定正常驱动保留。

### 与第一轮历史实验逐项对照

进一步找到新容器持久目录中的11月3日实验及gpu-after-screen.txt：历史实验于2026-11-03 19:01:13结束，GPU bus为0000:35:00.0，结束后826MiB、0%、无进程。当前GPU bus为0000:0e:00.0、2272MiB。因此历史目录虽保留，不足以归因当前占用；现有证据反而表明第一轮已在另一GPU释放资源。原始gpu-after-screen及对照JSON另存远端previous-test-comparison目录。继续保持未启动，不将未知XPID当作已确认旧测试进程清理。

### 后续用户指示与短诊断

用户明确指示：若不影响补测则接受当前状态，不要求重启或删除。依据该指示先运行独立短诊断，五次空闲状态采样后，对O与Gate/Up的M64原始BF16完整调用各做七轮计时，预先以max-min相对中位数不超过2%作为当前稳定性门禁。该门禁只能识别明显干扰，不能证明未知占用完全无影响；未知XPID归属保留，不清理其他进程、不reset GPU。诊断进程退出后才允许启动主搜索，主搜索保留每点两套独立BF16对照与2%稳定性标记。短诊断脚本resource_smoke.py、原始空闲样本idle-samples.txt和计时resource-smoke.json位于远端实验根目录，执行结果待回收。

### 短诊断通过，启动Dense搜索

短诊断七轮O M64原始BF16为0.032512～0.032768ms，跨度/中位数0.787396%；Gate/Up M64七轮均0.064000003ms，跨度0%。两项均低于预设2%稳定性门禁，未发现明显计时干扰，不等价于证明未知上下文毫无影响。短诊断进程PID1181已正常退出。

按用户接受当前状态的指示启动主搜索，后台入口/root/experiments/2026-11-05-w8a16-dense-mid-01/run.sh，日志run.log，包装进程PID1352；退出码将保存exit-code.txt。未知2272MiB占用保留，无删除或GPU reset，原始归属未知结论不变。主任务每点继续进行BF16独立输入稳定性检查。此前“尚未启动”描述保留为当时状态，以本节最新状态为准。整批完成及GPU数值、性能结论仍待验证。

启动门禁通过，已核对metadata、plan及实际源码指纹，首个Gate/Up M24已产生正确性通过及编译资源失败的逐候选记录，主搜索确实进入GPU筛选阶段。启动/短诊断证据已取回本地experiment-artifacts/2026-11-05-w8a16-dense-mid-01/startup-evidence/。完整16点尚未结束。

### 完整运行与本地归档核验

16/16点完整结束，exit-code=0，status.complete=true，无失败点。完整原始计时、候选、错误、资源、环境及源码快照已取回本地experiment-artifacts/2026-11-05-w8a16-dense-mid-01/completed-evidence/。归档SHA256为3370b477cb7e8a785073e97f43efe82dbf72862dc42b7152ba2d6e01fd411806。

本地核验1590条终测原始样本，逐项重算190行候选/seed中位数均一致；9个源码指纹一致。16点中没有两套输入均稳定快于原始BF16超过1%的Dense赢家。此结论仅限本轮完整显式入口、有限配置池和synthetic eager计时，不替代后续专用入口及图重放验证。详细核验见audit.json。

| 投影 | M | 两套输入Dense耗时 µs | 两套输入相对BF16耗时降低率 |
|---|---:|---|---|
| gate_up | 24 | 78.848 / 79.360 | -38.74% / -40.27% |
| gate_up | 64 | 99.840 / 99.328 | -42.86% / -43.17% |
| gate_up | 96 | 148.736 / 140.288 | -39.33% / -32.37% |
| gate_up | 16 | 68.608 / 68.608 | -26.42% / -29.47% |
| gate_up | 32 | 74.752 / 74.752 | -32.13% / -33.94% |
| gate_up | 48 | 94.976 / 95.744 | -46.64% / -47.83% |
| gate_up | 80 | 124.160 / 123.904 | -18.29% / -18.05% |
| gate_up | 104 | 126.720 / 127.488 | -19.57% / -20.58% |
| o | 64 | 56.832 / 56.832 | -65.67% / -65.67% |
| o | 80 | 57.344 / 57.344 | -69.70% / -70.99% |
| o | 96 | 57.600 / 56.832 | -66.67% / -62.04% |
| o | 56 | 56.064 / 56.064 | -67.18% / -67.18% |
| o | 72 | 55.552 / 55.552 | -65.65% / -65.65% |
| o | 88 | 56.832 / 55.808 | -63.24% / -60.29% |
| o | 104 | 64.256 / 64.256 | -69.59% / -69.59% |
| o | 112 | 64.768 / 64.512 | -68.67% / -68.00% |
