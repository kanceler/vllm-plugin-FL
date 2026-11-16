# 静态KV与Down组合诊断接入

## 1. 基本信息与摘要
2026-11-16，天数BI-V150。用户授权继续叠加Down、Down+QKV、Down+QKV+O，并要求06:00前能完成单场景两轮才启动。当前仅启动Down组合。远端隔离目录：`/workspace/experiments/tianshu-k-static-down-perf-20261116-01`，未覆盖生产仓库，未提交推送。

## 2. 问题与原因分析
静态KV冻结源码没有GEMM量化安装入口，单设Down环境变量不会启用。需复用独立已验证实现并明确检查实际覆盖。

## 3. 具体实现
复制静态KV精度04目录source，加入`tianshu-large-m-runtime-20261113-01/source/vllm_fl/quantization/w8a16`，仅在`vllm_fl/__init__.py`的量化注册前调用`install_tianshu_separate`。静态服务manifest重新计算源码绑定，并断言已有绑定变化只能是该安装入口；静态Kernel和校准不变。新的GEMM缓存身份包含实现及四投影开关。控制器只开启Down，检查42条Down初始化且QKV/O/GateUp未启用后才计时。

## 4. 实验与验证
控制器本地Python语法检查通过，远端PID52878，服务PID52881，已观察`static-perf-starting`。组合服务覆盖、数值及吞吐尚待验证；静态KV单独精度不代表叠加精度。此前静态KV独立16K两轮2597.72/2842.64 tok/s，4K两轮2986.77/3243.05 tok/s，全请求成功。组合没有新模型精度成绩。

## 5. 与已有改动的关系
静态KV来自已完成官方精度的隔离04源码；GEMM来自11月13日已验证离线策略。没有重写Kernel或改变量化路线。INT8权重、FP32 per-output-channel scale，BF16激活，计算时恢复BF16；不是原生INT8×INT8。

## 6. 合规性与风险
官方serve参数不变，原始Benchmark文件不改。两轮诊断调用其原始命令构造和指标提取函数，保留两轮原始值，不视为官方四轮summary。组合精度、实际策略命中及收益待验证。每个场景开始前按截止时间预算判断；16K预算1900秒，4K预算1050秒，不够则跳过。组间停止旧服务，检查GPU/端口，新服务隔离。

## 7. 复现方式
本机控制器：`artifacts/k-static-perf-20261116-01/run_down.py`。远端命令：`/workspace/tianshu-mixed-env-20261109/bin/python -u /workspace/experiments/run-k-static-down-perf-20261116-01.py`。必须新目录运行。命令、环境、源码Python哈希、manifest变化、覆盖、逐轮日志与指标在实验目录，轮后自动归档，结束停止服务并生成完整tar.gz。当前启动状态不代表计时已开始或完成。

## 用户暂停后的检查更正
05:40按用户要求停止本任务控制器、客户端及服务，GPU无计算进程、70MiB保留。Down确有42层INT8权重，但专用策略未命中：manifest要求BF16提供者`fd120991...`，实际site-packages为`4c2e23da...`，policy_matches因此false，走separate_generic。首轮2301.70 tok/s、TTFT103857.12ms、128/128成功仅代表通用回退；第二轮中断，无完整成绩。未执行4K或后续组合。
原验证环境PYTHONPATH除source外还包含`/workspace/experiments/tianshu-boundaries-20261113-01/gems`；本轮遗漏此提供者路径是接入错误。已定位匹配提供者，未改哈希绕过绑定，未重启测试。暂停现场归档为实验目录同名`-user-paused.tar.gz`。恢复前需检查整个提供者绑定、同版本BF16对照及三算法配置命中，不能据回退负结果否定融合优化。

## 提供者路径修正（用户授权修改）
新增另存控制器`artifacts/k-static-perf-20261116-01/run_down_fixed.py`及`check-k-static-down-policy.py`。新实验02不覆盖01；显式追加原验证gems路径，保留原策略与provider哈希，删除已撤销的06:00截止逻辑。模型启动前单独进程检查policy_matches、实际provider文件路径/哈希、Torch/Triton/设备，并记录M=1/16/32/56/64/128/256/2048/4096的真实choose结果；拒绝separate_generic，并逐点执行dispatch相对FP32参考relative L2≤0.005。VALIDATE_ONLY=1模式仅验证、不启模型性能服务。启动时GPU资源预检；配置覆盖不是模型耗时覆盖，不代表全部实际M已测。当前验证进程已启动，最终结果待归档；组合精度仍未验证。

02验证完成：policy_matches=True，9/9代表M专用策略命中，无generic回退，全部relative L2≤0.005。最大relative L2=3.483359978417866e-05。结果已取回`artifacts/k-static-perf-20261116-01/policy-check-fixed.json`，远端state为validation-complete-performance-paused。此项是已有算法正确性与分发检查，无性能收益结论；未恢复模型测评。

## 修正提供者后的两轮重测03（12:03授权启动）
用户要求先重新测叠加Down。新目录`/workspace/experiments/tianshu-k-static-down-perf-20261116-03`，控制器`run_down_retest.py`另存，原01错误回退与02验证记录不覆盖。重新GPU检查无进程、70MiB保留。正式启动前再次执行9点策略/正确性门槛，随后新服务42层覆盖及35图检查，再16K→4K各两轮。全模式环境变量另存full-env.json，控制器复制进归档。没有06:00截止，不并发INT4测试。当前只授权并启动Down组，未启动QKV/O组合。远端主仓库当前d799978且有既有改动，实际运行使用归档隔离source，不覆盖主仓库。测试结果待完成，组合模型精度未验证，历史静态KV对照与本次MM提供者不同，不能将其直接当同提供者严格量化归因A/B。

## 03两轮诊断完成
12:53全部结束，服务停止，GPU无进程。16K首/次轮2331.43/2801.23 tok/s、TTFT101403.69/79513.72ms，128/128成功；4K首/次轮2805.45/3117.49 tok/s、TTFT6091.17/5066.99ms，256/256成功。所有原始轮次保留，不作为四轮官方summary。对历史静态KV第二轮：16K吞吐-1.4568%左右，4K-3.8717%左右；提供者不同，不作严格Down单因素归因或正式收益判定。03完整tar.gz已生成，下一组未启动。
