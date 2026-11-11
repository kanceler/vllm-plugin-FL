> 2026-11-05 完成后审计：发现显式候选的LibEntry启动参数缓存别名。原始计时保留，名义warps/stages与完整联合搜索结论不能直接认证；完整归档、入口实测及修正见[完成后审计](2026-11-05-w8a16-completed-sweeps-analysis.md)。

# Split-K 补测协议与完成状态修复

## 1. 基本信息与摘要

2026-11-05，本地修复，目标平台 MetaX C500，未连接 GPU。沿用 `perf/gemm-w8a16-splitk-wide-search`，基点 `f93489fa1afb74969b76701c8b551552a55ae3bf`。涉及 `benchmarks/diagnostics/tune_metax_w8a16_splitk.py`、`test_w8a16_splitk_search.py` 及本报告。保留已有未提交内容，未提交、未推送。

## 2. 问题与原因分析

原脚本允许修改 baseline-seeds，但 metadata 将独立输入数量固定写为2，续跑身份检查不能识别参数变化。另一个问题是没有合法终测赢家时仍将单点写为 complete=true，总汇总仅统计60个文件，可能将数值验证失败误报为补测成功。

## 3. 具体实现

将 baseline-seeds 的合法值限定为2，并从实际参数生成协议字段。单点只有选出通过边界与两套独立输入验证、具有完整终测样本的配置才 complete=true；execution_complete 单独表示执行结束，无赢家记录 no_valid_finalist。

汇总增加 execution_complete 和 failed_points，complete 要求60点结果均有效。最终 status 使用实际汇总，存在无效点时控制器返回异常。有效结果不要求跑赢BF16：数值合法与性能改善仍分别记录。旧结果缺失 complete 字段时保守标为无效，复测使用新目录。

## 4. 实验与验证

CPU unittest 共12项通过，新增回归构造60个无赢家结果文件，确认执行结束但 complete=false 且失败清单包含60点。单独验证 baseline-seeds=3 在创建实验目录前被 argparse 拒绝，退出码2。Python编译检查、git diff --check通过。

GPU编译、数值误差、性能、模型精度、吞吐和TTFT均待验证。本次没有GPU实验数据。

## 5. 与已有改动的关系

依赖11月4日宽搜索工具；沿用20宽点加40精简点、候选生成与计时协议。历史实验文件保留。修复后脚本指纹变化，不能续跑旧指纹实验，应新建目录。

## 6. 合规性与风险

仅修复诊断控制与证据状态。上卡前仍需核对进程、显存、利用率、端口、仓库差异与实际import路径。通过CPU检查不代表GPU验证完成，也不代表有性能收益。

## 7. 复现方式

在本地工作区执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s vllm-plugin-FL/benchmarks/diagnostics -p test_w8a16_splitk_search.py -v
```

结果为12项通过。参数拒绝检查使用 `--baseline-seeds 3 --output-dir /private/tmp/unused-splitk-review`，在参数解析阶段退出，未创建实验目录。远端命令及GPU日志待取得SSH并完成环境核验后归档。

### 远端启动状态（2026-11-05）

新实例已启动后台补测，实验目录 `/root/experiments/2026-11-05-w8a16-splitk-wide-01/`。启动入口 `run.sh`，控制台 `run.log`，最终退出状态将写入 `exit-code.txt`。本条记录仅确认启动，尚未完成60点。

设备 C500、104 SM、warp64，驱动3.8.30、MACA3.7.0.38、Torch2.8.0+metax3.7.0.7。启动前GPU利用率0%、显存826MiB、无GPU进程；脚本端口及竞争任务门禁通过。原仓库无未提交差异：插件13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0、FlagGems a7620cc191a0b42e040194622c5758b22a7a25dc。保留原仓库，在实验目录复制隔离源码并覆盖必要后端/reference/工具，显式设置PYTHONPATH；原始提交和diff、环境、AGENTS及脚本指纹已在远端保存。上传包两端SHA256一致：fd4bae804702a3225dde79cfb90655e76a89d87aeaf32e1d732927721f957686。

首个Down M1已进入终测，筛选中出现shared memory OutOfResources，记录具体配置及所需/可用资源后继续其他候选，无设备崩溃证据。最终数值、性能、60点完成状态及运行后资源状态待任务结束归档；不将后台启动写成完成。
