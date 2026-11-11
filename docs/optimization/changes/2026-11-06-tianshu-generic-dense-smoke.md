# 天数通用 W8A16 Dense 首轮能力验证

## 1. 基本信息与摘要

2026-11-06，Iluvatar BI-V150、驱动/CoreX 4.5.0，PyTorch 2.10.0、Triton 3.6.0。新增独立诊断脚本 `experiment-artifacts/2026-11-06-tianshu-w8a16-01/dense_smoke.py`（工作区路径，不随插件自动提交），生产源码无修改。通用Dense的7个合成Shape编译与数值检查通过；不代表性能、图兼容或模型验收。

## 2. 问题与原因分析

通用weight_int8pack_mm源码已存在，但此前缺少天数执行证据。历史BF16候选工具的grid/warmup错误属于另一测试入口，本次直接调用通用公开函数，不据此宣称旧工具已修复。

## 3. 具体实现

独立进程直接导入已安装的flag_gems.ops.weight_int8pack_mm。BF16 A、INT8 Q、FP32 per-channel scale；参考为FP32 A@Q.T后乘scale并转BF16。seed=42，关闭参考TF32。预设smoke门槛为输出有限且relative L2<0.005；仅作初步Kernel检查，完整逐元素、量化误差及算法舍入验证待补。

## 4. 实验与验证

| M,N,K | relative L2 | 结果 |
|---|---:|---|
| 1,17,33 | 0 | 通过 |
| 3,129,257 | 0 | 通过 |
| 16,128,256 | 0 | 通过 |
| 2048,2560,2048 | 1.32189e-5 | 通过 |
| 2048,2048,2048 | 1.50344e-5 | 通过 |
| 2048,12288,2048 | 1.77766e-5 | 通过 |
| 2048,2048,6144 | 2.85239e-5 | 通过 |

全部输出有限。first_call_wall_s包含编译、调优及执行，不作为稳态性能。日志提示replay benchmarker不可用，调优内部回退event；本轮不报告replay计时。非连续输入、多seed、空输入、bias/回退、全部真实M、图重放、模型精度及性能均待验证。

## 5. 与已有改动的关系

沿用天数计划T0/T2，只验证现成通用Dense，未调用沐曦专属后端。插件f82fa1670509f081aa13b58464cb6e5627fc10c4、FlagGems仓库4df52d9168ac55c1c1061cefb4638c570292d896均干净；实际import来自site-packages，实际Dense源码路径/哈希见dense-metadata.json，不能只用仓库提交替代安装包指纹。

远端已核实历史summary含4K/16K两行、raw含八轮且全部SUCCESS，原本缺失的是本地旧副本。历史成绩不改写，不作为本轮候选收益。

## 6. 合规性与风险

测试前GPU无计算进程、68MiB驱动占用，端口9031空闲。未启动模型服务、未改官方命令/数据/Benchmark；仅合成权重与输入，无评测集校准。当前无模型量化覆盖、无吞吐收益声明。结束GPU状态见gpu-after-dense.txt。

## 7. 复现方式

远端独立目录 `/workspace/experiments/tianshu-w8a16-20261106-01/`，执行 `python3 dense_smoke.py > dense-smoke.log 2>&1`。本地证据目录 `experiment-artifacts/2026-11-06-tianshu-w8a16-01/`。

测试前完整历史归档包含90个文件，history-manifest.json记录逐文件大小/哈希；history.tar.gz SHA256为2315f595273f825c676162d6e2ab398ce17e793d7638b949c6c821d44858106c，本地下载哈希一致。归档涵盖三个历史实验目录及当时benchmark_results全部内容，保留原目录。原始日志、metadata和结果JSON保留；未提交/推送Git。
