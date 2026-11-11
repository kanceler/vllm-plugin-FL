# 转换优化与 scale 融合：默认 M2048 测评启动记录

## 1. 基本信息与摘要

2026-11-11，MetaX C500。本次仅修复容器实验 smoke 脚本的结果打印错误，并启动已有转换优化与 scale 融合候选的官方性能测评；未修改 Kernel。实验根目录：`/data/scale-direct-official-default-mm-20261111-02`。

## 2. 问题与原因分析

前次实验 `...-01/smoke.log` 在第39行报错：`JSONEncoder.__init__() got an unexpected keyword argument 'flush'`。`flush` 被传入 `json.dumps`，应属于外层 `print`。原脚本已备份为 `smoke_default_m2048.py.before-fix`，原失败日志保留。

## 3. 具体实现

将末尾打印改为 `print(json.dumps({...}), flush=True)`。修复文件位于 `/data/scale-direct-official-default-mm-20261111-01/smoke_default_m2048.py`。基于已有官方测评控制器生成 `/data/run_scale_direct_default_20261111.py`，仅运行 default 候选、复用已准备 overlay。M2048 使用缓存默认 BM/BN/BK=64/64/64、4 warps、4 stages、basic pipeline；不启用此前 wide override。

已有候选对 Down `(N,K)=(2048,6144)` 使用 BLOCK=4096 转换 INT8 权重至 BF16，并调用 BF16 NT MM 的 scale 融合 epilogue。此路径是 BF16 计算，不是原生 INT8×INT8 GEMM。权重为 INT8，scale FP32，输出 BF16；完整测量保留转换开销。

## 4. 实验与验证

修复后 M=2048、N=2048、K=6144 smoke：eager 与 CUDA Graph 相对 L2 均为 `0.0016593619948253036`，通过原定 `<0.005` 阈值。日志：`...-01/smoke-fixed.log`。仅证明该 shape 的 smoke，不代表全模型精度。

15:13:58 控制器开始启动新服务；15:17:23 健康检查返回 HTTP200，42层 Down 量化覆盖及默认 MM 模式检查通过。16:06:29 官方性能流程完成，退出码0；raw CSV共8轮，4K每轮256/256、16K每轮128/128，均SUCCESS。summary按后三轮平均：

| 场景 | total tokens/s | Mean TTFT ms | 相对旧未融合默认MM组合吞吐 | TTFT变化 |
|---|---:|---:|---:|---:|
| 4K | 4854.76 | 3521.26 | +7.31%，有效提升 | 增加0.12%，正常波动 |
| 16K | 6847.52 | 28353.69 | +0.12%，正常波动 | 增加0.16%，正常波动 |

旧未融合组合为 `/data/downonly-mm-official-ab-20261111-01/default/`。本轮核对证据表明命令、记录环境（除overlay路径）、仓库指纹相同，但实际运行overlay另有M56固定Split-K特判移除，故不能把7.31%全部归因于scale融合。辅助函数的128→113确有源码差异，但实际开启的shape dispatch在旧版本也已于M>=113回退Separate，不构成该路径分界变化。direct scale在FP32累加后直接乘scale，去掉旧Separate中间BF16舍入；精度尚未验证。

16:06:42收尾端口bind检查报Address already in use，控制器标FAILED；发生于CSV完整生成与测评成功退出之后，不重写原状态、不生成伪complete。后续17:23检查无相关服务/评测进程、GPU无进程、9031可绑定。原因未进一步证明，不将其自动写成TIME_WAIT。全模型math_500精度待验证，不宣称完成验收。

## 5. 与已有改动的关系

复用 `...-01/overlay` 中已有转换和 scale 融合实现，不重写 BF16 提供者。旧 BF16 wide summary 不能直接当作默认配置的可比基线；最终比较须核对提供者、源码、命令和环境。

## 6. 合规性与风险

沿用官方服务和双场景 benchmark 参数，RUNS=4、SKIP_FIRST=1；归档8轮原始数据与2行 summary。启动前检查空闲 GPU、9031端口、进程，归档仓库与源码指纹。服务健康后要求42层 down_proj量化且未开启 wide override。模型精度、实际回退及收益尚未验证。运行代码来自独立 overlay，容器仓库未修改；本地已有无关改动未纳入。

## 7. 复现方式

容器交互环境已加载 Conda/MACA。修复复验：

```bash
PYTHONPATH=/data/scale-direct-official-default-mm-20261111-01/overlay python /data/scale-direct-official-default-mm-20261111-01/smoke_default_m2048.py
```

本次控制器由 `nohup python -u /data/run_scale_direct_default_20261111.py` 启动；不要并发重复执行。控制日志：`/data/scale-direct-default-20261111-controller.log`。服务、benchmark、环境、hash、覆盖和结果在新实验目录 `default/` 下；状态及失败原因在根目录 `status.txt`、`failure.txt`，完成后生成 `complete.txt`。实际官方命令与优化开关见 `default/commands.json`。尚无正式结果。
