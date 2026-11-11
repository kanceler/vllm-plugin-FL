# Down 更大转换 block、M 与 MM 提供者交叉验证

## 1. 基本信息与摘要

2026-11-11，MetaX C500。用户要求扩大转换 block/M，并验证搭配官方默认 MM 是否仍有收益；明确旧基线为官方默认，当前已固化较优配置可覆盖。新增诊断脚本 `benchmark_metax_convert_provider_cross.py`，独立目录 `/data/down-convert-provider-cross-20261111-01/`。生产配置暂不修改，未提交或推送。

## 2. 问题与原因分析

转换权重规模只依赖 N/K，但完整 Separate 的 MM 耗时、布局与缓存条件影响收益比例。不得把旧 tiling 分支误称官方默认。既有 wide 仅覆盖四个 M2048 Shape，其他 M 与默认同路径；不重复把相同路径计成不同提供者。

## 3. 具体实现

default：注册 FlagGems MetaX MM，不调用配置覆盖安装。wide：同版本注册 MM，安装现有四形状 wide 配置。两者均用相同转换 kernel、BF16中间舍入与scale1024/warps4。转换候选block/warps为1024/4、4096/4、8192/4、8192/8、16384/4、16384/8。

default测M2048/4096/8192/16384，两种子；wide测M2048，两种子。所有候选保留五轮交错计时与图重放，不择优隐藏控制。M2048另测独立转换，区分阶段与完整调用，阶段时间不相加替代完整调用。

## 4. 实验与验证

模型Shape固定Down N2048/K6144，M8192/16384为扩展诊断，不称已观测负载。复用same overlay但每个提供者用新进程，串行执行；切换前核对GPU释放、9031端口、环境。实际源文件哈希、注册模块、命令和日志归档。数值参考包含BF16中间舍入，门槛relativeL2<0.005。GPU结果已汇总，见下表。

### 实测结果与结论

default完成60条初筛和60条finalist，wide完成24条和24条。全部数值L2为0，图重放无失败，两进程退出码0，最终GPU无进程。

M2048完整Separate，单位μs，转换1024/4改为4096/4、scale不变：

| MM提供者 | seed | 原调用 | 新调用 | 延迟降低 |
|---|---:|---:|---:|---:|
| 官方默认 | 20261110 | 642.83 | 627.09 | 2.449% |
| 官方默认 | 20261111 | 643.80 | 628.46 | 2.383% |
| wide覆盖 | 20261110 | 475.446 | 461.183 | 3.000% |
| wide覆盖 | 20261111 | 477.508 | 460.791 | 3.501% |

两个提供者的完整调用均节省约14–17μs。默认MM更慢，因此相同比量级的绝对节省对应更小的百分比。独立转换default约57.6降至43.2μs，wide约55.3–55.6降至40.9μs；不同进程采样状态有差异，不将其跨进程绝对差归因为MM。

8192/16384转换block（warps4/8）没有在两种子和eager/graph中稳定超过4096，维持4096。M4096的4096转换完整调用改善约1.19%–1.25%。M8192的eager改善为+1.005%/-0.728%，graph为+0.164%/+1.538%；M16384 eager约+0.504%/+0.266%，graph约-1.941%/-0.067%。其他更大block同样有跨种子与模式分歧，不择优宣称有效改善；该波动未归因。

本轮不新增生产改动，保留Down N2048/K6144、113≤M≤4096使用block4096，范围外1024。大M诊断未证明稳定收益，不扩大范围。当前wide仅覆盖M2048，其余大M仍是默认MM，没有进行新BF16 MM调优。

转换候选可以复用，但完整Separate配置必须绑定MM提供者源码、配置及软件指纹；MM更新后复验代表M、边界和graph，不能沿用旧百分比。

## 5. 与已有改动的关系

承接Down block4096推广验证；当前生产仍为N2048/K6144且113≤M≤4096用4096，其他为1024。本轮不假定更大block更好，不将有限wide搜索称全局最优。不修改旧实验结果，不改BF16 MM实现。

## 6. 合规性与风险

转换仍计入完整调用，没有持久BF16副本。W8A16 BF16 dot＋FP32累加＋FP32scale＋BF16输出。官方serve/Benchmark和数据不改。有限micro结果不能替代模型精度、吞吐和TTFT；提供者版本变化须重新验证Separate。提升≤1%归正常波动，不能声称有效优化。未知提供者或范围外保留明确合法回退。

## 7. 复现方式

```bash
PYTHONPATH=/data/downonly-hotspots-20261110-01/overlay \
python benchmark_metax_convert_provider_cross.py --provider default --output <新目录>
PYTHONPATH=/data/downonly-hotspots-20261110-01/overlay \
python benchmark_metax_convert_provider_cross.py --provider wide --output <另一个新目录>
```

实际远端run.py按default→wide顺序启动，检查退出码与GPU释放。screen.jsonl/finalists.jsonl保留全部候选、五轮值、graph和数值误差；provider.json记录模式。完整MM配置效应仅在M2048对照，其余M没有wide覆盖，不虚报两组不同配置实验。

本机原始计时与元数据位于 `artifacts/down-convert-provider-cross-20261111-01/`，已取回两个提供者的screen/finalists/provider及hashes/environment。远端完整归档 `/data/down-convert-provider-cross-20261111-01-results.tgz`。本轮未提交、未推送。
