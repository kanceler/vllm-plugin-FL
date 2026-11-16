# FlagGems跨平台源码定向整合

## 1. 基本信息与摘要

2026-11-16，分支 `perf/quantization-integration`。目标设备为MetaX C500和Iluvatar BI-V150；本次仅在本机完成源码整合与CPU静态核对，未在设备上运行。涉及仓库内 `FlagGems/`、根目录来源清单及 `benchmarks/diagnostics/audit_integrated_flaggems.py`。

采用完整已确认沐曦配套提交为基础，定向加入天数Iluvatar scale MM，使两平台的既有提供者源码共存。不合并最新上游，不修改插件量化实现或量化模式默认值。

## 2. 问题与原因分析

天数配套快照不能覆盖沐曦快照：对比13d3db1与47fb427，MetaX的MM融合实现、weight_int8pack_mm及分发表均存在差异，其中后者在天数快照中缺失。直接覆盖会丢失沐曦已采用的量化优化。另一方面，天数W8A16需要Iluvatar的scale MM提供者，单独复制沐曦快照也不完整。

提供者身份参与离线策略及缓存绑定。改变MM字节后仅更新hash会造成旧计时和新代码混用；本次要求首先保留对应已测提供者的准确字节。

## 3. 具体实现

基础为FlagGems提交 `13d3db134ab01e2495e17ac05aac1b49e07fd5a2` 的完整git archive，包含MetaX三算法、scale MM与分发表。仅用 `47fb4278f7d8bac5e48d22f4a6851af6e1186895` 中 `_iluvatar/ops/mm.py` 替换对应基础文件。

AST对比确认该文件原有函数/类全部保持不变，新增函数仅 `mm_kernel_scale`。MetaX与Iluvatar由原设备后端选择，源码路径独立，未发现需要新增公共接口或开关的冲突。天数自研Split-K和hotspot仍在原插件中，本次没有迁移或冒充已接入。

根目录 `flaggems-source-manifest.json` 记录基础提交、覆盖提交、覆盖文件及全部4782个文件的SHA256。该清单用于源码完整性审计，不作为新运行版本已获性能验证的声明。

## 4. 实验与验证

| 核对项 | 结果 |
|---|---|
| 全快照文件hash | 4782/4782匹配 |
| Python AST | 4176个文件通过 |
| Iluvatar原函数AST | 全部保持一致 |
| 新增Iluvatar函数 | 仅mm_kernel_scale |
| 天数MM提供者SHA256 | fd120991ed1c5cab988d02cbdb1d97ccde1e9e2eedad55affe90321d875f32ed，与历史绑定一致 |
| GPU数值、编译及Graph重放 | 本次未做，待验证 |
| microbenchmark与官方模型评测 | 本次未做，无新增性能数据 |

审计不导入Torch、Triton或FlagGems，不创建pyc，不读取/删除运行时调优数据库，也不启动GPU任务。历史micro和模型成绩仍归属各自实验，不自动成为该组合快照的成绩。

## 5. 与已有改动的关系

保留沐曦W8A16三算法、共享候选、转换和输出融合，并提供天数历史W8A16所需scale接口。静态KV本地配套来源为4ae85e2，兼容实验实际安装版为4df52d91；当前未将它们替换为本快照后重新验证，因此不能继承静态KV或兼容版精度。原工作树和归档不修改。

## 6. 合规性与缓存风险

不修改官方serve、Benchmark、模型和数据。不改原BF16主体，不删除框架主要分发，不为获取性能合并最新上游。

缓存审查重点：

- MetaX原显式JIT启动保留，其完整warps/stages等参数参与Triton缓存；不退回可能省略启动选项的旧缓存路径。
- 保留MetaX提供者及Kernel字节、分发表；保留天数fd120991提供者身份，不以改hash绕过策略校验。
- 插件cache identity、量化开关和联合KV图ABI尚未在本分支接入；源码快照存在不代表加载模式生效，后续须逐项核对。
- 原进程中的provider校验及lru_cache不热更新；后续切换源码或模式必须新进程、新服务并核对实际import。
- 不清理磁盘Triton/JIT缓存、不读取旧计时冒充新测；冷热缓存micro分开报告，官方运行不插入清缓存。
- 不维护持久BF16权重副本；新增接口不等于量化转换可以移出计时。

设备上仍须做实际import、编译metadata、同provider BF16对照、完整调用数值和Graph复验。未完成前不宣称组合可部署或有性能收益。

## 7. 复现方式

本机CPU审计命令：

```bash
cd /Users/tianfr/muxi_2b/worktrees/vllm-plugin-FL-quantization-integration
python3 benchmarks/diagnostics/audit_integrated_flaggems.py
```

预期输出包含files_verified=4782、provider_identity_matches=true、gpu_validation=not_run、runtime_caches_touched=false。脚本检查完整hash、额外文件和Python语法；发生新增pyc等额外文件时显式失败，不静默把缓存写入发布清单。

重新构建快照时，从已确认FlagGems仓库对13d3db1执行git archive，并仅覆盖47fb427中Iluvatar mm.py；保留原提交供追溯，不能复制含未提交差异的工作树。来源清单中的所有hash为本次实际生成。GPU复现命令需待插件接入及设备环境确认后补充，本次未编造未执行的GPU验证结果。
