# 三个final插件与缓存身份整合

## 1. 基本信息与摘要

2026-11-16，perf/quantization-integration。在天数#013底座上，整合muxifinal、天数W8A16 final、静态KV final及既有兼容工作树。配套FlagGems前一步已经定向整合。本次本机静态检查及CPU测试，不启动模型评测。

## 2. 问题与原因分析

两平台同名w8a16包具有不同Linear、参考与策略，直接覆盖会破坏某一平台。静态KV和#011–#013共同修改Attention安装链与平台入口，旧manifest还绑定公共路由源码；改变路由后不能继续声称旧hash或旧图有效。旧插件初始化没有新的跨平台路由，需要在模型注册前安装相应量化实现。

## 3. 具体实现

天数保留quantization/w8a16，沐曦移至quantization/metax_w8a16；integration按实际FlagGems vendor选择安装器。原投影开关及KV互斥保持，MetaX三算法仍调用对应设备后端；worker接入C500原生BF16配置映射。基点已有Prefill自适应预算与MetaX同步消除不整段覆盖。

Attention包采用已有兼容快照，保留#011分相、#012 Sampling和#013紧凑网格；仅应用静态KV的缓存分配增量到现有runner，不覆盖整个runner。保留V-only、动态top4、动态joint及静态KV；运行时所需的冻结诊断模块和配置一并补齐，同名不同诊断文件另存来源目录，不静默覆盖。

生成独立integration版KV manifest并保留旧文件：只更新整合源码绑定，原数值配置、校准与provider约束保留，显式标记GPU未验证。新的全源清单绑定插件与运行时依赖，初始化时逐文件验证，并将其身份、优化开关及Torch/Triton版本纳入VllmConfig和CompilationConfig缓存hash。旧天数W8A16的provider/Kernel核对继续保留，不能绕过策略失效。

## 4. 实验与验证

14项CPU兼容测试通过；覆盖密集前缀/尾padding、Mixed只写一次、launcher隔离、紧凑grid与合法回退。插件Python AST通过。静态KV、动态joint与旧top4的新清单本地绑定及候选源码hash一致；天数W8A16数值Kernel及MM provider字节保持原final。

GPU import、编译metadata、policy_matches、实际候选配置、数值与图重放均未在设备执行；没有新增micro、模型精度或官方性能。CPU mock不能作为GPU通过。原101/105、沐曦102/105及历史micro不自动属于整合版。

## 5. 与已有改动的关系

来源基点210c775、沐曦79684e9、天数W8A16 13efaa1、静态KV a28788f，以及兼容工作树未提交快照。完整文件hash见quantization-integration-manifest.json，FlagGems来源另见flaggems-source-manifest.json。说明#014–#025和整合报告已迁入，但历史编号不等于这些优化已在当前底座依次测过。

## 6. 合规性与风险

不修改官方serve、精度、Benchmark参数或数据。未删除框架主要选择逻辑，设备路径独立，未升级依赖或合并最新上游获取优化。不清理磁盘JIT/调优缓存，不热切换模式。import前必须核对PYTHONPATH与provider；版本不符应拒绝或合法回退，而不是更新hash假称已测。

本分支未完成实际GPU启动，仍可能存在不同镜像的ABI/依赖差异；需要两设备分别串行复验，不能宣称生产可部署。静态KV兼容旧精度已取消，不自动恢复。历史两个final中的根级报告与源码版本不同，整合版用新来源清单，不覆盖旧实验。

## 7. 复现方式

```bash
python3 benchmarks/diagnostics/audit_integrated_flaggems.py
python3 benchmarks/diagnostics/audit_quantization_integration.py
python3 -m unittest discover -s tests/compat_cpu -v
```

完整环境、开关、加载路径与官方命令见根目录“量化整合运行与复现.md”。本次没有执行其中GPU或模型命令。每次任务使用新实验目录、新服务，归档版本、开关、实际覆盖与原始结果。当前未提交或推送。

## 本轮天数验证补充

用户要求先放下沐曦验证。本轮天数方向完成原14项兼容测试及新增4项入口/缓存身份CPU测试，共18/18通过。新增检查包括：天数仅调用天数安装器、开关改变在新配置中改变缓存身份、重复安装不重复包裹、绑定源码改变时在安装hash前拒绝执行。天数W8A16 Kernel绑定与静态KV整合清单本地源码全部匹配，官方Benchmark保留RUNS=4、SKIP_FIRST=1。

本机未安装Torch/Triton，当前没有已确认可用的天数连接，未运行83项依赖真实栈的回归、实际import、GPU编译/配置命中、图重放及模型评测。上述18项为CPU mock/逻辑测试，不能代替设备验证。详情见 ../evidence/quantization-integration-validation/local-results.json。

## 推送前文档更新

#020与量化报告第6.3节已补齐兼容分支的83项回归、49项静态算子检查和两场景各两轮模型诊断，按用户要求展示第二轮对官方基线的整体吞吐变化，保留全部原始轮次和micro负结果。该证据绑定兼容分支，不作为本整合分支GPU验证。组合精度按要求未测。

提交检查已核对新增文件AST、文件体积、私钥及临时连接信息，未发现相关敏感内容。`git diff --cached --check`报告的CRLF、尾空白和EOF空行来自保留的历史源码/CSV，包括FlagGems provider与静态KV冻结源码；为维持归档及运行时指纹，保留原始字节，不为格式检查改写。全源与FlagGems审计和18项CPU测试再次通过。
