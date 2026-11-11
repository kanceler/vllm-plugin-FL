# muxi-quantization 分支汇总

## 1. 基本信息与摘要

插件分支从origin/main的71f61488c5bd62855eac147d3dbad11af331b97b新建，独立工作树保留原开发目录。按用户要求纳入001～007说明、相关诊断和实现，以及完整FlagGems源码快照；FlagGems自身也建立同名分支提交。

## 2. 问题与原因分析

优化分散在工作目录、独立工作树及容器overlay，不能直接覆盖main中的无关框架代码。本机旧linear.py仍含四投影量化选择，需以实际Down-only参考整理，避免错误扩大量化范围。

## 3. 具体实现

迁入W8A16目录并在register_model注册；加载选择整理为明确Down-only，其他三投影BF16。启用已验证GateUp/QKV/O三条M2048 BF16 MM配置；Down使用自身融合区间策略。保留main其他框架实现。完整FlagGems快照位于FlagGems/，不含嵌套.git、编译缓存、模型或数据；它与独立仓库同一提交的源码一致。

FlagGems无本地main；其master参考与被测版本有数千文件差异，因此独立分支基于已验证f6a4257e3d042acd26a7d3c496848025a4b32c07，未引入最新上游优化。该仓库提交为13d3db1，完整标识和逐文件SHA256见quantization-source-manifest.json。

## 4. 实验与验证

Shape汇总unittest 10项通过；量化实现、诊断及改动调用入口AST检查通过；两份FlagGems共4782文件逐字节一致。敏感内容及大文件检查未发现目标凭据、私钥或大于10MiB文件。完整快照的diff --check报告既有CRLF、尾空白及历史Markdown硬换行；为保留原始源码字节和两份快照一致性，未批量格式化。新增量化代码与七篇说明另行检查。pytest在本机缺失，未安装依赖；可独立运行的unittest已通过。GPU不可在本机复验；main基点组合的GPU和模型验证仍待执行，历史micro结果以七篇原始证据为准。

## 5. 与已有改动的关系

原分支及脏工作树保留。子目录七篇说明按技术主线引用历史证据；此分支仅汇总，不将历史或容器正式结果当作本main基点组合已通过的结果。其他投影BF16配置收益不归为Down量化收益。

## 6. 合规性与风险

未修改官方参数、benchmark、模型或数据。未推送远端。原始实验大包和本机绝对路径不纳入源码快照；部分报告引用工作区artifacts，分享分支时需另行提供这些证据归档。提供者版本校验失配时沿原回退，main的软件依赖与被测镜像可能不同，正式运行前须重新核对。

## 7. 复现方式

安装本分支所附FlagGems与插件，核对实际import路径。使用原官方serve/benchmark参数；按文档设置Down/INT8PACK/SHAPE_DISPATCH量化模式。验证42层Down量化、其他投影BF16、三项原生MM专用配置，以及M193/385分界。CPU检查：python3 -m unittest discover -s benchmarks/diagnostics -p test_summarize_gemm_shapes.py -q。GPU操作前遵守根工作区环境清理规则。
