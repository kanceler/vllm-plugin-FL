# W8A16插件内附FlagGems源码快照

## 1. 基本信息与摘要
2026-11-16，feat/tianshu-w8a16-final。内附FlagGems完整配套源码，来源47fb4278f7d8bac5e48d22f4a6851af6e1186895，4786文件。无需队友另拉配套仓库。

## 2. 问题与原因
插件独立克隆不含匹配provider，误用site-packages可能触发策略回退而测错路径。内附源码使代码交付齐全。

## 3. 具体实现
以git archive导出指定提交，复制所有已提交文件，排除.git及未跟踪缓存。flaggems-source-manifest.json记录来源、提交和逐文件SHA256。004、README和索引改为插件内FlagGems/src优先导入。没有改算子或策略。

## 4. 实验与验证
4786文件与独立仓库逐字节一致，provider哈希fd120991与策略匹配，无大于10MiB文件，无目标凭据模式。命令语法和暂存完整性检查。快照保留原空白、CRLF及Markdown换行，不批量格式化，因此全量diff --check会报告历史空白；单独检查本次说明。没有新GPU测试，不增加性能或精度结论。

## 5. 与已有优化点的关系
仅固定W8A16 Separate及版本绑定依赖，不改变沐曦复用Split-K、天数配置、独立开关、s2final基线或Down组合负结果。

## 6. 合规性与风险
没有安装/升级依赖，没有嵌套Git、模型、数据或编译缓存。仍需匹配现有CoreX软件；内附源码不等于新环境GPU验收，禁止绕过策略校验。

## 7. 复现方式
按根目录004完整命令；PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/vllm-plugin-FL/FlagGems/src。确认flag_gems.__file__位于内附路径，mm.py哈希匹配；独立九点检查通过后再新服务。来源仓库https://github.com/srtdtv/FlagGems，分支feat/tianshu-w8a16-final。
