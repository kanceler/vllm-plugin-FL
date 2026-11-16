# 静态KV内附FlagGems配套源码

## 1. 基本信息与摘要
2026-11-16，同名静态KV分支；内附4ae85e260d9ceadf4da462b9ce0dff3ddf0aa2f0完整已提交源码，共4785文件。

## 2. 问题与原因
插件单独克隆原先需要另取配套仓库，容易误用版本。按此前muxifinal交付形式补齐。

## 3. 具体实现
git archive原提交，排除.git及未跟踪缓存；逐文件字节比对及SHA256。更新004/README/索引为内附FlagGems/src导入。没有改计算代码或校准。

## 4. 实验与验证
源码逐字节一致，命令语法检查，未新增GPU测试。历史源码CRLF/空白保留，避免改变源码指纹。没有新增精度或吞吐结论。

## 5. 与已有优化点的关系
静态KV Kernel仍在插件，配套FlagGems仅固定依赖，不引入W8A16提供者或Down组合，s2final基线不变。

## 6. 合规性与风险
无模型/数据/密钥/编译缓存，不升级依赖；仍须匹配现有镜像与vLLM provider，配套源码齐全不等于新环境验收。原镜像整体FlagGems未核对限制保留。

## 7. 复现方式
根目录004完整命令；PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/vllm-plugin-FL/FlagGems/src。确认实际import路径，版本及逐文件指纹见flaggems-source-manifest.json。
