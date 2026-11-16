# 静态KV配套版本说明

分支feat/tianshu-static-kv-final基于ea34f2e。保留已有天数BF16实现，本次没有新增FlagGems算子：静态KV Kernel和联合接入在插件同名分支内。未加入其他工作树未提交GEMM改动。

本提交固定分发依赖版本，未核对远端精度镜像整个FlagGems安装，不宣称配套组合已重新通过GPU验收。原静态KV服务绑定vLLM Attention提供者SHA256，复现保留匹配镜像。完整命令及限制见插件004-完整复现与版本说明.md。
