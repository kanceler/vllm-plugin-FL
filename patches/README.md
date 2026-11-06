# 天数 GEMM tiling #001

插件基础：`s2-track2-final` @ `f82fa1670509f081aa13b58464cb6e5627fc10c4`。
测评FlagGems基础：`4df52d9168ac55c1c1061cefb4638c570292d896`；原始Iluvatar mm.py SHA256为`4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090`。

本仓库保存跨仓库算子补丁，插件源码未因此自动修改。进入上述FlagGems工作树后，先执行 `git apply --unidiff-zero --check /absolute/path/to/001-iluvatar-down-fixed-tiling.patch`，再执行 `git apply --unidiff-zero /absolute/path/to/001-iluvatar-down-fixed-tiling.patch`。运行前确认实际import使用应用补丁的FlagGems，而非旧安装包。

原状与候选均使用同一个插件基点。官方4K吞吐+1.21962%，16K+0.011297%（正常波动）；模型命中覆盖和精度待验证。详细证据及限制见docs/optimization/changes中的#001说明。
