# MetaX M2048/M56 扩展搜索与实际调用链修正（进行中）

## 1. 基本信息与摘要
2026-11-10，C500。用户批准扩大M2048四类BF16和M56 GateUp/QKV/O三算法搜索，Down M56保留对照。修改诊断search_metax_hotspots.py、插件int8pack.py、FlagGems weight_int8pack_mm.py，未提交或推送。

## 2. 问题与原因分析
远端入口审计：flag_gems.mm.__module__为_metax.ops.mm，weight_int8pack_mm.__module__为flag_gems.ops.weight_int8pack_mm。原模型实际调用通用Kernel，不能将既有所谓三算法/全Separate结果视为真实算法消融。性能原始事实保留，其解释撤回。完整路径导入专用模块不等于替换注册入口。

## 3. 具体实现
shape dispatch显式开启时，插件量化custom op调用MetaX专用实现；其他模式保持公开入口。Separate改用flag_gems.mm作为实际注册提供者，避免双重导入绕过tiling覆盖。不改变BF16主算子或官方命令。
新增离线脚本：M2048 BF16扩展BM/BN=32/64/128/256、BK=32/64/128、warps=4/8/16、stages=1/2/3/4、basic/cpasync；超过192KiB估计分阶段存储的组合剪枝，不沿用性能模型top-k。M56扩展Dense/Split-K配置及split=2/4/8/16/32，Separate和当前dispatch作对照。初筛后前8及对照五轮交错复测，正确性relative L2<0.005。候选原始失败和计时逐项落盘，仅生成provisional finalists，不自动启用未验证配置。

## 4. 实验与验证
本地AST检查通过；py_compile因macOS默认缓存目录不可写失败，已用不写缓存的AST检查。GPU测试、启动配置实传、正确性、图重放、实际模型Kernel命中、模型精度均待验证。不得宣称收益或完成固化。

## 5. 与已有改动的关系
在现有shape dispatch分支续做，保留已有未提交工作。参考d61ee2f提供者配置；Separate调优必须绑定最终相同BF16提供者，provider变化后复测边界。

## 6. 合规性与风险
不改官方参数、数据和Benchmark，不扩至其他算子。有限候选最优不等于全局最优。M56 Down不重搜；M2048 Down仍搜索。新接入必须先Kernel验证再模型运行。直接搜索包含资源失败可能，保留日志；无trace不宣称实际图重放命中。

## 7. 复现
本地脚本benchmarks/diagnostics/search_metax_hotspots.py，参数--stage bf16或w8a16 --output新目录。远端工作目录/data/metax-hotspot-search-20261110-01/。BF16阶段先行，量化阶段待BF16提供者版本确定后同机对照；后续再生成版本绑定manifest及固定接入，当前尚未完成。

## M56量化阶段启动
用户再次确认只搜索W8A16路径的M56，允许必要算法修改。远端新目录/data/metax-m56-search-20261110-01/，复制原安装包与既有量化实现形成独立overlay，覆盖修正后的插件int8pack入口及Separate注册BF16提供者调用，加载官方A/B已验证的新BF16 NT配置。该NT固定策略仅作用M2048，M56 BF16基线仍沿原合法分发；不能声称BF16 M56已重新最优化。
脚本新增--provider-override显式启用已验证提供者；所有合法算法各保留两个初筛领先者进入五轮复测，避免全局top8排除某算法。GateUp/QKV/O联合枚举Dense、Split-K配置以及Separate完整调用；Down M56暂不重搜。数值门槛relative L2<0.005，BF16基线对原始权重参考、量化Kernel对量化参考分开。当前搜索仅eager，图重放、跨seed与模型接入待验证。允许算法优化不意味着预先认定现有候选已胜过BF16。静态AST检查通过，GPU数据待更新。

## 历史去重审计与当前初筛（18:37）
历史M56 QKV/O checkfree赢家均为(64,64,128,4,1,16)，Down为(64,64,128,4,1,8)，有两seed五轮交错记录；模型入口接入问题不能否定这些独立算子测量。历史Dense O M56也已终测，没有翻正。控制项索引保存于experiment-artifacts/2026-11-10-m56-history-audit/historical-controls.json，不能将历史时延直接移入本轮排名。
本轮GateUp已计时2395项、失败288项、剪枝432项，尚未完成；初筛最快Dense(64,64,64,8,4)为90.64µs，BF16为65.71µs，未胜出；Split-K已测最好约94.82µs，Separate约156.66µs。均为初筛，不是五轮finalist结论。后续应对新增候选/算法变化做验证，而非把相同控制项反复测量说成新增优化。
