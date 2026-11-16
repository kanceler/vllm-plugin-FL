# 天数W8A16已测BF16提供者整理

## 1. 基本信息与摘要
2026-11-16，BI-V150，配套feat/tianshu-w8a16-final；固定原微测mm.py，不开展新算子优化。

## 2. 问题与原因
策略严格绑定provider；系统版及其他固定Down配置版不同，曾导致separate_generic回退。须提供原字节用于复现。

## 3. 具体实现
原样复制tianshu-boundaries-20261113-01/gems中的mm.py，SHA256 fd120991ed1c5cab988d02cbdb1d97ccde1e9e2eedad55affe90321d875f32ed。相对ea34f2e移除后加固定Down入口，恢复原选择逻辑，加入原实验scale-after-dot Kernel；不是新增性能成果。保留公开BF16 MM与调优选择。

## 4. 实验与验证
provider哈希与插件manifest匹配，AST检查通过，无新GPU验证。历史性能、数值阈值及Down组合负结果见插件001/003/005，不重复声称完整模型已验收。

## 5. 与已有优化点的关系
Separate依赖该BF16提供者并融合scale，Split-K在插件内复用沐曦算术。更换provider后必须重测Separate及分界，不能绕过哈希。

## 6. 合规与风险
其他FlagGems文件基于ea34f2e，整套镜像软件仍须核对。未引入最新上游优化，未改官方命令；没有原生INT8×INT8计算或持久BF16副本。

## 7. 复现方式
插件004包含固定路径、import、sha256sum、9点验证、micro及官方命令。源码路径src/flag_gems/runtime/backend/_iluvatar/ops/mm.py。量化四开关默认关闭，s2final为模型基线，开关切换须新服务。

提交检查：AST及策略/提供者SHA256通过。原样证据CSV保留CRLF；diff --check还提示历史基线报告或provider原有末尾空行，为保持归档/提供者字节及哈希不格式化。没有新增数值/性能测试。
