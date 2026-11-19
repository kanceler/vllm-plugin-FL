# 天数 BF16 热点配置统一开关

## 1. 基本信息
天数 BI-V150，submission 整合分支。两份 FlagGems 的 Iluvatar ops 新增 `bf16_hotspots.py`，由 `ops/__init__.py` 按开关加载。

## 2. 问题与原因
将已有 Down M2048 固定分支与另外六个 BF16 NT 测点统一管理为可选配置。

## 3. 实现
启动前设置 `FLAGGEMS_ILUVATAR_BF16_HOTSPOTS=1` 启用。精确匹配形状、BF16 dtype 和 NT stride 后选择实测配置。大 M 四点采用 256/256/32、16 warps、2 stages，Gate/Up 分组 4，QKV 和 O 行优先，均为整除写回。M64 O 和 QKV 使用原 MM 主体，采用 64/64/128、16 warps、2 stages。Down M2048 复用原 MM 主体，采用 256/256/32、16 warps、2 stages。七点均为 Split-K=1。默认关闭，全部七点回到原自动调优。开启但不命中时也使用原自动调优。

## 4. 验证
本地源码解析、镜像文件比对、完整性清单检查及 Git 空白检查。先前隔离诊断覆盖六个候选，整合后的 GPU 数值、图重放与模型验证待执行。已有计时及逐轮模型数据见工作区 `output/BF16热点配置与消融交接.md`。

## 5. 与已有改动的关系
将 Down 固定分支移至开关模块后，`mm.py` 恢复至整合分支 HEAD 的字节内容及历史 W8A16 提供者哈希。沐曦路径及 W8A16 策略清单沿用现有源码。

## 6. 合规与风险
维持官方启动与 Benchmark 参数。变更开关需重启服务。当前为本地工作树修改，未提交或同步容器。

## 7. 复现
执行插件目录下 `benchmarks/diagnostics/audit_integrated_flaggems.py` 与 `benchmarks/diagnostics/audit_quantization_integration.py`。模型验证时分别在未设置开关及开关为 1 的独立服务进程运行相同官方流程。
