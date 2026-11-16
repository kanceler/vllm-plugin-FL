# 改动 #014：BI-V150 V-only缓存量化

> 平台：天数 BI-V150；模型：MiniCPM5-2B。编号接续#013，表示技术演进，不代表历史实验已按此顺序在perf/BI-V150形成独立提交。

## 1. 为什么这样改

V保持BF16增加缓存与读取开销，先单独量化V并建立后续对照。

## 2. 改动内容

K保持BF16；V按token/head动态对称量化为INT8，FP32 scale作用于V，转换BF16参与PV，FP32累加。每token/head缓存388字节。

V写入时按每个token和KV head计算scale并量化；读取时在Attention内将INT8 V乘scale后转BF16，避免保留完整BF16 V副本。K仍使用原BF16路径。scale-on-V改变了相对先缩放概率P的舍入位置，必须按对应参考独立验证；仅Attention＋reduce的诊断不包含KV写入。

## 3. 开关、适用范围与Graph安全

仅适用记录中的已验证设备、模型与布局；未知规格使用合法回退或明确拒绝。切换缓存格式须新服务及图捕获状态。

历史V-only服务使用 `VLLM_FL_INT8_KV_REPAIR=1` 与 `VLLM_FL_V_ONLY=1`，默认关闭；缓存采用uint8原始页，实际每token/head 388字节。加载后不可在原服务上热切换；必须核对实际backend，按原实验的BF16模型和head配置执行。

## 4. 收益与正确性验证

独立FP32参考10项通过；math_500报告0.981、105请求成功，输出未全面审计。两轮诊断第二轮4K 3282.66 tok/s、TTFT 5101.54 ms；16K 2748.25 tok/s、TTFT 142902.05 ms。并非四轮官方summary。

### 独立算子诊断对照

下表是16K合成输入、20次交替图重放的Attention＋reduce中位数，未计写入；仅用于定位scale位置，不是最终完整调用或官方性能。

| batch | BF16 ms | 原V-only ms | scale作用于V ms |
|---:|---:|---:|---:|
| 16 | 1.01402 | 1.10375 | 0.85407 |
| 40 | 2.34976 | 2.58911 | 2.11616 |
| 64 | 3.68096 | 4.11825 | 3.35875 |

实验改变舍入顺序，不逐位等价；完整模型结果见上方，不能把这个算子差异直接当吞吐收益。

## 5. 复现与已有改动的关系

后续K量化版本不能继承该精度。

实际16K与4K两轮原始日志及命令分别归档在工作区 `experiment-artifacts/2026-11-15-tianshu-scalev-two-rounds/` 和 `experiment-artifacts/2026-11-15-tianshu-scalev-4k-two-rounds/`；精度和算子诊断在 `experiment-artifacts/2026-11-15-tianshu-vonly-16k-01/`。使用原始隔离源码及同一provider，先检查设备、环境、GPU和端口，再使用新目录、新服务；不复用已有结果目录。

### 完整命令与历史环境

以下为原实验记录的完整命令块，按原记录顺序保留，部分命令属于诊断、后续建议或不同尝试，须结合上文实验状态判断，不能全部视为已执行通过。命令依赖当时隔离源码和归档环境；本分支当前尚不包含这些实现。复跑时使用新的结果目录，绝不覆盖原目录。当前文档整理未执行这些命令。

```bash
cd /workspace/experiments/tianshu-vonly-20261113-02
/workspace/tianshu-mixed-env-20261109/bin/python -u \
 source/benchmarks/diagnostics/run_tianshu_vonly_validation.py "$PWD"
```

原始记录已复制到本仓库：[记录1](docs/optimization/evidence/quantization-history/014-1.md)、[记录2](docs/optimization/evidence/quantization-history/014-2.md)。副本来源和SHA256见同目录清单。

## 6. 合规性与保留范围

完整调用计入实际写入、转换、Attention和归约。独立算子不套用官方±1%规则；正式评测保持官方命令、四轮和后三轮平均。容量节省不等于吞吐收益。详细命令、环境、开关、原始日志及失败记录见[原始实验记录](../vllm-plugin-FL-tianshu-int8-kv/docs/optimization/changes/2026-11-15-vonly-scalev-results.md)。本说明整理历史证据，当前新分支仅复核文档，没有重新执行上述实验或官方测评。

## 整合版接入补充

三个final及既有静态KV兼容实现已在当前分支定向接入；本文前述数据仍是历史实验数据。整合版仅通过CPU静态与兼容测试，GPU编译、策略命中、完整调用和模型评测待验证。当前开关、加载路径、独立缓存身份及完整官方命令见[整合运行与复现](量化整合运行与复现.md)。历史命令与旧manifest保留，不当作整合版新测。
