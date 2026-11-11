# MetaX NT W8A16 dtype 对照恢复与编译证据

## 1. 基本信息与摘要

2026-10-28，独立任务ID `flaggems-w8a16-dtype-nt-recovery-20261028-08`。恢复已批准但未完成的NT-only归因诊断，沿用 `benchmark_w8a16_nt_dtype_control.py --nt-only`，不运行之前触发编译器断言的generic BF16对照。复用大M NT候选主体、同一FP32 scale epilogue，INT8输入与预转换的BF16(Q)做同场比较。不改生产算子，无新增开关，未提交推送。

## 2. 问题与原因分析

大M融合仍慢于独立反量化，需区分数据值、scale位置、加载dtype和编译流水线的影响。BF16(Q)是INT8整数的精确表示，不乘scale；与INT8路径数学及缩放顺序一致。该控制用双倍权重存储，转换在计时外，只用于诊断，不是量化部署收益。

## 3. 具体实现

使用已存在的NT适配代码，两条路径均分配BF16输出、同样FP32 dot及scale-before-output epilogue，唯一输入控制为Q INT8或BF16(Q)。固定BM64/BN64/BK64/warps4/stages3/pipeline cpasync，M256/2048、N2048/K6144，seed281028。每条路径正确性先验证，再三轮轮换do_bench(warmup100ms,rep300ms)，保存全部原始计时、资源和IR。

## 4. 实验与验证

本次容器1eb2caf770fd、GPU bus0000:ac:00.0，C500 64GiB、driver3.9.35、MACA3.7.0.38，PyTorch2.8.0+metax3.7.0.7、Triton3.0.0，使用已有Conda/MACA与FlagGems源码路径。小M任务完成且资源释放后串行运行，单独preflight/postflight。非旧实例同场数据不拼接绝对收益。

| M | INT8路径（微秒） | BF16(Q)控制（微秒） | INT8/BF16(Q)耗时比 |
|---|---:|---:|---:|
| 256 | 246.016 | 135.168 | 1.820 |
| 2048 | 2482.944 | 648.448 | 3.829 |

两种dtype对FP32 dot后scale参考relative L2分别约6.75e-5、7.32e-5，数值检查通过。退出0，12条计时、两shape中位数逐项核对一致。未测模型精度或正式bench。

### 关键编译差异

INT8路径82寄存器、0 spill、shared8192字节；BF16(Q)为66寄存器、0 spill、shared49152字节。两个M资源相同。

BF16(Q) TTGIR包含A、B的 `3x64x64xbf16` shared环形缓冲和 `async_copy_global_to_local`，可见三阶段异步组织。INT8路径没有相同的三缓冲/异步拷贝结构，循环内global load INT8后转换BF16，再经local_alloc/local_load组织dot；虽然metadata同样标注cpasync/stages3，实际IR并未实现相同流水线。资源数不能当成同配置就同算法执行组织的证明。

这支持“混合dtype触发了不同流水线/布局lowering”，比仅猜转换算术贵更具体；但不是硬件stall采样，也不能将全部耗时差量化归因于异步流水线一个因素。完整机理和可修复性仍需后续单项改造验证。不将大M差异直接套到小Mbasic双缓冲路径。

## 5. 与已有改动的关系

上轮generic BF16 cpasync触发validMaskSwizzle断言，退出134的记录保留，本次仅NT分支退出0，不声称失败分支已修复。旧主实验105条计时已恢复归档；本次独立归因对照与小M整除特化结果分别记录，不覆盖旧日志。

## 6. 合规性与风险

BF16(Q)控制不是完整W8A16部署，不计预转换开销且权重存储翻倍，不能当作实现已加速。scale-after-dot契约不变，不涉及模型或官方参数修改。当前可以排除“相同配置字符串保证相同三缓冲”的假设，尚未获得可接入的大M融合优化。最终仍需数值、图编译/图捕获、模型精度、吞吐/TTFT验收。

## 7. 复现与归档

远端与本地目录标识一致；本地 `artifacts/flaggems-w8a16-dtype-nt-recovery-20261028-08/` 保留脚本/helpers、指纹、环境、preflight/postflight、run.log、exit、完整raw/summary/metadata、IR/资源及audit.json。运行方式：

```bash
export PYTHONPATH=/workspace/FlagGems/src
python benchmark_w8a16_nt_dtype_control.py --nt-only --output-dir results
```

沿用小M任务已核对的同一shell环境，四个历史W8A16开关已清除。环境继承记录与helpers SHA分别保存；运行后GPU无进程、util0、显存860MiB、9031可bind，两生产仓库clean，SSH已退出。
