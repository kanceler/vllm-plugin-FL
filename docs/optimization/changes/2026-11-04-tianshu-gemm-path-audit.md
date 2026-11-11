# 天数 BF16 GEMM：第一步调用路径与配置审计

## 1. 基本信息与摘要

日期：2026-11-04；平台：天数 BI-V150 / CoreX 4.5.0。只读检查已安装源码、队友工作目录和既有编译/日志/缓存，没有修改源码、运行 GPU 测试或重跑 Benchmark。本文独立于沐曦实验。

队友基线 `/workspace/tianshu-final-20261104` 提交为 `f82fa1670509f081aa13b58464cb6e5627fc10c4`，检查时工作树干净，无 vLLM 服务。历史服务环境 PYTHONPATH 指向该目录；交互 shell 默认 vllm_fl 来自 site-packages，后续不能遗漏该差异。

## 2. 问题与原因分析

目标是识别四类 GEMM 实际编译路径及后处理边界，并检查是否有不适配设备的配置。源码存在某实现、enabled-ops 中存在某名称、FX 图有几个节点，均不足以确定最终 GPU Kernel 和性能瓶颈。

此前基线审计报告把 GEMM 归为 vendor BLAS、小算子整体归为 FlagGems，证据强度过高。此次明确区分上层源码、生成调用、调优缓存和实际 GPU Trace。历史结论按下文收窄，不能据此声称已确定厂商库 Kernel 或热点比例。

## 3. 具体审计结果

### 3.1 GEMM 上层与编译路径

已安装 vLLM 源码：`UnquantizedLinearMethod.apply → dispatch_unquantized_gemm → default_unquantized_gemm → torch.nn.functional.linear`。非 CPU/ROCm 的 default 分支不经过此前报告所述通用 `_POSSIBLE_KERNELS` 选择；后续仍须与服务实际 import 的源码版本绑定。

既有实验 `/workspace/experiments/tianshu-final-20261104-01/` 中两个生成产物都包含四次外部 MM 调用：

| 产物 | SHA256 |
|---|---|
| artifact_compile_range_1_2048_subgraph_1 | e6538f14083f44be45542c15afef7fc2cac53f616264fbe80259975938ab4d17 |
| artifact_compile_range_2049_4096_subgraph_1 | 436889e35fa037919515ef88a29edc05cd5bce7c315f09906c57a58efb352092 |

| 投影 | MM 输入 / 转置权重视图 | 后处理生成调用 |
|---|---|---|
| O | [M,2048] / [2048,2048]，权重视图 stride [1,2048] | triton_red_fused_fused_add_rms_norm_0 |
| Gate/Up | [M,2048] / [2048,12288]，stride [1,2048] | triton_poi_fused_mul_silu_slice_1 |
| Down | [M,6144] / [6144,2048]，stride [1,6144] | triton_red_fused_fused_add_rms_norm_2 |
| QKV | [M,2048] / [2048,2560]，stride [1,2048] | triton_poi_fused_3，旋转后输出Q/K，V保留视图 |

四者均调用 `extern_kernels.mm`，BF16 输出。这里的转置是 reinterpret/view，不能当作实际转置复制。未在该 Runner 主调用链见显式 contiguous/copy；不代表其他分支无复制。

生成产物包含初始化/预热代码和 Runner 调用，字符串 `.run` 计数不能当作每请求启动数。产物示例值 M=4096 不等于范围1～2048中实际负载 M=4096；动态变量 s72、compile range、warmup example 和真实执行 M 必须区分。

### 3.2 激活与残差数值数据流

Gate/Up Kernel 从 `[M,12288]` 的前6144与后6144读取两半，执行 `float32(gate) * sigmoid(float32(gate)) * float32(up)`，最终写BF16。已生成代码没有独立 SiLU BF16 中间存储。后续融合不能盲目插入额外 BF16 cast，也不能仅凭 FX 中间 dtype 判断最终数值顺序。

O 后 Kernel 读取 O 输出与 residual，在 FP32 中相加并计算 RMS，输出 BF16 normalized activation。Down 后 Kernel 同时读取 Down 输出、O 输出与传入 residual，在 FP32 中重建残差链路，写出 residual 及下一层 normalized activation。不能简化为“每个投影后各独立 residual add”，也不能把 residual 移进 GEMM 后默认与当前跨算子编译结果等价。

上述仅为指定生成分支的表达式记录，未做数值误差测量，也不据此认定当前实现存在精度错误。

### 3.3 tile 与调优缓存

生成 Kernel 的 DeviceProperties 为16个 multiprocessor、warp_size=64、cc=71；当前证据没有显示它使用 Intel 设备元数据。队友插件及已安装 FlagGems Iluvatar 后端 Python 搜索未发现 Intel 字样，此搜索不覆盖全部依赖，不能单独证明配置完全适配。

当前 GEMM 是外部 MM，因此 FlagGems Iluvatar `mm.py` 中的 `runtime.get_tuned_config("mm")` 不自动控制该路径。现阶段没有证据支持“修改 FlagGems MM tile 就能优化当前 BF16 模型”。外部 MM 的实际GPU实现及tile仍待Trace/提供者证据。

当前编译缓存 `torch_aot_compile/f059778b472ac470c8eb00151cd0476d48577fc12fd7b44af0aa443a580b69fc/inductor_cache/` 存在 best_config：pointwise记录有XBLOCK=512、warps=4、stages=1；归约记录包括XBLOCK=2、R0_BLOCK=2048、warps=16，以及XBLOCK=1、R0_BLOCK=256、warps=2。目录级记录尚未逐个绑定全部Kernel及M，不能将任一条直接称为某投影最终配置。

`time_taken_ms` 为缓存记录字段，不作为Kernel执行耗时。16 warps 本身不是错误证据，需结合warp64、寄存器和实际性能判断。

### 3.4 量化与图捕获

历史服务配置 quantization=None、quantization_config=None、TP=1；本次不启用量化。W8A16源码存在不等于实际调用，尚无天数W8A16收益。

历史日志有FULL Decode图捕获进度，以及FlagGems libentry警告：后端未提供 replay benchmarker，回退 event timing。此警告针对调优计时能力，不能解释为“推理CUDA Graph失效”。它提示后续micro需单列实际图重放测量，不能把event计时当作replay计时。

## 4. 实验与验证

仅只读验证，没有新性能或精度数据。两个compile range确认相同外部MM与融合后处理边界；已核对commit、工作树、服务状态、历史启动环境及生成设备属性。

待补充：Decode replay实际GPU Kernel、四投影实际M分布、外部MM底层提供者与tile、Kernel耗时占比、best_config逐Kernel关联、其他子图/层边界和回退。现有产物不足以分别宣称Prefill与Decode全部覆盖。

## 5. 与已有改动的关系

补充并收窄 `2026-11-04-tianshu-final-baseline-and-dispatch-audit.md` 的归属结论，不改变其历史官方基线。执行入口为 [天数独立计划](../plans/2026-11-04-tianshu-gemm-audit-and-experiments.md)。不改沐曦实现、配置或结果，不直接移植其算法分界。

## 6. 合规性与风险

没有改官方启动参数、Benchmark、数据、依赖或Kernel。尚未发现可直接修复的已确认Intel tile错误。当前无法依据生成代码排出新的耗时瓶颈排名，不作性能收益结论。

第一步静态/归档审计已完成；实际负载覆盖仍有缺口。最小补采应在新实验目录按环境清理规则，使用官方参数的新服务，采集短时Prefill/Decode代表窗口，并关联CPU MM Shape与GPU Kernel；诊断负载与官方成绩分开。现有环境检测未找到ixprof/nsys/ncu，但这不排除PyTorch profiler可用，能力须另行验证。补采不需要跑完整官方八轮。

## 7. 复现方式

读取指定实验目录下两份 `artifact_compile_range_*_subgraph_1`、serve.log、environment.txt与serve-command.txt。生成产物为二进制封装，审计只提取可读字符串，不执行或反序列化其中代码。核对上层源码 `/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/{linear,utils}.py`。

本地生成产物与日志已存在于 `experiment-artifacts/2026-11-04-tianshu-final-01/remote-evidence/`。缓存配置为本次远端只读观察，尚未完整下载归档，不将其列为已完成的逐Kernel配置清单。

后续实施须先讨论准确改动文件与验证计划并取得同意。本次未提交或推送Git。
