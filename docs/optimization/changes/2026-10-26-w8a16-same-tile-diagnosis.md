# Down 融合 GEMM 的同分块 BF16 诊断对照

## 1. 基本信息与摘要

日期：2026-10-26（UTC+8）。任务 ID：`w8a16-control-20261026-04`。用户明确批准“先做诊断对照”。

新增文件：`benchmarks/diagnostics/benchmark_w8a16_control.py`。不修改生产 Kernel、加载时量化、融合封装、分派或官方 Benchmark。沿用分支 `perf/gemm-w8a16-down-fused`、基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，保留已有未提交文件；未提交、未推送。

目的：区分当前融合实现的基础 GEMM 效率与量化融合带来的额外执行成本。结果表明两方面均有差距：同分块 BF16 对照本身慢于原 BF16；融合又慢于同分块对照。编译 IR 显示权重侧流水线发生变化，后端资源信息未显示 spill。不能将全部差值解释成纯 INT8 转换成本或寄存器溢出。

## 2. 问题与原因分析

前次九种 M 的完整路径补测已确认，当前融合在全部测试点慢于独立 Triton 反量化。仅把融合与 `F.linear` 比较，无法确定其差距来自基础 GEMM 分块、布局、流水线，还是反量化增量，因此新增同分块 BF16 对照。

对照直接读取 `BF16(FP32(q)*scale)`，与融合使用相同数学权重；保持逻辑 M/N/K 分块、M 分组映射、掩码、BF16 dot、FP32 累加、BF16 输出及编译参数。对照实际读取 BF16，而融合读取 INT8，两者存储流量和编译器生成代码不同，因此不是仅删除几条算术指令的严格等价机器码对照。

## 3. 具体实现

诊断 `_control_kernel` 复制生产融合的 GEMM 骨架，仅去除 scale 读取、INT8 转 FP32、乘 scale、转 BF16，改为直接加载连续 BF16 `[N,K]` 权重。

配置：M=1 使用 BM=16；M=64/256/2048 使用 BM=32；BN=64、BK=64、warps=4、stages=2、M 分组大小 8、`enable_fp_fusion=False`，与生产融合对应配置一致。真实 Down N=2048、K=6144，K 循环 96 次，无 bias。

四条完整路径：

```text
bf16:            F.linear(x, original BF16 weight)
control:         same-tile BF16 Triton(x, rounded quantized weight)
separate_triton: F.linear(x, Triton dequant(q,scale))
fused:           production fused_linear(x,q,scale)
```

另用预分配输出直接启动 control/fused Kernel，移除输出分配和融合自定义算子封装，以判断差距是否主要来自调用层。预分配输出不代表正式推理接口，仅为诊断。

每个 M、每条路径三轮，按两条路径的位置偏移轮换顺序，保存全部 72 条原始计时。每轮 `do_bench(warmup=100,rep=300,return_mode='median')`，汇总取三个中位数的中位数。独立量化预处理不计入推理计时。输出目录必须不存在，避免覆盖历史。

## 4. 实验与验证

环境：同一容器 `6951ed69dce3`、单卡 C500 64 GiB、bus `0000:35:00.0`；Python 3.12.11、PyTorch `2.8.0+metax3.7.0.7`、Triton 3.0.0、MACA 3.7.0.38、驱动 3.8.30。128 GiB RAM，swap 0，实际 import 根目录 `/workspace/vllm-plugin-FL`。

启动前保留进程/GPU/仓库状态和生产源码指纹，确认无 GPU 进程、显存 826 MiB、9031 可绑定。关闭模型量化接入开关，直接调用诊断路径，没有启动模型服务或官方评测。最初 SSH 两次在认证阶段断开，随后使用用户确认的相同信息重连成功；失败连接没有启动实验。

脚本 SHA256：`a31d14abb91aa49e3e101e7654dd848e7926feab2edf1dc8caffcee0ddbe9fba`，本机与远端一致。

### 完整路径 GPU 延迟（ms）

| M | 原 BF16 | 同分块 BF16 对照 | 独立 Triton＋GEMM | 融合 | 对照/原 BF16 | 融合/对照 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.032768 | 0.077056 | 0.079616 | 0.141568 | 2.35× | 1.84× |
| 64 | 0.052736 | 0.086528 | 0.099072 | 0.157952 | 1.64× | 1.83× |
| 256 | 0.075520 | 0.150528 | 0.119808 | 0.262144 | 1.99× | 1.74× |
| 2048 | 0.291328 | 0.832512 | 0.334848 | 2.018560 | 2.86× | 2.42× |

### 预分配输出、直接 Kernel 调用（ms）

| M | 同分块 BF16 Kernel | 融合 Kernel |
|---|---:|---:|
| 1 | 0.077056 | 0.141312 |
| 64 | 0.086272 | 0.156928 |
| 256 | 0.150784 | 0.261888 |
| 2048 | 0.833792 | 2.039808 |

去掉输出分配和自定义算子封装后，主要差距仍然存在，不支持“调用层造成主要性能损失”的解释。完整路径与预分配路径分别计时，微小差异不解释为固定封装开销；例如 M=2048 预分配融合略慢，不能据此得出分配会加速。

### 编译产物与资源

| M | BF16 对照 n_regs | 融合 n_regs | 对照 shared | 融合 shared | n_spills |
|---|---:|---:|---:|---:|---|
| 1 | 70 | 66 | 20 KiB | 12 KiB | 两者均 0 |
| 64/256/2048 | 102 | 86 | 24 KiB | 16 KiB | 两者均 0 |

以上为 C500 Triton 后端报告的字段，不是实测占用率。融合寄存器和 shared 占用反而更少，且未报告 spill；不能把本次差距归因于已证实的寄存器溢出。实际 active warps、带宽、stall 和硬件利用率尚未测量，不给出占比。

M=2048 的 TTGIR 证据：

- BF16 对照在循环外分配 `2x32x64` 激活和 `2x64x64` 权重双缓冲；循环内预取下一块 A/B，当前块以两个 K=32 的 BF16 dot 完成逻辑 BK=64 的计算。
- 融合只有 A 侧持久双缓冲；权重侧在 K 循环内执行 `tt.load i8 → sitofp → mulf → truncf bf16 → local_alloc/local_load → dot`，对应 dot 的 K 为 64。
- B 侧 blocked layout 也不同：对照 `sizePerThread=[8,1]`，融合 `[16,1]`；两者均标记 `maca_mma`。

这说明同样的源码分块和 stages 并未生成同样的 B 侧流水线。**权重转换及其依赖链、流水线/布局变化是下一步应验证的方向**；目前未通过消融证明哪项占主导，也没有最终机器指令级反汇编或硬件性能计数器结果。归档保存 TTIR/TTGIR 和资源 metadata，编译对象列出 `mcfatbin`，但本脚本未保存或反汇编其二进制。

### 正确性与完整性

- 新对照先通过 `(M,N,K)=(1,17,33)`、`(64,129,257)` 掩码正确性检查。
- Triton 独立反量化权重与 reference 逐元素相等，四种 M 的独立输出与舍入权重 `F.linear` 完全相等。
- 四种 M 的 control/fused 相对舍入权重参考的 L2 误差均低于 0.005；实际误差约 `5.64e-5～2.43e-4`。两者各自误差相同，不据此声称输出已逐元素比较完全相等。
- 退出码 0；本地核对四个 Shape、六条路径、每项三轮共 72 条计时，中位数与汇总完全一致。
- 结束后测试进程已退出，GPU 无进程、利用率 0%、显存 826/65536 MiB，9031 可绑定，见 `postflight.txt`。

## 5. 与已有改动的关系

依赖已通过动态编译修复的生产融合接口，不修改其算法或性能配置。前次 `2026-10-26-w8a16-triton-microbench.md` 的历史数据保留。本次融合绝对耗时与前次不同，例如 M=2048 从约 2.498 ms 到本次 2.019 ms，生产 Kernel 未改，输入随机序列及测量时段也不同，因此不将其解释成优化收益。

本次各 M 的融合三轮计时在本轮内较接近；后续候选仍需同场对照及重复验证，不能与旧轮次直接拼接收益。新增对照仅用于定位，不接入模型，也不作为已验证优化策略。

## 6. 合规性与风险

仅做独立 GEMM 诊断，不改正式启动/评测参数、数据、模型输出或量化范围。无并发 GPU 测试、模型服务或 Benchmark；未清理持久编译缓存。

当前结论是基础 GEMM 与融合流水线均需改进，尚未找到有效配置；不宣称性能提升。量化仍为 Down-only INT8 存储、FP32 per-channel scale、BF16 激活、反量化后 BF16 dot、FP32 累加；没有调整舍入顺序。

建议下一阶段围绕 B 侧转换与预取流水线、BF16 GEMM 基础分块分别做消融，再决定配置搜索范围；小 M 专用实现可单独讨论。生产调优和分派更改需要后续计划确认，本次未实施。模型精度、TTFT 和完整官方验收未执行。

## 7. 复现方式

远端：`/workspace/benchmark_results/w8a16-control-20261026-04/`。

本地：工作区 `artifacts/w8a16-control-20261026-04/`，原始归档 `remote-evidence.tar.gz` 已下载展开。`results/summary.json`、`raw_timings.json`、`metadata.json` 保存结果及口径；`*-resources.json`、`*.ttir/ttgir` 保存编译证据；`audit.json` 保存本地完整性核对。`preflight.txt`、`run-environment.txt`、`run.log`、`exit.txt`、`postflight.txt` 保存版本、命令、退出和资源状态。

加载已有 Conda/MACA、确认资源空闲后：

```bash
export PYTHONPATH=/workspace/vllm-plugin-FL VLLM_PLUGINS=fl
export VLLM_FL_W8A16_DOWN=0 VLLM_FL_W8A16_DOWN_FUSED=0 VLLM_FL_W8A16_OFFLINE=0
unset VLLM_FL_W8A16_CHECKPOINT
python /workspace/vllm-plugin-FL/benchmarks/diagnostics/benchmark_w8a16_control.py \
  --output-dir /workspace/benchmark_results/w8a16-control-20261026-04/results
```

以上目录已存在，重跑需使用新实验 ID，不覆盖历史。语法和 diff 空白检查通过；GPU 正确性及完整诊断运行本身作为本次必要验证，不重复执行模型级测试。
