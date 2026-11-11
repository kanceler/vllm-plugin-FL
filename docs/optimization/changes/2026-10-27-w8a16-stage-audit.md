# W8A16 阶段诊断复核与修正

## 1. 基本信息与摘要

2026-10-27，MetaX C500。修正 `benchmarks/diagnostics/benchmark_flaggems_w8a16_stage.py`，只涉及诊断代码，不改 FlagGems 生产算子、注册、分派和开关，不运行模型或官方 Benchmark。沿用任务分支 `perf/gemm-w8a16-down-fused`，插件基点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。新实验标识 `flaggems-w8a16-stage-fixed-20261027-04`，不覆盖旧结果。

## 2. 问题与原因分析

旧脚本把 `perf_counter()` 的秒数保存为 wall_ms；raw 在每个 M 内重置，覆盖前面形状的三轮明细。独立 kernel 的事件计时不能相加成为完整路径，partial_share/reduce_share 不能解释为阶段实际占比；截断负差得到 launch=0 没有归因依据。wall fresh 与 prealloc 的差含 Python 调用和同步等因素，不能命名为纯分配成本。

## 3. 具体实现

wall 统一乘 1000 保存毫秒。raw 移至整个 shape 循环外累计保存。真实 Down 四个 M/S 增加预分配输出对 FP32 scale-after-dot 参考及 fresh 输出的断言，并归档正确性。删除误导性占比和 launch 派生字段，改为保留原始差值及不可做精确归因的解释。保持 M/S、tile、warps、stages、pipeline、计时轮数不变。

## 4. 实验与验证

本地 AST 语法检查通过，git diff --check 通过。补测环境：容器 `5e5ec8dfe1a6`，C500 bus `0000:58:00.0`，驱动3.9.35、MACA3.7.0.38，Conda Python3.12.11，48GiB主机内存。运行前GPU无进程、util0、显存860MiB，两生产仓库无差异；端口9031用socket bind核对（ss不可用）。未清理持久编译缓存。退出码0，60条计时（4M×5路径×3轮）完整归档且逐项重算中位数一致；8项正确性记录通过，四种真实shape预分配输出与fresh逐元素一致。本地/远端脚本相同。运行后GPU无进程、util0、显存860MiB。

## 5. 与已有改动的关系

旧 `2026-10-27-flaggems-w8a16-stage.md` 增加复核更正说明，保留历史数据；撤回阶段占比和分配/启动成本为0的结论。Split-K候选算法不变。

## 6. 合规性与风险

仅 Down N=2048、K=6144，M/S=(1,8)/(16,8)/(32,8)/(64,4)。INT8权重、FP32每输出通道scale、BF16激活输出、FP32 partial；无完整BF16权重反量化。数值测试不能替代模型精度。独立计时不能精确分离流水线stall、缓存或启动成本。没有硬件计数器证据，不宣称定位了具体硬件瓶颈。

## 7. 复现方式与编译证据

远端目录 `/workspace/benchmark_results/flaggems-w8a16-stage-fixed-20261027-04/`。实际命令：

```bash
export PYTHONPATH=/workspace/FlagGems/src
unset VLLM_FL_W8A16_DOWN VLLM_FL_W8A16_DOWN_FUSED VLLM_FL_W8A16_DOWN_OFFLINE VLLM_FL_W8A16_DOWN_CHECKPOINT
python benchmark_flaggems_w8a16_stage.py --helpers-dir . --output-dir results > run.log 2>&1
```

旧轮partial TTGIR确认A双缓冲 `2x16x128xbf16`、B双缓冲 `2x128x64xi8`。每BK128拆两个K64 dot，B从共享内存加载为dot布局后执行INT8到BF16转换；资源记录82寄存器、0 spill、24576字节shared。已有软件缓冲，不支持“未实现双缓冲”的判断；也不能仅凭缓冲存在证明访存与计算充分重叠。M=1的BM16有15行被mask，提示小M算力利用待优化；S=8时每CTA仅6次BK128循环，增加stage可能加重启动/收尾，须测量，不能盲目增stage。M=32/64增加M tile同时重复加载权重，需权衡更大BM的复用收益与寄存器/驻留成本。上述是后续候选分析，未改kernel、未验证收益。

### 补测结果与分析

以下为同场GPU事件计时，单位μs，三轮中位数；不能把独立阶段相加。

| M/S | partial | reduce | 完整fresh | 完整prealloc | BF16 MetaX |
|---|---:|---:|---:|---:|---:|
| 1/8 | 41.472 | 12.032 | 44.544 | 44.288 | 41.984 |
| 16/8 | 43.776 | 13.056 | 46.848 | 46.592 | 41.984 |
| 32/8 | 60.160 | 13.568 | 63.744 | 63.744 | 50.944 |
| 64/4 | 97.792 | 13.568 | 101.632 | 101.632 | 72.448 |

预分配在GPU事件计时仅改善0～0.256μs，但同步wall fresh/prealloc差值约32～178μs（每项仅5次中位数，原始wall样本未保存，不作稳定成本估计）；这些差值不能解释为纯分配成本，更不能推导到图捕获或模型推理。M=32/64 partial单独耗时已大于BF16完整GEMM，因此单纯去除reduce不足以弥补差距。当前优先研究partial分块与计算/访存组织，同时在接入时关注CPU包装和workspace管理。

后续方案：保持FlagGems weight_int8pack_mm数值与权重接口，基于现有MetaX实现进行有限候选实验。小M关注BM浪费、Split-K并行与短循环流水线开销；M32/64关注BM增大后的权重复用和资源压力。以完整fresh调用为主指标，同时保留prealloc及partial诊断。先形成明确参数候选再修改kernel，不扩大为全模型量化或官方bench。本轮未增加任何生产配置。

本地证据：`artifacts/flaggems-w8a16-stage-fixed-20261027-04/`（含preflight、environment、postflight、run.log、exit.txt、脚本、资源/IR、原始及汇总JSON）。旧轮和新轮不可混算收益。本次未提交、未推送。
