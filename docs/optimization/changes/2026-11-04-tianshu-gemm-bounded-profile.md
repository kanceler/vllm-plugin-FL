# 天数 GEMM 短时运行采样：Prefill 形状与 Decode 可见性

## 1. 基本信息与摘要

2026-11-04，天数 BI-V150、CoreX/驱动4.5.0。基于队友插件f82fa167，保持官方serve参数，用现有外部observer进行有限步PyTorch profiler采样；没有修改队友仓库、安装包或Kernel。独立实验 `/workspace/experiments/tianshu-gemm-path-20261104-02/`。

Prefill取得四类投影的M=2048、168次MM GPU事件；Decode取得8步FULL图重放上下文，但图内Kernel未展开，不能报告Decode GEMM耗时。无新官方Benchmark或精度成绩。记录独立于沐曦。

## 2. 问题与原因分析

编译产物只能确认外部MM调用，不能给出实际Shape及GPU耗时。此次补采实际事件，避免把capture尺寸、示例M或Kernel名字推断为执行覆盖。

内置profiler需额外serve参数，本次未采用。复用 `benchmarks/diagnostics/collect_gemm_shapes.py` 原样复制，SHA256为 `62ef8273e8d48ebe15c5af1007e058d85cc629f6ffeab6fc56f7be97ad3d3587`。该工具使用进程内诊断包装、有限步torch.profiler，保持官方参数和vLLM profiler配置；存在观测开销，不用于正式性能成绩。

## 3. 具体执行

### 环境隔离

启动前GPU无计算进程，68MiB驱动占用，9031端口连接返回111。队友工作树此前已核对干净。PYTHONPATH=/workspace/tianshu-final-20261104，VLLM_PLUGINS=fl，COREX_VERSION=4.5.0；vllm_fl实际解析到队友目录，vllm与FlagGems来自site-packages。服务PID19657、EngineCore19800，健康检查200；observer保存模型层清单、dtype、量化方法及配置。

官方参数保持：

```bash
python3 collect_gemm_shapes.py serve --output /workspace/experiments/tianshu-gemm-path-20261104-02/observer -- \
  vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

### 诊断负载

客户端使用aiohttp调用 `/v1/completions`，输入是合成token IDs，无评测集内容。warmup为1请求、128输入/16输出；Prefill诊断为8并发、4096输入/64输出，采匹配prefill的1步；Decode诊断为8并发、256输入/64输出，采匹配decode的8步。temperature=0、ignore_eos=true仅属于诊断客户端负载，不是官方评测配置。prefix cache保留原服务行为，未清缓存；不同请求首token不同。

两组各8/8请求返回200，使用量均与设定一致。并发8不等于采样M=8；Prefill实际采样M=2048。

## 4. 实验与验证

### Prefill：单个已观测窗口

Trace `trace-ab4843d416eb4392a6fe90816a45dc5d-pid19800.json`，433个GPU Kernel事件、累计158227.465μs。此为事件dur之和，不是墙钟时间；仅该窗口分母，不是官方负载占比。

CPU `aten::mm`的Input Dims/Strides与GPU `mm_kernel`通过相同External id关联，168/168个MM Kernel均有对应CPU MM Shape。下表是42层的按Shape分组，不能恢复每个事件的确切层号。

| 投影 | M,N,K | 调用数 | GPU合计μs | 平均μs | block | grid | shared memory bytes |
|---|---|---:|---:|---:|---|---|---:|
| QKV | 2048,2560,2048 | 42 | 12754.204 | 303.672 | 512,1,1 | 160,1,1 | 49152 |
| O | 2048,2048,2048 | 42 | 10326.460 | 245.868 | 512,1,1 | 128,1,1 | 49152 |
| Gate/Up | 2048,12288,2048 | 42 | 59153.009 | 1408.405 | 1024,1,1 | 384,1,1 | 65536 |
| Down | 2048,2048,6144 | 42 | 30453.924 | 725.093 | 512,1,1 | 128,1,1 | 49152 |

四类MM总计112687.597μs，占此窗口已记录GPU Kernel时间约71.22%。Gate/Up和Down是本窗口GEMM优化的优先覆盖Shape。

| 已记录Kernel | 次数 | 合计μs | 窗口占比约 |
|---|---:|---:|---:|
| kernel_unified_attention | 42 | 28587.720 | 18.07% |
| triton_poi_fused_mul_silu_slice_1 | 42 | 5752.790 | 3.64% |
| triton_red_fused_fused_add_rms_norm_2 | 42 | 3482.240 | 2.20% |
| triton_poi_fused_3 | 41 | 3342.720 | 2.11% |
| triton_red_fused_fused_add_rms_norm_0 | 42 | 2123.840 | 1.34% |

Attention仅用于分母说明，不扩展优化范围。Gate/Up融合激活平均约136.971μs；直接消除该Kernel的理想上限仅约本窗口3.64%，实际融合仍需做激活，并可能改变GEMM效率，不能将该上限当作收益。GEMM中间流量节省另需实测。

GPU名称确为 `mm_kernel`，launch block/grid/shared-memory已归档，但名称不足以唯一识别厂商库或FlagGems。不能由 `extern_kernels.mm`或enabled-ops推断vendor归属，也不能将launch block当作BM/BN/BK。此前vendor BLAS归属结论继续保留待核验状态。

已有通用summarizer在此天数Trace上报告unknown correlation=168、exact_M coverage=0，说明它现有关联逻辑未兼容本Trace；不能据此说Trace没有Shape。此次另以External id直接与CPU MM关联生成 `prefill-mm-by-shape.csv`，未修改用户已有summarizer源码。

### Decode：可见性限制

Trace `trace-208ed4c46555424f995afe7a10f8ae90-pid19800.json`，8个step标记均为decode，active_tokens=8，forward标记graph_tokens=8、mode=FULL。记录到124个GPU Kernel，包含8个linear_kernel等图外执行，但未展开图内168×8个投影MM或融合激活事件。

因此这份Trace不能用作Decode各投影耗时、GEMM占比或tile结论。不能把图外linear_kernel（还需确认是否LM Head）当作模型全部GEMM。不关闭CUDA Graph、不更改官方参数来制造替代路径。后续需查天数profiler对graph节点活动的支持，或只读检查对应capture产物；修改诊断工具前单独讨论。

## 5. 与已有方案的关系

补足[第一步静态审计](2026-11-04-tianshu-gemm-path-audit.md)的Prefill运行证据。BF16融合候选仍需对照当前MM+已融合激活，不能将SiLU+Mul视作未融合。W8A16仍未启用，三算法能力及收益要在天数独立验证；不搬用沐曦M分界和tile。

scale-before-dot继续弃用。当前没有已确认的Intel tile错配；本轮没有修复配置或Kernel。

## 6. 合规性与风险

诊断负载不替代官方Benchmark，未改官方参数、数据、评测脚本。仅覆盖小规模合成负载的一个Prefill窗口和八步Decode上下文，无16K、64并发、Mixed/尾部完整覆盖；无模型精度、TTFT改善或total tokens/s候选成绩。

结束前确认客户端退出，核对PID归属后TERM服务19657。服务退出后ixsmi无计算进程、利用率0%、显存68MiB；不删除历史日志或持久缓存。后续新GPU任务需再次按规则检查环境。

## 7. 复现与归档

完整原始日志、客户端、脚本、Trace和元数据归档到本地：

`experiment-artifacts/2026-11-04-tianshu-gemm-path-02/remote/`。

聚合表：`experiment-artifacts/2026-11-04-tianshu-gemm-path-02/prefill-mm-by-shape.csv`。原summarizer结果保留在prefill-summary，标注其关联失败，不作为最终Shape覆盖结论。

控制命令：

```bash
python3 collect_gemm_shapes.py control --output observer --label prefill-4k-c8 --phase prefill --steps 1
python3 diagnostic_client.py 8 4096 64 prefill-4k-c8
python3 collect_gemm_shapes.py control --output observer --label decode-c8 --phase decode --steps 8
python3 diagnostic_client.py 8 256 64 decode-c8
```

此前传输失败的observer.b64保留作为异常记录，实际执行的是SCP传入并核对哈希的collect_gemm_shapes.py。当前未提交或推送Git。
