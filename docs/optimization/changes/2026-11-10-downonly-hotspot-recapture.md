# Down-only 双场景热点补采

## 1. 基本信息与摘要

2026-11-10，C500。用户要求先补采 Down-only 在 4K/16K 的热点排序和占比。本次独立实验副本 `/data/downonly-hotspots-20261110-01/`；不将此前全投影 generic 量化 Trace 作为本组合结果。四个窗口已完成，4K 256/256、16K 128/128 请求 HTTP200；服务退出后 GPU 无进程。

## 2. 问题与原因分析

原 SHAPE_DISPATCH 同时选择 Down、Gate/Up、QKV、O，不等于 Down-only；此前模型还存在 public generic 与专用 MetaX 入口差异。需重启服务，核对实际量化覆盖与 kernel 后采样。采样窗口统计不能外推为整个官方运行耗时。

## 3. 具体实现

复用独立 overlay 中已修正的 int8pack 入口，仅在本实验副本限制 linear 选择 Down；明确 M56/64 Down 调用配置 BM64/BN64/BK128、4 warps、1 stage、Split8。复用 BF16 wide 版本 worker hook 与 nt_gemm 配置；Down 大 M 为转换＋注册 BF16 MM＋scale，仍承担完整转换成本。配置与实际源文件哈希保留，不改共享安装和主工作区生产代码。

## 4. 实验与验证

启动前 GPU 无进程，826 MiB，9031 bind 成功。启动使用沐曦官方 serve 原参数，外围 observer 仅诊断。负载使用确定性 synthetic token IDs，分别 4096/16384 输入、1024 输出、并发64、256/128请求；不是官方 Benchmark，也不报告计分吞吐。每场景先采 Prefill 16 个匹配 step，再采 Decode 128 个匹配 step，实际 M、有效样本和 unknown 以 Trace 为准。不同 phase 窗口分别报告，不直接相加为场景总耗时。

服务健康后强制检查恰好42个 Down 权重量化启用记录。实测确认42个启用记录均为 Down。M2048新BF16配置四组命中；Down M56/64使用Split8。


### 采样结果

占比均以各窗口全部 GPU kernel duration 总和为分母；排序不等于端到端 wall time 排序。Prefill 标签窗口包含 mixed step。

#### 16384-decode

Trace `trace-9c7433b09bd042b48b4d62f29e8a1e48-pid419148.json`；GPU kernel 时间总和 5237.098 ms；匹配 GEMM 时间 795.845 ms；精确 M 耗时关联覆盖 23.44%。

| 排名 | Kernel | 调用数 | GPU ms | 占全部GPU时间 |
|---:|---|---:|---:|---:|
| 1 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, false, false, false, false, true, false, true, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 5376 | 4117.337 | 78.619% |
| 2 | `mm_kernel_nt` | 16128 | 528.515 | 10.092% |
| 3 | `weight_int8pack_mm_splitk_partial` | 5376 | 242.958 | 4.639% |
| 4 | `linear_kernel` | 128 | 77.627 | 1.482% |
| 5 | `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, 8, 3, true, false>(mcFlashAttn::Flash_fwd_params)` | 5376 | 55.561 | 1.061% |
| 6 | `triton_red_fused_fused_add_rms_norm_2` | 5376 | 28.557 | 0.545% |
| 7 | `triton_red_fused_fused_add_rms_norm_0` | 5376 | 27.443 | 0.524% |
| 8 | `weight_int8pack_mm_splitk_reduce_scale` | 5376 | 24.373 | 0.465% |
| 9 | `void vllm::reshape_and_cache_flash_kernel<__maca_bfloat16, __maca_bfloat16, (vllm::Fp8KVCacheDataType)0>(__maca_bfloat16 const*, __maca_bfloat16 const*, __maca_bfloat16*, __maca_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*)` | 5376 | 24.085 | 0.460% |
| 10 | `triton_poi_fused_cat_3` | 5248 | 23.200 | 0.443% |
| 11 | `triton_poi_fused_cat_4` | 5248 | 22.765 | 0.435% |
| 12 | `triton_poi_fused_mul_silu_1` | 5376 | 21.918 | 0.419% |
| 13 | `_copy_kernel_kernel_rank_3` | 5376 | 18.058 | 0.345% |
| 14 | `argmax_kernel_inner` | 127 | 12.360 | 0.236% |
| 15 | `_to_copy_func_kernel_rank_1` | 382 | 4.967 | 0.095% |

#### 16384-prefill

Trace `trace-680182c148a940c58afea3f11a9385f4-pid419148.json`；GPU kernel 时间总和 2005.464 ms；匹配 GEMM 时间 920.549 ms；精确 M 耗时关联覆盖 100.00%。

| 排名 | Kernel | 调用数 | GPU ms | 占全部GPU时间 |
|---:|---|---:|---:|---:|
| 1 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 128, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 672 | 910.640 | 45.408% |
| 2 | `mm_kernel_nt` | 2688 | 881.120 | 43.936% |
| 3 | `triton_poi_fused_mul_silu_1` | 672 | 42.095 | 2.099% |
| 4 | `triton_red_fused_fused_add_rms_norm_2` | 672 | 31.695 | 1.580% |
| 5 | `weight_int8pack_convert` | 672 | 28.407 | 1.416% |
| 6 | `triton_red_fused_fused_add_rms_norm_0` | 672 | 21.090 | 1.052% |
| 7 | `triton_poi_fused_cat_3` | 656 | 16.943 | 0.845% |
| 8 | `void vllm::reshape_and_cache_flash_kernel<__maca_bfloat16, __maca_bfloat16, (vllm::Fp8KVCacheDataType)0>(__maca_bfloat16 const*, __maca_bfloat16 const*, __maca_bfloat16*, __maca_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*)` | 672 | 13.212 | 0.659% |
| 9 | `_copy_kernel_kernel_rank_3` | 1008 | 12.488 | 0.623% |
| 10 | `weight_int8pack_output_scale` | 672 | 11.022 | 0.550% |
| 11 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, false, false, false, false, true, false, true, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 336 | 10.809 | 0.539% |
| 12 | `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, 8, 7, true, false>(mcFlashAttn::Flash_fwd_params)` | 336 | 7.651 | 0.381% |
| 13 | `linear_kernel` | 16 | 6.866 | 0.342% |
| 14 | `triton_poi_fused_cat_4` | 656 | 4.567 | 0.228% |
| 15 | `reduce_then_scan_root_scan_kernel_row` | 672 | 1.898 | 0.095% |

#### 4096-decode

Trace `trace-0df9ea5b034d4b4c809c0492023f3355-pid419148.json`；GPU kernel 时间总和 2231.454 ms；匹配 GEMM 时间 796.456 ms；精确 M 耗时关联覆盖 100.00%。

| 排名 | Kernel | 调用数 | GPU ms | 占全部GPU时间 |
|---:|---|---:|---:|---:|
| 1 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, false, false, false, false, true, false, true, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 5376 | 1111.353 | 49.804% |
| 2 | `mm_kernel_nt` | 16128 | 528.916 | 23.703% |
| 3 | `weight_int8pack_mm_splitk_partial` | 5376 | 243.142 | 10.896% |
| 4 | `linear_kernel` | 128 | 77.653 | 3.480% |
| 5 | `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, 8, 3, true, false>(mcFlashAttn::Flash_fwd_params)` | 5376 | 55.351 | 2.480% |
| 6 | `triton_red_fused_fused_add_rms_norm_2` | 5376 | 28.514 | 1.278% |
| 7 | `triton_red_fused_fused_add_rms_norm_0` | 5376 | 27.422 | 1.229% |
| 8 | `weight_int8pack_mm_splitk_reduce_scale` | 5376 | 24.397 | 1.093% |
| 9 | `void vllm::reshape_and_cache_flash_kernel<__maca_bfloat16, __maca_bfloat16, (vllm::Fp8KVCacheDataType)0>(__maca_bfloat16 const*, __maca_bfloat16 const*, __maca_bfloat16*, __maca_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*)` | 5376 | 24.077 | 1.079% |
| 10 | `triton_poi_fused_cat_3` | 5248 | 23.198 | 1.040% |
| 11 | `triton_poi_fused_cat_4` | 5248 | 22.756 | 1.020% |
| 12 | `triton_poi_fused_mul_silu_1` | 5376 | 21.931 | 0.983% |
| 13 | `_copy_kernel_kernel_rank_3` | 5376 | 18.061 | 0.809% |
| 14 | `argmax_kernel_inner` | 127 | 12.376 | 0.555% |
| 15 | `_to_copy_func_kernel_rank_1` | 382 | 4.947 | 0.222% |

#### 4096-prefill

Trace `trace-a7f2f02a935647ee921dcbc5a420a362-pid419148.json`；GPU kernel 时间总和 1356.103 ms；匹配 GEMM 时间 924.496 ms；精确 M 耗时关联覆盖 100.00%。

| 排名 | Kernel | 调用数 | GPU ms | 占全部GPU时间 |
|---:|---|---:|---:|---:|
| 1 | `mm_kernel_nt` | 2688 | 884.999 | 65.260% |
| 2 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 128, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 420 | 184.415 | 13.599% |
| 3 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 64, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 252 | 64.455 | 4.753% |
| 4 | `triton_poi_fused_mul_silu_1` | 672 | 42.010 | 3.098% |
| 5 | `triton_red_fused_fused_add_rms_norm_2` | 672 | 31.698 | 2.337% |
| 6 | `weight_int8pack_convert` | 672 | 28.482 | 2.100% |
| 7 | `triton_red_fused_fused_add_rms_norm_0` | 672 | 21.018 | 1.550% |
| 8 | `triton_poi_fused_cat_3` | 656 | 16.958 | 1.250% |
| 9 | `_copy_kernel_kernel_rank_3` | 1260 | 13.741 | 1.013% |
| 10 | `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, false, false, false, false, true, false, true, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 588 | 13.547 | 0.999% |
| 11 | `void vllm::reshape_and_cache_flash_kernel<__maca_bfloat16, __maca_bfloat16, (vllm::Fp8KVCacheDataType)0>(__maca_bfloat16 const*, __maca_bfloat16 const*, __maca_bfloat16*, __maca_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*)` | 672 | 13.146 | 0.969% |
| 12 | `weight_int8pack_output_scale` | 672 | 11.015 | 0.812% |
| 13 | `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 16, 16, 1, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, 16, 16, 1, mctlass::bfloat16_t> >, 8, 7, true, false>(mcFlashAttn::Flash_fwd_params)` | 462 | 10.040 | 0.740% |
| 14 | `linear_kernel` | 16 | 7.076 | 0.522% |
| 15 | `triton_poi_fused_cat_4` | 656 | 4.573 | 0.337% |

三个窗口匹配 GEMM 的精确 M 关联为100%：Prefill为M2048、4K Decode为M64。16K Decode只有23.44%耗时关联M64，其余M未知；不能依据active_tokens64补齐。匹配器未证明覆盖全部GEMM（例如linear_kernel应单列），因此匹配GEMM分母只代表匹配集合。BF16 mm_kernel_nt共用名称，无法可靠拆分Gate/Up、QKV、O和Separate内部Down MM，故合并统计，不按顺序猜测。

## 5. 与已有改动的关系

基于三算法量化和 BF16 wide 配置的组合诊断；与之前全层量化采样不构成同版本直接性能 A/B。M56 配置来自已有 Down 实测，不采用 Gate/Up 新实验原型。

## 6. 合规性与风险

不改官方 serve 参数、Benchmark 或数据集；诊断负载单独标识。量化为 per-channel INT8 权重、FP32 scale、BF16 激活与 dot 输入、FP32 累加、BF16 输出。没有原生 INT8×INT8 GEMM。未做模型精度或官方性能验收。Kernel 时间占比为窗口内 GPU kernel duration 总和分母，不等于端到端 wall time；名字匹配覆盖与 CPU correlation 的 unknown 单列。

## 7. 复现方式

远端 `run.py` 归档启动、环境、健康检查、覆盖、两场景请求结果和控制步骤；`rank.py` 生成全部 GPU kernel 排名及匹配 GEMM 的 M 排名。原始证据保存 `observer/trace-*.json`、`server.log`、`commands.json`、`hashes.json`、`quantization-coverage.txt`。执行命令 `python /data/downonly-hotspots-20261110-01/run.py`；输出目录固定用于本次首次执行，复现须改用新的实验目录，不覆盖历史。

完整Trace保留远端原实验目录；汇总原始JSON已取回 `artifacts/downonly-hotspots-20261110-01/ranking.json`。SCP汇总包两次连接中断，改经现有SSH连接压缩传回JSON并解析校验；不宣称完整Trace已下载。
