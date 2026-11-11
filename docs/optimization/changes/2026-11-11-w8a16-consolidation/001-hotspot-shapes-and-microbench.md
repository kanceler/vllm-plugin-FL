# 001 热点M、真实Shape与算子测评

## 1. 基本信息与摘要

整理日期：2026-11-11。平台：MetaX C500；模型MiniCPM5-2B。本阶段是诊断与证据建设，不单独声称带来吞吐提升。目标是确定需要优化的真实GEMM Shape及小M/大M路径，而不是重新分析整个推理框架。

## 2. 问题与原因分析

历史Profiling已定位GEMM为热点，但“并发64”“输入长度16384”和某次GEMM的M不是同一概念；Prefill、Decode、图捕获也可能有不同Shape。仅按层名或请求数设计Kernel容易测错对象。需要同时核对本地N/K、stride、dtype、算法分支及实际导入来源。

## 3. 具体实现

复用已有模型Profiling与Shape采样工具，按投影类型及本地N/K分组：Down N2048/K6144；Packed Gate/Up N12288/K2048；Packed QKV N2560/K2048；O N2048/K2048。M来自执行时输入展平后的行数。

采集和分析入口包括 `benchmarks/diagnostics/collect_gemm_shapes.py`、`summarize_gemm_shapes.py`；诊断负载与正式Benchmark分别归档。明确初始化、warmup、capture与负载采样窗口，未关联的Kernel标为unknown。

## 4. 实验与验证：microbenchmark

Down-only重采样报告中，已匹配Prefill GEMM的精确M关联为2048，4K Decode为64；16K Decode只有23.44%的已匹配耗时可关联M64，其余未知。不能将某窗口100%关联率当作未来输入全覆盖。

M56保留为历史调优与专用配置对照，不能声称它已经取代M64成为当前官方负载热点。M8192/16384主要属于扩展诊断点，不因16K输入长度就认定实际GEMM使用这些M。

本阶段建立microbench口径：Dense完整融合；Split-K包含workspace、partial和reduce/scale；Separate包含转换、MM和scale。独立阶段计时只用于归因，不相加替代完整调用。

### 详细数据与原始依据

**来源：[2026-11-10-downonly-hotspot-recapture.md](../2026-11-10-downonly-hotspot-recapture.md)。各表仅在原实验内比较，不跨版本合并。**

**16384-decode**

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

**16384-prefill**

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

**4096-decode**

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

**4096-prefill**

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

以上为原采样窗口中的Kernel统计，不是独立Kernel microbench，不将窗口占比拼接成全程占比。

## 5. 与已有改动的关系

为002提供Shape矩阵，为003提供投影比较依据，为004提供分发边界输入。后续优化沿用这些证据，不重新开展无范围的框架分析。

## 6. 合规性与风险

采样窗口和匹配集合均有限。Python调用/捕获计数不是图重放次数；共名BF16 MM不能凭顺序拆分投影耗时。此项没有独立模型性能验收，也不涉及精度阈值变更。

## 7. 复现方式与分支汇总清单

主要依据：[Down-only热点重采样](../2026-11-10-downonly-hotspot-recapture.md)、[历史M热点记录](../2026-11-10-quantized-m-hotspots.md)。

汇总分支纳入必要Shape采样与汇总工具，并保留采样协议；测试入口参数以归档脚本的 `--help` 和实际commands.json为准，不编造历史命令。每次GPU复现前按工作区规则核对资源和环境。
