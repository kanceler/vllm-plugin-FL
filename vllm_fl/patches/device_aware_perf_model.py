# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""把 Triton 自带的 matmul 分析模型改成设备感知的。

背景
----
FlagGems 的 Iluvatar 后端在自动调优时，用 Triton 教程自带的
``estimate_matmul_time`` 给候选配置排序，只把排名靠前的若干组真正跑一遍
（``top_k``）。这个分析模型里有三个写死的硬件参数：

    active_cta_ratio_bw1 = min(1, num_ctas / 32)   # 32 个 CTA 视为打满带宽
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)
    l2_bw = dram_bw * 4                            # 注释原话 "should be 4.7 for A100?"

其中 108 是 A100 的 SM 数。在一个只有 16 个 SM 的设备上，这个模型会系统性
低估小 grid 的可达带宽，从而偏好"很多小 CTA"的配置，把大 tile 的配置排除在
``top_k`` 之外——即使大 tile 在实测里明显更快。

做法
----
不修改 Triton 本身，而是在插件层把 ``estimate_matmul_time`` 换成一个"饱和点
由设备 SM 数推导"的等价实现。其余部分（计算/加载/写回三段式、L2 占比假设、
写回带宽系数）保持与上游完全一致，只有饱和点从写死的 32/108 改成
``num_sm`` / ``2 * num_sm``。

注意
----
这个改动只影响"选哪个配置"，不改变任何 kernel 的语义。实测收益集中在
中小 M（M=512/2048 上 GEMM 快 21%/9.6%），而真实 chunked prefill 的 M
在万级，收益衰减到 0.4%，端到端不可辨识。保留它是为了记录这个缺陷本身。
"""

import logging

logger = logging.getLogger(__name__)

_installed = False


def _make_estimate_matmul_time(perf_model):
    """基于上游模块，构造一个饱和点由设备 SM 数推导的 estimate_matmul_time。"""

    def estimate_matmul_time(
        A, B, C, M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,
        debug=False, **kwargs,
    ):
        import torch

        num_warps = kwargs.get("num_warps", 4)
        device = torch.cuda.current_device()
        dtype = A.dtype
        dtsize = A.element_size()

        if (K % 64 == 0) and (K % (BLOCK_K * SPLIT_K) != 0):
            return float("inf")
        num_cta_m = perf_model.cdiv(M, BLOCK_M)
        num_cta_n = perf_model.cdiv(N, BLOCK_N)
        num_ctas = num_cta_m * num_cta_n * SPLIT_K

        # 输入比 tile 还小时按 tile 计
        M, N = max(M, BLOCK_M), max(N, BLOCK_N)

        total_ops = 2 * M * N * K / (1024 * 1024 * 1024)   # GOPS
        tput = perf_model.get_tflops(device, num_ctas, num_warps, dtype)
        compute_ms = total_ops / tput

        num_sm = perf_model.driver.active.utils.get_device_properties(
            device)["multiprocessor_count"]
        # —— 与上游唯一的差别：饱和点由设备推导，而不是写死 32/108 ——
        sat_ctas = max(8, num_sm)
        full_ctas = max(sat_ctas + 1, num_sm * 2)
        active_cta_ratio_bw1 = min(1, num_ctas / sat_ctas)
        active_cta_ratio_bw2 = max(
            min(1, (num_ctas - sat_ctas) / (full_ctas - sat_ctas)), 0)

        dram_bw = perf_model.get_dram_gbps(device) * (
            active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05)
        l2_bw = dram_bw * 4

        load_a_dram = M * K * dtsize * (1 + 0.2 * (num_cta_n - 1))
        load_a_l2 = M * K * dtsize * 0.8 * (num_cta_n - 1)
        load_b_dram = N * K * dtsize * (1 + 0.2 * (num_cta_m - 1))
        load_b_l2 = N * K * dtsize * 0.8 * (num_cta_m - 1)
        total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)
        total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
        load_ms = total_dram / dram_bw + total_l2 / l2_bw

        store_bw = dram_bw * 0.6
        store_c_dram = M * N * dtsize * SPLIT_K / (1024 * 1024)
        if SPLIT_K == 1:
            store_ms = store_c_dram / store_bw
        else:
            store_ms = store_c_dram / store_bw
            store_ms += M * N * 2 / (1024 * 1024) / store_bw

        total_time_ms = max(compute_ms, load_ms) + store_ms
        if debug:
            logger.debug(
                "perf model(device-aware): total=%.4f compute=%.4f load=%.4f "
                "store=%.4f ctas=%d sm=%d", total_time_ms, compute_ms,
                load_ms, store_ms, num_ctas, num_sm)
        return total_time_ms

    return estimate_matmul_time


def install_device_aware_perf_model() -> bool:
    """安装设备感知的 matmul 分析模型。失败时保持上游行为，只记日志。"""
    global _installed
    if _installed:
        return True
    try:
        from triton.ops import matmul_perf_model as perf_model
    except Exception as exc:  # noqa: BLE001
        logger.debug("device-aware perf model: triton unavailable: %r", exc)
        return False

    needed = ("cdiv", "get_tflops", "get_dram_gbps", "driver")
    if not all(hasattr(perf_model, n) for n in needed):
        logger.debug("device-aware perf model: 上游接口不符，跳过")
        return False

    new_fn = _make_estimate_matmul_time(perf_model)
    perf_model.estimate_matmul_time = new_fn

    # 若 FlagGems 的后端模块已经先导入，它的名字绑定还指向旧函数，一并替换。
    import sys
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if getattr(mod, "estimate_matmul_time", None) is not None and \
                getattr(mod, "__name__", "").startswith("flag_gems"):
            try:
                mod.estimate_matmul_time = new_fn
            except Exception:  # noqa: BLE001
                pass

    _installed = True
    logger.info(
        "device-aware matmul perf model installed: CTA 饱和点由设备 SM 数推导")
    return True
