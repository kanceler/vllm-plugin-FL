# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""按设备属性推导注意力内核的分块与流水配置。

## 为什么需要推导而不是枚举

vLLM 的 Triton attention 启动器里，分块参数是**为 NVIDIA 调的常量**：

```python
BLOCK_M = (16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv))
BLOCK_Q = BLOCK_M // num_queries_per_kv
```

这组参数在 SM 数很多的卡上靠并行度就能喂饱带宽；在 SM 很少的加速器上，
K/V 会被重复读取数倍。要在这类设备上取得性能，需要重新选参。

枚举（把能试的都试一遍）可行但不可移植。本模块改为**从设备属性推导**：

## 推导链

1. **分块**：FlashAttention 的算术强度约等于 `BLOCK_M`（Dao et al., arXiv:2205.14135），
   而每增大一档分块，K/V 的重复读取次数减半。所以分块越大越好，
   上限由片上资源决定——经典的占用率约束：

       occ(BM, s) = smem // (s · per_stage + BM · d · elem)

   `occ` 至少要达到 `OCC_MIN`，否则单个 CTA 的加载延迟无法被其它 CTA 掩盖。

2. **warp 数**：沿用"每 16 个 M 行一个 warp"的比例（`BM=128 → 8`），
   上限 8，避免过度拆分。

3. **流水深度**：PipeThreader（OSDI'25）指出分块与流水深度会争抢同一块片上内存。
   在本类设备上，**驻留 CTA 数**比流水深度更能掩盖延迟，故取占用率最大的那个；
   平局时取更浅的流水（不损失占用率就不多占 smem）。

## 与上游的关系

上游只对 `head_size == 256` 且 NVIDIA 大卡启用了一个特调分支
（`tuned_large_head`，注释称"B200 上快 ~2 倍"）。本模块把同一件事推广成
按设备属性推导，因此对任何架构都成立。
"""

from dataclasses import dataclass
from typing import Optional

# ---- 推导用的常量（均有物理含义，见模块文档）----
OCC_MIN = 2          # 每 SM 至少常驻的 CTA 数：单个 CTA 无法自隐藏加载延迟
WARP_CAP = 8         # warp 数上限
REG_BUDGET = 128     # 每个线程允许的累加器寄存器数（BLOCK_M × d 的 fp32）
SMEM_BUDGET = 0.8    # 共享内存只用到 80%，留出余量
BM_MIN = 16
BM_MAX = 256


@dataclass(frozen=True)
class DeviceProfile:
    sm_count: int
    smem_per_sm: int
    warp_size: int


def device_profile() -> Optional[DeviceProfile]:
    """读取当前设备的片上资源；取不到时返回 None。"""
    try:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None
    sms = int(getattr(props, "multi_processor_count", 0) or 0)
    smem = int(getattr(props, "shared_memory_per_multiprocessor", 0) or 0)
    warp = int(getattr(props, "warp_size", 32) or 32)
    if sms <= 0 or smem <= 0:
        return None
    return DeviceProfile(sm_count=sms, smem_per_sm=smem, warp_size=warp)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def smem_needed(block_m: int, stages: int, head_size: int, tile: int,
                elem_size: int) -> int:
    """一个 CTA 需要的共享内存：stages 份 K/V 瓦片 + 一份 Q 瓦片。"""
    per_stage = 2 * tile * head_size * elem_size      # K + V
    q_tile = block_m * head_size * elem_size
    return stages * per_stage + q_tile


def residency(block_m: int, stages: int, head_size: int, tile: int,
              elem_size: int, smem_per_sm: int) -> int:
    """每 SM 能常驻的 CTA 数。

    注意：``smem_per_sm`` 必须是**真实容量**——占用率是硬件事实，
    不应被"预算折扣"（SMEM_BUDGET）影响。折扣只用来判断"是否允许使用"。
    """
    return max(0, smem_per_sm // max(1, smem_needed(block_m, stages, head_size,
                                                    tile, elem_size)))


def derive_tile(
    block_m: int,
    stages: int,
    head_size: int,
    elem_size: int,
    smem_per_sm: int,
    max_tile: int = 128,
) -> int:
    """推导 K/V 瓦片宽度（TILE_SIZE）。

    与分块用的是同一条约束：K/V 瓦片每加宽一倍，迭代次数减半（更少的固定开销），
    但每级流水的共享内存也随之翻倍。所以取**仍能满足占用率下界的最大瓦片**。

    BM=128 / stages=1 / 128 KB smem 时：
        TILE=32   → 16 KB(KV) + 32 KB(Q) = 48 KB → occ=2
        TILE=64   → 32 KB     + 32 KB     = 64 KB → occ=2   ← 取这个
        TILE=128  → 64 KB     + 32 KB     = 96 KB → occ=1   ✗
    """
    best = 16
    tile = 16
    while tile <= max_tile:
        if residency(block_m, stages, head_size, tile, elem_size, smem_per_sm) < OCC_MIN:
            break
        best = tile
        tile *= 2
    return best


def derive_prefill_config(
    head_size: int,
    num_queries_per_kv: int,
    max_seqlen_q: int,
    elem_size: int = 2,
) -> tuple[int, Optional[int], Optional[int], Optional[int]]:
    """返回 (BLOCK_M, num_warps, num_stages, TILE_SIZE)；None 表示沿用上游默认。"""
    default_bm = (
        16 if num_queries_per_kv <= 16 else _next_pow2(num_queries_per_kv)
    )
    profile = device_profile()
    if profile is None:
        return default_bm, None, None, None
    budget = int(profile.smem_per_sm * SMEM_BUDGET)
    smem = profile.smem_per_sm

    best_bm = 0
    bm = BM_MIN
    while bm <= BM_MAX:
        if smem_needed(bm, 1, head_size, 32, elem_size) > budget:
            break
        if residency(bm, 1, head_size, 32, elem_size, smem) < OCC_MIN:
            break
        warps = min(WARP_CAP, max(4, bm // 16))
        threads = warps * profile.warp_size
        if bm * head_size // max(1, threads) > REG_BUDGET:
            break
        if bm > 2 * max(1, max_seqlen_q):
            break
        best_bm = bm
        bm <<= 1

    if best_bm < default_bm:
        return default_bm, None, None, None

    warps = min(WARP_CAP, max(4, best_bm // 16))
    occ_by_stage = {
        s: residency(best_bm, s, head_size, 32, elem_size, smem) for s in (1, 2, 3)
    }
    top = max(occ_by_stage.values())
    stages = min(s for s, occ in occ_by_stage.items() if occ == top)
    tile = derive_tile(best_bm, stages, head_size, elem_size, smem)
    return best_bm, warps, stages, tile


def derive_prefill_tiles(
    head_size: int,
    num_queries_per_kv: int,
    max_seqlen_q: int,
    elem_size: int = 2,
    tile: int = 32,
) -> tuple[int, Optional[int], Optional[int]]:
    """推导 prefill 的 (BLOCK_M, num_warps, num_stages)。

    返回 ``(block_m, None, None)`` 表示沿用上游默认，不做改动。

    prefill 一次处理很多 query，分块越大、K/V 的重复读取越少；
    因此取"占用率仍达标"的最大分块。
    """
    default_bm = (
        16 if num_queries_per_kv <= 16 else _next_pow2(num_queries_per_kv)
    )
    profile = device_profile()
    if profile is None:
        return default_bm, None, None

    budget = int(profile.smem_per_sm * SMEM_BUDGET)
    smem = profile.smem_per_sm

    # ① 满足占用率下界的最大 2 的幂
    best_bm = 0
    bm = BM_MIN
    while bm <= BM_MAX:
        if residency(bm, 1, head_size, tile, elem_size, smem) < OCC_MIN:
            break
        # ② 寄存器：累加器是 BLOCK_M × head_size 的 fp32
        warps = min(WARP_CAP, max(4, bm // 16))
        threads = warps * profile.warp_size
        if bm * head_size // max(1, threads) > REG_BUDGET:
            break
        # ③ 尾块：分块不应超过本步 query 长度的两倍
        if bm > 2 * max(1, max_seqlen_q):
            break
        best_bm = bm
        bm <<= 1

    if best_bm < default_bm:
        return default_bm, None, None

    warps = min(WARP_CAP, max(4, best_bm // 16))

    # ④ 流水深度：占用率最大者；平局取更浅的
    occ_by_stage = {
        s: residency(best_bm, s, head_size, tile, elem_size, smem)
        for s in (1, 2, 3)
    }
    top = max(occ_by_stage.values())
    stages = min(s for s, occ in occ_by_stage.items() if occ == top)

    return best_bm, warps, stages


def derive_decode_tiles(
    head_size: int,
    num_queries_per_kv: int,
    elem_size: int = 2,
    tile: int = 16,
) -> tuple[int, Optional[int], Optional[int]]:
    """推导 decode 的 (BLOCK_M, num_warps, num_stages)。

    decode 每个请求只有 1 个 query，上游的 ``BLOCK_M`` 已经把同一 KV 组的
    多个 q head 打包进 M 维，因此**分块沿用上游**；只推导流水深度——
    与 prefill 用同一条"占用率最大、平局取浅"的规则。
    """
    default_bm = (
        16 if num_queries_per_kv <= 16 else _next_pow2(num_queries_per_kv)
    )
    profile = device_profile()
    if profile is None:
        return default_bm, None, None

    budget = int(profile.smem_per_sm * SMEM_BUDGET)
    smem = profile.smem_per_sm
    occ_by_stage = {
        s: residency(default_bm, s, head_size, tile, elem_size, smem)
        for s in (1, 2, 3)
    }
    top = max(occ_by_stage.values())
    if top <= 0:
        return default_bm, None, None
    stages = min(s for s, occ in occ_by_stage.items() if occ == top)
    warps = min(WARP_CAP, max(4, default_bm // 16))
    return default_bm, warps, stages
