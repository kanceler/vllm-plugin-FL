# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""#005: 按设备属性为 vLLM 的 Triton attention 选择分块参数。

## 问题

``vllm.v1.attention.ops.triton_unified_attention`` 里：

```python
BLOCK_M = (16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv))
BLOCK_Q = BLOCK_M // num_queries_per_kv
```

这组参数是为**解码**选的（解码时 CTA 数 = 请求数 × KV head 数，在 NVIDIA 的
上百个 SM 上足以铺满，单个 CTA 做小一点无所谓）。但 **prefill 复用了同一个常量**。

一次注意力迭代里：

    运算量 = 4 · BLOCK_M · BLOCK_N · d
    搬运量 = 4 · BLOCK_N · d          （K + V）
    算术强度 = 运算量 / 搬运量 = BLOCK_M

即**算术强度恰好等于 BLOCK_M**。BLOCK_M=16 意味着每搬 1 字节只换 16 次运算，
远低于加速器的"算力/带宽"平衡点，于是 prefill 全程卡在等内存上。

在 SM 数很多的卡上，并行度本身就能把带宽喂饱，这个问题被掩盖；
在 SM 数很少的加速器上（例如 16 个），必须靠"每个 CTA 多干活"来提高复用。

## 做法

按设备属性（SM 数、共享内存、warp 大小）和本次请求形状推导 BLOCK_M：

* decode（``max_seqlen_q <= 1``）保持 vLLM 默认 —— 该路径已被上游专门调过
* SM 数足够多的设备保持默认 —— 默认值在那里不是瓶颈
* 其余情况在 **寄存器 / 共享内存 / 尾块浪费** 三个约束内取最大的 2 的幂

然后用一个**最小文本替换**把这段计算接到 vLLM 的启动器上（见 ``install_tile_policy``），
Triton 内核本身一行不动。

## 为什么用文本替换而不是 fork

只需要改 4 行（``BLOCK_M`` / ``launch_num_warps`` / ``launch_num_stages``），
而这三个都是启动器里的**局部变量**，无法从外部注入。fork 整个启动器会带来
数百行上游代码的维护负担；文本替换则：

* 对上游代码做**精确整块匹配**，匹配不上就放弃（fail-safe，退回默认行为）
* 升级 vLLM 后若上游改动，会安全地退回，不会算错

## 实验数据（天数 BI-V150，官方 benchmark）

    prefill 内核     : 5.3~5.95×（三种形状一致）
    4k 端到端        : 1969.80 → 2571.23 tok/s  (+30.5%)
    16k 端到端       : 886.55  → 1962.14  tok/s  (+121.3%)
    精度 math_500 L3 : 0.962(基线) → 0.971（门槛 0.95）

详见仓库根目录 ``改动说明-005-Attention分块.md``。
"""

import ast
import importlib
import os
import sys

_TARGET_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
_BINDING_MODULE = "vllm.v1.attention.backends.triton_attn"

# 上游启动器里计算分块的那一段（逐字匹配）
_ORIG_BLOCK = """    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None"""

_PATCHED_BLOCK = """    BLOCK_M, launch_num_warps, launch_num_stages = _fl_prefill_tiles(
        max_seqlen_q=max_seqlen_q,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
    )
    BLOCK_Q = max(1, BLOCK_M // num_queries_per_kv)"""

# 上游启动器里 BLOCK_M 已被 ``tuned_large_head`` 覆盖过；这里只关心我们自己的策略。
_STAGE_TILE = 32          # vLLM 的 TILE_SIZE_PREFILL 默认值
_SMEM_MARGIN = 0.8        # 共享内存只用到 80%，留出余量


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _floor_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _device_limits():
    """返回 (sm_count, smem_bytes_per_sm, warp_size)；取不到时返回 None。"""
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
    return sms, smem, warp


def compute_prefill_tiles(
    max_seqlen_q: int,
    num_queries_per_kv: int,
    head_size: int,
):
    """返回 (BLOCK_M, num_warps, num_stages)。``None`` 表示沿用 Triton 默认。

    三个约束：

    1. **寄存器**：累加器 ``acc`` 是 ``BLOCK_M × head_size`` 的 fp32，
       每线程不宜超过 128 个寄存器。
    2. **共享内存**：``stages`` 级流水的 K/V 缓冲 + Q 瓦片。
    3. **尾块浪费**：BLOCK_M 不应超过本步 query 长度的两倍，否则尾部空转。

    另外两条"不做"的判据：

    * ``max_seqlen_q <= 1``：decode 形状，vLLM 的 3D 路径已专门调优，不动。
    * SM 数 ≥ 64：并行度本身足以摊薄带宽，默认值不是瓶颈，不动。
    """
    default_bm = 16 if num_queries_per_kv <= 16 else _next_pow2(num_queries_per_kv)

    if max_seqlen_q <= 1:
        return default_bm, None, None

    limits = _device_limits()
    if limits is None:
        return default_bm, None, None
    sms, smem, warp = limits

    if sms >= 64:
        return default_bm, None, None

    warps = 8
    stages = 2
    threads = warps * warp

    # 约束 1：寄存器（acc 为 BLOCK_M × head_size 个 fp32）
    bm_reg = (128 * threads) // max(1, head_size)

    # 约束 2：共享内存
    kv_per_stage = 2 * _STAGE_TILE * head_size * 2      # K + V，bf16
    q_bytes_per_row = head_size * 2
    budget = int(smem * _SMEM_MARGIN) - stages * kv_per_stage
    bm_smem = budget // max(1, q_bytes_per_row)

    # 约束 3：尾块浪费
    bm_q = 2 * max_seqlen_q

    bm = _floor_pow2(min(bm_reg, bm_smem, bm_q))
    bm = max(default_bm, min(bm, 128))

    if bm <= default_bm:
        return default_bm, None, None
    return bm, warps, stages


def _enabled() -> bool:
    return os.environ.get("VLLM_FL_ATTN_TILE_POLICY", "1").lower() not in (
        "0", "false", "no",
    )


_installed = False


def install_tile_policy() -> bool:
    """把分块策略接到 vLLM 的 Triton attention 启动器上。

    返回是否安装成功。任何一步失败都只记日志并返回 False —— 此时行为与上游一致，
    不会比不打补丁更糟。
    """
    global _installed
    if _installed:
        return True
    if not _enabled():
        print("[FL] attention tile policy disabled by env; using upstream defaults.")
        return False

    try:
        src_mod = importlib.import_module(_TARGET_MODULE)
        binding_mod = importlib.import_module(_BINDING_MODULE)
    except Exception as exc:  # noqa: BLE001
        print("[FL] attention tile policy: cannot import vLLM modules: %r" % (exc,))
        return False

    src_path = getattr(src_mod, "__file__", None)
    if not src_path:
        return False

    try:
        with open(src_path, encoding="utf-8") as f:
            source = f.read()
    except OSError as exc:
        print("[FL] attention tile policy: cannot read %s: %r" % (src_path, exc))
        return False

    hits = source.count(_ORIG_BLOCK)
    if hits != 1:
        print("[FL] attention tile policy: upstream block matched %d times (expected 1); "
              "skipping (vLLM may have changed its launcher)." % (hits,))
        return False

    patched = source.replace(_ORIG_BLOCK, _PATCHED_BLOCK, 1)

    try:
        tree = ast.parse(patched)
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "unified_attention"
        )
        ns = dict(src_mod.__dict__)
        ns["_fl_prefill_tiles"] = compute_prefill_tiles
        exec(compile(ast.Module(body=[fn], type_ignores=[]), src_path, "exec"), ns)  # noqa: S102
        tuned = ns["unified_attention"]
    except Exception as exc:  # noqa: BLE001
        print("[FL] attention tile policy: patch failed: %r" % (exc,))
        return False

    binding_mod.unified_attention = tuned
    _installed = True

    limits = _device_limits()
    print(
        "[FL] attention tile policy installed (device: sms=%s, smem=%.0f KB, warp=%s). "
        "Prefill uses a larger BLOCK_M derived from device limits; decode is unchanged."
        % (limits[0] if limits else "?",
           (limits[1] / 1024.0) if limits else 0,
           limits[2] if limits else "?")
    )
    return True


if __name__ == "__main__":  # 便于本地自测
    logging.basicConfig(level=logging.DEBUG)
    print("compute_prefill_tiles(2048, 8, 128) =", compute_prefill_tiles(2048, 8, 128))
    print("compute_prefill_tiles(1, 8, 128)    =", compute_prefill_tiles(1, 8, 128))
    print("install:", install_tile_policy())
