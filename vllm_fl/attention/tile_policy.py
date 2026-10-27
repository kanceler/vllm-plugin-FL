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

* decode（``max_seqlen_q <= 1``）：BLOCK_M 与并行 softmax 分段沿用上游，
  但流水级数取 1（实测快 ~30%）
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


# prefill 的 K/V 瓦片宽度，由 compute_prefill_tiles() 写入、_get_tile_size 钩子读取。
# 上游启动器里 BLOCK_M 的计算（约 860 行）早于 TILE_SIZE 的计算（约 906 行），
# 所以同一次调用里顺序是正确的。
_PREFILL_TILE: dict = {"value": None}


def compute_prefill_tiles(
    max_seqlen_q: int,
    num_queries_per_kv: int,
    head_size: int,
):
    """返回 (BLOCK_M, num_warps, num_stages)。``None`` 表示沿用 Triton 默认。

    具体推导见 :mod:`vllm_fl.attention.config_model` —— 分块与流水深度
    都是从设备的片上资源与占用率约束推出来的，不依赖平台名。
    """
    from vllm_fl.attention.config_model import (
        derive_decode_tiles,
        derive_prefill_config,
    )

    if max_seqlen_q <= 1:
        return derive_decode_tiles(head_size, num_queries_per_kv)

    bm, warps, stages, tile = derive_prefill_config(
        head_size, num_queries_per_kv, max_seqlen_q
    )
    _PREFILL_TILE["value"] = tile
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

        # K/V 瓦片宽度同样由推导给出；decode 沿用上游的默认值。
        _orig_get_tile = getattr(src_mod, "_get_tile_size", None)

        def _fl_get_tile_size(head_size, sliding_window, element_size, is_prefill):
            if is_prefill and _PREFILL_TILE["value"]:
                return _PREFILL_TILE["value"]
            return _orig_get_tile(head_size, sliding_window, element_size, is_prefill)

        if _orig_get_tile is not None:
            ns["_get_tile_size"] = _fl_get_tile_size
        exec(compile(ast.Module(body=[fn], type_ignores=[]), src_path, "exec"), ns)  # noqa: S102
        tuned = ns["unified_attention"]
    except Exception as exc:  # noqa: BLE001
        print("[FL] attention tile policy: patch failed: %r" % (exc,))
        return False

    binding_mod.unified_attention = tuned
    _installed = True

    from vllm_fl.attention.config_model import device_profile

    prof = device_profile()
    print(
        "[FL] attention tile policy installed (device: sms=%s, smem=%.0f KB, warp=%s). "
        "Prefill tile/pipeline/TILE and decode launch parameters are all derived "
        "from device limits."
        % (prof.sm_count if prof else "?",
           (prof.smem_per_sm / 1024.0) if prof else 0,
           prof.warp_size if prof else "?")
    )
    return True


if __name__ == "__main__":  # 便于本地自测
    logging.basicConfig(level=logging.DEBUG)
    print("compute_prefill_tiles(2048, 8, 128) =", compute_prefill_tiles(2048, 8, 128))
    print("compute_prefill_tiles(1, 8, 128)    =", compute_prefill_tiles(1, 8, 128))
    print("install:", install_tile_policy())
