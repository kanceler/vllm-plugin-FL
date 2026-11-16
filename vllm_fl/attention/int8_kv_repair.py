"""Opt-in Iluvatar INT8 KV repair, independent of cache-dtype selection.

Keep the original launcher and JIT object intact for BF16 and A/B controls.
Exact source matching deliberately fails closed on incompatible vLLM builds.
"""
import os
import types


def install_int8_kv_repair():
    if os.environ.get("VLLM_FL_INT8_KV_REPAIR", "0") != "1":
        return False
    import torch
    from vllm.triton_utils import triton
    from vllm.v1.attention.backends import triton_attn as backend
    from vllm.v1.kv_cache_interface import KVQuantMode

    original = backend.unified_attention
    if getattr(original, "_fl_int8_repair", False):
        return True
    ns = dict(original.__globals__)
    kernel = ns["kernel_unified_attention"]
    old = "L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
    if kernel.src.count(old) != 1:
        raise RuntimeError("INT8 KV repair: unsupported online-softmax source")
    fixed = triton.jit(kernel.fn)
    # With no sink the empty prefix has zero probability mass. Explicit zero
    # avoids dependence on exp(-inf) elimination in the device compiler.
    fixed._unsafe_update_src(kernel.src.replace(
        old, "L = tl.full([BLOCK_M], 1.0 if USE_SINKS else 0.0, dtype=tl.float32)"))
    ns["kernel_unified_attention"] = fixed
    if "_fl_prefill_tiles" not in ns:
        raise RuntimeError("INT8 KV repair requires final tile policy installed first")
    # Preserve final's device-derived Prefill BM, tile, warps and stages.
    repaired = types.FunctionType(original.__code__, ns, original.__name__,
                                  original.__defaults__, original.__closure__)
    repaired.__kwdefaults__ = original.__kwdefaults__

    def dispatch(*args, **kwargs):
        # Backend invokes the launcher with keywords; diagnostic callers follow
        # the same convention. Other modes retain the original implementation.
        if kwargs.get("kv_quant_mode") == KVQuantMode.INT8_PER_TOKEN_HEAD:
            if kwargs["q"].dtype != torch.bfloat16:
                raise RuntimeError("INT8 KV repair validated only for BF16 queries")
            return repaired(*args, **kwargs)
        return original(*args, **kwargs)

    dispatch._fl_int8_repair = True
    dispatch._fl_int8_repaired = repaired
    backend.unified_attention = dispatch
    print("[FL] INT8 KV repair enabled: explicit empty-prefix mass; final tile policy", flush=True)
    return True
