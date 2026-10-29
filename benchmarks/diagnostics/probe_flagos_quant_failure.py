#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
"""Pinpoint the ``default.flagos`` activation-quantization failure.

Runs in the MetaX container with the installed flag_gems/flaggems_vllm stack and
prints, step by step, where ``dynamic_per_token_quant_int8_flaggems_vllm`` fails:
import of ``flaggems_vllm``, import of ``scaled_int8_quant``, its signature, a
direct call with a small BF16 tensor, and finally the plugin wrapper itself. No
model weights or serving parameters are changed.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("=== environment ===", flush=True)
    print("torch:", torch.__version__, flush=True)
    try:
        import importlib.metadata as md

        for name in ("flag-gems", "vllm", "vllm-plugin-fl", "triton"):
            try:
                print(f"{name}: {md.version(name)}", flush=True)
            except Exception:
                print(f"{name}: <not installed>", flush=True)
    except Exception:
        pass
    print("gpu:", torch.cuda.get_device_name(0), flush=True)

    x = torch.randn((8, 6144), device="cuda", dtype=torch.bfloat16)

    print("\n=== step 1: import flaggems_vllm ===", flush=True)
    try:
        import flaggems_vllm  # noqa: F401

        print("OK:", getattr(flaggems_vllm, "__file__", "<no __file__>"), flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)

    print("\n=== step 2: import scaled_int8_quant ===", flush=True)
    scaled_int8_quant = None
    try:
        from flaggems_vllm.ops.scaled_int8_quant import scaled_int8_quant

        print("OK:", scaled_int8_quant, flush=True)
        try:
            print("signature:", inspect.signature(scaled_int8_quant), flush=True)
        except Exception:
            print("signature unavailable:\n", traceback.format_exc(), flush=True)
        print("source file:", inspect.getsourcefile(scaled_int8_quant), flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)

    print(
        "\n=== step 3: direct call "
        "scaled_int8_quant(x, scale=None, azp=None, symmetric=True) ===",
        flush=True,
    )
    if scaled_int8_quant is not None:
        try:
            out = scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
            print("OK, return type:", type(out), flush=True)
            if isinstance(out, (tuple, list)):
                print("  arity:", len(out), flush=True)
                for i, o in enumerate(out):
                    if hasattr(o, "shape"):
                        print(f"  [{i}] shape={tuple(o.shape)} dtype={o.dtype}", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)

    print("\n=== step 4: call the plugin wrapper ===", flush=True)
    try:
        from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
            dynamic_per_token_quant_int8_flaggems_vllm,
        )

        out, scale = dynamic_per_token_quant_int8_flaggems_vllm(x)
        print(
            "OK: out", tuple(out.shape), out.dtype, "scale",
            tuple(scale.shape), scale.dtype, flush=True,
        )
    except Exception:
        print(traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
