# SPDX-License-Identifier: Apache-2.0
"""Validate and time the production MetaX W8A16 int8pack dispatch."""

import argparse
import importlib
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
import flag_gems

from flag_gems.ops.weight_int8pack_mm import weight_int8pack_mm as generic_int8pack
from flag_gems.runtime.backend._metax.ops.weight_int8pack_mm import (
    _can_use_small_m_splitk,
)
from vllm_fl.quantization.w8a16.dequant import dequantize_weight
from vllm_fl.quantization.w8a16.int8pack import int8pack_linear
from vllm_fl.quantization.w8a16.reference import quantize_weight


@triton.jit
def _output_scale(Y, S, O, N: tl.constexpr, SIZE: tl.constexpr,
                  BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SIZE
    y = tl.load(Y + idx, mask, other=0).to(tl.float32)
    scale = tl.load(S + idx % N, mask, other=0)
    tl.store(O + idx, (y * scale).to(tl.bfloat16), mask)


def scale_output(y, scale):
    output = torch.empty_like(y)
    _output_scale[(triton.cdiv(y.numel(), 1024),)](
        y, scale, output, y.shape[1], y.numel(), BLOCK=1024)
    return output


@triton.jit
def _convert_weight(Q, W, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SIZE
    value = tl.load(Q + idx, mask, other=0).to(tl.bfloat16)
    tl.store(W + idx, value, mask)


def convert_weight(q):
    output = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    _convert_weight[(triton.cdiv(q.numel(), 1024),)](
        q, output, q.numel(), BLOCK=1024)
    return output


def relative_l2(actual, expected):
    return (
        (actual.float() - expected.float()).norm() / expected.float().norm()
    ).item()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--large-m", action="store_true",
                        help="Compare complete large-M scale-after-dot fallback")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metax_mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

    torch.manual_seed(20261028)
    n, k = 2048, 6144
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scale = quantize_weight(weight)
    dequantized = dequantize_weight(q, scale)
    metadata = {
        "seed": 20261028,
        "shape": {"n": n, "k": k, "m": ([128, 256, 512, 1024, 2048, 4096]
                  if args.large_m else [1, 16, 32, 64, 128, 512, 2048])},
        "contract": "FP32 dot -> FP32 channel scale -> BF16 output",
        "timing": "three GPU median rounds; warmup=100ms, rep=300ms",
        "official_benchmark": False,
    }
    dump(args.output_dir / "metadata.json", metadata)

    rows = []
    raw = []
    for m in metadata["shape"]["m"]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        paths = {
            "production_custom_op": lambda: int8pack_linear(x, q, scale),
            "production_flaggems": lambda: flag_gems.weight_int8pack_mm(x, q, scale),
            "generic_flaggems": lambda: generic_int8pack(x, q, scale),
            "separate_dequant_full": lambda: metax_mm.mm(
                x, dequantize_weight(q, scale).t()
            ),
            "predequantized_bf16_mm": lambda: metax_mm.mm(x, dequantized.t()),
            "bf16_metax": lambda: metax_mm.mm(x, weight.t()),
        }
        if args.large_m:
            # This explicit candidate materializes a BF16 GEMM result before
            # FP32 scaling. Its extra rounding must be measured, not assumed
            # identical to scaling the FP32 accumulator in the fused epilogue.
            converted = convert_weight(q)
            unscaled = metax_mm.mm(x, converted.t())

            def separate_after_dot():
                result = metax_mm.mm(x, convert_weight(q).t())
                return scale_output(result, scale)

            paths = {
                "bf16_metax": paths["bf16_metax"],
                "separate_dequant_full": paths["separate_dequant_full"],
                "generic_flaggems": paths["generic_flaggems"],
                "separate_scale_after_full": separate_after_dot,
                "component_int8_to_bf16": lambda: convert_weight(q),
                "component_unscaled_mm": lambda: metax_mm.mm(x, converted.t()),
                "component_output_scale": lambda: scale_output(unscaled, scale),
            }
        errors = {}
        for name, fn in paths.items():
            output = fn()
            if name == "bf16_metax" or name.startswith("component_"):
                continue
            error = relative_l2(output, expected)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.125)
            errors[name] = error

        names = list(paths)
        samples = {name: [] for name in names}
        for round_id in range(3):
            ordered = names[round_id:] + names[:round_id]
            for name in ordered:
                value = triton.testing.do_bench(
                    paths[name], warmup=100, rep=300, return_mode="median"
                )
                samples[name].append(value)
                raw.append(
                    {"m": m, "round": round_id + 1, "path": name, "ms": value}
                )
                dump(args.output_dir / "raw.json", raw)

        row = {
            "m": m,
            "small_m_splitk": _can_use_small_m_splitk(x, q, scale),
            "errors": errors,
            "timings_ms": {name: median(values) for name, values in samples.items()},
        }
        rows.append(row)
        dump(args.output_dir / "summary.json", rows)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
