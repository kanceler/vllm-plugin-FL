# SPDX-License-Identifier: Apache-2.0
"""Compare host entry overhead while sharing exactly the same Split-K kernels.

The historical helper is extracted from the recorded production commit. It is
executed in a copied namespace, never installed into the production backend.
This isolates entry structure; it does not reproduce historical device timing.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import time

from tune_metax_w8a16_splitk import (
    append, fingerprint, gpu_modules, preflight, tensor_info, write_json,
)


def load_historical_helper(path, backend):
    source = path.read_text()
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("expected one archived helper function")
    if tree.body[0].name != "_small_m_splitk":
        raise ValueError("unexpected helper name")
    namespace = dict(vars(backend))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_small_m_splitk"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    preflight(root)
    write_json(root / "metadata.json", {
        "fingerprint": fingerprint(),
        "historical_reference_commit": "f6a4257e3d042acd26a7d3c496848025a4b32c07",
        "helper_sha256": hashlib.sha256(args.historical_helper.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "synthetic eager full call; shared current kernels; no historical hardware reproduction",
        "protocol": {"seeds": [20261028, 20261104], "ms": [1, 16],
                     "warmup_ms": 100, "rep_ms": 300, "rounds": 5},
    })
    torch, triton, backend, mm = gpu_modules()
    import triton.testing
    from vllm_fl.quantization.w8a16.reference import quantize_weight

    assert (backend._SPLIT_K, backend._BLOCK_N, backend._BLOCK_K,
            backend._REDUCE_BLOCK) == (8, 64, 128, 256)
    old = load_historical_helper(args.historical_helper, backend)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    results = []
    try:
        with torch.inference_mode():
            for seed in (20261028, 20261104):
                torch.manual_seed(seed)
                weight = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16) * 0.02
                q, scale = quantize_weight(weight)
                for m in (1, 16):
                    x = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16)
                    expected = (torch.nn.functional.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
                    config = (16, 64, 128, 4, 2, 8)
                    paths = {
                        "historical_entry_shared_kernels": lambda: old(x, q, scale),
                        "configured_entry": lambda: backend._configured_splitk(x, q, scale, config),
                        "public_explicit_entry": lambda: backend.run_w8a16_algorithm(x, q, scale, "splitk", config),
                        "bf16_original": lambda: mm.mm(x, weight.t()),
                    }
                    errors = {}
                    for name, fn in paths.items():
                        output = fn()
                        if name != "bf16_original":
                            error = ((output.float()-expected.float()).norm() /
                                     expected.float().norm().clamp_min(1e-12)).item()
                            if not math.isfinite(error) or error >= 0.005:
                                raise AssertionError(f"{name}: relative L2 {error}")
                            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.125)
                            errors[name] = error
                    torch.cuda.synchronize()
                    names = list(paths)
                    samples = {name: [] for name in names}
                    for round_id in range(5):
                        order = names[round_id % len(names):] + names[:round_id % len(names)]
                        for name in order:
                            value = float(triton.testing.do_bench(
                                paths[name], warmup=100, rep=300, return_mode="median"))
                            if not math.isfinite(value) or value <= 0:
                                raise RuntimeError("invalid timing")
                            samples[name].append(value)
                            append(root / "raw.jsonl", {"seed": seed, "m": m,
                                "round": round_id+1, "path": name, "ms": value})
                    timing = {name: median(values) for name, values in samples.items()}
                    results.append({"seed": seed, "m": m, "config": config,
                        "timings_ms": timing, "errors": errors,
                        "public_vs_historical_entry_increase_pct": 100 * (
                            timing["public_explicit_entry"] /
                            timing["historical_entry_shared_kernels"] - 1),
                        "layouts": {name: tensor_info(t) for name, t in
                                    (("x", x), ("q", q), ("scale", scale), ("weight", weight))}})
                    write_json(root / "summary.json", {"complete": False, "results": results})
        write_json(root / "summary.json", {"complete": True, "results": results,
            "limitation": "entry difference only; no standalone kernel/CPU wall/graph timing claim"})
    finally:
        status = subprocess.run(["mx-smi"], capture_output=True, text=True)
        (root / "gpu-before-process-exit.txt").write_text(status.stdout+status.stderr)


if __name__ == "__main__":
    main()
