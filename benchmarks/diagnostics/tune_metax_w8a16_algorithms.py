# SPDX-License-Identifier: Apache-2.0
"""Bounded full-call tuning of MetaX W8A16 Dense, Split-K, and Separate.

This is an offline diagnostic.  It never changes the production dispatch and
does not replace model-level accuracy or the official throughput benchmark.
"""

import argparse
import importlib
import json
import platform
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
import triton.testing

from vllm_fl.quantization.w8a16.reference import quantize_weight


SHAPES = {
    "down": (2048, 6144),
    "gate_up": (12288, 2048),
    "qkv": (2560, 2048),
    "o": (2048, 2048),
}
DEFAULT_MS = (1, 16, 32, 64, 128, 512)
ERROR_RTOL = 0.02
ERROR_ATOL = 0.125
ERROR_RELATIVE_L2 = 0.005


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_l2(actual, expected):
    denominator = expected.float().norm()
    if denominator == 0:
        return (actual.float() - expected.float()).norm().item()
    return ((actual.float() - expected.float()).norm() / denominator).item()


def dense_features(m, n, k, config):
    bm, bn, bk, _, _ = config
    grid_m = triton.cdiv(m, bm)
    grid_n = triton.cdiv(n, bn)
    return {
        "output_programs": grid_m * grid_n,
        "k_loops": triton.cdiv(k, bk),
        "eta_mn": (m * n) / (grid_m * bm * grid_n * bn),
        "eta_k": k / (triton.cdiv(k, bk) * bk),
    }


def candidate_sources(backend):
    official = {
        (16, 64, 256, 4, 3),
        (32, 32, 128, 4, 3),
        (32, 64, 128, 4, 3),
        (64, 64, 128, 4, 3),
        (64, 128, 64, 8, 3),
        (32, 128, 64, 2, 3),
    }
    production = {(16, 64, 128, 4, 2), (64, 64, 128, 4, 2)}
    historical = {(32, 64, 128, 4, 2), (128, 128, 32, 8, 2)}
    sources = {}
    for config in backend.DENSE_CONFIGS:
        names = []
        if config in official:
            names.append("official_bf16")
        if config == (64, 128, 64, 8, 3):
            names.append("official_fp16_migration")
        if config in production:
            names.append("production_control")
        if config in historical:
            names.append("historical_control")
        sources.setdefault(tuple(config), []).extend(names or ["shared_pool"])
    return {config: sorted(set(names)) for config, names in sources.items()}


def select_dense_configs(backend, m, n, k, budget):
    sources = candidate_sources(backend)
    official = backend.official_bf16_dense_config(m, n, k)
    scored = []
    for config, names in sources.items():
        features = dense_features(m, n, k, config)
        # Soft ranking only.  The official tile and fixed controls survive the
        # budget even when their tile efficiency is low.
        score = (
            features["eta_mn"] * features["eta_k"],
            min(features["output_programs"], 256),
            -features["k_loops"],
        )
        scored.append((score, config, names, features))
    scored.sort(reverse=True)
    required = {
        official,
        (16, 64, 128, 4, 2) if m <= 32 else (32, 64, 128, 4, 2),
    }
    chosen = []
    for _, config, names, features in scored:
        if config in required or len(chosen) < budget:
            chosen.append((config, names, features))
    deduplicated = {}
    for config, names, features in chosen:
        deduplicated[config] = (names, features)
    return [(config, *deduplicated[config]) for config in deduplicated]


def check_output(output, expected):
    error = relative_l2(output, expected)
    if error >= ERROR_RELATIVE_L2:
        raise AssertionError(f"relative L2 {error} >= {ERROR_RELATIVE_L2}")
    torch.testing.assert_close(output, expected, rtol=ERROR_RTOL, atol=ERROR_ATOL)
    return error


def bench(function, warmup, rep):
    return triton.testing.do_bench(
        function, warmup=warmup, rep=rep, return_mode="median"
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--ms", nargs="+", type=int, default=list(DEFAULT_MS))
    parser.add_argument("--dense-budget", type=int, default=4)
    parser.add_argument("--screen-warmup", type=int, default=20)
    parser.add_argument("--screen-rep", type=int, default=60)
    parser.add_argument("--final-warmup", type=int, default=50)
    parser.add_argument("--final-rep", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    backend = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.weight_int8pack_mm"
    )
    metax_mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    torch.manual_seed(20261101)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    metadata = {
        "scope": "offline synthetic operator tuning; production dispatch unchanged",
        "shapes": {name: {"n": SHAPES[name][0], "k": SHAPES[name][1]}
                   for name in args.shapes},
        "ms": args.ms,
        "contract": "BF16 activation, INT8 weight converted to BF16, FP32 accumulation/reduction, FP32 scale, BF16 output",
        "separate_contract": "INT8-to-BF16 conversion, BF16 MM output, FP32 scale, BF16 output",
        "thresholds": {"relative_l2_lt": ERROR_RELATIVE_L2,
                       "rtol": ERROR_RTOL, "atol": ERROR_ATOL},
        "protocol": {"screen_warmup_ms": args.screen_warmup,
                     "screen_rep_ms": args.screen_rep,
                     "final_warmup_ms": args.final_warmup,
                     "final_rep_ms": args.final_rep, "rounds": args.rounds},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "triton": triton.__version__, "backend_file": backend.__file__,
                     "bf16_provider_file": metax_mm.__file__},
        "official_benchmark": False,
    }
    dump(args.output_dir / "metadata.json", metadata)

    screening = []
    raw_final = []
    summary = []
    for shape_name in args.shapes:
        n, k = SHAPES[shape_name]
        original_weight = (
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        q, scales = quantize_weight(original_weight)
        quantized_weight = (q.float() * scales[:, None]).to(torch.bfloat16)

        for m in args.ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            expected = (F.linear(x.float(), q.float()) * scales).to(torch.bfloat16)
            candidates = []
            for config, sources, features in select_dense_configs(
                    backend, m, n, k, args.dense_budget):
                candidates.append(("dense", config, sources, features))
            if m == 1:
                for config in backend.M1_DENSE_CONFIGS:
                    candidates.append(("dense_m1", tuple(config),
                                       ["official_m1_broadcast_bf16"],
                                       dense_features(m, n, k, config)))
            for config in backend.SPLITK_CONFIGS:
                candidates.append(("splitk", tuple(config),
                                   ["shared_splitk_pool"],
                                   dense_features(m, n, k, config[:5])))
            candidates.append(("separate", None, ["production_separate"], None))

            passed = []
            for algorithm, config, sources, features in candidates:
                row = {"shape": shape_name, "m": m, "n": n, "k": k,
                       "algorithm": algorithm, "config": config,
                       "sources": sources, "features": features}
                if algorithm in ("dense", "dense_m1"):
                    reason = backend.dense_rejection_reason(
                        x, q, scales, config, m1=algorithm == "dense_m1"
                    )
                elif algorithm == "splitk":
                    reason = backend.splitk_rejection_reason(x, q, scales, config)
                else:
                    reason = backend.separate_rejection_reason(x, q, scales)
                row["workspace_bytes"] = backend.algorithm_workspace_bytes(
                    x, q, "dense" if algorithm == "dense_m1" else algorithm, config
                )
                if reason is not None:
                    row.update(status="pruned_illegal", rejection_reason=reason)
                    screening.append(row)
                    dump(args.output_dir / "screening.json", screening)
                    continue
                function = lambda algorithm=algorithm, config=config: (
                    backend.run_w8a16_algorithm(x, q, scales, algorithm, config)
                )
                try:
                    row["relative_l2_error"] = check_output(function(), expected)
                    row["median_ms"] = bench(
                        function, args.screen_warmup, args.screen_rep
                    )
                    row["status"] = "passed"
                    passed.append(row)
                except Exception as exc:  # preserve compile/resource failures
                    row.update(status="failed", error_type=type(exc).__name__,
                               error=repr(exc))
                screening.append(row)
                dump(args.output_dir / "screening.json", screening)
                print("SCREEN", json.dumps(row, sort_keys=True), flush=True)

            winners = {}
            for algorithm in ("dense", "dense_m1", "splitk", "separate"):
                eligible = [row for row in passed if row["algorithm"] == algorithm]
                if eligible:
                    winners[algorithm] = min(eligible, key=lambda row: row["median_ms"])
            functions = {
                "bf16_original": lambda: metax_mm.mm(x, original_weight.t()),
                "bf16_quantized_weight": lambda: metax_mm.mm(x, quantized_weight.t()),
                "production_dispatch": lambda: backend.weight_int8pack_mm(
                    x, q, scales
                ),
                "generic_flaggems_dense": lambda: backend._generic_weight_int8pack_mm(
                    x, q, scales
                ),
            }
            for algorithm, winner in winners.items():
                functions[algorithm] = (
                    lambda algorithm=algorithm, config=winner["config"]:
                    backend.run_w8a16_algorithm(x, q, scales, algorithm, config)
                )
            samples = {name: [] for name in functions}
            control_errors = {
                name: check_output(functions[name](), expected)
                for name in ("production_dispatch", "generic_flaggems_dense")
            }
            names = list(functions)
            for round_id in range(args.rounds):
                order = names[round_id:] + names[:round_id]
                for name in order:
                    value = bench(functions[name], args.final_warmup, args.final_rep)
                    samples[name].append(value)
                    raw_final.append({"shape": shape_name, "m": m, "path": name,
                                      "round": round_id + 1, "ms": value})
                    dump(args.output_dir / "raw_final.json", raw_final)
            result = {
                "shape": shape_name, "m": m, "n": n, "k": k,
                "winners": {name: {"config": row["config"],
                                    "screen_ms": row["median_ms"],
                                    "relative_l2_error": row["relative_l2_error"],
                                    "workspace_bytes": row["workspace_bytes"],
                                    "sources": row["sources"]}
                            for name, row in winners.items()},
                "final_median_ms": {name: median(values)
                                    for name, values in samples.items()},
                "control_relative_l2_error": control_errors,
                "quantized_weight_vs_original_relative_l2": relative_l2(
                    functions["bf16_quantized_weight"](),
                    functions["bf16_original"](),
                ),
            }
            summary.append(result)
            dump(args.output_dir / "summary.json", summary)
            print("FINAL", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
