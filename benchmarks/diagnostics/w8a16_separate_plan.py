# SPDX-License-Identifier: Apache-2.0
"""CPU-only protocol and result reduction for the bounded Separate experiment."""

import math
from statistics import median


SHAPES = {
    "down": (2048, 6144),
    "gate_up": (12288, 2048),
    "qkv": (2560, 2048),
    "o": (2048, 2048),
}
MAIN_MS = (160, 1554, 2048)
VALIDATION_MS = (320,)
BOUNDARY_MS = {
    "down": (64, 72, 104, 112),
    "gate_up": (96, 104),
    "qkv": (64, 72),
    "o": (96, 104),
}
CONFIGS = tuple((block, warps) for block in (512, 1024, 2048, 4096)
                for warps in (2, 4, 8))
SEEDS = (20261105, 20271105)
PROTOCOL = {
    "rounds": 5,
    "warmup_ms": 100,
    "rep_ms": 300,
    "seeds": SEEDS,
    "relative_l2_lt": 0.005,
    "rtol": 0.02,
    "atol": 0.125,
    "diagnostic_stability_limit_pct": 2.0,
    "timer": "triton.testing.do_bench median; default cache treatment",
    "scope": "synthetic operator diagnostic, not official benchmark",
}


def config_id(config):
    return "b%d-w%d" % tuple(config)


def plan(stage="convert"):
    if stage not in ("convert", "scale"):
        raise ValueError("unknown stage")
    return {
        "stage": stage,
        "shapes": SHAPES,
        "configs": CONFIGS,
        "search_count": len(SHAPES) * len(CONFIGS)
        * (1 if stage == "convert" else 2),
        "search_ms": None if stage == "convert" else (160, 2048),
        "main_points": [(shape, m) for shape in SHAPES for m in MAIN_MS],
        "validation_points": [(shape, m) for shape in SHAPES
                              for m in (*VALIDATION_MS, *BOUNDARY_MS[shape])],
        "protocol": PROTOCOL,
        "scale_requires": "explicit invocation and eligible conversion summary",
        "selection": "two finalists from component search; full-call selection on main points only",
        "validation": "no configuration reselection on validation/boundary points",
        "numerics": "exact INT8->BF16; FP32 output scale after BF16 MM",
    }


def summarize(samples):
    values = [float(x) for x in samples]
    if len(values) != PROTOCOL["rounds"]:
        raise ValueError("expected five independently recorded timing rounds")
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("invalid timing")
    mid = median(values)
    return {"median_ms": mid, "samples_ms": values,
            "min_ms": min(values), "max_ms": max(values),
            "range_pct": 100 * (max(values) - min(values)) / mid}


def classify(candidate, baseline):
    if not all(math.isfinite(x) and x > 0 for x in (candidate, baseline)):
        raise ValueError("invalid comparison")
    if candidate < baseline * 0.99:
        return "effective_improvement"
    if candidate > baseline * 1.01:
        return "effective_regression"
    return "normal_variation"


def shortlist(records, count=2):
    """Rank by worst seed/representative-point ratio, never by a lucky sample."""
    scores = {}
    for record in records:
        if record.get("status") != "valid":
            continue
        config = tuple(record["config"])
        ratios = record["ratios_to_direct_default"]
        if not ratios or any(not math.isfinite(x) or x <= 0 for x in ratios):
            raise ValueError("invalid candidate ratios")
        scores[config] = max(ratios)
    return [list(c) for c in sorted(scores, key=lambda c: (scores[c], c))[:count]]


def full_call_gate(rows, path, control="direct_default"):
    """Conservative diagnostic adoption gate, not a competition scoring rule.

    Every main point/seed must improve by >1% against a matched entry and the
    existing wrapper. This deliberately avoids hiding a point's regression in
    a cross-shape mean. Validation is an additional veto, not a selection set.
    """
    relevant = [r for r in rows if r["role"] == "main"]
    if {(r["m"], r["seed"]) for r in relevant} != {
            (m, seed) for m in MAIN_MS for seed in SEEDS}:
        return False
    for row in relevant:
        timings = row["timings"]
        if path not in timings or not row["bf16_seed_stable"]:
            return False
        for name in (path, control, "production", "bf16_original"):
            if timings[name]["range_pct"] > PROTOCOL["diagnostic_stability_limit_pct"]:
                return False
        for baseline in (control, "production"):
            if classify(timings[path]["median_ms"],
                        timings[baseline]["median_ms"]) != "effective_improvement":
                return False
    return True


def validation_gate(rows, path, control="direct_default"):
    if not rows:
        return False
    for row in rows:
        if not row.get("graph_passed") or not row["bf16_seed_stable"]:
            return False
        for name in (path, control, "production", "bf16_original"):
            if row["timings"][name]["range_pct"] > PROTOCOL["diagnostic_stability_limit_pct"]:
                return False
        for baseline in (control, "production"):
            if classify(row["timings"][path]["median_ms"],
                        row["timings"][baseline]["median_ms"]) == "effective_regression":
                return False
    return True
