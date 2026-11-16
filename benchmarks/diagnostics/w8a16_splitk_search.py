# SPDX-License-Identifier: Apache-2.0
"""CPU-only candidate planning for the approved 20+40 MetaX Split-K sweep.

Heuristic exclusions are not legality failures. Their audit representatives and
all six historical controls survive, so the effect of pruning remains visible.
"""

from collections import Counter
from itertools import product
from math import ceil


SHAPES = {"down": (2048, 6144), "gate_up": (12288, 2048),
          "qkv": (2560, 2048), "o": (2048, 2048)}
WIDE_MS = {"down": (1, 4, 16, 32, 64), "gate_up": (1, 4, 16, 24, 32),
           "qkv": (1, 4, 16, 32, 64), "o": (1, 4, 16, 32, 64)}
REFINED_MS = {
    "down": (2, 3, 8, 24, 48, 56, 72, 80, 96, 104, 112),
    "gate_up": (2, 3, 8, 40, 48, 64, 88, 96, 104, 112),
    "qkv": (2, 3, 8, 24, 48, 56, 72, 80, 96),
    "o": (2, 3, 8, 24, 48, 56, 72, 80, 96, 104),
}
RANGES = ((16, 32, 64), (32, 64, 128), (64, 128, 256),
          (2, 4, 8), (1, 2, 3), (2, 4, 8, 16, 32))
CONTROLS = tuple((bm, 64, 128, 4, 2, s)
                 for bm in (16, 32, 64) for s in (4, 8))
POLICY = {
    "version": 1, "pipeline": "basic", "reduce_block": 256, "reduce_warps": 4,
    "max_m_padding_ratio": 2, "max_waves_for_short_split": 16,
    "short_split_iters": 2, "top_k": 5, "near_tie_ratio": 1.01,
    "control_stability_threshold_pct": 2.0,
    "scope": "finite candidate pool; no S=1; no M>=160; no runtime dispatch",
}


def config_id(config):
    return "-".join(map(str, config))


def legality(n, k, config):
    bm, bn, bk, warps, stages, splits = config
    if any(x <= 0 for x in config):
        return "nonpositive_config"
    if n % bn:
        return "N_not_divisible_by_BN"
    if k % (bk * splits):
        return "K_not_divisible_by_BK_times_S"
    # These are the implemented sweep domain, not universal Triton limits.
    if any(x & (x - 1) for x in (bm, bn, bk, warps)):
        return "outside_power_of_two_tile_domain"
    return None


def features(m, n, k, config, sm_count, warp_size):
    bm, bn, bk, warps, stages, splits = config
    tiles = ceil(m / bm) * ceil(n / bn)
    return {
        "output_tiles": tiles, "partial_programs": tiles * splits,
        "programs_per_sm": tiles * splits / sm_count,
        "k_iters_per_split": k / (bk * splits),
        "eta_m": m / (ceil(m / bm) * bm),
        "eta_mn": m * n / (ceil(m / bm) * bm * ceil(n / bn) * bn),
        "workspace_bytes": 4 * splits * m * n,
        "accumulator_elements_per_thread_proxy": bm * bn / (warps * warp_size),
        # Planning estimate only: compiler layouts/pipeline can differ.
        "staged_operand_bytes_proxy": (2 * bm + bn) * bk * max(1, stages),
    }


def generate_candidates(m, n, k, sm_count, warp_size, workspace_budget):
    if min(m, n, k, sm_count, warp_size, workspace_budget) <= 0:
        raise ValueError("positive Shape, hardware values and budget required")
    rows = []
    for config in product(*RANGES):
        row = {"config": list(config), "config_id": config_id(config),
               "sources": ["approved_finite_pool"], "heuristic_reasons": []}
        reason = legality(n, k, config)
        if reason:
            row.update(status="pruned_illegal", rejection_reason=reason)
            rows.append(row)
            continue
        f = features(m, n, k, config, sm_count, warp_size)
        row["features"] = f
        if f["workspace_bytes"] > workspace_budget:
            row.update(status="pruned_resource_budget",
                       rejection_reason="FP32_workspace_exceeds_declared_budget")
            rows.append(row)
            continue
        bm, bn, bk, warps, stages, splits = config
        # BM=16 remains usable even when M=1. Larger padding is only soft-pruned.
        if bm > max(16, POLICY["max_m_padding_ratio"] * m):
            row["heuristic_reasons"].append("excess_M_padding")
        if (f["programs_per_sm"] > POLICY["max_waves_for_short_split"]
                and f["k_iters_per_split"] <= POLICY["short_split_iters"]):
            row["heuristic_reasons"].append("many_programs_with_short_K_split")
        if config in CONTROLS:
            row["sources"].append("historical_control")
        # Retain each excluded tile/S structure at the established launch setting;
        # this also keeps S=32 boundary probes whenever Shape legality permits.
        if row["heuristic_reasons"] and warps == 4 and stages == 2:
            row["sources"].append("heuristic_audit")
        retained = (not row["heuristic_reasons"] or config in CONTROLS
                    or "heuristic_audit" in row["sources"])
        row["status"] = "scheduled" if retained else "pruned_heuristic"
        rows.append(row)
    return rows


def shortlist(measurements, top_k=5, near_tie_ratio=1.01):
    passed = sorted((r for r in measurements if r["status"] == "passed"),
                    key=lambda r: (r["median_ms"], tuple(r["config"])))
    if not passed:
        return []
    selected = {tuple(r["config"]): r for r in passed[:top_k]}
    # Retain the fastest candidate for each tile/S structure if within 1%.
    structures = {(r["config"][0], r["config"][1], r["config"][2], r["config"][5])
                  for r in selected.values()}
    for row in passed:
        if row["median_ms"] > passed[0]["median_ms"] * near_tie_ratio:
            break
        structure = tuple(row["config"][:3]) + (row["config"][5],)
        if structure not in structures:
            selected.setdefault(tuple(row["config"]), row)
            structures.add(structure)
    return list(selected.values())


def pool_from_wide(points):
    pool = {config: {"historical_control"} for config in CONTROLS}
    for point in points:
        rejected = {tuple(r["config"]) for r in point.get("independent_rejections", [])}
        valid = [r for r in point["screening"] if tuple(r["config"]) not in rejected]
        for row in shortlist(valid):
            pool.setdefault(tuple(row["config"]), set()).add(
                f"wide_M={point['m']}")
    return [{"config": list(c), "sources": sorted(s)} for c, s in sorted(pool.items())]


def classify_latency(candidate_ms, baseline_ms, baseline_spread_pct=None):
    if candidate_ms <= 0 or baseline_ms <= 0:
        raise ValueError("latencies must be positive")
    # A BF16 control that varies more than the policy threshold across baseline
    # seeds cannot support a ">1% improvement" claim; report it honestly instead
    # of letting a noisy baseline manufacture a false winner.
    if (baseline_spread_pct is not None
            and baseline_spread_pct > POLICY["control_stability_threshold_pct"]):
        return "control_unstable"
    # Compare unrounded latencies to the two exact policy boundaries.
    if candidate_ms < baseline_ms * 0.99:
        return "effective_improvement"
    if candidate_ms > baseline_ms * 1.01:
        return "effective_regression"
    return "normal_fluctuation"


def plan(sm_count, warp_size, workspace_budget):
    points = []
    for shape, ms in WIDE_MS.items():
        n, k = SHAPES[shape]
        for m in ms:
            rows = generate_candidates(m, n, k, sm_count, warp_size, workspace_budget)
            points.append({"shape": shape, "m": m, "n": n, "k": k,
                           "counts": dict(Counter(r["status"] for r in rows)),
                           "candidates": rows})
    return {"policy": POLICY, "wide_points": points, "refined_ms": REFINED_MS,
            "hardware_inputs": {"sm_count": sm_count, "warp_size": warp_size},
            "workspace_budget_bytes": workspace_budget}
