# SPDX-License-Identifier: Apache-2.0
"""Serial, resumable full-call Dense sweep; production dispatch is unchanged.

The controller has no GPU context. A disposable subprocess executes each Shape,
isolating compiler aborts without paying an import/process cost per candidate.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import socket
from statistics import median
import subprocess
import sys
import time

from w8a16_splitk_search import classify_latency, config_id
from w8a16_dense_search import (
    POLICY, SHAPES, WIDE_MS, REFINED_MS, generate_candidates, legality,
    plan, pool_from_wide, shortlist,
)



def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Do not silently discard a torn append, including the last line.
            raise RuntimeError(f"invalid journal at {path}:{i + 1}; preserve and inspect")
    return rows


def gpu_modules():
    import torch
    import triton
    backend = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.weight_int8pack_mm")
    mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    return torch, triton, backend, mm


def fingerprint():
    torch, triton, backend, mm = gpu_modules()
    if not all(callable(getattr(backend, name, None)) for name in (
            "run_w8a16_algorithm", "dense_rejection_reason")):
        raise RuntimeError("wrong backend import: explicit Dense API missing")
    if getattr(backend, "_REDUCE_BLOCK", None) != POLICY["reduce_block"]:
        raise RuntimeError("Reduce configuration changed; review before starting sweep")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; no sweep started")
    props = torch.cuda.get_device_properties(0)
    sm_count = getattr(props, "multi_processor_count", None)
    warp_size = getattr(props, "warp_size", None)
    if not warp_size:
        # MetaX target from the active compiler, never an NVIDIA warp assumption.
        target = triton.runtime.driver.active.get_current_target()
        warp_size = target.warp_size
    if not sm_count or not warp_size:
        raise RuntimeError("actual SM count/warp size unavailable; refuse fallback estimates")
    root = Path(mm.__file__).resolve().parents[1]
    libentry = importlib.import_module("flag_gems.utils.libentry")
    ref = importlib.import_module("vllm_fl.quantization.w8a16.reference")
    files = [Path(backend.__file__), Path(mm.__file__), Path(libentry.__file__),
             Path(ref.__file__), root / "tune_configs.yaml", root / "mm_metax_expand.yaml",
             Path(__file__), Path(__file__).with_name("w8a16_splitk_search.py"),
             Path(__file__).with_name("w8a16_dense_search.py")]
    driver = subprocess.run(["mx-smi"], capture_output=True, text=True, check=True).stdout
    # Driver status is stable metadata; current utilization/temperature are not
    # part of resume identity. Full raw status is preserved in preflight files.
    driver_versions = [line.strip() for line in driver.splitlines()
                       if "Driver Version" in line or "MACA Version" in line]
    return {"torch": torch.__version__, "triton": triton.__version__,
            "python": platform.python_version(), "hostname": socket.gethostname(),
            "device": str(props), "device_name": torch.cuda.get_device_name(0),
            "sm_count": sm_count, "warp_size": warp_size,
            "driver_versions": driver_versions,
            "source_hashes": {str(p.resolve()): digest(p) for p in files},
            "flags": {k: v for k, v in os.environ.items()
                      if k.startswith(("VLLM_FL_W8A16", "USE_FLAGTUNE"))
                      or k in ("PYTHONPATH", "MACA_PATH", "LD_LIBRARY_PATH",
                               "TRITON_CACHE_DIR", "TRITON_PRINT_AUTOTUNING")}}


def preflight(folder):
    driver = subprocess.run(["mx-smi"], capture_output=True, text=True)
    (folder / "gpu-before.txt").write_text(driver.stdout + driver.stderr)
    if driver.returncode or "no process found" not in driver.stdout.lower():
        raise RuntimeError("GPU not confirmed idle by mx-smi; inspect gpu-before.txt")
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("port 9031 occupied; no independent GPU sweep started")
    processes = subprocess.run(["ps", "-eo", "pid,ppid,args"],
                               capture_output=True, text=True, check=True).stdout
    (folder / "processes-before.txt").write_text(processes)
    conflicts = [line for line in processes.splitlines()
                 if any(term in line for term in (
                     "vllm serve", "EngineCore", "benchmark_throughput_serve.py",
                     "evalscope eval", "pytest", "benchmark_flaggems_"))
                 and "ps -eo" not in line]
    if conflicts:
        raise RuntimeError("possible competing task; inspect processes-before.txt")


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "address_mod_256": tensor.data_ptr() % 256}


def kernel_snapshot(module):
    result = {}
    for name in ("mm_kernel", "mm_kernel_nn", "mm_kernel_nt", "mm_kernel_splitk",
                 "mm_kernel_splitk_partial"):
        entry = getattr(module, name, None)
        current, seen, tuner = entry, set(), None
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "best_config"):
                tuner = current
                break
            current = getattr(current, "fn", None)
        if tuner is not None:
            result[name] = {"best_config": str(tuner.best_config),
                            "raw_key_fields": getattr(tuner, "keys", None),
                            "new_timing_count_last_tuner_run": getattr(
                                tuner, "benchmark_success_count", None),
                            "timing_cache_hits_last_tuner_run": getattr(
                                tuner, "benchmark_cache_hit_count", None)}
    # A best_config snapshot can be stale if LibEntry bypasses its tuner. Label
    # it honestly; capture the actual called public subpath independently below.
    return {"kind": "tuner_state_snapshot_not_per_launch_proof", "kernels": result}


def compiled_resources(entry):
    rows = []
    for cache in getattr(entry, "kernel_cache", ()):
        for value in cache.values():
            if not isinstance(value, tuple) or len(value) < 2:
                continue
            kernel, constexprs = value[:2]
            meta = getattr(kernel, "metadata", None)
            rows.append({"constexprs": constexprs,
                         "tuned_constants": value[2] if len(value) > 2 else None,
                         "n_regs": getattr(kernel, "n_regs", None),
                         "n_spills": getattr(kernel, "n_spills", None),
                         "metadata": meta._asdict() if hasattr(meta, "_asdict") else str(meta)})
    return rows


def traced_mm(module, a, b):
    calls, original = [], {}
    for name in ("gemv_mm", "gemv_mm_k_parallel", "splitk_mm_two_step",
                 "splitk_mm", "general_mm_nn", "general_mm_nt", "general_mm"):
        if not hasattr(module, name):
            continue
        original[name] = getattr(module, name)

        def wrapper(*args, _name=name, _function=original[name], **kwargs):
            calls.append(_name)
            return _function(*args, **kwargs)

        setattr(module, name, wrapper)
    try:
        output = module.mm(a, b)
        evidence = {"called_subpaths": calls, "tuner": kernel_snapshot(module),
                    "compiled_kernel_cache_snapshot": {
                        name: compiled_resources(getattr(module, name, None))
                        for name in ("mm_kernel", "mm_kernel_nn", "mm_kernel_nt",
                                     "mm_kernel_splitk", "mm_kernel_splitk_partial")}}
        return output, evidence
    finally:
        for name, function in original.items():
            setattr(module, name, function)


def worker(args):
    folder = args.output_dir / args.shape / f"m{args.m}"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    if fingerprint() != metadata["fingerprint"]:
        raise RuntimeError("source/environment/device changed; use a new experiment directory")
    torch, triton, backend, mm = gpu_modules()
    from vllm_fl.quantization.w8a16.reference import quantize_weight
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    n, k = SHAPES[args.shape]
    shape_index = list(SHAPES).index(args.shape)
    # Per-problem seeds make inputs invariant to ordering, pruning and resumes.
    base_seed = 20261104 + shape_index * 100000
    baseline_seeds = [base_seed + 10000 * i for i in range(args.baseline_seeds)]

    def inputs(seed):
        torch.manual_seed(seed)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        weight[0] = 0
        q, scale = quantize_weight(weight)
        torch.manual_seed(seed + args.m * 101)
        x = torch.randn(args.m, k, device="cuda", dtype=torch.bfloat16)
        expected = (torch.nn.functional.linear(x.float(), q.float()) * scale).to(
            torch.bfloat16)
        return x, weight, q, scale, expected

    def check(output, expected):
        delta = (output.float() - expected.float()).norm()
        denominator = expected.float().norm()
        error = (delta / denominator).item() if denominator else delta.item()
        if not math.isfinite(error) or error >= 0.005:
            raise AssertionError(f"relative L2 {error} >= 0.005")
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.125)
        return error

    def measure(function, warmup, rep):
        value = float(triton.testing.do_bench(function, warmup=warmup, rep=rep,
                                              return_mode="median"))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"invalid timing {value}")
        return value

    with torch.inference_mode():
        x, weight, q, scale, expected = inputs(base_seed)
        if args.stage == "wide":
            candidates = generate_candidates(args.m, n, k,
                metadata["fingerprint"]["sm_count"], metadata["fingerprint"]["warp_size"],
                metadata["protocol"]["workspace_budget_bytes"])
        else:
            pool = json.loads((args.output_dir / args.shape / "pool.json").read_text())
            candidates = [{**r, "config_id": config_id(r["config"]),
                           "status": "scheduled"} for r in pool]
        write_json(folder / "candidates.json", candidates)
        journal = folder / "screening.jsonl"
        done = {r["config_id"]: r for r in read_rows(journal)}
        screening = []
        for candidate in candidates:
            cid = candidate["config_id"]
            if candidate["status"] != "scheduled":
                continue
            if cid in done:
                screening.append(done[cid])
                continue
            row = {**candidate, "shape": args.shape, "m": args.m,
                   "stage": args.stage, "measurement_source": "fresh_full_call",
                   "timing_cache_hits": 0, "new_compilation_count": None,
                   "compilation_cache_source": "not_observed"}
            config = candidate["config"]
            reason = legality(n, k, config) or backend.dense_rejection_reason(
                x, q, scale, config)
            if reason:
                row.update(status="pruned_illegal", rejection_reason=reason)
            else:
                write_json(folder / "active.json", {"phase": "screen", "config_id": cid,
                                                     "config": config})
                start = time.monotonic()
                execution_phase = "first_call_and_correctness"
                try:
                    function = lambda: backend.run_w8a16_algorithm(
                        x, q, scale, "dense", config)
                    row["relative_l2_error"] = check(function(), expected)
                    torch.cuda.synchronize()
                    row["first_call_wall_s"] = time.monotonic() - start
                    execution_phase = "screen_timing"
                    row["median_ms"] = measure(function, 20, 60)
                    row["status"] = "passed"
                except AssertionError as exc:
                    row.update(status="correctness_failed", error=repr(exc))
                except Exception as exc:
                    # Device corruption is not a harmless candidate failure.
                    if any(s in str(exc).lower() for s in (
                            "illegal memory", "device-side", "device lost", "hiperror")):
                        raise
                    row.update(status=("timing_failed" if execution_phase == "screen_timing"
                                       else "compile_resource_or_execution_failed"),
                               error_stage=execution_phase, error=repr(exc),
                               error_type=type(exc).__name__)
                row["screen_wall_s"] = time.monotonic() - start
            append(journal, row)
            screening.append(row)
            write_json(folder / "active.json", {"phase": "idle"})
            print("SCREEN", args.shape, args.m, cid, row["status"], flush=True)
        write_json(folder / "screening.json", screening)
        finalists = shortlist(screening)
        # Old direct-call Dense is always retained for final comparison.
        old = backend.official_bf16_dense_config(args.m, n, k)
        finalist_configs = {tuple(r["config"]) for r in finalists} | {old}
        if not finalists:
            raise RuntimeError("no valid Dense candidate; preserve failure evidence")
        # Fresh independent inputs; none of the final samples enters the screen cache.
        raw_path = folder / "raw_final.jsonl"
        existing = {(r["seed"], r["round"], r["path"]): r for r in read_rows(raw_path)}
        errors, layouts, evidence, rejected = {}, {}, {}, []
        # Adversarial integer endpoints, zero channel, scale range, M-tail and
        # differing K partitions. These gate finalists, not just random data.
        torch.manual_seed(base_seed + 9999)
        edge_x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
        edge_q = torch.randint(-128, 128, (128, k), device="cuda", dtype=torch.int8)
        edge_q[0] = 0
        edge_q[1, ::2] = 127
        edge_q[1, 1::2] = -128
        edge_scale = torch.logspace(-3, 0, 128, device="cuda", dtype=torch.float32)
        edge_expected = (torch.nn.functional.linear(edge_x.float(), edge_q.float())
                         * edge_scale).to(torch.bfloat16)
        for config in sorted(finalist_configs):
            try:
                error = check(backend.run_w8a16_algorithm(
                    edge_x, edge_q, edge_scale, "dense", config), edge_expected)
                errors["edge:" + config_id(config)] = error
            except AssertionError as exc:
                rejected.append({"config": config, "status": "edge_correctness_failed",
                                 "error": repr(exc)})
        edge_rejected = {tuple(r["config"]) for r in rejected}
        write_json(folder / "correctness.json", {"errors": errors, "rejected": rejected})
        for seed in baseline_seeds:
            x, weight, q, scale, expected = inputs(seed)
            rounded = (q.float() * scale[:, None]).to(torch.bfloat16)
            original_y, original_evidence = traced_mm(mm, x, weight.t())
            rounded_y, rounded_evidence = traced_mm(mm, x, rounded.t())
            evidence[str(seed)] = {"original": original_evidence,
                                   "BF16_Q_diagnostic": rounded_evidence}
            layouts[str(seed)] = {name: tensor_info(tensor) for name, tensor in (
                ("x", x), ("original_weight", weight), ("q", q), ("scale", scale),
                ("rounded_weight", rounded))}
            quant_error = ((rounded_y.float() - original_y.float()).norm()
                           / original_y.float().norm().clamp_min(1e-12)).item()
            functions = {"bf16_original": lambda: mm.mm(x, weight.t()),
                         "bf16_quantized_weight": lambda: mm.mm(x, rounded.t())}
            for label, algorithm, cfg in (
                    ("historical_splitk", "splitk", (16, 64, 128, 4, 2, 8)),
                    ("separate", "separate", None)):
                fn = lambda alg=algorithm, c=cfg: backend.run_w8a16_algorithm(x, q, scale, alg, c)
                check(fn(), expected)
                functions[label] = fn
            seed_errors = {"quantized_vs_original_relative_l2": quant_error}
            for config in sorted(finalist_configs - edge_rejected):
                name = "dense:" + config_id(config)
                function = lambda c=config: backend.run_w8a16_algorithm(
                    x, q, scale, "dense", c)
                try:
                    seed_errors[name] = check(function(), expected)
                except AssertionError as exc:
                    rejected.append({"seed": seed, "config": config,
                                     "status": "independent_correctness_failed", "error": repr(exc)})
                    continue
                functions[name] = function
            errors[str(seed)] = seed_errors
            names = list(functions)
            for round_id in range(1, 6):
                offset = (round_id - 1) % len(names)
                order = names[offset:] + names[:offset]
                for name in order:
                    key = (seed, round_id, name)
                    if key in existing:
                        continue
                    write_json(folder / "active.json", {"phase": "final", "path": name,
                                                         "seed": seed, "round": round_id})
                    row = {"seed": seed, "round": round_id, "path": name,
                           "ms": measure(functions[name], 100, 300),
                           "measurement_source": "fresh_full_call",
                           "layouts": layouts[str(seed)], "worker_pid": os.getpid()}
                    append(raw_path, row)
                    existing[key] = row
            write_json(folder / "active.json", {"phase": "idle"})
        raw = list(existing.values())
        results = []
        rejected_configs = {tuple(r["config"]) for r in rejected}
        samples_by_seed = {}
        baseline_by_seed = {}
        for seed in baseline_seeds:
            samples = {}
            for row in raw:
                if row["seed"] == seed:
                    samples.setdefault(row["path"], []).append(row["ms"])
            samples_by_seed[seed] = samples
            baseline_by_seed[seed] = median(samples["bf16_original"])
        baseline_values = list(baseline_by_seed.values())
        baseline_median = median(baseline_values) if baseline_values else 0.0
        bf16_original_spread_pct = (
            100.0 * (max(baseline_values) - min(baseline_values)) / baseline_median
            if baseline_median > 0 else float("inf"))
        bf16_control_stable = (
            bf16_original_spread_pct <= POLICY["control_stability_threshold_pct"])
        for seed in baseline_seeds:
            samples = samples_by_seed[seed]
            baseline = baseline_by_seed[seed]
            for config in sorted(finalist_configs - rejected_configs):
                name = "dense:" + config_id(config)
                if name not in samples or len(samples[name]) != 5:
                    continue
                timing = median(samples[name])
                results.append({"seed": seed, "config": list(config),
                    "median_ms": timing, "samples_ms": samples[name],
                    "bf16_original_ms": baseline,
                    "bf16_Q_ms": median(samples["bf16_quantized_weight"]),
                    "latency_reduction_pct": 100 * (1 - timing / baseline),
                    "classification": classify_latency(
                        timing, baseline, bf16_original_spread_pct)})
        by_config = {}
        for row in results:
            by_config.setdefault(tuple(row["config"]), []).append(row)
        eligible = {c: rs for c, rs in by_config.items()
                   if len(rs) == len(baseline_seeds)}
        winner = min(eligible, key=lambda c: median(
            r["median_ms"] / r["bf16_original_ms"] for r in eligible[c])) if eligible else None
        write_json(folder / "compiled-resources.json", {
            "dense": compiled_resources(backend.weight_int8pack_mm_dense)})
        write_json(folder / "result.json", {
            "complete": winner is not None, "execution_complete": True,
            "failure_reason": None if winner is not None else "no_valid_finalist",
            "shape": args.shape, "m": args.m, "n": n, "k": k,
            "stage": args.stage, "m_source": "boundary_probe" if args.shape == "o" and args.m == 56 else "observed", "selection_source": "offline_tuned_finite_pool",
            "screen_counts": dict(Counter(r["status"] for r in screening)),
            "resume_reused_screen_rows": len(done), "screening": screening,
            "finalists": [list(c) for c in sorted(finalist_configs)],
            "old_control_config": list(old), "results": results,
            "selected_config": list(winner) if winner else None,
            "selected_by": "median normalized latency across baseline seeds",
            "baseline_seed_count": len(baseline_seeds),
            "bf16_original_spread_pct": bf16_original_spread_pct,
            "bf16_control_stable": bf16_control_stable,
            "control_stability_threshold_pct": POLICY["control_stability_threshold_pct"],
            "beats_bf16_over_1pct_on_both_seeds": bool(
                bf16_control_stable and winner is not None and all(
                    r["classification"] == "effective_improvement"
                    for r in eligible[winner])),
            "independent_rejections": rejected, "errors": errors, "layouts": layouts,
            "bf16_evidence": evidence,
            "scope": "synthetic eager steady-state; no model/graph/official performance claim"})


def aggregate(folder):
    rows, complete, winners, failed = [], [], [], []
    for shape in SHAPES:
        for m in (*WIDE_MS[shape], *REFINED_MS[shape]):
            path = folder / shape / f"m{m}" / "result.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            complete.append({"shape": shape, "m": m, "stage": data["stage"]})
            if data["selected_config"] is None or not data.get("complete", False):
                failed.append({"shape": shape, "m": m,
                               "reason": data.get("failure_reason") or "invalid_or_incomplete_result"})
            winners.append({"shape": shape, "m": m, "stage": data["stage"],
                "config": data["selected_config"],
                "beats_bf16_over_1pct_on_both_seeds": data[
                    "beats_bf16_over_1pct_on_both_seeds"],
                "per_seed_results": [r for r in data["results"]
                                     if r["config"] == data["selected_config"]]})
            for row in data["results"]:
                rows.append({"shape": shape, "m": m, "stage": data["stage"],
                             "selected": row["config"] == data["selected_config"], **row})
    summary = {"complete": len(complete) == 16 and not failed,
        "execution_complete": len(complete) == 16, "failed_points": failed,
        "completed_points": len(complete), "expected_points": 16, "points": complete,
        "results": rows, "winners": winners,
        "winner_scope": "screen shortlist, separately reported per seed"}
    write_json(folder / "summary.json", summary)
    columns = ["shape", "m", "stage", "seed", "config", "selected", "median_ms",
               "bf16_original_ms", "bf16_Q_ms", "latency_reduction_pct", "classification"]
    with (folder / "details.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sm-count", type=int)
    parser.add_argument("--warp-size", type=int)
    parser.add_argument("--workspace-budget-mib", type=int, default=256)
    parser.add_argument("--baseline-seeds", type=int, choices=(2,), default=2,
                        help="fixed two independent BF16 control seeds for this protocol")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", choices=list(SHAPES), help=argparse.SUPPRESS)
    parser.add_argument("--m", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("wide", "refined"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print("FINGERPRINT_JSON=" + json.dumps(fingerprint(), sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir required")
    if args.worker:
        worker(args)
        return
    if args.plan_only:
        if not args.sm_count or not args.warp_size:
            parser.error("CPU planning requires explicit --sm-count and --warp-size")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        data = plan(args.sm_count, args.warp_size, args.workspace_budget_mib * 1024**2)
        data["hardware_input_source"] = "user_supplied_not_live_verified"
        write_json(args.output_dir / "plan.json", data)
        print(json.dumps({"wide_points": len(data["wide_points"]),
                          "refined_points": sum(map(len, REFINED_MS.values())),
                          "status": "CPU_plan_only_no_GPU_measurements"}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    # O_EXCL protects against concurrent controllers/resume. Stale locks require
    # PID/ownership verification, never automatic deletion.
    lock = args.output_dir / "controller.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        audit = args.output_dir / ("preflight-" + stamp)
        audit.mkdir()
        previous_flags = {key: os.environ.pop(key) for key in list(os.environ)
                          if key.startswith("VLLM_FL_W8A16")}
        write_json(audit / "cleared-model-optimization-flags.json", previous_flags)
        preflight(audit)
        probe = subprocess.run([sys.executable, __file__, "--probe"],
                               capture_output=True, text=True)
        (audit / "probe.log").write_text(probe.stdout + probe.stderr)
        probe.check_returncode()
        fingerprint_data = json.loads(next(line.split("=", 1)[1]
            for line in probe.stdout.splitlines() if line.startswith("FINGERPRINT_JSON=")))
        metadata = {"fingerprint": fingerprint_data, "policy": POLICY,
                    "protocol": {"screen_warmup_ms": 20, "screen_rep_ms": 60,
                        "final_warmup_ms": 100, "final_rep_ms": 300, "rounds": 5,
                        "independent_seeds": args.baseline_seeds, "workspace_budget_bytes":
                        args.workspace_budget_mib * 1024**2},
                    "official_benchmark": False}
        meta_path = args.output_dir / "metadata.json"
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != metadata:
                raise RuntimeError("resume identity mismatch; use a new directory")
        else:
            write_json(meta_path, metadata)
        write_json(args.output_dir / "plan.json", plan(fingerprint_data["sm_count"],
            fingerprint_data["warp_size"], args.workspace_budget_mib * 1024**2))
        for stage, matrix in (("wide", WIDE_MS), ("refined", REFINED_MS)):
            for shape, ms in matrix.items():
                if stage == "refined":
                    points = [json.loads((args.output_dir / shape / f"m{m}" /
                              "result.json").read_text()) for m in WIDE_MS[shape]]
                    write_json(args.output_dir / shape / "pool.json", pool_from_wide(points))
                for m in ms:
                    folder = args.output_dir / shape / f"m{m}"
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / "result.json").exists():
                        continue
                    command = [sys.executable, __file__, "--worker", "--output-dir",
                               str(args.output_dir), "--shape", shape, "--m", str(m),
                               "--stage", stage, "--baseline-seeds",
                               str(args.baseline_seeds)]
                    # GPU tests stay serial. Abort/segfault is preserved and the
                    # controller stops; resume explicitly after investigating.
                    with (folder / ("worker-" + stamp + ".log")).open("w") as log:
                        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    aggregate(args.output_dir)
                    if status.returncode:
                        write_json(args.output_dir / "status.json", {
                            "complete": False, "stage": stage, "shape": shape, "m": m,
                            "returncode": status.returncode, "active_file": str(folder / "active.json")})
                        raise RuntimeError(f"worker failed at {shape} M={m}; inspect preserved logs")
                    print("COMPLETE", stage, shape, m, flush=True)
        summary = aggregate(args.output_dir)
        write_json(args.output_dir / "status.json", {
            "complete": summary["complete"], "execution_complete": True,
            "points": summary["completed_points"], "failed_points": summary["failed_points"]})
        if not summary["complete"]:
            raise RuntimeError("sweep finished with invalid points; inspect summary.json")
    finally:
        post = subprocess.run(["mx-smi"], capture_output=True, text=True)
        (args.output_dir / "gpu-after.txt").write_text(post.stdout + post.stderr)
        lock.unlink()


if __name__ == "__main__":
    main()
