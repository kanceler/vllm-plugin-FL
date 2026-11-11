#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
"""Synthetic Down-projection capability probe; not an official benchmark.

Uses the existing activation dispatch and installed FlagGems scaled_mm.
No model weights, serving parameters, or runtime policies are changed.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def command_output(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def source_info(fn):
    path = inspect.getsourcefile(fn)
    return {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if path else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+",
                        default=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--profile-m", type=int, nargs="*", default=[64, 1024])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.m) <= 0 or min(args.iterations, args.repeats) <= 0 or args.warmup < 0:
        parser.error("M, iterations and repeats must be positive; warmup >= 0")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic single-GPU Down, K=6144 N=2048; M is not observed workload",
              "args": {**vars(args), "output": str(args.output)}, "rows": []}

    def save():
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    save()
    try:
        import torch
        import torch.nn.functional as functional

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        print("Loading existing FL activation dispatch and FlagGems", flush=True)
        from vllm_fl.quantization.w8a8.linear import _dynamic_per_token_quant_int8
        from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8
        from flag_gems import scaled_mm
        from vllm_fl.dispatch import get_default_manager

        report["environment"] = {
            "python": sys.version, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("flag-gems", "vllm", "vllm-plugin-fl", "triton")},
            "maca_path": os.environ.get("MACA_PATH"),
            "gpu_status": command_output("mx-smi"),
            "framework_commit": command_output("git", "-C", str(root), "rev-parse", "HEAD"),
            "framework_status": command_output("git", "-C", str(root), "status", "--short"),
            "flaggems_commit": command_output("git", "-C", "/workspace/FlagGems", "rev-parse", "HEAD"),
            "scaled_mm_source": source_info(scaled_mm),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # FP32 reference checks must not silently use reduced-precision TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        K, N = 6144, 2048
        weight = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        w_float = weight.float()
        w_scale = w_float.abs().amax(dim=1) / 127.0
        safe_scale = torch.where(w_scale > 0, w_scale, torch.ones_like(w_scale))
        w_q_nk = (w_float / safe_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        # Match process_weights_after_loading(), not a favorable alternate layout.
        w_q = w_q_nk.t().contiguous()
        del w_float, safe_scale

        def measure(fn):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            samples = []
            wall_samples = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(args.iterations):
                    fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / args.iterations)
                wall_samples.append((time.perf_counter() - wall_start) * 1000 / args.iterations)
            return {"gpu_ms": samples, "median_gpu_ms": statistics.median(samples),
                    "wall_ms": wall_samples, "median_wall_ms": statistics.median(wall_samples)}

        def check_output(x_q, x_scale, actual):
            # Exact CPU INT64 accumulation on a sample avoids assuming a GPU
            # integer reference API works, and avoids any FP32 accumulation error.
            ids = sorted(set([0, actual.shape[0] // 2, actual.shape[0] - 1]))
            cols = list(range(8)) + list(range(N - 8, N))
            acc = x_q[ids].cpu().to(torch.int64) @ w_q_nk[cols].cpu().to(torch.int64).t()
            expected = (acc.float() * x_scale[ids].cpu() * w_scale[cols].cpu()[None, :])
            expected = expected.to(torch.bfloat16).float()
            observed = actual[ids][:, cols].cpu().float()
            torch.testing.assert_close(observed, expected, rtol=0.01, atol=0.02)
            return {"sample_max_abs_error": (observed - expected).abs().max().item(),
                    "sample_rows": ids, "sample_columns": cols,
                    "tolerance": {"rtol": 0.01, "atol": 0.02},
                    "reference": "exact INT64 accumulation, FP32 scale, BF16 cast"}

        with torch.inference_mode():
            zero_x = torch.zeros((1, K), device="cuda", dtype=torch.bfloat16)
            zero_q, zero_scale = _dynamic_per_token_quant_int8(zero_x)
            zero_y = scaled_mm(zero_q, w_q, zero_scale, w_scale, out_dtype=torch.bfloat16)
            if (torch.count_nonzero(zero_q).item() or torch.count_nonzero(zero_scale).item()
                    or not torch.isfinite(zero_y).all().item() or torch.count_nonzero(zero_y).item()):
                raise AssertionError("Zero activation check failed")
            zero_weight_y = scaled_mm(
                torch.ones_like(zero_q), torch.zeros_like(w_q),
                torch.ones_like(zero_scale), torch.zeros_like(w_scale), out_dtype=torch.bfloat16)
            if not torch.isfinite(zero_weight_y).all().item() or torch.count_nonzero(zero_weight_y).item():
                raise AssertionError("Zero weight/scale check failed")
            report["zero_checks"] = "passed"
            del zero_x, zero_q, zero_scale, zero_y, zero_weight_y
            save()

            for M in args.m:
                row = {"M": M, "N": N, "K": K, "status": "started"}
                report["rows"].append(row)
                save()
                print(f"Probing M={M}, K={K}, N={N}", flush=True)
                try:
                    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                    first_start = time.perf_counter()
                    x_q, x_scale = _dynamic_per_token_quant_int8(x)
                    actual = scaled_mm(x_q, w_q, x_scale, w_scale, out_dtype=x.dtype)
                    torch.cuda.synchronize()
                    row["first_call_wall_seconds"] = time.perf_counter() - first_start
                    row["note_first_call"] = "includes dispatch/autotune/cache effects; not isolated compilation time"
                    ref_q, ref_scale = dynamic_per_token_quant_int8(x)
                    mismatches = x_q != ref_q
                    mismatch_count = mismatches.count_nonzero().item()
                    coordinates = mismatches.nonzero()[:8].cpu().tolist()
                    max_code_diff = (x_q.to(torch.int16) - ref_q.to(torch.int16)).abs().max().item()
                    row["activation_contract"] = {
                        "passed": max_code_diff <= 1,
                        "mismatch_count": mismatch_count,
                        "total_codes": x_q.numel(),
                        "max_code_difference": max_code_diff,
                        "scale_max_abs_error": (x_scale - ref_scale).abs().max().item(),
                        "examples": [{"row": i, "column": j, "input": x[i, j].item(),
                                      "actual_code": x_q[i, j].item(), "reference_code": ref_q[i, j].item(),
                                      "actual_scale": x_scale[i, 0].item(),
                                      "reference_scale": ref_scale[i, 0].item(),
                                      "row_absmax": x[i].abs().max().item()}
                                     for i, j in coordinates]}
                    row["quant_dispatch_failed_impls"] = {
                        name: sorted(impls) for name, impls in get_default_manager().get_failed_impls(
                            "dynamic_per_token_quant_int8").items()}
                    # Codes may differ by +-1 exactly at |input| == absmax/2, where the
                    # scaled value is 63.5 and round-half-to-even flips on the last FP32
                    # ULP of 127/absmax between the Triton kernel and the PyTorch
                    # reference. Both are valid round-to-nearest results, so accept +-1;
                    # anything larger indicates a real quantization divergence.
                    if max_code_diff > 1:
                        raise AssertionError(
                            f"Activation codes differ from reference by {max_code_diff} "
                            "codes (expected <= 1 for midpoint rounding)"
                        )
                    torch.testing.assert_close(x_scale, ref_scale, rtol=1e-6, atol=1e-8)
                    if actual.dtype != torch.bfloat16 or actual.shape != (M, N):
                        raise AssertionError("Output shape/dtype mismatch")
                    if not torch.isfinite(actual).all().item():
                        raise AssertionError("Nonfinite output")
                    row["correctness"] = check_output(x_q, x_scale, actual)
                    baseline = functional.linear(x, weight)
                    diff = actual.float() - baseline.float()
                    row["vs_original_bf16"] = {
                        "relative_l2": (diff.norm() / baseline.float().norm().clamp_min(1e-12)).item(),
                        "max_abs_error": diff.abs().max().item(),
                        "note": "random-tensor quantization error, not model accuracy"}
                    del ref_q, ref_scale, actual, baseline, diff

                    def bf16():
                        return functional.linear(x, weight)

                    def int8_only():
                        return scaled_mm(x_q, w_q, x_scale, w_scale, out_dtype=x.dtype)

                    def full_path():
                        q, scale = _dynamic_per_token_quant_int8(x)
                        return scaled_mm(q, w_q, scale, w_scale, out_dtype=x.dtype)

                    # Warm all candidates before timing, then reverse measurement
                    # order in alternate Shapes to reduce systematic order bias.
                    candidates = {"bf16": bf16, "int8_gemm": int8_only, "full_w8a8": full_path}
                    for fn in candidates.values():
                        for _ in range(args.warmup):
                            fn()
                    torch.cuda.synchronize()
                    order = list(candidates)
                    if args.m.index(M) % 2:
                        order.reverse()
                    row["timing"] = {name: measure(candidates[name]) for name in order}
                    base_ms = row["timing"]["bf16"]["median_gpu_ms"]
                    row["int8_only_speedup"] = base_ms / row["timing"]["int8_gemm"]["median_gpu_ms"]
                    row["full_path_speedup"] = base_ms / row["timing"]["full_w8a8"]["median_gpu_ms"]
                    row["quant_dispatch_failed_impls"] = {
                        name: sorted(impls) for name, impls in get_default_manager().get_failed_impls(
                            "dynamic_per_token_quant_int8").items()}
                    if M in args.profile_m:
                        try:
                            with torch.profiler.profile(activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]) as profile:
                                full_path()
                                torch.cuda.synchronize()
                            trace_path = args.output / f"trace_m{M}.json"
                            profile.export_chrome_trace(str(trace_path))
                            trace = json.loads(trace_path.read_text())
                            row["kernel_names"] = sorted({e.get("name", "") for e in trace.get("traceEvents", [])
                                                          if "kernel" in e.get("cat", "").lower()})
                            row["trace_file"] = str(trace_path)
                        except Exception:
                            row["profile_error"] = traceback.format_exc()
                    row["status"] = "passed"
                    print(f"M={M}: full path speedup {row['full_path_speedup']:.3f}x", flush=True)
                    del x, x_q, x_scale
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = traceback.format_exc()
                    print(row["error"], flush=True)
                    # Known numerical assertions do not poison the device.
                    # Skip timings for that Shape, but inspect remaining Shapes.
                    # Unknown runtime/GPU failures still terminate the probe.
                    if not isinstance(exc, AssertionError):
                        raise
                finally:
                    save()
        report["status"] = ("completed" if all(r["status"] == "passed" for r in report["rows"])
                            else "completed_with_failures")
        print(f"Results: {args.output / 'results.json'}", flush=True)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
