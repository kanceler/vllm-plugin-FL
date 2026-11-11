# SPDX-License-Identifier: Apache-2.0
"""Bounded offline INT8->BF16/scale tuning with an unchanged MetaX BF16 MM.

Use --plan-only without Torch/Triton. GPU runs require a reviewed idle-memory
ceiling and a fresh directory. Scale is a separate, explicitly gated invocation.
No production dispatch, global LibEntry, model or official benchmark is changed.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
from statistics import median
import subprocess
import sys
import traceback

from w8a16_separate_plan import (
    BOUNDARY_MS, CONFIGS, MAIN_MS, PROTOCOL, SEEDS, SHAPES, VALIDATION_MS,
    config_id, full_call_gate, plan, shortlist, summarize, validation_gate,
)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def append(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {"args": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def preflight(root, memory_limit):
    """Read-only gate. Occupied resources are never killed or reset here."""
    state = command(["mx-smi"])
    write_json(root / "gpu-before.json", state)
    processes = command(["ps", "-eo", "pid,ppid,args"])
    write_json(root / "processes-before.json", processes)
    if state["returncode"] or "no process found" not in state["stdout"].lower():
        raise RuntimeError("GPU not confirmed idle; inspect gpu-before.json")
    status = state["stdout"]
    usage = re.findall(r"(\d+)\s*/\s*(\d+)\s*MiB", status)
    utilization = re.findall(r"(\d+)\s*%", status)
    if len(usage) != 1 or not utilization:
        raise RuntimeError("cannot establish single-GPU memory/utilization state")
    if int(usage[0][0]) > memory_limit or any(int(x) for x in utilization):
        raise RuntimeError("GPU busy or memory exceeds explicitly reviewed ceiling")
    if processes["returncode"]:
        raise RuntimeError("process inventory failed")
    # Ignore our own ancestry (including a shell whose command mentions us).
    records = []
    parents = {}
    for line in processes["stdout"].splitlines()[1:]:
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            pid, ppid = map(int, fields[:2])
            parents[pid] = ppid
            records.append((pid, fields[2]))
    ignored, pid = set(), os.getpid()
    while pid and pid not in ignored:
        ignored.add(pid)
        pid = parents.get(pid, 0)
    markers = ("vllm serve", "EngineCore", "benchmark_throughput_serve.py",
               "evalscope eval", "pytest", "tune_metax_w8a16", "benchmark_w8a16",
               "benchmark_flaggems", "verify_splitk_launch")
    conflicts = [(pid, args) for pid, args in records if pid not in ignored
                 and any(marker in args for marker in markers)]
    if conflicts:
        raise RuntimeError(f"possible competing task: {conflicts}")
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("service port 9031 occupied")


def fingerprint(torch, triton, backend, mm):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this protocol requires one visible GPU")
    libentry = importlib.import_module("flag_gems.utils.libentry")
    ref = importlib.import_module("vllm_fl.quantization.w8a16.reference")
    backend_root = Path(mm.__file__).resolve().parents[1]
    files = {
        "backend": Path(backend.__file__), "bf16_provider": Path(mm.__file__),
        "libentry": Path(libentry.__file__), "quantizer": Path(ref.__file__),
        "tune_configs": backend_root / "tune_configs.yaml",
        "expand_configs": backend_root / "mm_metax_expand.yaml",
        "diagnostic": Path(__file__),
        "protocol": Path(__file__).with_name("w8a16_separate_plan.py"),
    }
    props = torch.cuda.get_device_properties(0)
    target = triton.runtime.driver.active.get_current_target()
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "triton": triton.__version__, "hostname": socket.gethostname(),
        "device": str(props), "uuid": str(getattr(props, "uuid", "unavailable")),
        "sm_count": props.multi_processor_count, "warp_size": target.warp_size,
        "compiler_target": str(target),
        "source_hashes": {name: digest(path) for name, path in files.items()},
        "import_paths": {name: str(path.resolve()) for name, path in files.items()},
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("VLLM_FL_W8A16", "USE_FLAGTUNE"))
                        or k in ("MACA_PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
                                 "TRITON_CACHE_DIR")},
    }


def repository_evidence(root, backend, mm):
    for label, path, relative in (
        ("flaggems", Path(backend.__file__),
         ["src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py",
          "src/flag_gems/runtime/backend/_metax/ops/mm.py",
          "src/flag_gems/runtime/backend/_metax/tune_configs.yaml",
          "src/flag_gems/runtime/backend/_metax/mm_metax_expand.yaml"]),
        ("plugin", Path(__file__), ["benchmarks/diagnostics/benchmark_w8a16_separate.py",
                                   "benchmarks/diagnostics/w8a16_separate_plan.py"]),
    ):
        evidence = {}
        for args in (["rev-parse", "HEAD"], ["branch", "--show-current"],
                     ["status", "--short"], ["diff", "--", *relative]):
            evidence[" ".join(args)] = command(["git", "-C", str(path.parent), *args])
        write_json(root / f"repository-{label}.json", evidence)


def compiled_info(compiled, block, warps=None, stages=None, block_index=3):
    meta = compiled.metadata
    actual_warps, actual_stages = int(meta.num_warps), int(meta.num_stages)
    if warps is not None and actual_warps != warps:
        raise RuntimeError(f"launch mismatch: warps requested {warps}, actual {actual_warps}")
    if stages is not None and actual_stages != stages:
        raise RuntimeError(f"launch mismatch: stages requested {stages}, actual {actual_stages}")
    constants = getattr(getattr(compiled, "src", None), "constants", None)
    if constants is None:
        raise RuntimeError("compiled source constants unavailable; cannot certify BLOCK")
    # Both current named keys and older Triton positional constexpr keys occur.
    actual_block = constants.get("BLOCK", constants.get(
        block_index, constants.get((block_index,))))
    if actual_block != block:
        raise RuntimeError(f"BLOCK cannot be certified: {constants}")
    hashes = {}
    for name, value in compiled.asm.items():
        data = value.encode() if isinstance(value, str) else bytes(value)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"requested": {"block": block, "warps": warps, "stages": stages},
            "actual": {"block": actual_block, "warps": actual_warps,
                       "stages": actual_stages},
            "kernel_hash": str(getattr(compiled, "hash", "unavailable")),
            "asm_sha256": hashes, "object_id": id(compiled),
            "n_regs": getattr(compiled, "n_regs", None),
            "n_spills": getattr(compiled, "n_spills", None),
            "shared": getattr(meta, "shared", None),
            "cache_note": "JIT artifact identity; fresh compilation versus disk cache is unknown"}


class Launches:
    def __init__(self, torch, triton, backend, root):
        self.torch, self.triton, self.backend, self.root = torch, triton, backend, root
        self.defaults = {}

    def entry(self, kind):
        return getattr(self.backend, "weight_int8pack_convert" if kind == "convert"
                       else "weight_int8pack_output_scale")

    def arguments(self, kind, source, target, scale):
        if kind == "convert":
            return (source, target, source.numel())
        return (source, scale, target, source.shape[1], source.numel())

    def run(self, kind, source, target, scale=None, config=None):
        args = self.arguments(kind, source, target, scale)
        block, warps = config if config is not None else (1024, None)
        grid = (self.triton.cdiv(source.numel(), block),)
        if config is None:
            value = self.entry(kind).run(*args, BLOCK=block, grid=grid, warmup=False)
            return value[0] if isinstance(value, tuple) else value
        return self.entry(kind).jit_function.run(
            *args, BLOCK=block, num_warps=warps,
            num_stages=self.defaults[kind]["actual"]["stages"],
            grid=grid, warmup=False)

    def audit(self, kind, source, target, scale=None, config=None):
        kernel = self.run(kind, source, target, scale, config)
        block, warps = config if config is not None else (1024, None)
        info = compiled_info(kernel, block, warps,
                             None if config is None else self.defaults[kind]["actual"]["stages"],
                             block_index=3 if kind == "convert" else 5)
        append(self.root / "launches.jsonl", {"kind": kind, **info})
        artifacts = self.root / "compiled" / kind / str(info["kernel_hash"])
        artifacts.mkdir(parents=True, exist_ok=True)
        for name, value in kernel.asm.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                continue
            path = artifacts / name
            if not path.exists():
                path.write_bytes(value.encode() if isinstance(value, str) else bytes(value))
        return info

    def establish_defaults_and_guard(self):
        torch = self.torch
        q = torch.arange(-128, 128, device="cuda", dtype=torch.int16).to(torch.int8)
        q = q.repeat(3)[:513].contiguous()
        converted = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
        y = torch.linspace(-5, 5, 513, device="cuda", dtype=torch.float32).to(torch.bfloat16).view(3, 171)
        scale = torch.logspace(-4, 4, 171, device="cuda", dtype=torch.float32)
        scale[0] = 0
        output = torch.empty_like(y)
        for kind, src, dst, scales in (("convert", q, converted, None),
                                      ("scale", y, output, scale)):
            self.defaults[kind] = self.audit(kind, src, dst, scales)
            observations = [self.audit(kind, src, dst, scales, (1024, w)) for w in (2, 8, 2)]
            if observations[0]["object_id"] == observations[1]["object_id"]:
                raise RuntimeError("different warps reused one compiled object")
            if observations[0]["object_id"] != observations[2]["object_id"]:
                raise RuntimeError("same configuration did not reuse its JIT artifact")
            expected = src.to(torch.bfloat16) if kind == "convert" else (
                src.float() * scales).to(torch.bfloat16)
            torch.testing.assert_close(dst, expected, rtol=0, atol=0)
            for config in CONFIGS:
                # Every tile must handle a non-divisible size, not just 1024.
                self.audit(kind, src, dst, scales, config)
                torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        write_json(self.root / "launch-guard.json", {"passed": True, "defaults": self.defaults})

    def default_config(self, kind):
        return [1024, self.defaults[kind]["actual"]["warps"]]


def bench(torch, testing, root, identity, paths):
    # All paths compile/warm before interleaving, including the native MM tuner.
    for function in paths.values():
        function()
    torch.cuda.synchronize()
    samples = {name: [] for name in paths}
    names = list(paths)
    for round_index in range(PROTOCOL["rounds"]):
        offset = round_index % len(names)
        for name in names[offset:] + names[:offset]:
            value = float(testing.do_bench(paths[name], warmup=PROTOCOL["warmup_ms"],
                                          rep=PROTOCOL["rep_ms"], return_mode="median"))
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("invalid GPU timing")
            samples[name].append(value)
            append(root / "raw.jsonl", {**identity, "round": round_index + 1,
                                       "path": name, "ms": value,
                                       "measurement_source": "fresh_timing_not_timing_cache"})
    return {name: summarize(values) for name, values in samples.items()}


def check(torch, actual, expected):
    error = ((actual.float() - expected.float()).norm()
             / expected.float().norm().clamp_min(1e-12)).item()
    if not math.isfinite(error) or error >= PROTOCOL["relative_l2_lt"]:
        raise AssertionError(f"relative L2 {error}")
    torch.testing.assert_close(actual, expected, rtol=PROTOCOL["rtol"], atol=PROTOCOL["atol"])
    return error


def tensors(torch, shape, seed, m):
    from vllm_fl.quantization.w8a16.reference import quantize_weight
    n, k = SHAPES[shape]
    torch.manual_seed(seed + list(SHAPES).index(shape) * 100000)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    weight[0] = 0
    q, scale = quantize_weight(weight)
    torch.manual_seed(seed + m * 101)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    return x, weight, q, scale


@contextmanager
def trace_mm(mm):
    """Trace Python branch selection; snapshots are not graph replay counts."""
    calls, originals = [], {}
    try:
        for name in ("gemv_mm", "gemv_mm_k_parallel", "splitk_mm_two_step",
                     "splitk_mm", "general_mm_nn", "general_mm_nt", "general_mm"):
            if hasattr(mm, name):
                originals[name] = getattr(mm, name)
                def wrapper(*args, _name=name, _fn=originals[name], **kwargs):
                    calls.append(_name)
                    return _fn(*args, **kwargs)
                setattr(mm, name, wrapper)
        yield calls
    finally:
        for name, function in originals.items():
            setattr(mm, name, function)


def mm_evidence(mm):
    evidence = {}
    for name in ("mm_kernel", "mm_kernel_nn", "mm_kernel_nt", "mm_kernel_splitk",
                 "mm_kernel_splitk_partial", "mm_kernel_splitk_reduce"):
        entry = getattr(mm, name, None)
        rows = []
        for cache in getattr(entry, "kernel_cache", ()):
            for value in cache.values():
                kernel = value[0]
                rows.append({"hash": str(getattr(kernel, "hash", None)),
                             "metadata": str(kernel.metadata),
                             "constants": value[1]})
        evidence[name] = rows
    return {"kind": "cache_snapshot_not_per_launch_proof", "entries": evidence}


def full(torch, mm, launches, x, q, scale, convert_config, scale_config):
    converted = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    launches.run("convert", q, converted, config=convert_config)
    unscaled = mm.mm(x, converted.t())
    output = torch.empty_like(unscaled)
    launches.run("scale", unscaled, output, scale, scale_config)
    return output


def profile_after_conversion(torch, mm, launches, x, q, scale, convert_config, scale_config):
    # Events are diagnostic intervals (may include CPU launch gaps). Never sum
    # these intervals in place of separately timed, uninstrumented full calls.
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    converted = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    result = []
    for _ in range(PROTOCOL["rounds"]):
        events[0].record()
        launches.run("convert", q, converted, config=convert_config)
        events[1].record()
        unscaled = mm.mm(x, converted.t())
        events[2].record()
        output = torch.empty_like(unscaled)
        launches.run("scale", unscaled, output, scale, scale_config)
        events[3].record()
        events[3].synchronize()
        result.append({"convert_ms": events[0].elapsed_time(events[1]),
                       "mm_after_convert_ms": events[1].elapsed_time(events[2]),
                       "scale_ms": events[2].elapsed_time(events[3])})
    return {"kind": "instrumented_intervals_not_full_call_benchmark", "samples": result}


def graph_check(torch, fn, x, q, scale):
    # Warm on a side stream before capture; graph allocations stay alive in g.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    graph.replay()
    check(torch, output, fn())
    saved_q, saved_scale = q[1, 0].clone(), scale[1].clone()
    try:
        x.mul_(0.5)
        q[1, 0] = 17
        scale[1].mul_(0.5)
        graph.replay()
        expected = (torch.nn.functional.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        check(torch, output, expected)
        check(torch, output, fn())
    finally:
        x.mul_(2)
        q[1, 0], scale[1] = saved_q, saved_scale
    torch.cuda.synchronize()


def component_search(torch, testing, mm, launches, root, shape, stage, fixed_convert):
    records = {config: {"config": config, "status": "valid",
                        "ratios_to_direct_default": []} for config in CONFIGS}
    for m in ((160,) if stage == "convert" else (160, 2048)):
        for seed in SEEDS:
            x, weight, q, scale = tensors(torch, shape, seed, m)
            if stage == "convert":
                source = q
            else:
                converted = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
                launches.run("convert", q, converted, config=fixed_convert)
                source = mm.mm(x, converted.t())
            target = torch.empty(source.shape, device="cuda", dtype=torch.bfloat16)
            expected = source.to(torch.bfloat16) if stage == "convert" else (
                source.float() * scale).to(torch.bfloat16)
            paths = {
                "production_component": lambda: launches.run(stage, source, target, scale),
                "direct_default": lambda: launches.run(
                    stage, source, target, scale, launches.default_config(stage)),
            }
            for config, record in records.items():
                if record["status"] != "valid":
                    continue
                try:
                    launches.audit(stage, source, target, scale, config)
                    torch.testing.assert_close(target, expected, rtol=0, atol=0)
                except Exception as exc:
                    # Metadata/configuration mismatch invalidates the whole run.
                    if isinstance(exc, RuntimeError) and type(exc).__name__ not in (
                            "OutOfResources", "CompilationError", "CompileTimeAssertionFailure"):
                        raise
                    record.update(status="failed", error=repr(exc))
                    append(root / "failures.jsonl", {"shape": shape, "stage": stage,
                                                       "config": config, "error": repr(exc)})
                    continue
                paths[config_id(config)] = lambda c=config: launches.run(stage, source, target, scale, c)
            values = bench(torch, testing, root, {"phase": "component_search",
                "shape": shape, "m": None if stage == "convert" else m, "seed": seed}, paths)
            for config, record in records.items():
                name = config_id(config)
                if name in values:
                    record["ratios_to_direct_default"].append(
                        values[name]["median_ms"] / values["direct_default"]["median_ms"])
            # Includes allocation like the original historical component helper.
            def allocation_control():
                dst = torch.empty_like(target)
                launches.run(stage, source, dst, scale)
                return dst
            bench(torch, testing, root, {"phase": "component_allocation_control",
                "shape": shape, "m": m if stage == "scale" else None, "seed": seed},
                {"allocation_plus_original_component": allocation_control})
            if stage == "convert":
                logical_bytes = 3 * q.numel()
                append(root / "bandwidth.jsonl", {"shape": shape, "seed": seed,
                    "logical_bytes": logical_bytes,
                    "logical_GB_per_s": {k: logical_bytes / (v["median_ms"] * 1e6)
                                         for k, v in values.items()},
                    "limitation": "logical traffic, not measured DRAM bytes or peak bandwidth"})
    result = list(records.values())
    expected_ratios = 2 if stage == "convert" else 4
    for record in result:
        if record["status"] == "valid" and len(record["ratios_to_direct_default"]) != expected_ratios:
            raise RuntimeError("incomplete component candidate")
    finalists = shortlist(result)
    if not finalists:
        raise RuntimeError("no valid component finalists")
    write_json(root / f"component-{shape}.json", {"stage": stage, "records": result,
                                                "finalists": finalists})
    return finalists


def full_points(torch, testing, backend, mm, launches, root, shape, stage,
                points, finalists, fixed_convert, role, graph=False):
    rows = []
    for m in points:
        point_rows = []
        for seed in SEEDS:
            x, weight, q, scale = tensors(torch, shape, seed, m)
            converted = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
            launches.run("convert", q, converted, config=fixed_convert)
            unscaled = mm.mm(x, converted.t())
            destination = torch.empty_like(unscaled)
            for config in finalists:
                launches.audit(stage, q if stage == "convert" else unscaled,
                               converted if stage == "convert" else destination, scale, config)
            with trace_mm(mm) as converted_calls:
                mm.mm(x, converted.t())
            with trace_mm(mm) as original_calls:
                mm.mm(x, weight.t())
            evidence = {"converted_calls": converted_calls, "original_calls": original_calls,
                        "mm_cache": mm_evidence(mm),
                        "converted_stride": list(converted.stride()),
                        "gemm_weight_stride": list(converted.t().stride())}
            paths = {
                "production": lambda: backend._large_m_separate(x, q, scale),
                "direct_default": lambda: full(torch, mm, launches, x, q, scale,
                    fixed_convert if stage == "scale" else launches.default_config("convert"),
                    launches.default_config("scale") if stage == "scale" else None),
                "bf16_original": lambda: mm.mm(x, weight.t()),
            }
            for config in finalists:
                paths[config_id(config)] = lambda c=config: full(
                    torch, mm, launches, x, q, scale,
                    c if stage == "convert" else fixed_convert,
                    c if stage == "scale" else None)
            expected = (torch.nn.functional.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
            errors = {name: check(torch, fn(), expected) for name, fn in paths.items()
                      if name != "bf16_original"}
            timing = bench(torch, testing, root, {"phase": "full", "role": role,
                "stage": stage, "shape": shape, "m": m, "seed": seed}, paths)
            components = {
                "convert_preallocated": lambda: launches.run("convert", q, converted,
                    config=fixed_convert if stage == "scale" else None),
                "mm_preconverted_diagnostic": lambda: mm.mm(x, converted.t()),
                "scale_preallocated": lambda: launches.run("scale", unscaled, destination, scale),
            }
            bench(torch, testing, root, {"phase": "components", "shape": shape,
                "m": m, "seed": seed}, components)
            profiles = {}
            for name, cc, sc in [("production_components", None, None),
                                  ("direct_default", fixed_convert if stage == "scale"
                                   else launches.default_config("convert"),
                                   launches.default_config("scale") if stage == "scale" else None)] + [
                    (config_id(c), c if stage == "convert" else fixed_convert,
                     c if stage == "scale" else None) for c in finalists]:
                profiles[name] = profile_after_conversion(torch, mm, launches, x, q, scale, cc, sc)
            passed = False
            if graph:
                for c in finalists:
                    graph_check(torch, paths[config_id(c)], x, q, scale)
                passed = True
            point_rows.append({"shape": shape, "m": m, "seed": seed, "role": role,
                "timings": timing, "errors": errors, "mm_evidence": evidence,
                "interval_profiles": profiles, "graph_passed": passed})
        medians = [row["timings"]["bf16_original"]["median_ms"] for row in point_rows]
        stable = 100 * (max(medians) - min(medians)) / median(medians) <= PROTOCOL["diagnostic_stability_limit_pct"]
        for row in point_rows:
            row["bf16_seed_stable"] = stable
            append(root / "full-points.jsonl", row)
        rows.extend(point_rows)
    return rows


def load_conversion(path, current):
    data = json.loads((path / "summary.json").read_text())
    previous = json.loads((path / "metadata.json").read_text())
    if (data.get("stage") != "convert" or not data.get("protocol_complete")
            or not data.get("eligible_shapes")):
        raise ValueError("scale requires a completed conversion run with verified full-call benefit")
    # Paths may change when copied to an isolated experiment; content and device
    # identity must match. A provider update always forces a new conversion run.
    for key in ("source_hashes", "torch", "triton", "python", "device", "uuid",
                "compiler_target", "driver_versions"):
        if previous["fingerprint"].get(key) != current.get(key):
            raise ValueError(f"conversion provenance expired: {key}")
    if not data.get("execution_complete") or set(data.get("shapes", {})) != set(SHAPES):
        raise ValueError("conversion protocol incomplete")
    for shape in data["eligible_shapes"]:
        result = data["shapes"][shape]
        if not result.get("benefit_verified") or not result.get("validation_complete"):
            raise ValueError("conversion benefit missing validation evidence")
    configs = {shape: data["shapes"][shape]["accepted_convert_config"] for shape in SHAPES}
    if any(not isinstance(c, list) or len(c) != 2 for c in configs.values()):
        raise ValueError("conversion manifest missing per-shape configuration")
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("convert", "scale"), default="convert")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--conversion-run", type=Path)
    parser.add_argument("--idle-memory-limit-mib", type=int,
                        help="previously inspected and explicitly accepted idle/driver memory ceiling")
    args = parser.parse_args()
    if not args.plan_only and (args.idle_memory_limit_mib is None or args.idle_memory_limit_mib <= 0):
        parser.error("GPU execution requires --idle-memory-limit-mib after resource review")
    if args.stage == "scale" and not args.plan_only and args.conversion_run is None:
        parser.error("scale requires --conversion-run")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan(args.stage))
    write_json(root / "status.json", {"execution_complete": False, "complete": False,
                                     "plan_only": args.plan_only})
    if args.plan_only:
        print(json.dumps(plan(args.stage), indent=2))
        write_json(root / "status.json", {"plan_only": True, "gpu_executed": False,
                                         "execution_complete": True, "complete": False})
        return
    summary = {"stage": args.stage, "execution_complete": False, "complete": False,
               "eligible_shapes": [], "shapes": {}}
    write_json(root / "summary.json", summary)
    try:
        preflight(root, args.idle_memory_limit_mib)
        import torch
        import triton
        import triton.testing
        backend = importlib.import_module("flag_gems.runtime.backend._metax.ops.weight_int8pack_mm")
        mm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        identity = fingerprint(torch, triton, backend, mm)
        idle = json.loads((root / "gpu-before.json").read_text())["stdout"]
        identity["driver_versions"] = [line.strip() for line in idle.splitlines()
                                       if "Driver Version" in line or "MACA Version" in line]
        write_json(root / "metadata.json", {
            "fingerprint": identity, "argv": sys.argv,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL, "official_benchmark": False,
            "conversion_run": str(args.conversion_run) if args.conversion_run else None,
            "conversion_summary_sha256": digest(args.conversion_run / "summary.json")
            if args.conversion_run else None,
            "memory_ceiling_mib": args.idle_memory_limit_mib})
        repository_evidence(root, backend, mm)
        launches = Launches(torch, triton, backend, root)
        with torch.inference_mode():
            launches.establish_defaults_and_guard()
            fixed = load_conversion(args.conversion_run, identity) if args.stage == "scale" else {
                shape: launches.default_config("convert") for shape in SHAPES}
            for shape in SHAPES:
                finalists = component_search(torch, triton.testing, mm, launches, root,
                                             shape, args.stage, fixed[shape])
                rows = full_points(torch, triton.testing, backend, mm, launches, root,
                    shape, args.stage, MAIN_MS, finalists, fixed[shape], "main")
                # Select once on main points. Boundary/validation cannot reselect.
                ranked = sorted(finalists, key=lambda c: max(
                    row["timings"][config_id(c)]["median_ms"] /
                    row["timings"]["direct_default"]["median_ms"] for row in rows))
                accepted = next((c for c in ranked if full_call_gate(rows, config_id(c))), None)
                result = {"finalists": finalists, "main_complete": True,
                          "validation_complete": False, "accepted_config": None,
                          "accepted_convert_config": fixed[shape], "benefit_verified": False}
                if accepted is None:
                    result["stop_reason"] = "no stable full-call improvement on main points; no expanded search"
                else:
                    verification = full_points(torch, triton.testing, backend, mm, launches, root,
                        shape, args.stage, (*VALIDATION_MS, *BOUNDARY_MS[shape]),
                        [accepted], fixed[shape], "validation", graph=True)
                    result["validation_complete"] = True
                    result["benefit_verified"] = validation_gate(verification, config_id(accepted))
                    if result["benefit_verified"]:
                        result["accepted_config"] = accepted
                        if args.stage == "convert":
                            result["accepted_convert_config"] = accepted
                        summary["eligible_shapes"].append(shape)
                    else:
                        result["stop_reason"] = "validation regression, instability or graph failure"
                summary["shapes"][shape] = result
                write_json(root / "summary.json", summary)
        summary["execution_complete"] = True
        summary["protocol_complete"] = len(summary["shapes"]) == len(SHAPES)
        summary["complete"] = (len(summary["shapes"]) == len(SHAPES)
            and all(x["validation_complete"] for x in summary["shapes"].values()))
        summary["full_matrix_complete"] = summary["complete"]
        summary["limitation"] = "diagnostic shortlist, not global optimum or production/model acceptance"
        write_json(root / "summary.json", summary)
        write_json(root / "status.json", {"execution_complete": True,
            "complete": summary["complete"], "eligible_shapes": summary["eligible_shapes"],
            "protocol_complete": summary["protocol_complete"],
            "full_matrix_complete": summary["full_matrix_complete"],
            "gpu_executed": True})
    except BaseException as exc:
        write_json(root / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(root / "gpu-before-process-exit.json", command(["mx-smi"]))


if __name__ == "__main__":
    main()
