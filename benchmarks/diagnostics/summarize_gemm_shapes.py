#!/usr/bin/env python3
"""Correlate profiler GPU kernels through CPU launch IDs, never GPU wall time.

Graph token extent is reported separately from exact per-operation GEMM M.
Missing/ambiguous correlations and projections stay unknown.
"""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

STEP = "GEMM_DIAG_STEP "
GRAPH = "GEMM_DIAG_GRAPH "
GEMM = re.compile(r"mm_kernel|gemm|(?:^|::)(?:mm|addmm)(?:_|$)", re.I)
OPS = {"aten::mm", "aten::addmm", "aten::linear"}


def dimensions(event):
    args = event.get("args", {})
    shapes = args.get("Input Dims", args.get("Input Shapes", []))
    offset = 1 if event["name"] == "aten::addmm" else 0
    if len(shapes) < offset + 2:
        return None
    left, right = shapes[offset:offset + 2]
    if len(left) != 2 or len(right) != 2:
        return None
    m, k = left
    n = right[0] if event["name"] == "aten::linear" else right[1]
    expected_k = right[1] if event["name"] == "aten::linear" else right[0]
    if k != expected_k or not all(isinstance(v, int) and v > 0 for v in (m, n, k)):
        return None
    return {"M": m, "N": n, "K": k}


def summarize(trace, inventory=()):
    events = trace["traceEvents"]
    streams = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") != "kernel":
            streams[(event.get("pid"), event.get("tid"))].append(event)
    launches = defaultdict(list)
    for thread_events in streams.values():
        stack = []
        for event in sorted(thread_events, key=lambda e: (e["ts"], -e.get("dur", 0))):
            start, end = event["ts"], event["ts"] + event.get("dur", 0)
            stack = [parent for parent in stack if parent["ts"] + parent.get("dur", 0) > start]
            name = event.get("name", "")
            if name.startswith((STEP, GRAPH)) or name in OPS:
                stack.append(event)
            if event.get("cat") not in ("cuda_runtime", "hip_runtime"):
                continue
            correlation = event.get("args", {}).get("correlation")
            if correlation is None or correlation == 0:
                continue
            parents = [p for p in stack if p["ts"] <= start and p["ts"] + p.get("dur", 0) >= end]
            context = {"phase": "unknown", "label": "unknown", "M": None, "N": None,
                       "K": None, "graph_tokens": None, "graph_action": None,
                       "M_source": "unknown", "active_tokens": None}
            for parent in parents:
                pname = parent["name"]
                if pname.startswith(STEP):
                    data = json.loads(pname[len(STEP):])
                    context.update({k: data[k] for k in ("label", "phase", "active_tokens") if k in data})
                elif pname.startswith(GRAPH):
                    data = json.loads(pname[len(GRAPH):])
                    context.update(graph_tokens=data["graph_tokens"], graph_action=data["action"])
                elif pname in OPS:
                    dims = dimensions(parent)
                    if dims:
                        context.update(dims, M_source="operator_input_shapes")
            launches[correlation].append(context)

    groups = defaultdict(lambda: {"kernel_calls": 0, "gpu_us": 0.0})
    total = attributed = graph_extent = capture = 0.0
    total_kernels = unknown_correlation = 0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel" or not GEMM.search(event.get("name", "")):
            continue
        duration = event.get("dur", 0)
        if duration < 0:
            continue
        total += duration
        total_kernels += 1
        correlation = event.get("args", {}).get("correlation")
        matches = launches.get(correlation, []) if correlation not in (None, 0) else []
        unique = {json.dumps(item, sort_keys=True) for item in matches}
        if len(unique) == 1:
            ctx = json.loads(next(iter(unique)))
        else:
            ctx = {"phase": "unknown", "label": "unknown", "M": None, "N": None, "K": None,
                   "graph_tokens": None, "graph_action": None, "M_source": "unknown", "active_tokens": None}
            unknown_correlation += 1
        if ctx["graph_action"] == "capture":
            capture += duration
            ctx.update(M=None, N=None, K=None, M_source="capture_excluded")
        elif ctx["M"] is not None:
            attributed += duration
        elif ctx["graph_tokens"] is not None:
            graph_extent += duration
            ctx["M_source"] = "graph_extent_only_not_exact_gemm_M"
        candidates = [item.get("prefix", item["module"]) for item in inventory
                      if ctx["K"] is not None and item.get("shape") == [ctx["N"], ctx["K"]]]
        projection_types = sorted({p.rsplit(".", 1)[-1] for p in candidates})
        projection = projection_types[0] if len(projection_types) == 1 else "unknown"
        key = (ctx["label"], ctx["phase"], projection, ctx["M"], ctx["N"], ctx["K"],
               ctx["graph_tokens"], ctx["active_tokens"], ctx["M_source"], event["name"])
        groups[key]["kernel_calls"] += 1
        groups[key]["gpu_us"] += duration
    fields = ["scene", "phase", "projection_group", "M", "N", "K", "graph_tokens",
              "active_tokens", "M_source", "kernel"]
    rows = [{**dict(zip(fields, key)), **value} for key, value in groups.items()]
    rows.sort(key=lambda row: row["gpu_us"], reverse=True)
    return {"duration_unit": "us (torch profiler Chrome trace)", "matched_kernel_calls": total_kernels,
            "matched_gpu_us": total, "exact_M_gpu_us": attributed,
            "exact_M_duration_coverage": attributed / total if total else 0,
            "graph_extent_only_gpu_us": graph_extent, "capture_excluded_gpu_us": capture,
            "unknown_correlation_kernel_calls": unknown_correlation,
            "limitations": ["Kernel-name matcher is not proven complete GEMM coverage",
                            "projection_group is shape-matched inventory candidate, not block identity",
                            "graph token extent does not automatically establish exact GEMM M",
                            "kernel_calls counts kernels, not logical GEMM invocations"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as stream:
        trace = json.load(stream)
    inventory = json.loads(args.inventory.read_text()) if args.inventory else []
    result = summarize(trace, inventory)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["rows"]:
        with (args.output / "shapes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
