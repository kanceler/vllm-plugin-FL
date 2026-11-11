# SPDX-License-Identifier: Apache-2.0
"""Serial official BF16/candidate runs; never alter the official benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def capture(command, destination):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    destination.write_text(result.stdout)
    return result


def idle(directory):
    result = capture(["mx-smi"], directory / "gpu-idle.txt")
    if result.returncode or "no process found" not in result.stdout.lower():
        raise RuntimeError("GPU has unexplained processes; see gpu-idle.txt")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Port 9031 is occupied")


def run_mode(mode, root):
    directory = root / mode
    directory.mkdir()
    idle(directory)
    capture(["ps", "-eo", "pid,ppid,args"], directory / "processes-before.txt")
    capture(["free", "-h"], directory / "memory-before.txt")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("VLLM_FL_W8A16")}
    env.update(PYTHONPATH="/workspace/vllm-plugin-FL:/workspace/FlagGems/src",
               VLLM_PLUGINS="fl", GIT_PAGER="cat")
    # DOWN defaults to enabled in the plugin; absence is not a BF16 baseline.
    env.update(VLLM_FL_W8A16_DOWN="0", VLLM_FL_W8A16_DOWN_INT8PACK="0",
               VLLM_FL_W8A16_DOWN_FUSED="0")
    if mode == "candidate":
        env.update(VLLM_FL_W8A16_DOWN="1",
                   VLLM_FL_W8A16_DOWN_INT8PACK="1",
                   VLLM_FL_W8A16_DOWN_FUSED="0")
    command = ["vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
               "--served-model-name", "minicpm", "--gpu-memory-utilization",
               "0.85", "--max-model-len", "131072"]
    benchmark = ["python3", "/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py",
                 "--model", "/workspace/MiniCPM5-2B", "--served-model-name",
                 "minicpm", "--port", "9031", "--test-cases",
                 "[[4096,1024,64,256],[16384,1024,64,128]]"]
    (directory / "commands.json").write_text(json.dumps({
        "serve": command, "benchmark": benchmark,
        "env": {k: v for k, v in env.items()
                if k.startswith("VLLM") or k in ("PYTHONPATH", "MACA_PATH")},
    }, indent=2))
    for name, repo in (("plugin", "/workspace/vllm-plugin-FL"),
                       ("flaggems", "/workspace/FlagGems")):
        for label, args in (("head", ["rev-parse", "HEAD"]),
                            ("status", ["status", "--short"]),
                            ("diff", ["diff"])):
            capture(["git", "--no-pager", "-C", repo, *args],
                    directory / f"{name}-{label}.txt")
    tracked_sources = [
        Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py"),
        *Path("/workspace/vllm-plugin-FL/vllm_fl/quantization/w8a16").glob("*.py"),
        Path("/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py"),
        Path("/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py"),
    ]
    (directory / "source-hashes.json").write_text(json.dumps({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_sources
    }, indent=2))
    service = None
    try:
        with (directory / "server.log").open("w") as server_log:
            service = subprocess.Popen(command, env=env, stdout=server_log,
                                       stderr=subprocess.STDOUT, cwd="/workspace")
            (directory / "server.pid").write_text(str(service.pid))
            deadline = time.monotonic() + 1800
            while True:
                if service.poll() is not None:
                    raise RuntimeError(f"{mode} service exited during startup")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:9031/health",
                                                timeout=5) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Service health timeout")
                time.sleep(5)
            text = (directory / "server.log").read_text()
            coverage = text.count("mode=int8pack_scale_after_dot")
            if (mode == "candidate" and coverage != 42) or (
                    mode == "bf16" and "W8A16 enabled" in text):
                raise RuntimeError(f"Unexpected quantization coverage {coverage}")
            (directory / "health.json").write_text(json.dumps(
                {"http": 200, "down_int8pack_layers": coverage}))
            before = set(Path("/workspace/benchmark_results").glob("*.csv"))
            with (directory / "benchmark.log").open("w") as log:
                result = subprocess.run(benchmark, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, cwd="/workspace")
            (directory / "benchmark-exit-code.txt").write_text(str(result.returncode))
            files = set(Path("/workspace/benchmark_results").glob("*.csv")) - before
            for path in files:
                (directory / path.name).write_bytes(path.read_bytes())
            summaries = [p for p in files if p.name.startswith("summary_")]
            raws = [p for p in files if p.name.startswith("raw_runs_")]
            if result.returncode or len(summaries) != 1 or len(raws) != 1:
                raise RuntimeError("Official benchmark incomplete")
            with summaries[0].open() as stream:
                rows = list(csv.DictReader(stream))
            with raws[0].open() as stream:
                raw = list(csv.DictReader(stream))
            if len(rows) != 2 or len(raw) != 8 or any(
                    r.get("Run Status") != "SUCCESS" for r in raw):
                raise RuntimeError("Official cases/rounds/requests incomplete")
            (directory / "complete.txt").write_text("2 cases, 8 successful raw rounds\n")
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=120)
            except subprocess.TimeoutExpired:
                raise RuntimeError("Service did not stop; refusing next mode")
        time.sleep(5)
        capture(["ps", "-eo", "pid,ppid,args"], directory / "processes-after.txt")
        idle(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for mode in ("candidate", "bf16"):
        print(f"Starting {mode}", flush=True)
        run_mode(mode, args.output_dir)
        print(f"Completed {mode}", flush=True)
    (args.output_dir / "complete.txt").write_text("BF16 and candidate completed\n")


if __name__ == "__main__":
    main()
