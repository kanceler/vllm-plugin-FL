#!/usr/bin/env python3
"""Opt-in process-local vLLM observer. Does not modify installed packages.

serve --output DIR -- vllm serve ... preserves all supplied serving arguments.
control --output DIR --label 4k --steps 64 arms a bounded worker trace.
No tensor values, quantization, model hooks, or eager-mode switches are used.
"""

import argparse
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

PREFIX_STEP = "GEMM_DIAG_STEP "
PREFIX_GRAPH = "GEMM_DIAG_GRAPH "
_observer = None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class Observer:
    def __init__(self, worker, module):
        self.worker = worker
        self.module = module
        self.root = Path(os.environ["GEMM_DIAG_DIR"])
        self.pid = os.getpid()
        self.profiler = None
        self.control = {}
        self.seen = set()
        self.steps = 0
        self.last_poll = 0.0
        self.status = self.root / f"worker-{self.pid}-status.json"
        config = worker.vllm_config
        write_json(self.root / f"worker-{self.pid}-config.json", {
            "pid": self.pid, "model": str(config.model_config.model),
            "dtype": str(config.model_config.dtype),
            "quantization": config.model_config.quantization,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "compilation_config": repr(config.compilation_config),
            "profiler_config": repr(config.profiler_config),
            "observation": "direct bounded torch.profiler; vLLM config unchanged",
        })

    def inventory(self):
        result = []
        for name, layer in self.worker.model_runner.get_model().named_modules():
            weight = getattr(layer, "weight", None)
            if weight is None or getattr(weight, "ndim", 0) != 2:
                continue
            result.append({"module": name, "prefix": getattr(layer, "prefix", name),
                           "class": type(layer).__name__, "shape": list(weight.shape),
                           "stride": list(weight.stride()), "dtype": str(weight.dtype),
                           "method": type(getattr(layer, "quant_method", None)).__name__})
        write_json(self.root / f"worker-{self.pid}-inventory.json", result)

    def poll(self):
        if time.monotonic() - self.last_poll < 0.2:
            return
        self.last_poll = time.monotonic()
        path = self.root / "control.json"
        if not path.exists():
            return
        control = json.loads(path.read_text())
        if control["id"] in self.seen:
            return
        self.seen.add(control["id"])
        if self.profiler is not None:
            self.finish("replaced_by_control")
        if control["action"] == "stop":
            return
        import torch

        self.control = control
        self.steps = 0
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=False, profile_memory=False)
        self.profiler.start()
        write_json(self.status, {"state": "recording", **control, "pid": self.pid})

    def finish(self, reason):
        profiler, self.profiler = self.profiler, None
        if profiler is None:
            return
        profiler.stop()
        path = self.root / f"trace-{self.control['id']}-pid{self.pid}.json"
        profiler.export_chrome_trace(str(path))
        write_json(self.status, {"state": "saved", "reason": reason,
                                "steps": self.steps, "trace": str(path),
                                "control": self.control, "pid": self.pid})

    def execute(self, original, *args, **kwargs):
        scheduler = args[0] if args else kwargs["scheduler_output"]
        self.poll()
        if self.profiler is None:
            return original(self.worker, *args, **kwargs)
        import torch

        from vllm.v1.utils import compute_iteration_details
        details = compute_iteration_details(scheduler)
        ctx = details.num_ctx_tokens
        gen = details.num_generation_tokens
        phase = "mixed" if ctx and gen else "prefill" if ctx else "decode" if gen else "empty"
        data = {"label": self.control["label"], "step": self.steps,
                "phase": phase, "context_tokens": ctx, "generation_tokens": gen,
                "active_tokens": scheduler.total_num_scheduled_tokens}
        with torch.profiler.record_function(PREFIX_STEP + json.dumps(data)):
            result = original(self.worker, *args, **kwargs)
        self.steps += 1
        if self.steps >= self.control["steps"]:
            self.finish("step_limit")
        else:
            self.profiler.step()
        return result


def patch_worker(module):
    Worker = module.WorkerFL if hasattr(module, "WorkerFL") else module.Worker
    if getattr(Worker, "_gemm_diag_installed", False):
        return
    Worker._gemm_diag_installed = True
    original_init = Worker.__init__
    original_load = Worker.load_model
    original_execute = Worker.execute_model

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        global _observer
        original_init(self, *args, **kwargs)
        _observer = Observer(self, module)

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        _observer.inventory()
        return result

    @functools.wraps(original_execute)
    def execute(self, *args, **kwargs):
        try:
            return _observer.execute(original_execute, *args, **kwargs)
        except Exception:
            # Fail visibly; never retry a partially executed model step.
            (Path(os.environ["GEMM_DIAG_DIR"]) / f"error-{os.getpid()}.txt").write_text(
                traceback.format_exc())
            raise

    Worker.__init__, Worker.load_model, Worker.execute_model = initialize, load, execute

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.forward_context import get_forward_context, is_forward_context_available

    def wrap_graph_class(cls, breakable):
        if getattr(cls, "_gemm_diag_graph", False):
            return
        cls._gemm_diag_graph = True
        original_graph = cls.__call__

        @functools.wraps(original_graph)
        def graph(self, *args, **kwargs):
            if _observer is None or _observer.profiler is None or not is_forward_context_available():
                return original_graph(self, *args, **kwargs)
            import torch

            context = get_forward_context()
            descriptor = context.batch_descriptor
            mode = context.cudagraph_runtime_mode
            matching = mode.name != "NONE" if breakable else mode == self.runtime_mode
            if not matching or descriptor is None:
                return original_graph(self, *args, **kwargs)
            entries = self.entries if breakable else self.concrete_cudagraph_entries
            entry = entries.get(descriptor)
            captured = (getattr(entry, "capture" if breakable else "cudagraph", None) is not None)
            data = {"mode": str(mode), "graph_tokens": descriptor.num_tokens,
                    "action": "replay" if captured else "capture"}
            with torch.profiler.record_function(PREFIX_GRAPH + json.dumps(data)):
                return original_graph(self, *args, **kwargs)

        cls.__call__ = graph

    wrap_graph_class(CUDAGraphWrapper, False)
    wrap_graph_class(BreakableCUDAGraphWrapper, True)


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_worker(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ("vllm.v1.worker.gpu_worker", "vllm_fl.worker.worker"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader)
        return spec


def install():
    if os.environ.get("GEMM_DIAG_DIR"):
        sys.meta_path.insert(0, Finder())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("command", nargs=argparse.REMAINDER)
    control = sub.add_parser("control")
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--label", default="pilot")
    control.add_argument("--steps", type=int, default=64)
    control.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == "control":
        if not root.is_dir() or args.steps < 1:
            parser.error("existing output directory and positive steps required")
        write_json(root / "control.json", {"id": uuid.uuid4().hex, "label": args.label,
                                          "steps": args.steps, "action": "stop" if args.stop else "start"})
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command[:2] != ["vllm", "serve"]:
        parser.error("expected -- vllm serve <unchanged official arguments>")
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    (root / script.name).write_bytes(script.read_bytes())
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "from collect_gemm_shapes import install\ninstall()\n")
    env = dict(os.environ)
    env["GEMM_DIAG_DIR"] = str(root)
    env["PYTHONPATH"] = os.pathsep.join([str(bootstrap), str(script.parent), env.get("PYTHONPATH", "")])
    write_json(root / "launcher.json", {"argv": command, "pid": os.getpid(),
                                       "source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                                       "diagnostic_only": True})
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
