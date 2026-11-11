# SPDX-License-Identifier: Apache-2.0
"""Explicit MetaX BF16 vs unmodified FlagGems weight-only scaled GEMM."""
import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from statistics import median
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.testing
import flag_gems


def load_helper(name, folder):
    spec = importlib.util.spec_from_file_location('diagnostic_' + name, folder / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dump(folder, name, value):
    (folder / name).write_text(json.dumps(value, indent=2, default=str))


def relative(actual, expected):
    return ((actual.float() - expected.float()).norm() /
            expected.float().norm().clamp_min(1e-12)).item()


def save_kernel(folder, label, kernel, constexprs):
    meta = getattr(kernel, 'metadata', None)
    dump(folder, label + '-resources.json', dict(constexprs=constexprs,
        n_regs=getattr(kernel, 'n_regs', None), n_spills=getattr(kernel, 'n_spills', None),
        metadata=meta._asdict() if hasattr(meta, '_asdict') else str(meta),
        assembly_kinds=list(kernel.asm)))
    for kind, value in kernel.asm.items():
        if isinstance(value, str):
            (folder / (label + '.' + kind)).write_text(value)
        elif isinstance(value, bytes):
            (folder / (label + '.' + kind)).write_bytes(value)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--helpers-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    ref = load_helper('reference', args.helpers_dir)
    deq = load_helper('dequant', args.helpers_dir)
    dense = importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    weight = importlib.import_module('flag_gems.ops.weight_int8pack_mm')
    # Use the actual MetaX BF16 entry explicitly, not a native F.linear proxy.
    mm = dense.mm
    fused = weight.weight_int8pack_mm
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    metadata = dict(torch=torch.__version__, triton=triton.__version__,
        flag_gems_version=getattr(flag_gems, '__version__', None),
        flag_gems_path=flag_gems.__file__, bf16_source=inspect.getsourcefile(mm),
        fused_source=inspect.getsourcefile(fused), device=torch.cuda.get_device_name(),
        seed=42, n=2048, k=6144, rounds=3, warmup_ms=100, rep_ms=300,
        statistic='median of three per-round GPU latency medians',
        contract='FP32 accumulate(x_bf16 * q_bf16), then FP32 per-channel scale, then BF16',
        first_call='wall time includes compile/tune; excluded from steady-state timings',
        paths='explicit MetaX BF16 mm; unmodified generic weight_int8pack_mm; Triton dequant + same MetaX mm',
        sources={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                 for p in (inspect.getsourcefile(mm), inspect.getsourcefile(fused))})
    dump(out, 'metadata.json', metadata)
    torch.manual_seed(42)
    correctness = []
    # Reference has FP32 accumulation and FP32 epilogue, no intermediate BF16 matmul output.
    for m, n, k in ((1, 17, 33), (3, 129, 257), (16, 128, 256)):
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
        w[0] = 0
        q, scale = ref.quantize_weight(w)
        expected = (F.linear(x.float(), q.float()) * scale[None, :]).to(torch.bfloat16)
        actual = fused(x, q, scale)
        err = relative(actual, expected)
        assert err < .005, (m, n, k, err)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
        torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), rtol=0, atol=0)
        correctness.append(dict(m=m, n=n, k=k, relative_l2_error=err, zero_channel_passed=True))
        dump(out, 'correctness.json', correctness)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = ref.quantize_weight(w)
    rounded = ref.dequantize_reference(q, scale)
    torch.testing.assert_close(deq.dequantize_weight(q, scale), rounded, rtol=0, atol=0)
    summaries, raw, first = [], [], []
    captured = set()
    for m in (1, 16, 32, 64, 128, 256, 512, 1024, 2048):
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        paths = dict(bf16_metax=lambda: mm(x, w.t()),
                     weight_int8pack=lambda: fused(x, q, scale),
                     separate_triton_metax=lambda: mm(x, deq.dequantize_weight(q, scale).t()))
        outputs = {}
        for name, fn in paths.items():
            start = time.perf_counter()
            outputs[name] = fn()
            torch.cuda.synchronize()
            first.append(dict(m=m, path=name, first_call_wall_s=time.perf_counter() - start))
            dump(out, 'first_calls.json', first)
        expected = (F.linear(x.float(), q.float()) * scale[None, :]).to(torch.bfloat16)
        fused_err = relative(outputs['weight_int8pack'], expected)
        assert fused_err < .005, (m, fused_err)
        torch.testing.assert_close(outputs['weight_int8pack'], expected, rtol=.02, atol=.125)
        dense_ref = F.linear(x.float(), w.float()).to(torch.bfloat16)
        dense_err = relative(outputs['bf16_metax'], dense_ref)
        assert dense_err < .005, (m, dense_err)
        separate_ref = F.linear(x.float(), rounded.float()).to(torch.bfloat16)
        separate_err = relative(outputs['separate_triton_metax'], separate_ref)
        assert separate_err < .005, (m, separate_err)
        # Obtain the exact selected fused config and its compiled kernel from LibEntry.
        y = torch.empty((m, 2048), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(2048, meta['BLOCK_N']),)
        compiled, constexprs = weight.weight_int8pack_mm_kernel.run(
            x, q, scale, y, m, 2048, 6144, x.stride(0), x.stride(1),
            q.stride(0), q.stride(1), y.stride(0), y.stride(1), scale.stride(0),
            GROUP_M=8, grid=grid)
        save_kernel(out, 'weight-m' + str(m), compiled, constexprs)
        dense_entries = []
        for entry_name in ('mm_kernel', 'mm_kernel_nn', 'mm_kernel_nt', 'mm_kernel_splitk',
                           'mm_kernel_splitk_partial', 'mm_kernel_splitk_reduce'):
            entry = getattr(dense, entry_name, None)
            if entry is None or not hasattr(entry, 'kernel_cache'):
                continue
            for index, (key, value) in enumerate(entry.kernel_cache[torch.cuda.current_device()].items()):
                dense_entries.append(dict(entry=entry_name, key=str(key), constexprs=value[1]))
                token = (entry_name, str(key))
                if token not in captured:
                    save_kernel(out, 'dense-m' + str(m) + '-' + entry_name + '-' + str(index), value[0], value[1])
                    captured.add(token)
        dump(out, 'dense-entries-m' + str(m) + '.json', dense_entries)
        names = list(paths)
        samples = {name: [] for name in names}
        for repeat in range(3):
            for name in names[repeat:] + names[:repeat]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300, return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, round=repeat + 1, path=name, ms=ms))
                dump(out, 'raw_timings.json', raw)
        row = dict(m=m, n=2048, k=6144, scale_dtype=str(scale.dtype), fused_config=constexprs,
                   relative_l2_errors=dict(fused_contract=fused_err, bf16=dense_err, separate=separate_err),
                   quantization_relative_l2_vs_bf16=relative(outputs['weight_int8pack'], outputs['bf16_metax']),
                   timings_ms={name: median(values) for name, values in samples.items()})
        summaries.append(row)
        dump(out, 'summary.json', summaries)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
