# SPDX-License-Identifier: Apache-2.0
"""Diagnostic same-tile BF16 control; no production dispatch changes."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

from vllm_fl.quantization.w8a16.reference import quantize_weight, dequantize_reference
from vllm_fl.quantization.w8a16.dequant import dequantize_weight
from vllm_fl.quantization.w8a16.fused import fused_linear, _fused_kernel


@triton.jit
def _control_kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Mirror _fused_kernel's grouped mapping, masks, dot and output rounding.
    # The only mathematical change is loading already rounded BF16 weights.
    pid = tl.program_id(0)
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (8 * gn)
    size = tl.minimum(gm - group * 8, 8)
    pm = group * 8 + pid % size
    pn = (pid % (8 * gn)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + rk
        a = tl.load(X + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(W + rn[None, :] * K + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K), other=0)
        acc += tl.dot(a, b)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


def control(x, w, output=None):
    m, k = x.shape
    n = w.shape[0]
    if output is None:
        output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    bm = 16 if m <= 32 else 32
    with torch.cuda.device(x.device):
        _control_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, 64),)](
            x, w, output, m, n, k, bm, 64, 64,
            num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output


def launch_fused(x, q, scale, output):
    m, k = x.shape
    n = q.shape[0]
    bm = 16 if m <= 32 else 32
    with torch.cuda.device(x.device):
        _fused_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, 64),)](
            x, q, scale, output, output, m, n, k, k, False, bm, 64, 64,
            num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output


def save_compiled(out, name, compiled):
    # Backend-dependent fields may be unavailable; preserve that explicitly.
    info = {key: getattr(compiled, key, None) for key in ('n_regs', 'n_spills')}
    metadata = getattr(compiled, 'metadata', None)
    info['metadata'] = metadata._asdict() if hasattr(metadata, '_asdict') else str(metadata)
    info['shared'] = getattr(metadata, 'shared', None)
    info['assembly_kinds'] = list(compiled.asm)
    (out / (name + '-resources.json')).write_text(json.dumps(info, indent=2, default=str))
    for kind, assembly in compiled.asm.items():
        if isinstance(assembly, str):
            (out / (name + '.' + kind)).write_text(assembly)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    # Masked shapes verify the diagnostic control before real Down measurements.
    for m, n, k in ((1, 17, 33), (64, 129, 257)):
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
        torch.testing.assert_close(control(x, w), F.linear(x, w), rtol=.02, atol=.125)
    n, k = 2048, 6144
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = quantize_weight(w)
    rounded = dequantize_reference(q, scale)
    torch.testing.assert_close(dequantize_weight(q, scale), rounded, rtol=0, atol=0)
    metadata = dict(seed=42, n=n, k=k, torch=torch.__version__, triton=triton.__version__,
                    rounds=3, warmup_ms=100, rep_ms=300,
                    statistic='median of three per-round GPU latency medians',
                    hardware=torch.cuda.get_device_name(),
                    full_paths=['bf16', 'control', 'separate_triton', 'fused'],
                    kernel_only=['control_preallocated', 'fused_preallocated'],
                    note='same tiling is not identical generated code; control uses BF16 weight storage')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    rows, raw = [], []
    for m in (1, 64, 256, 2048):
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        yc = torch.empty((m, n), device=x.device, dtype=x.dtype)
        yf = torch.empty_like(yc)
        expected = F.linear(x, rounded)
        results = dict(control=control(x, rounded), fused=fused_linear(x, q, scale),
                       separate=F.linear(x, dequantize_weight(q, scale)))
        errors = {}
        for name, actual in results.items():
            error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
            assert error < .005, (m, name, error)
            errors[name] = error
        torch.testing.assert_close(results['separate'], expected, rtol=0, atol=0)
        paths = dict(bf16=lambda: F.linear(x, w), control=lambda: control(x, rounded),
                     separate_triton=lambda: F.linear(x, dequantize_weight(q, scale)),
                     fused=lambda: fused_linear(x, q, scale),
                     control_preallocated=lambda: control(x, rounded, yc),
                     fused_preallocated=lambda: launch_fused(x, q, scale, yf))
        names = list(paths)
        samples = {name: [] for name in names}
        for repeat in range(3):
            shift = repeat * 2
            for name in names[shift:] + names[:shift]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300,
                                             return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, round=repeat + 1, path=name, median_ms=ms))
                (out / 'raw_timings.json').write_text(json.dumps(raw, indent=2))
        bm = 16 if m <= 32 else 32
        row = dict(m=m, n=n, k=k, bm=bm, bn=64, bk=64, num_warps=4, num_stages=2,
                   relative_l2_errors=errors,
                   **{name + '_ms': median(values) for name, values in samples.items()})
        rows.append(row)
        (out / 'summary.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, 64),)
        c = _control_kernel.warmup(x, rounded, yc, m, n, k, bm, 64, 64,
            grid=grid, num_warps=4, num_stages=2, enable_fp_fusion=False)
        f = _fused_kernel.warmup(x, q, scale, yf, yf, m, n, k, k, False, bm, 64, 64,
            grid=grid, num_warps=4, num_stages=2, enable_fp_fusion=False)
        save_compiled(out, 'control-m' + str(m), c)
        save_compiled(out, 'fused-m' + str(m), f)


if __name__ == '__main__':
    main()
