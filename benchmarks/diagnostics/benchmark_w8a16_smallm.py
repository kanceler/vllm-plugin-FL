# SPDX-License-Identifier: Apache-2.0
"""Experimental small-M W8A16 candidates. No production registration or switches."""
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
def _gemv(X, Q, S, Y, P, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, KB: tl.constexpr, SPLIT: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    kk = split * KB + tl.arange(0, KB)
    x = tl.load(X + kk, kk < K, other=0).to(tl.float32)
    q = tl.load(Q + rn[:, None] * K + kk[None, :],
                (rn[:, None] < N) & (kk[None, :] < K), other=0)
    scale = tl.load(S + rn, rn < N, other=0)
    # Preserve BF16 weight rounding before any product or partial sum.
    w = (q.to(tl.float32) * scale[:, None]).to(tl.bfloat16).to(tl.float32)
    total = tl.sum(w * x[None, :], axis=1)
    if SPLIT == 1:
        tl.store(Y + rn, total.to(tl.bfloat16), rn < N)
    else:
        tl.store(P + rn * SPLIT + split, total, rn < N)


@triton.jit
def _finish(P, Y, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    rn = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rs = tl.arange(0, SPLIT)
    partial = tl.load(P + rn[:, None] * SPLIT + rs[None, :],
                      rn[:, None] < N, other=0)
    tl.store(Y + rn, tl.sum(partial, axis=1).to(tl.bfloat16), rn < N)


@triton.jit
def _prefetch(X, Q, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (8 * gn)
    size = tl.minimum(gm - group * 8, 8)
    pm, pn = group * 8 + pid % size, (pid % (8 * gn)) // size
    rm, rn = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    scale = tl.load(S + rn, rn < N, other=0)
    a = tl.load(X + rm[:, None] * K + rk[None, :],
                (rm[:, None] < M) & (rk[None, :] < K), other=0)
    q = tl.load(Q + rn[None, :] * K + rk[:, None],
                (rn[None, :] < N) & (rk[:, None] < K), other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        # Explicitly fetch the next raw tiles before consuming current tiles.
        kk = (block + 1) * BK + rk
        next_a = tl.load(X + rm[:, None] * K + kk[None, :],
                         (rm[:, None] < M) & (kk[None, :] < K), other=0)
        next_q = tl.load(Q + rn[None, :] * K + kk[:, None],
                         (rn[None, :] < N) & (kk[:, None] < K), other=0)
        b = (q.to(tl.float32) * scale[None, :]).to(tl.bfloat16)
        acc += tl.dot(a, b)
        a, q = next_a, next_q
    tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


def execute(config, x, q, scale):
    m, k = x.shape
    n = q.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        if config['kind'] == 'gemv':
            assert m == 1
            splits = config['split']
            kb = triton.next_power_of_2(triton.cdiv(k, splits))
            partial = torch.empty((n, splits), device=x.device, dtype=torch.float32) if splits > 1 else y
            _gemv[(triton.cdiv(n, config['bn']), splits)](
                x, q, scale, y, partial, n, k, config['bn'], kb, splits,
                num_warps=config['warps'], enable_fp_fusion=False)
            if splits > 1:
                _finish[(triton.cdiv(n, 128),)](partial, y, n, splits, 128,
                                               num_warps=4, enable_fp_fusion=False)
        else:
            bm, bn, bk = config['bm'], config['bn'], config['bk']
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
            options = dict(num_warps=config['warps'], num_stages=config['stages'], enable_fp_fusion=False)
            if config['kind'] == 'prefetch':
                _prefetch[grid](x, q, scale, y, m, n, k, bm, bn, bk, **options)
            else:
                _fused_kernel[grid](x, q, scale, y, y, m, n, k, k, False, bm, bn, bk, **options)
    return y


def candidates(m):
    if m == 1:
        return [dict(id=f'gemv-n{bn}-w{warps}-s{split}', kind='gemv', bn=bn, warps=warps, split=split)
                for split, bns, warps_list in [(1, (1, 2, 4), (4, 8)), (2, (1, 4), (4,)), (4, (1, 4), (4,))]
                for bn in bns for warps in warps_list]
    bm = 16 if m <= 32 else 32
    base = dict(kind='tile', bm=bm, bn=64, bk=64, warps=4, stages=2)
    result = [dict(base, id='tile-default')]
    for key, value in [('bn', 128), ('bk', 128), ('bk', 256), ('warps', 8),
                       ('stages', 3), ('bm', 32 if bm == 16 else 16)]:
        result.append(dict(base, id=f'tile-{key}{value}', **{key: value}))
    result.append(dict(base, id='prefetch-default', kind='prefetch'))
    return result


def error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def validate(config, x, q, scale, expected):
    actual = execute(config, x, q, scale)
    rel = error(actual, expected)
    assert rel < .005, (config['id'], x.shape, rel)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
    return rel


def dump(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    dump(out, 'metadata.json', dict(torch=torch.__version__, triton=triton.__version__,
        device=torch.cuda.get_device_name(), screen_seed=42, validation_seed=314159,
        n=2048, k=6144, screen_warmup_ms=50, screen_rep_ms=120,
        final_warmup_ms=100, final_rep_ms=300, final_rounds=3,
        timing='eager full path; median of three GPU latency medians; includes allocations and split reduction',
        production_changes=False))
    torch.manual_seed(42)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = quantize_weight(w)
    rounded = dequantize_reference(q, scale)
    torch.testing.assert_close(dequantize_weight(q, scale), rounded, rtol=0, atol=0)
    screening, selected = [], {}
    for m in (1, 16, 32, 64):
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        expected = F.linear(x, rounded)
        passed = []
        for config in candidates(m):
            row = dict(m=m, config=config)
            try:
                # Each candidate must also pass a masked shape before timing.
                odd_x = torch.randn(1 if m == 1 else 3, 257, device='cuda', dtype=torch.bfloat16)
                odd_w = torch.randn(129, 257, device='cuda', dtype=torch.bfloat16) * .02
                odd_q, odd_s = quantize_weight(odd_w)
                validate(config, odd_x, odd_q, odd_s, F.linear(odd_x, dequantize_reference(odd_q, odd_s)))
                row['relative_l2_error'] = validate(config, x, q, scale, expected)
                row['ms'] = triton.testing.do_bench(lambda: execute(config, x, q, scale),
                                                   warmup=50, rep=120, return_mode='median')
                row['status'] = 'passed'
                passed.append(row)
            except Exception as exc:
                row.update(status='failed', error=repr(exc))
                screening.append(row)
                dump(out, 'screening.json', screening)
                if type(exc).__name__ not in ('OutOfResources', 'CompilationError', 'AssertionError'):
                    raise
                continue
            screening.append(row)
            dump(out, 'screening.json', screening)
            print(json.dumps(row), flush=True)
        selected[str(m)] = [r['config'] for r in sorted(passed, key=lambda r: r['ms'])[:2]]
    dump(out, 'selected.json', selected)
    torch.manual_seed(314159)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = quantize_weight(w)
    rounded = dequantize_reference(q, scale)
    raw, summaries = [], []
    for m in (1, 16, 32, 64, 256, 2048):
        configs = selected[str(m)] if m <= 64 else selected['64'][:1]
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        expected = F.linear(x, rounded)
        torch.testing.assert_close(F.linear(x, dequantize_weight(q, scale)), expected, rtol=0, atol=0)
        paths = dict(bf16=lambda: F.linear(x, w),
                     separate_triton=lambda: F.linear(x, dequantize_weight(q, scale)),
                     old_fused=lambda: fused_linear(x, q, scale))
        errors = {}
        for config in configs:
            errors[config['id']] = validate(config, x, q, scale, expected)
            paths[config['id']] = lambda config=config: execute(config, x, q, scale)
        names = list(paths)
        samples = {name: [] for name in names}
        for repeat in range(3):
            for name in names[repeat:] + names[:repeat]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300, return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, round=repeat + 1, path=name, ms=ms))
                dump(out, 'raw_final.json', raw)
        row = dict(m=m, configs=configs, relative_l2_errors=errors,
                   timings_ms={name: median(values) for name, values in samples.items()})
        summaries.append(row)
        dump(out, 'final_summary.json', summaries)
        print('FINAL ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
