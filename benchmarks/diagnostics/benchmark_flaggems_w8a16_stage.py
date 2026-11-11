# SPDX-License-Identifier: Apache-2.0
"""Diagnostic stage-cost breakdown of FlagGems MetaX two-step Split-K W8A16.

Measures, at the previously selected optimal Split-K per M, the separate cost of
the partial kernel, the reduce+scale kernel, the full fresh-allocated call, and
the full pre-allocated call. No production operator registration.
"""
import argparse
import importlib
import json
from pathlib import Path
from statistics import median
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

from benchmark_flaggems_w8a16 import load_helper, dump, relative, save_kernel


@triton.jit
def _partial(A, B, P, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
             SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // grid_n, pid % grid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum((pid_k + 1) * k_per_split, total_k_iters)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
                    mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K), other=0.0)
        b = tl.load(B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
                    mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N), other=0)
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)
    p_ptrs = P + pid_k * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(p_ptrs, acc, mask=(offsets_m < M)[:, None] & (offsets_n < N)[None, :])


@triton.jit
def _reduce_scale(P, S, C, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        acc += tl.load(P + split_id * M * N + offsets, mask=mask, other=0.0)
    scales = tl.load(S + offsets % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(C + offsets, (acc * scales).to(tl.bfloat16), mask=mask)


def _partial_grid(m, n, splits):
    return (triton.cdiv(m, 16) * triton.cdiv(n, 64), splits)


def _reduce_grid(m, n):
    return (triton.cdiv(m * n, 256),)


def partial_only(x, q, partials, splits):
    m, k = x.shape
    n = q.shape[0]
    with torch.cuda.device(x.device):
        return _partial[_partial_grid(m, n, splits)](
            x, q, partials, m, n, k, x.stride(0), x.stride(1),
            q.stride(1), q.stride(0), splits, 16, 64, 128,
            num_warps=4, num_stages=2, pipeline='basic')


def reduce_only(partials, scale, output, splits):
    m, n = output.shape
    with torch.cuda.device(partials.device):
        return _reduce_scale[_reduce_grid(m, n)](
            partials, scale, output, m, n, splits, 256, num_warps=4)


def splitk_fresh(x, q, scale, splits, capture=False):
    m, k = x.shape
    n = q.shape[0]
    partials = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        part = partial_only(x, q, partials, splits)
        finish = reduce_only(partials, scale, output, splits)
    return (output, part, finish) if capture else output


def splitk_prealloc(x, q, scale, splits, partials, output, capture=False):
    with torch.cuda.device(x.device):
        part = partial_only(x, q, partials, splits)
        finish = reduce_only(partials, scale, output, splits)
    return (output, part, finish) if capture else output


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
    mm = dense.mm
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    seed = 20261027
    torch.manual_seed(seed)
    dump(out, 'metadata.json', dict(
        torch=torch.__version__, triton=triton.__version__,
        device=torch.cuda.get_device_name(), seed=seed, n=2048, k=6144,
        contract='FP32 accumulate(x_bf16 * BF16(q_int8)), then FP32 per-channel scale, then BF16',
        selected=[{'m': m, 'splits': s} for m, s in SELECTED],
        timing='do_bench(warmup=100,rep=300,return_mode=median); wall=median of 5 synchronized runs',
        paths='partial_only; reduce_only; full_fresh; full_prealloc; bf16_metax'))
    correctness = []
    for m, n, k in ((1, 17, 33), (3, 129, 257)):
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        ww = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
        qq, ss = ref.quantize_weight(ww)
        expected = (F.linear(x.float(), qq.float()) * ss).to(torch.bfloat16)
        for splits in (2, 4):
            actual = splitk_fresh(x, qq, ss, splits)
            err = relative(actual, expected)
            assert err < .005, (m, n, k, splits, err)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
            correctness.append(dict(m=m, n=n, k=k, splits=splits, relative_l2_error=err))
            dump(out, 'correctness.json', correctness)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = ref.quantize_weight(w)
    torch.testing.assert_close(deq.dequantize_weight(q, scale),
                               ref.dequantize_reference(q, scale), rtol=0, atol=0)
    rows, raw = [], []
    for m, splits in SELECTED:
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        partials = torch.empty((splits, m, 2048), device='cuda', dtype=torch.float32)
        output = torch.empty((m, 2048), device='cuda', dtype=torch.bfloat16)
        actual, part, finish = splitk_fresh(x, q, scale, splits, capture=True)
        err = relative(actual, expected)
        assert err < .005, (m, splits, err)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
        save_kernel(out, f'm{m}-s{splits}-partial', part,
                    dict(BM=16, BN=64, BK=128, SPLIT_K=splits))
        save_kernel(out, f'm{m}-s{splits}-reduce', finish,
                    dict(BLOCK_SIZE=256, SPLIT_K=splits))
        paths = dict(
            partial=lambda: partial_only(x, q, partials, splits),
            reduce=lambda: reduce_only(partials, scale, output, splits),
            full_fresh=lambda: splitk_fresh(x, q, scale, splits),
            full_prealloc=lambda: splitk_prealloc(x, q, scale, splits, partials, output),
            bf16_metax=lambda: mm(x, w.t()),
        )
        preallocated = splitk_prealloc(x, q, scale, splits, partials, output)
        torch.testing.assert_close(preallocated, expected, rtol=.02, atol=.125)
        torch.testing.assert_close(preallocated, actual, rtol=0, atol=0)
        correctness.append(dict(m=m, n=2048, k=6144, splits=splits,
            relative_l2_error=err, prealloc_matches_fresh=True))
        dump(out, 'correctness.json', correctness)
        for fn in paths.values():
            fn()
        torch.cuda.synchronize()
        wall = {}
        for name in ('full_fresh', 'full_prealloc'):
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                paths[name]()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            wall[name] = median(samples)
        samples = {name: [] for name in paths}
        for repeat in range(3):
            names = list(paths)
            for name in names[repeat:] + names[:repeat]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300,
                                             return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, splits=splits, round=repeat + 1, path=name, ms=ms))
                dump(out, 'raw_timings.json', raw)
        timings = {name: median(values) for name, values in samples.items()}
        row = dict(
            m=m, splits=splits, n=2048, k=6144,
            timings_ms=timings,
            wall_ms=wall,
            wall_fresh_minus_prealloc_ms=wall['full_fresh'] - wall['full_prealloc'],
            gpu_fresh_minus_prealloc_ms=timings['full_fresh'] - timings['full_prealloc'],
            full_minus_isolated_partial_ms=timings['full_prealloc'] - timings['partial'],
            interpretation='Independent timings are not additive; differences do not isolate allocation or launch cost.',
        )
        rows.append(row)
        dump(out, 'stages.json', rows)
        print(json.dumps(row), flush=True)


SELECTED = [(1, 8), (16, 8), (32, 8), (64, 4)]


if __name__ == '__main__':
    main()
