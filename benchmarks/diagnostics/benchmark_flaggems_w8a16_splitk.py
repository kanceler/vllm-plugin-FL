# SPDX-License-Identifier: Apache-2.0
"""Diagnostic W8A16 adaptation of FlagGems MetaX two-step Split-K.

Partial/finish structure follows _metax/ops/mm.py; only weight conversion
and final channel scaling are added. No production operator registration.
"""
import argparse
import importlib
import inspect
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
from benchmark_flaggems_w8a16_configs import launch as configured_fused


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
        # INT8 values are exactly representable in BF16. Scale stays after reduction.
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


def splitk(x, q, scale, splits, capture=False):
    m, k = x.shape
    n = q.shape[0]
    partials = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    # Same tile as the prior small-M configuration candidate; vary only Split-K.
    with torch.cuda.device(x.device):
        part = _partial[(triton.cdiv(m, 16) * triton.cdiv(n, 64), splits)](
            x, q, partials, m, n, k, x.stride(0), x.stride(1), q.stride(1), q.stride(0),
            splits, 16, 64, 128, num_warps=4, num_stages=2, pipeline='basic')
        finish = _reduce_scale[(triton.cdiv(m * n, 256),)](
            partials, scale, output, m, n, splits, 256, num_warps=4)
    return (output, part, finish) if capture else output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helpers-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    ref = load_helper('reference', args.helpers_dir)
    deq = load_helper('dequant', args.helpers_dir)
    dense = importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    weight = importlib.import_module('flag_gems.ops.weight_int8pack_mm')
    jit = weight.weight_int8pack_mm_kernel.jit_function
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dump(out, 'metadata.json', dict(seed=20261027, torch=torch.__version__, triton=triton.__version__,
        n=2048, k=6144, m_values=[1,16,32,64], splits=[1,2,4,8],
        tile=[16,64,128], warps=4, stages=2, pipeline='basic',
        partial_source=inspect.getsourcefile(dense.mm_kernel_splitk_partial.jit_function.fn),
        contract='FP32 partials -> FP32 reduce -> FP32 channel scale -> BF16',
        statistic='median of three per-round GPU latency medians', warmup_ms=100, rep_ms=300,
        includes='output and workspace allocations, partial kernel and reduce+scale kernel',
        split1_note='two-stage control, also compared with previous single-kernel configuration',
        production_switches_added=False))
    torch.manual_seed(20261027)
    correctness = []
    for splits in (1, 2, 4, 8):
        for m, n, k in ((1,17,33), (3,129,257)):
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
            w[0] = 0
            q, scale = ref.quantize_weight(w)
            expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
            actual = splitk(x, q, scale, splits)
            err = relative(actual, expected)
            assert err < .005, (splits, m, n, k, err)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
            torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), rtol=0, atol=0)
            correctness.append(dict(m=m,n=n,k=k,splits=splits,relative_l2_error=err,zero_channel=True))
            dump(out, 'correctness.json', correctness)
    w = torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16) * .02
    q, scale = ref.quantize_weight(w)
    torch.testing.assert_close(deq.dequantize_weight(q,scale), ref.dequantize_reference(q,scale),rtol=0,atol=0)
    raw, summaries, first = [], [], []
    for m in (1,16,32,64):
        x = torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        expected = (F.linear(x.float(),q.float()) * scale).to(torch.bfloat16)
        paths = dict(bf16_metax=lambda: dense.mm(x,w.t()),
            separate_triton_metax=lambda: dense.mm(x,deq.dequantize_weight(q,scale).t()),
            original_fused=lambda: weight.weight_int8pack_mm(x,q,scale),
            configured_fused=lambda: configured_fused(jit,(16,64,128,4,2),x,q,scale))
        errors = {}
        for splits in (1,2,4,8):
            name = 'split' + str(splits)
            start = time.perf_counter()
            actual, part, finish = splitk(x,q,scale,splits,capture=True)
            torch.cuda.synchronize()
            first.append(dict(m=m,splits=splits,wall_s=time.perf_counter()-start))
            errors[name] = relative(actual,expected)
            assert errors[name] < .005, (m,splits,errors[name])
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.125)
            save_kernel(out,f'm{m}-s{splits}-partial',part,dict(BM=16,BN=64,BK=128,SPLIT_K=splits))
            save_kernel(out,f'm{m}-s{splits}-reduce',finish,dict(BLOCK_SIZE=256,SPLIT_K=splits))
            paths[name] = lambda splits=splits: splitk(x,q,scale,splits)
        for name in ('original_fused','configured_fused'):
            errors[name] = relative(paths[name](),expected)
            assert errors[name] < .005
        for fn in paths.values():
            fn()
        torch.cuda.synchronize()
        dump(out,'first_calls.json',first)
        names = list(paths)
        samples = {name: [] for name in names}
        for repeat in range(3):
            shift = 3*repeat
            for name in names[shift:] + names[:shift]:
                ms = triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m,round=repeat+1,path=name,ms=ms))
                dump(out,'raw_timings.json',raw)
        timings = {name: median(values) for name,values in samples.items()}
        winner = min(('split1','split2','split4','split8'),key=lambda name:timings[name])
        row = dict(m=m,n=2048,k=6144,relative_l2_errors=errors,timings_ms=timings,
            best_split=winner,best_split_speedup_vs_bf16=timings['bf16_metax']/timings[winner],
            workspace_bytes={str(s):s*m*2048*4 for s in (1,2,4,8)})
        summaries.append(row)
        dump(out,'summary.json',summaries)
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
