# SPDX-License-Identifier: Apache-2.0
"""Bounded MetaX config experiment on the unmodified FlagGems weight-only JIT."""
import argparse
import importlib
import json
from pathlib import Path
from statistics import median
import time

import torch
import torch.nn.functional as F
import triton
import triton.testing

from benchmark_flaggems_w8a16 import load_helper, dump, relative, save_kernel


CONFIGS = [(16,32,64,2,3), (16,64,64,4,2), (16,64,64,4,4),
           (16,64,128,4,2), (16,64,128,4,4), (32,64,64,4,2),
           (32,64,64,4,4), (32,64,128,4,2), (32,64,128,4,4),
           (64,64,64,4,4), (64,128,64,4,4), (128,128,32,8,2),
           (64,128,128,8,2), (32,64,256,4,2)]


def label(config):
    return '-'.join(map(str, config))


def launch(jit, config, x, q, scale, return_kernel=False):
    m, k = x.shape
    n = q.shape[0]
    bm, bn, bk, warps, stages = config
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        compiled = jit[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            x, q, scale, y, m, n, k, x.stride(0), x.stride(1),
            q.stride(0), q.stride(1), y.stride(0), y.stride(1), scale.stride(0),
            bm, bn, bk, 8, num_warps=warps, num_stages=stages, pipeline='basic')
    return (y, compiled) if return_kernel else y


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
    jit = weight.weight_int8pack_mm_kernel.jit_function
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dump(out, 'metadata.json', dict(configs=CONFIGS, pipeline='basic', seed_screen=73,
        seed_validate=314159, screen_warmup_ms=50, screen_rep_ms=150,
        final_warmup_ms=100, final_rep_ms=300, final_rounds=3,
        kernel_source=jit.fn.__code__.co_filename,
        scope='configuration-only diagnostic; same FlagGems JIT body, no production registration',
        timing='full path includes output allocation; median of three GPU latency medians'))
    torch.manual_seed(73)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = ref.quantize_weight(w)
    screening, chosen = [], {}
    for m in (1, 16, 32, 64, 128, 256, 512, 1024, 2048):
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        passed = []
        for config in CONFIGS:
            row = dict(m=m, config=config, id=label(config))
            try:
                start = time.perf_counter()
                actual = launch(jit, config, x, q, scale)
                torch.cuda.synchronize()
                row['compile_and_first_call_s'] = time.perf_counter() - start
                error = relative(actual, expected)
                assert error < .005, (config, m, error)
                torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
                row['relative_l2_error'] = error
                row['ms'] = triton.testing.do_bench(lambda: launch(jit, config, x, q, scale),
                    warmup=50, rep=150, return_mode='median')
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
        assert passed, ('no valid config', m)
        best = min(passed, key=lambda r: r['ms'])
        chosen[str(m)] = best['config']
        dump(out, 'chosen.json', chosen)
        print('SCREEN ' + json.dumps(best), flush=True)
    # Independent input seed, fresh repeated comparison with all original paths.
    torch.manual_seed(314159)
    w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = ref.quantize_weight(w)
    torch.testing.assert_close(deq.dequantize_weight(q, scale), ref.dequantize_reference(q, scale), rtol=0, atol=0)
    raw, summary = [], []
    for m in (1, 16, 32, 64, 128, 256, 512, 1024, 2048):
        config = chosen[str(m)]
        # Masked correctness for the selected config, not just divisible real shapes.
        xo = torch.randn(3, 257, device='cuda', dtype=torch.bfloat16)
        wo = torch.randn(129, 257, device='cuda', dtype=torch.bfloat16) * .02
        wo[0] = 0
        qo, so = ref.quantize_weight(wo)
        eo = (F.linear(xo.float(), qo.float()) * so).to(torch.bfloat16)
        ao = launch(jit, config, xo, qo, so)
        assert relative(ao, eo) < .005
        torch.testing.assert_close(ao, eo, rtol=.02, atol=.125)
        torch.testing.assert_close(ao[:, 0], torch.zeros_like(ao[:, 0]), rtol=0, atol=0)
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        actual, compiled = launch(jit, config, x, q, scale, return_kernel=True)
        err = relative(actual, expected)
        assert err < .005
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)
        original = weight.weight_int8pack_mm(x, q, scale)
        assert relative(original, expected) < .005
        save_kernel(out, 'candidate-m' + str(m), compiled,
            dict(zip(('BLOCK_M','BLOCK_N','BLOCK_K','num_warps','num_stages'), config), pipeline='basic'))
        paths = dict(bf16_metax=lambda: dense.mm(x, w.t()),
            original_fused=lambda: weight.weight_int8pack_mm(x, q, scale),
            candidate=lambda: launch(jit, config, x, q, scale),
            separate_triton_metax=lambda: dense.mm(x, deq.dequantize_weight(q, scale).t()))
        # Compile/tune all paths before the measured comparisons.
        for fn in paths.values():
            fn()
        torch.cuda.synchronize()
        names = list(paths)
        samples = {name: [] for name in names}
        for repeat in range(3):
            for name in names[repeat:] + names[:repeat]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300, return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, round=repeat + 1, path=name, ms=ms))
                dump(out, 'raw_final.json', raw)
        timings = {name: median(values) for name, values in samples.items()}
        row = dict(m=m, config=config, relative_l2_error=err, masked_correctness_passed=True,
            timings_ms=timings, candidate_speedup_vs_bf16=timings['bf16_metax']/timings['candidate'],
            candidate_speedup_vs_original=timings['original_fused']/timings['candidate'])
        summary.append(row)
        dump(out, 'final_summary.json', summary)
        print('FINAL ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
