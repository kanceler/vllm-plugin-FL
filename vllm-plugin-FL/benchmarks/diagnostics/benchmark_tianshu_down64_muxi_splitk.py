"""Offline complete-call search. Run only after the GPU environment audit."""
import hashlib
import importlib
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from vllm_fl.quantization.w8a16.backend import run
from vllm_fl.quantization.w8a16.reference import quantize_weight
from tianshu_muxi_splitk import run as muxi_splitk
import w8a16_splitk_search as planner


def main():
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False, parents=True)
    provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    kernel = provider.mm_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(20261113)
    configs = list(itertools.product((32, 64, 128, 256), (32, 64, 128, 256),
                                    (32, 64, 128), (4, 8, 16), (1, 2, 3)))
    configs += [(bm, bn, 256, w, st) for bm, bn, w, st in itertools.product(
        (64, 128, 256), (64, 128, 256), (4, 8, 16), (1, 2))]
    configs += [(bm, bn, bk, w, st) for (bm, bn), bk, w, st in itertools.product(
        ((512, 64), (512, 128), (64, 512), (128, 512)), (32, 64), (8, 16), (1, 2))]
    assert len(configs) == 518
    configs += list(itertools.product((16,), (32,64,128,256), (32,64,128), (4,8), (1,2,3)))
    assert len(configs) == 590
    converts = {'nk': list(itertools.product((512, 1024, 2048, 4096, 8192), (4, 8, 16))),
                'kn': list(itertools.product((16, 32, 64), (32, 64, 128), (4, 8, 16)))}
    manifest = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                    triton=triton.__version__, configs=configs, conversions=converts,
                    provider=provider.__file__, provider_sha256=hashlib.sha256(
                        Path(provider.__file__).read_bytes()).hexdigest(),
                    thresholds={'relative_l2': .005, 'normalized_max': .01},
                    timing_cache_reused=False, scope='Down M64; MetaX reused Split-K full call vs native BF16; full legal finite pool')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    def emit(row):
        with (out / 'raw.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')

    def check(y, ref):
        err = y.float() - ref
        l2 = (err.norm() / ref.norm().clamp_min(1e-12)).item()
        mx = (err.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        assert torch.isfinite(y).all() and l2 <= .005 and mx <= .01, (l2, mx)
        return {'l2': l2, 'normalized_max': mx}

    summary = []
    for projection, n, k in [('down', 2048, 6144)]:
        for m in (64,):
            shape = dict(projection=projection, m=m, n=n, k=k)

            def data(seed):
                torch.manual_seed(seed)
                a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
                q, s = quantize_weight(w)
                # References are built before timing and never used as compute inputs.
                return a, w, q, s, a.float() @ w.float().T, (a.float() @ q.float().T) * s

            a, w, q, s, native_ref, quant_ref = data(20261113)

            def function(spec):
                if spec['algorithm'] == 'splitk_muxi':
                    return lambda: muxi_splitk(a,q,s,tuple(spec['config']),tuple(spec['convert']))
                if spec['algorithm'] == 'separate':
                    return lambda: run(a, q, s, 'separate', tuple(spec['config']),
                                       tuple(spec['convert']), spec['layout'])
                def native():
                    bm, bn, bk, warps, stages = spec['config']
                    y = torch.empty(m, n, device=a.device, dtype=a.dtype)
                    kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), 1)](
                        a, w.T, y, m, n, k, k, 1, 1, k, n, 1,
                        acc_dtype=tl.float32, input_precision=None, fp8_fast_accum=True,
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_K=1,
                        EVEN_K=k % bk == 0, AB_DTYPE=tl.bfloat16, UPGRADE=False,
                        UPGRADE_A_OFFS=m*k > 2**31-1, UPGRADE_B_OFFS=n*k > 2**31-1,
                        UPGRADE_C_OFFS=m*n > 2**31-1, num_warps=warps, num_stages=stages)
                    return y
                return native

            def measure(spec, stage, graph=False, **extra):
                row = dict(shape, **spec, stage=stage, mode='graph' if graph else 'eager', **extra)
                try:
                    fn = function(spec)
                    ref = native_ref if spec['algorithm'] == 'native' else quant_ref
                    row.update(check(fn(), ref))
                    if graph:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            y = fn()
                        g.replay()
                        torch.cuda.synchronize()
                        check(y, ref)
                        fn = g.replay
                    timing = triton.testing.do_bench(fn, warmup=5,
                        rep=30 if stage == 'confirm' else 10, quantiles=[.5])
                    row['us'] = (timing[0] if isinstance(timing, (list, tuple)) else timing) * 1000
                    row['status'] = 'ok'
                except Exception as exc:
                    row.update(status='failed', error=type(exc).__name__ + ': ' + str(exc)[:1200])
                    emit(row)
                    if any(t in str(exc).lower() for t in ('illegal memory', 'device-side assert')):
                        raise
                    return row
                emit(row)
                return row

            def best(rows, count):
                return sorted((r for r in rows if r['status'] == 'ok'), key=lambda r: r['us'])[:count]

            leaders = []
            for algorithm, layout in [('native', 'nk'), ('splitk_muxi', 'nk')]:
                initial_cv = (2048, 8) if layout == 'nk' else (32, 64, 8)
                seq = configs.copy()
                if algorithm == 'splitk_muxi':
                    # Reuse the MetaX finite pool and exact legality checks.
                    # Heuristics are recorded but do not eliminate legal configurations here.
                    props = torch.cuda.get_device_properties(0)
                    pool = planner.generate_candidates(m,n,k,props.multi_processor_count,32,256*1024*1024)
                    (out/'splitk-pool.json').write_text(json.dumps(pool,indent=2))
                    seq = [tuple(r['config']) for r in pool
                           if r['status'] not in ('pruned_illegal','pruned_resource_budget')]
                    initial_cv = ('serial',256)
                rng.shuffle(seq)
                rows = [measure(dict(algorithm=algorithm, layout=layout, config=cfg,
                                     convert=initial_cv), 'mm_screen') for cfg in seq]
                top = best(rows, 4)
                assert top, (shape, algorithm, layout)
                if algorithm == 'splitk_muxi':
                    pairs = list(itertools.product([r['config'] for r in top],
                        [('serial',128),('serial',256),('serial',512),
                         ('vector',128),('vector',256),('vector',512)]))
                    rng.shuffle(pairs)
                    rows = [measure(dict(algorithm=algorithm,layout=layout,config=cfg,convert=cv),
                                    'reduce_cross') for cfg,cv in pairs]
                if algorithm == 'separate':
                    pairs = list(itertools.product([r['config'] for r in top], converts[layout]))
                    rng.shuffle(pairs)
                    rows = [measure(dict(algorithm=algorithm, layout=layout, config=cfg,
                                         convert=cv), 'conversion_cross') for cfg, cv in pairs]
                leaders += [{key: r[key] for key in ('algorithm', 'layout', 'config', 'convert')}
                            for r in best(rows, 3)]
                print('SEARCH_DONE', shape, algorithm, layout, flush=True)
            finals = []
            for seed in (20261114, 20261115):
                a, w, q, s, native_ref, quant_ref = data(seed)
                for repeat in range(5):
                    seq = leaders.copy()
                    rng.shuffle(seq)
                    for spec in seq:
                        for graph in (False, True):
                            finals.append(measure(spec, 'confirm', graph, seed=seed, repeat=repeat))
            for spec in leaders:
                rows = [r for r in finals if all(r[key] == value for key, value in spec.items())]
                if len(rows) != 20 or any(r['status'] != 'ok' for r in rows):
                    continue
                result = dict(shape, **spec)
                for mode in ('eager', 'graph'):
                    values = [r['us'] for r in rows if r['mode'] == mode]
                    result[mode] = dict(median_us=statistics.median(values), min_us=min(values), max_us=max(values))
                summary.append(result)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))
            print('SHAPE_DONE', shape, flush=True)
    (out / 'COMPLETE').write_text('done\n')


if __name__ == '__main__':
    main()
