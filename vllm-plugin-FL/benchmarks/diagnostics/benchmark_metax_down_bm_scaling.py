"""Offline BM256 scaling validation for the fixed converted and fused path; no runtime policy changes."""
import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import random
import statistics

WIDE = (64, 128, 128, 4, 4, 'cpasync')


def main():
    import torch
    import triton
    import triton.testing
    import flag_gems
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--m', type=int, choices=[2048,4096], required=True)
    args = p.parse_args()
    m = args.m
    args.output.mkdir(parents=True, exist_ok=False)
    mm = importlib.import_module(flag_gems.mm.__module__)
    b = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    (args.output/'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'triton': triton.__version__,
        'device': torch.cuda.get_device_name(), 'mm': mm.__file__, 'backend': b.__file__,
        'mm_sha256': hashlib.sha256(Path(mm.__file__).read_bytes()).hexdigest(),
        'backend_sha256': hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
        'convert': [4096, 4], 'shape': [m, 2048, 6144],
        'threshold': .005, 'selection': 'fresh full-call timing; compilation cache allowed',
        'scope': 'fixed conversion and fused scale; only MM configuration varies; compare existing wide control'}, indent=2))

    def emit(file, row):
        with (args.output/file).open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    def inputs(seed):
        torch.manual_seed(seed)
        a = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(2048, 6144, device='cuda', dtype=torch.bfloat16)
        scales = w.float().abs().amax(1).clamp_min(1e-8)/127
        q = (w.float()/scales[:, None]).round().clamp(-127,127).to(torch.int8)
        refs = {'quant': (a.float()@q.float().T)*scales}
        return a, w, q, scales, refs

    def job(a, w, q, scales, cfg, kind):
        bm, bn, bk, warps, stages, pipe = cfg
        options = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                       EVEN_M=m%bm==0, EVEN_N=2048%bn==0, EVEN_K=6144%bk==0,
                       num_warps=warps, num_stages=stages, pipeline=pipe,
                       grid=(triton.cdiv(m,bm)*triton.cdiv(2048,bn),), warmup=False)
        def call():
            out = torch.empty((m,2048), device=a.device, dtype=a.dtype)
            weight = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
            b.weight_int8pack_convert.jit_function.run(
                q, weight, q.numel(), BLOCK=4096, num_warps=4,
                grid=(triton.cdiv(q.numel(),4096),), warmup=False)
            mm.mm_kernel_nt_scale_direct.run(a, weight, out, scales, m,2048,6144, **options)
            return out
        return call

    def check(fn, ref):
        y = fn()
        torch.cuda.synchronize()
        err = float((y.float()-ref).norm()/ref.norm())
        assert err < .005, err
        return err

    def fatal(exc):
        if 'illegal' in str(exc).lower() or 'device-side assert' in str(exc).lower():
            raise exc  # Do not continue with a poisoned GPU context.

    a,w,q,s,refs = inputs(20261129)
    emit('controls.jsonl', {'config':WIDE,'convert':[4096,4],
                           'epilogue':'mm_kernel_nt_scale_direct','selection':'explicit; no tuner cache'})
    # Keep the exact BM-only scaling pair, with nearby BN/BK resource alternatives.
    configs = [(256,bn,bk,warps,stages,pipe) for bn,bk,warps,stages,pipe in
               itertools.product([64,128],[32,64,128],[4,8],[2,4],['basic','cpasync'])]
    controls = [WIDE,(128,128,128,4,4,'cpasync'),(256,128,128,4,4,'cpasync')]
    configs = list(dict.fromkeys(configs+controls))
    random.Random(20261129).shuffle(configs)
    valid = []
    for cfg in configs:
        row = {'m':m,'config':cfg, 'kind':'quant', 'seed':20261129}
        try:
            fn = job(a,w,q,s,cfg,'quant')
            row['l2'] = check(fn,refs['quant'])
            row['screen_ms'] = triton.testing.do_bench(fn,warmup=15,rep=60)
            valid.append(row)
        except Exception as exc:
            row['failure'] = repr(exc)
            emit('screen.jsonl',row)
            fatal(exc)
            continue
        emit('screen.jsonl',row)
    assert valid, 'No valid candidates'
    selected = [tuple(x['config']) for x in sorted(valid,key=lambda x:x['screen_ms'])[:6]]
    selected = list(dict.fromkeys(selected+[c for c in controls if any(tuple(x["config"])==c for x in valid)]))
    (args.output/'selected.json').write_text(json.dumps(selected,indent=2))

    # New seeds, all selected candidates interleaved with old full-call controls.
    for seed in [20261130,20261201]:
        a,w,q,s,refs = inputs(seed)
        jobs = []
        for kind, cfgs in [('quant',selected)]:
            for cfg in cfgs:
                row = {'m':m,'seed':seed,'kind':kind,'config':cfg,'source':'explicit_full_config'}
                fn = job(a,w,q,s,cfg,kind)
                try:
                    row['l2'] = check(fn,refs[kind])
                    jobs.append((row,fn))
                except Exception as exc:
                    row['failure'] = repr(exc);emit('validation-failures.jsonl',row);fatal(exc)
        times = {id(row):[] for row,fn in jobs}
        for repeat in range(7):
            order = list(jobs);random.Random(seed+repeat).shuffle(order)
            for row,fn in order:
                times[id(row)].append(triton.testing.do_bench(fn,warmup=20,rep=100))
        graphs = []
        for row,fn in jobs:
            row['rounds_ms'] = times[id(row)]
            row['median_ms'] = statistics.median(row['rounds_ms'])
            try:
                stream = torch.cuda.Stream();g = torch.cuda.CUDAGraph();torch.cuda.synchronize()
                with torch.cuda.graph(g,stream=stream):
                    y = fn()
                g.replay();torch.cuda.synchronize()
                row['graph_l2'] = float((y.float()-refs[row['kind']]).norm()/refs[row['kind']].norm())
                assert row['graph_l2'] < .005
                graphs.append((row,g,y))
            except Exception as exc:
                row['graph_failure'] = repr(exc);fatal(exc)
        gt = {id(row):[] for row,g,y in graphs}
        for repeat in range(5):
            order = list(graphs);random.Random(seed+100+repeat).shuffle(order)
            for row,g,y in order:
                gt[id(row)].append(triton.testing.do_bench(g.replay,warmup=20,rep=100))
        for row,fn in jobs:
            if id(row) in gt:
                row['graph_rounds_ms'] = gt[id(row)]
                row['graph_ms'] = statistics.median(gt[id(row)])
            emit('finalists.jsonl',row)
    (args.output/'complete.txt').write_text('micro complete; no runtime integration or model claims')


if __name__ == '__main__':
    main()
