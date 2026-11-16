"""Bounded partial search, ranked by complete calls; graph checks stay separate."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton
    import triton.testing
    from metax_down_partial import down_partial_acc
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ms', default='16,32,56,64,96,112')
    p.add_argument('--wide', action='store_true', help='Wider bounded tile audit; basic pipeline only')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    def emit(file, row):
        with (args.output / file).open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    (args.output/'environment.json').write_text(json.dumps({
        'torch':torch.__version__, 'triton':triton.__version__,
        'backend':backend.__file__, 'backend_sha256':hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
        'threshold_l2':0.005, 'selection':'fresh offline full-call timings; no measurement cache',
        'seeds':[20261111,20261112]}, indent=2))
    n, k = 2048, 6144
    for seed in [20261111, 20261112]:
        torch.manual_seed(seed)
        w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        scale = w.float().abs().amax(1).clamp_min(1e-8)/127
        q = (w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        for m in map(int,args.ms.split(',')):
            a = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            ref = (a.float()@q.float().T)*scale
            # Intercept only this offline process to recover actual dispatch config.
            original = backend._configured_splitk
            seen = []
            def record(*xs, **kw):
                seen.append(tuple(xs[3])); return original(*xs, **kw)
            backend._configured_splitk = record
            try:
                control = backend._shape_dispatch(a,q,scale)
            finally:
                backend._configured_splitk = original
            if not seen:
                emit('skipped.jsonl', {'m':m,'seed':seed,'reason':'actual dispatch is not Split-K'})
                continue
            current = seen[0]
            def baseline(): return backend._shape_dispatch(a,q,scale)
            configs = list(dict.fromkeys([current,
                (current[0],current[1],current[2],current[3],1,current[5]),
                (current[0],64,128,4,2,8),
                (32 if m<=32 else 64,64,64,4,2,16)]))
            if args.wide:
                # Audit each axis at representative M, without claiming an exhaustive search.
                configs = list(dict.fromkeys(configs + [
                    (bm,bn,bk,warps,stages,sk)
                    for bm,bn,bk,warps,stages,sk in [
                        (16,64,128,4,1,8), (32,64,128,4,1,8),
                        (64,64,128,4,1,4), (64,64,128,4,1,16),
                        (64,64,128,4,1,32), (64,128,128,4,1,8),
                        (32,128,128,4,1,8), (32,128,64,4,1,16),
                        (32,64,256,4,1,8), (64,64,256,4,1,8),
                        (32,64,128,2,1,8), (64,64,128,8,1,8),
                        (32,128,128,8,1,8), (32,128,128,4,2,8),
                        (64,128,64,4,1,16), (16,128,128,4,1,8),
                    ] if bm <= max(16, 2*m)]))
            jobs = [('dispatch',current,None)]
            for cfg in configs:
                bm,bn,bk,warps,stages,sk = cfg
                if k%(bk*sk) or n%bn:
                    emit('pruned.jsonl', {'m':m, 'seed':seed, 'config':cfg,
                         'reason':'K split or N tile does not divide'})
                    continue
                variants = ([(False,False,'basic'),(True,True,'basic')] if args.wide else
                            [(False,True,'basic'),(True,True,'basic'),
                             (False,True,'cpasync'),(True,True,'cpasync'),
                             (False,False,'basic')])
                for nt,acc,pipe in variants:
                    jobs.append((f'nt{int(nt)}_acc{int(acc)}_{pipe}',cfg,(nt,acc,pipe)))
            valid=[]
            for name,cfg,variant in jobs:
                row={'m':m,'seed':seed,'name':name,'config':cfg,'variant':variant,'current':current}
                bm,bn,bk,warps,stages,sk=cfg
                partial=torch.empty((sk,m,n),device='cuda',dtype=torch.float32)
                out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                def partial_fn(cfg=cfg,variant=variant,partial=partial):
                    bm,bn,bk,warps,stages,sk=cfg
                    nt,acc,pipe=variant
                    down_partial_acc.run(a,q,partial,m,n,k,SPLIT_K=sk,BLOCK_M=bm,
                        BLOCK_N=bn,BLOCK_K=bk,NT_LOAD=nt,DOT_ACC=acc,
                        num_warps=warps,num_stages=stages,pipeline=pipe,
                        grid=(triton.cdiv(m,bm)*(n//bn),sk),warmup=False)
                def full(cfg=cfg,variant=variant):
                    bm,bn,bk,warps,stages,sk=cfg
                    pp=torch.empty((sk,m,n),device='cuda',dtype=torch.float32)
                    yy=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                    nt,acc,pipe=variant
                    down_partial_acc.run(a,q,pp,m,n,k,SPLIT_K=sk,BLOCK_M=bm,
                        BLOCK_N=bn,BLOCK_K=bk,NT_LOAD=nt,DOT_ACC=acc,
                        num_warps=warps,num_stages=stages,pipeline=pipe,
                        grid=(triton.cdiv(m,bm)*(n//bn),sk),warmup=False)
                    backend.weight_int8pack_mm_splitk_reduce_scale.jit_function.run(
                        pp,scale,yy,m,n,SPLIT_K=sk,BLOCK_SIZE=256,num_warps=4,
                        grid=(triton.cdiv(m*n,256),),warmup=False)
                    return yy
                fn=baseline if variant is None else full
                try:
                    y=fn();torch.cuda.synchronize()
                    row['l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                    assert row['l2']<0.005
                    row['full_ms']=triton.testing.do_bench(fn,warmup=15,rep=50)
                    if variant is not None:
                        row['partial_ms']=triton.testing.do_bench(partial_fn,warmup=15,rep=50)
                    valid.append((row,fn))
                except Exception as exc:
                    row['failure']=repr(exc)
                emit('screen.jsonl',row)
            # Full-call ordering, with dispatch always retained as control.
            finalists=sorted(valid,key=lambda x:x[0]['full_ms'])[:3]
            finalists += [x for x in valid if x[0]['name']=='dispatch' and x not in finalists]
            values={id(row):[] for row,fn in finalists}
            for repeat in range(5):
                order=list(finalists);random.Random(seed+repeat).shuffle(order)
                for row,fn in order:
                    values[id(row)].append(triton.testing.do_bench(fn,warmup=20,rep=80))
            for row,fn in finalists:
                original_row = row
                row=dict(row);row['rounds_ms']=values[id(original_row)]
                row['median_ms']=statistics.median(row['rounds_ms'])
                try:
                    stream=torch.cuda.Stream();g=torch.cuda.CUDAGraph();torch.cuda.synchronize()
                    with torch.cuda.graph(g,stream=stream): y=fn()
                    g.replay();torch.cuda.synchronize()
                    row['graph_l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                    assert row['graph_l2']<0.005
                    row['graph_rounds_ms']=[triton.testing.do_bench(g.replay,warmup=20,rep=80) for _ in range(3)]
                    row['graph_ms']=statistics.median(row['graph_rounds_ms'])
                except Exception as exc: row['graph_failure']=repr(exc)
                emit('finalists.jsonl',row)
    (args.output/'complete.txt').write_text('screen complete; no production integration; inspect per-seed full-call and graph results')


if __name__=='__main__': main()
