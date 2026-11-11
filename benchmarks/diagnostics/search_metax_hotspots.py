"""Offline full-call C500 search; never enable provisional winners at runtime."""
import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton
    import triton.testing
    import flag_gems
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['bf16', 'w8a16'], required=True)
    p.add_argument('--provider-override', action='store_true',
                   help='Install the independently validated C500 BF16 tiling provider')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.provider_override:
        from vllm_fl.dispatch.backends.vendor.metax.nt_gemm import apply_c500_nt_gemm
        if not apply_c500_nt_gemm(torch.device('cuda:0')):
            raise RuntimeError('Validated BF16 provider override failed')
    mm = importlib.import_module(flag_gems.mm.__module__)
    backend = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    (args.output/'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'triton': triton.__version__,
        'registered_mm': flag_gems.mm.__module__,
        'registered_int8': flag_gems.weight_int8pack_mm.__module__,
        'explicit_backend': backend.__file__,
        'mm_hash': hashlib.sha256(Path(mm.__file__).read_bytes()).hexdigest(),
        'backend_hash': hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
        'provider_override': args.provider_override,
        'scope': 'synthetic eager full-call screening; five interleaved finalist rounds; provisional only'}, indent=2))
    torch.manual_seed(20261110)
    results = (args.output/'candidates.jsonl').open('w')
    winners = []
    shapes = [(2048,6144),(12288,2048),(2560,2048),(2048,2048)]
    for n,k in shapes:
        m = 2048 if args.stage == 'bf16' else 56
        if args.stage == 'w8a16' and k == 6144:
            continue  # Down M56 is a retained control, not a new search target.
        a = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        out = torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
        scale = w.float().abs().amax(1).clamp_min(1e-8)/127
        q = (w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        ref = a.float()@w.float().T if args.stage=='bf16' else (a.float()@q.float().T)*scale
        jobs = [('registered_bf16', None, lambda: flag_gems.mm(a,w.t()))]
        if args.stage=='bf16':
            jit = mm.mm_kernel_nt.jit_function
            for bm,bn,bk,warps,stages,pipe in itertools.product([32,64,128,256],[32,64,128,256],[32,64,128],[4,8,16],[1,2,3,4],['basic','cpasync']):
                # Only a conservative explicit workspace/shared-memory budget;
                # do not reuse the old performance-model top-k pruning.
                if (bm+bn)*bk*2*stages > 192*1024:
                    continue
                cfg=(bm,bn,bk,warps,stages,pipe)
                def call(cfg=cfg):
                    bm,bn,bk,warps,stages,pipe=cfg
                    jit.run(a,w.t(),out,m,n,k,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=8,EVEN_M=m%bm==0,EVEN_N=n%bn==0,EVEN_K=k%bk==0,num_warps=warps,num_stages=stages,pipeline=pipe,grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),),warmup=False)
                    return out
                jobs.append(('bf16_nt',cfg,call))
        else:
            jobs += [('dispatch',None,lambda:backend._shape_dispatch(a,q,scale)),('separate',None,lambda:backend.run_w8a16_algorithm(a,q,scale,'separate'))]
            for bm,bn,bk,warps,stages in itertools.product([16,32,64],[32,64,128,256],[32,64,128,256],[2,4,8],[1,2,3,4]):
                cfg=(bm,bn,bk,warps,stages)
                for alg,splits in [('dense',[None]),('splitk',[2,4,8,16,32])]:
                    for split in splits:
                        c=cfg if split is None else cfg+(split,)
                        reason=backend.dense_rejection_reason(a,q,scale,c) if alg=='dense' else backend.splitk_rejection_reason(a,q,scale,c)
                        if reason:
                            results.write(json.dumps({'m':m,'n':n,'k':k,'algorithm':alg,'config':c,'pruned':reason})+'\n'); continue
                        jobs.append((alg,c,lambda alg=alg,c=c:backend.run_w8a16_algorithm(a,q,scale,alg,c)))
        valid=[]
        for name,cfg,fn in jobs:
            row={'m':m,'n':n,'k':k,'algorithm':name,'config':cfg}
            try:
                y=fn(); torch.cuda.synchronize()
                reference = a.float()@w.float().T if name=='registered_bf16' else ref
                err=float((y.float()-reference).norm()/reference.norm()); row['relative_l2']=err
                if err>=0.005: raise RuntimeError('correctness relative L2 >= .005')
                row['screen_ms']=triton.testing.do_bench(fn,warmup=10,rep=30)
                valid.append((row,fn))
            except Exception as e:
                row['failure']=repr(e)
            results.write(json.dumps(row)+'\n'); results.flush()
        finalists=sorted(valid,key=lambda x:x[0]['screen_ms'])[:8]
        # Preserve exploration of every legal algorithm even if screening
        # noise pushes it outside the combined top eight.
        for algorithm in {row['algorithm'] for row, _ in valid}:
            own = sorted((item for item in valid if item[0]['algorithm'] == algorithm),
                         key=lambda item: item[0]['screen_ms'])[:2]
            finalists += [item for item in own if item not in finalists]
        finalists += [x for x in valid if x[0]['config'] is None and x not in finalists]
        samples={id(row):[] for row,_ in finalists}
        for repeat in range(5):
            order=list(finalists); random.Random(repeat).shuffle(order)
            for row,fn in order: samples[id(row)].append(triton.testing.do_bench(fn,warmup=30,rep=100))
        for row,_ in finalists:
            row['final_ms']=samples[id(row)]; row['median_ms']=statistics.median(row['final_ms'])
        candidates=[row for row,_ in finalists]
        if not candidates: raise RuntimeError('No valid candidates')
        winners.append({'m':m,'n':n,'k':k,'finalists':sorted(candidates,key=lambda x:x['median_ms']),'status':'provisional_requires_graph_replay_and_provider_binding'})
        (args.output/'finalists.json').write_text(json.dumps(winners,indent=2))
    (args.output/'complete.txt').write_text('search complete; not production certification')


if __name__=='__main__':
    main()
