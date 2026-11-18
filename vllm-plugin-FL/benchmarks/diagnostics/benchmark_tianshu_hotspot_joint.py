"""Two large-M hotspots: complete Separate vs original BF16 and matched MM control.

Staged, bounded joint search; no timing-cache reuse, no deployment modifications.
All conversion, allocation and scale costs are inside each quantized timed call.
"""
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
from tianshu_hotspot_kernels import hotspot_mm
from vllm_fl.quantization.w8a16.backend import convert, convert_kn, run
from vllm_fl.quantization.w8a16.reference import quantize_weight

SHAPES = {'down': (2048,6144), 'gate_up': (12288,2048),
          'qkv': (2560,2048), 'o': (2048,2048)}
KEYS = ('algorithm','config','layout','convert','group','fast')


def main():
    out=Path(sys.argv[1]); out.mkdir(exist_ok=False)
    seeds=json.loads(Path(sys.argv[2]).read_text())
    provider=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    original=provider.mm_kernel
    while not isinstance(original,triton.runtime.JITFunction): original=original.fn
    torch.backends.cuda.matmul.allow_tf32=False
    rng=random.Random(20261119)
    manifest=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
        provider=provider.__file__,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).with_name('tianshu_hotspot_kernels.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds_sha256=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
        thresholds=dict(l2=.005,normalized_max=.01),ms=[2048,4096],
        scope='complete Separate: allocate + INT8 to BF16 conversion + MM with scale epilogue',
        search='tile/layout anchors and local neighbors; group cross top 4 per layout; mask and conversion cross finalists',
        timing_cache_reused=False,persistent_int8_layout='NK',persistent_bf16_copy=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def emit(row):
        with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def accuracy(y,ref):
        d=y.float()-ref
        l2=(d.norm()/ref.norm().clamp_min(1e-12)).item()
        mx=(d.abs().max()/ref.abs().max().clamp_min(1e-12)).item()
        assert torch.isfinite(y).all() and l2<=.005 and mx<=.01,(l2,mx)
        return dict(l2=l2,normalized_max=mx)
    def spec(alg,cfg,layout='nk',cv=(4096,8),group=0,fast=False):
        return dict(algorithm=alg,config=list(cfg),layout=layout,convert=list(cv),group=group,fast=fast)
    def key(s):return json.dumps({k:s[k] for k in KEYS},sort_keys=True)
    def stripped(r):return {k:r[k] for k in KEYS}
    def unique(items):return list({key(s):s for s in items}.values())
    # Validate both omitted masks and fallback masks, including partial CTA groups.
    for vm,vn,vk in ((128,128,128),(129,128,128),(128,129,128),(129,129,130)):
      va=torch.randn(vm,vk,device='cuda',dtype=torch.bfloat16)
      vb=torch.randn(vk,vn,device='cuda',dtype=torch.bfloat16)
      vs=torch.rand(vn,device='cuda',dtype=torch.float32)
      for vg,vfast in itertools.product((0,2,8),(False,True)):
        vy=torch.empty((vm,vn),device='cuda',dtype=torch.bfloat16)
        hotspot_mm[(triton.cdiv(vm,64)*triton.cdiv(vn,64),1)](
            va,vb,vy,vs,vm,vn,vk,*va.stride(),*vb.stride(),*vy.stride(),
            acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
            BLOCK_M=64,BLOCK_N=64,BLOCK_K=32,GROUP_M=vg,SPLIT_K=1,EVEN_K=vk%32==0,
            FAST_STORE=vfast,EVEN_M=vm%64==0,EVEN_N=vn%64==0,APPLY_SCALE=True,
            AB_DTYPE=tl.bfloat16,UPGRADE=False,UPGRADE_A_OFFS=False,UPGRADE_B_OFFS=False,
            UPGRADE_C_OFFS=False,num_warps=4,num_stages=1)
        emit(dict(stage='boundary_correctness',m=vm,n=vn,k=vk,group=vg,fast=vfast,
                  **accuracy(vy,(va.float()@vb.float())*vs)))
    all_summary=[]
    for projection,(n,k) in SHAPES.items():
      for m in (2048,4096):
        meta=dict(projection=projection,m=m,n=n,k=k)
        def data(seed):
            torch.manual_seed(seed)
            a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            q,s=quantize_weight(w)
            nr=a.float()@w.float().T
            qr=(a.float()@q.float().T)*s
            return a,w,q,s,nr,qr
        a,w,q,s,native_ref,quant_ref=data(20261119)
        emit(dict(meta,stage='quantization_error',output_l2=((quant_ref-native_ref).norm()/native_ref.norm()).item()))
        def function(item):
            alg=item['algorithm']; cfg=item['config']; layout=item['layout']; cv=item['convert']
            if alg=='frozen':
                return lambda:run(a,q,s,'separate',tuple(cfg),tuple(cv),layout)
            def fn():
                if alg=='separate':
                    if layout=='nk':
                        b=torch.empty(q.shape,device=q.device,dtype=torch.bfloat16)
                        block,warps=cv
                        convert[(triton.cdiv(q.numel(),block),)](q,b,q.numel(),block,num_warps=warps)
                        b=b.T
                    else:
                        b=torch.empty((k,n),device=q.device,dtype=torch.bfloat16)
                        tn,tk,warps=cv
                        convert_kn[(triton.cdiv(n,tn),triton.cdiv(k,tk))](q,b,n,k,tn,tk,num_warps=warps)
                else:b=w.T
                y=torch.empty((m,n),device=a.device,dtype=a.dtype)
                bm,bn,bk,warps,stages=cfg
                args=(a,b,y) if alg=='native' else (a,b,y,s)
                extra={} if alg=='native' else dict(FAST_STORE=item['fast'],EVEN_M=m%bm==0,EVEN_N=n%bn==0,APPLY_SCALE=alg=='separate')
                kernel=original if alg=='native' else hotspot_mm
                kernel[(triton.cdiv(m,bm)*triton.cdiv(n,bn),1)](*args,m,n,k,*a.stride(),*b.stride(),*y.stride(),
                    acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,
                    GROUP_M=item['group'],SPLIT_K=1,EVEN_K=k%bk==0,AB_DTYPE=tl.bfloat16,UPGRADE=False,
                    UPGRADE_A_OFFS=False,UPGRADE_B_OFFS=False,UPGRADE_C_OFFS=False,
                    num_warps=warps,num_stages=stages,**extra)
                return y
            return fn
        def measure(item,stage,graph=False,**extra):
            row=dict(meta,**item,stage=stage,mode='graph' if graph else 'eager',**extra)
            try:
                fn=function(item);ref=native_ref if item['algorithm'] in ('native','matched_native') else quant_ref
                row.update(accuracy(fn(),ref))
                if graph:
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):y=fn()
                    g.replay();torch.cuda.synchronize();accuracy(y,ref);fn=g.replay
                t=triton.testing.do_bench(fn,warmup=5,rep=30 if stage=='confirm' else 8,quantiles=[.5])
                row.update(status='ok',us=float(t[0] if isinstance(t,(tuple,list)) else t)*1000)
            except Exception as exc:
                row.update(status='failed',error=type(exc).__name__+': '+str(exc)[:1000])
                emit(row)
                if isinstance(exc,AssertionError) or any(x in str(exc).lower() for x in ('illegal memory','device-side assert')):raise
                return row
            emit(row);return row
        rows=[];seen=set()
        def screen(pool,stage):
            pool=[x for x in unique(pool) if key(x) not in seen]
            emit(dict(meta,stage='pool',name=stage,count=len(pool),candidates=pool))
            rng.shuffle(pool)
            for i,item in enumerate(pool):
                seen.add(key(item));rows.append(measure(item,stage))
                if i%25==0:print('PROGRESS',projection,m,stage,i,len(pool),flush=True)
        def best(alg,layout=None,count=4):
            good=[r for r in rows if r['status']=='ok' and r['algorithm']==alg and (layout is None or r['layout']==layout)]
            return [stripped(r) for r in sorted(good,key=lambda r:r['us'])[:count]]
        priors=[r for r in seeds if r['projection']==projection and r['m'] in (2048,4096)]
        cfgs={tuple(r['config']) for r in priors if r['algorithm'] in ('native','separate')}
        for bm,bn,bk,warps,stages in list(cfgs):
            cfgs.update((bm,bn,bk,w,st) for w in (8,16) for st in (1,2,3))
            cfgs.update((bm,bn,b,warps,stages) for b in (32,64))
        cfgs.update(itertools.product((128,256),(128,256),(32,64),(8,16),(1,2)))
        pool=[]
        cv_by_layout={}
        for layout in ('nk','kn'):
            p=sorted((r for r in priors if r['algorithm']=='separate' and r['layout']==layout),
                     key=lambda r:(abs(r['m']-m),r['graph']['median_us']))
            cv_by_layout[layout]=p[0]['convert'] if p else ([4096,8] if layout=='nk' else [64,128,4])
        for cfg in sorted(cfgs):
            pool.append(spec('native',cfg))
            for layout in ('nk','kn'):
                pool.append(spec('separate',cfg,layout,cv_by_layout[layout]))
        screen(pool,'tile_layout')
        pool=[]
        for layout in ('nk','kn'):
            for item in best('separate',layout,4):
                for group in (1,2,4,8,16):pool.append(dict(item,group=group))
        for item in best('native',count=4):
            for group in (0,1,2,4,8,16):pool.append(dict(item,algorithm='matched_native',group=group))
        screen(pool,'group_joint')
        pool=[]
        for layout in ('nk','kn'):
            cvs=([(1024,4),(2048,8),(4096,8),(4096,16),(8192,8)] if layout=='nk'
                 else [(32,64,4),(64,64,4),(64,128,4),(128,64,8),(128,128,8)])
            for item in best('separate',layout,3):
                for fast,cv in itertools.product((False,True),cvs):pool.append(dict(item,fast=fast,convert=list(cv)))
        for item in best('matched_native',count=4):pool.append(dict(item,fast=True))
        screen(pool,'store_conversion')
        # Graph validation selection is separate from the final paired confirmation.
        shortlist=unique(best('separate','nk',4)+best('separate','kn',4)+best('native',count=3)+best('matched_native',count=3))
        graph_rows=[measure(x,'graph_select',True) for x in shortlist]
        selected=[]
        for alg in ('separate','native','matched_native'):
            good=sorted((r for r in graph_rows if r['status']=='ok' and r['algorithm']==alg),key=lambda r:r['us'])
            assert good,alg
            selected.append(stripped(good[0]))
        frozen=[r for r in seeds if r['projection']==projection and r['m']==m and r['algorithm']=='separate']
        old=min(frozen,key=lambda r:r['graph']['median_us'])
        selected.append(spec('frozen',old['config'],old['layout'],old['convert']))
        emit(dict(meta,stage='selected',candidates=selected))
        print('SELECTED',projection,m,json.dumps(selected),flush=True)
        confirms=[]
        for seed in (20261120,20261121):
            a,w,q,s,native_ref,quant_ref=data(seed)
            for repeat in range(5):
                order=list(itertools.product(selected,(False,True)));rng.shuffle(order)
                for item,graph in order:
                    confirms.append(measure(item,'confirm',graph,seed=seed,repeat=repeat))
        for item in selected:
            rec=dict(meta,**item)
            for mode in ('eager','graph'):
                vals=[r['us'] for r in confirms if r['status']=='ok' and key(r)==key(item) and r['mode']==mode]
                assert len(vals)==10
                rec[mode]=dict(median_us=statistics.median(vals),min_us=min(vals),max_us=max(vals))
            all_summary.append(rec)
        (out/'summary.json').write_text(json.dumps(all_summary,indent=2))
        print('DONE',projection,m,flush=True)
    (out/'COMPLETE').write_text('8 hotspots completed; no model-level evaluation\n')


if __name__=='__main__':main()
