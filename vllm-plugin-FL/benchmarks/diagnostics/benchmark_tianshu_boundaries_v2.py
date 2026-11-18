"""Bounded shared-candidate boundary measurements; no runtime search or model patch."""
import hashlib
import importlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from vllm_fl.quantization.w8a16.backend import run
from vllm_fl.quantization.w8a16.reference import quantize_weight
from tianshu_muxi_splitk import run as splitk
from tianshu_small_m_kernels import gemv

SHAPES = {'down': (2048,6144), 'gate_up': (12288,2048),
          'qkv': (2560,2048), 'o': (2048,2048)}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    seeds_path = Path(__file__).with_name('tianshu_boundary_seeds.json')
    seeds = json.loads(seeds_path.read_text())
    provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    kernel = provider.mm_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(20261113)
    manifest = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        triton=triton.__version__, provider=provider.__file__,
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        seed_manifest_sha256=hashlib.sha256(seeds_path.read_bytes()).hexdigest(),
        initial_m=[64,128,256,512,1024,2048,4096], thresholds=dict(l2=.005,normalized_max=.01),
        search='prior candidates plus bounded controls; graph crossing bisection <=4 rounds',
        measured_timing_cache_reused=False, configuration_sources_reused=True, candidate_audit='include 64/2048/4096 anchor winners for both Separate layouts')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'candidate-seeds.json').write_text(seeds_path.read_text())

    def emit(row):
        with (out/'raw.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')

    def accuracy(y, ref):
        err=y.float()-ref
        l2=(err.norm()/ref.norm().clamp_min(1e-12)).item()
        mx=(err.abs().max()/ref.abs().max().clamp_min(1e-12)).item()
        assert torch.isfinite(y).all() and l2<=.005 and mx<=.01,(l2,mx)
        return dict(l2=l2,normalized_max=mx)

    def spec(algorithm, cfg, layout='nk', conversion=(2048,8), source='bounded_control'):
        return dict(algorithm=algorithm,config=list(cfg),layout=layout,
                    convert=list(conversion),source=source)

    def unique(items):
        result={}
        for x in items:
            key=json.dumps({k:x[k] for k in ('algorithm','config','layout','convert')},sort_keys=True)
            result.setdefault(key,x)
        return list(result.values())

    def candidates(projection,m):
        primary='splitk_muxi' if projection=='down' else 'fused_native'
        result=[]
        for algorithm in ('native',primary,'separate'):
            prior=[r for r in seeds if r['projection']==projection and r['algorithm']==algorithm]
            if algorithm=='separate':
                for layout in ('nk','kn'):
                    sub=sorted((r for r in prior if r['layout']==layout),
                        key=lambda r:(abs(math.log2(m/r['m'])),r['graph']['median_us']))[:3]
                    result += [spec(algorithm,r['config'],layout,r['convert'],r['candidate_source']) for r in sub]
                    for anchor in (64,2048,4096):
                        known=sorted((r for r in prior if r['layout']==layout and r['m']==anchor),
                                     key=lambda r:r['graph']['median_us'])[:1]
                        result += [spec(algorithm,r['config'],layout,r['convert'],
                                        r['candidate_source']+':anchor_'+str(anchor)) for r in known]
                # Give each layout a small chance to improve MM at a new M.
                for cfg in ((32,64,64,4,1),(64,128,64,4,2),(128,128,128,4,2)):
                    for layout,cv in [('nk',(4096,16)),('kn',(64,128,4))]:
                        result.append(spec(algorithm,cfg,layout,cv))
            elif algorithm=='splitk_muxi':
                result += [spec(algorithm,r['config'],'nk',r['convert'],r['candidate_source']) for r in prior]
                for bm,bk,sk in itertools.product((32,64),(64,128),(4,8)):
                    if SHAPES[projection][1]%(bk*sk)==0:
                        result.append(spec(algorithm,(bm,64,bk,4,1,sk),'nk',('vector',256)))
                result.append(spec(algorithm,(64,64,128,4,1,8),'nk',('serial',256),'MetaX_M64_control'))
            else:
                ordered=sorted(prior,key=lambda r:(abs(math.log2(m/r['m'])),r['graph']['median_us']))
                result += [spec(algorithm,r['config'],'nk',source=r['candidate_source']) for r in ordered[:5]]
                for cfg in ((16,64,64,4,1),(32,128,64,4,1),(64,64,128,8,1),
                            (64,128,128,4,2),(128,128,128,4,2),(256,256,32,16,2)):
                    result.append(spec(algorithm,cfg))
        # Cross-anchor MM configurations are necessary controls near tile transitions.
        for anchor in (2048,4096):
            native=sorted((r for r in seeds if r['projection']==projection and r['algorithm']=='native'
                           and r['m']==anchor),key=lambda r:r['graph']['median_us'])[:1]
            for r in native:
                result.append(spec('native',r['config'],source=r['candidate_source']+':large_anchor'))
                if projection!='down':
                    result.append(spec('fused_native',r['config'],source=r['candidate_source']+':large_anchor'))
        if projection=='down' and m==1:
            for rn,w in itertools.product((1,2,4,8,16),(4,8)):
                result.append(spec('gemv',(rn,1,w)))
            for cfg in ((16,64,64,4,1),(16,128,64,4,1),(32,64,128,4,1)):
                result.append(spec('fused_native',cfg))
        return unique(result)

    summary=[]
    points={p:{} for p in SHAPES}

    def evaluate(projection,m,phase):
        n,k=SHAPES[projection]
        meta=dict(projection=projection,m=m,n=n,k=k,phase=phase)
        def data(seed):
            torch.manual_seed(seed)
            a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            q,s=quantize_weight(w)
            return a,w,q,s,a.float()@w.float().T,(a.float()@q.float().T)*s
        a,w,q,s,native_ref,quant_ref=data(20261113)
        emit(dict(meta,stage='quantization_error',weight_l2=((q.float()*s[:,None]-w.float()).norm()/w.float().norm()).item(),
                  output_l2=((quant_ref-native_ref).norm()/native_ref.norm()).item()))

        def function(item):
            algorithm=item['algorithm']; cfg=tuple(item['config'])
            if algorithm=='splitk_muxi':
                return lambda:splitk(a,q,s,cfg,tuple(item['convert']))
            if algorithm=='gemv':
                return lambda:gemv(a,q,s,cfg)
            if algorithm!='native':
                return lambda:run(a,q,s,algorithm,cfg,tuple(item['convert']),item['layout'])
            def native():
                bm,bn,bk,warps,stages=cfg
                y=torch.empty(m,n,device=a.device,dtype=a.dtype)
                kernel[(triton.cdiv(m,bm)*triton.cdiv(n,bn),1)](
                    a,w.T,y,m,n,k,k,1,1,k,n,1,acc_dtype=tl.float32,input_precision=None,
                    fp8_fast_accum=True,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=8,
                    SPLIT_K=1,EVEN_K=k%bk==0,AB_DTYPE=tl.bfloat16,UPGRADE=False,
                    UPGRADE_A_OFFS=m*k>2**31-1,UPGRADE_B_OFFS=n*k>2**31-1,
                    UPGRADE_C_OFFS=m*n>2**31-1,num_warps=warps,num_stages=stages)
                return y
            return native

        def measure(item,stage,graph=False,**extra):
            row=dict(meta,**item,stage=stage,mode='graph' if graph else 'eager',**extra)
            try:
                fn=function(item); ref=native_ref if item['algorithm']=='native' else quant_ref
                row.update(accuracy(fn(),ref))
                if graph:
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        y=fn()
                    g.replay(); torch.cuda.synchronize(); accuracy(y,ref); fn=g.replay
                timing=triton.testing.do_bench(fn,warmup=5,rep=20 if stage=='confirm' else 8,quantiles=[.5])
                row.update(status='ok',us=float(timing[0] if isinstance(timing,(tuple,list)) else timing)*1000)
            except Exception as exc:
                row.update(status='failed',error=type(exc).__name__+': '+str(exc)[:1200])
                emit(row)
                if any(t in str(exc).lower() for t in ('illegal memory','device-side assert')):
                    raise
                return row
            emit(row); return row

        pool=candidates(projection,m); rng.shuffle(pool)
        emit(dict(meta,stage='pool',candidates=pool))
        rows=[measure(x,'screen') for x in pool]
        leaders=[]
        algorithms=sorted(set(x['algorithm'] for x in pool))
        for algorithm in algorithms:
            good=sorted((r for r in rows if r['algorithm']==algorithm and r['status']=='ok'),key=lambda r:r['us'])
            assert good,(meta,algorithm)
            leaders += [{key:r[key] for key in ('algorithm','config','layout','convert','source')} for r in good[:2]]
        finals=[]
        for seed in (20261114,20261115):
            a,w,q,s,native_ref,quant_ref=data(seed)
            for repeat in range(5):
                seq=leaders.copy(); rng.shuffle(seq)
                for item in seq:
                    for graph in (False,True):
                        finals.append(measure(item,'confirm',graph,seed=seed,repeat=repeat))
        results=[]
        for item in leaders:
            rows=[r for r in finals if all(r[key]==item[key] for key in ('algorithm','config','layout','convert'))]
            assert len(rows)==20 and all(r['status']=='ok' for r in rows),(meta,item)
            entry=dict(meta,**item)
            for mode in ('eager','graph'):
                vals=[r['us'] for r in rows if r['mode']==mode]
                entry[mode]=dict(median_us=statistics.median(vals),min_us=min(vals),max_us=max(vals))
            summary.append(entry); results.append(entry)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        if m>=64:
            primary='splitk_muxi' if projection=='down' else 'fused_native'
            win={a:min((r for r in results if r['algorithm']==a),key=lambda r:r['graph']['median_us'])
                 for a in (primary,'separate')}
            points[projection][m]=dict(primary=win[primary],separate=win['separate'],
                delta_us=win[primary]['graph']['median_us']-win['separate']['graph']['median_us'])
        print('POINT_DONE',projection,m,phase,flush=True)

    for m in (64,128,256,512,1024,2048,4096):
        for projection in SHAPES:
            evaluate(projection,m,'coarse')
    evaluate('down',1,'down_m1')
    # Refine every observed sign change rather than imposing a single threshold.
    for depth in range(4):
        tasks=[]
        for projection,values in points.items():
            ms=sorted(values)
            for left,right in zip(ms,ms[1:]):
                if right-left<=32:
                    continue
                if values[left]['delta_us']*values[right]['delta_us']<=0:
                    mid=((left+right)//16)*8
                    if mid not in values and left<mid<right:
                        tasks.append((projection,mid))
        if not tasks:
            break
        for projection,m in tasks:
            evaluate(projection,m,'refine_'+str(depth))
    (out/'boundary-points.json').write_text(json.dumps(points,indent=2))
    (out/'COMPLETE').write_text('done\n')


if __name__=='__main__':
    main()
