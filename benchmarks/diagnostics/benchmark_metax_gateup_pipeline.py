"""GateUp weight pipeline experiments; no production dispatch changes."""
import argparse
import importlib
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
    p.add_argument('--weight-only-lookahead', action='store_true')
    p.add_argument('--refine', choices=['gateup_depth1_raw','gateup_weightonly1_raw','gateup_weightonly1_converted','gateup_weightonly2_raw','gateup_weightonly2_converted'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    import metax_gateup_pipeline as experiments
    from metax_gateup_prefetch import gateup_prefetch
    variants = {name:getattr(experiments,name) for name in ['gateup_depth1_raw','gateup_depth2_converted','gateup_depth2_raw']}
    if args.weight_only_lookahead:
        variants = {name:getattr(experiments,name) for name in ['gateup_weightonly1_raw','gateup_weightonly1_converted','gateup_weightonly2_raw','gateup_weightonly2_converted']}
    if args.refine:
        variants = {args.refine:getattr(experiments,args.refine)}
    variants['old_prefetch'] = gateup_prefetch
    configs = [(64,64,64,8,4), (64,64,128,4,2), (64,128,64,4,4),
               (32,64,128,4,2), (64,32,128,8,2), (64,128,128,4,4)]
    if args.refine:
        configs = [(64,bn,bk,warps,stages) for bn in [64,128] for bk in [32,64] for warps in [4,8] for stages in [1,2,4]]
    metadata = {'torch':torch.__version__, 'triton':triton.__version__,
                'backend':backend.__file__, 'bf16_provider':flag_gems.mm.__module__,
                'device':torch.cuda.get_device_name(0),
                'note':'All runtime INT8 to BF16 conversion is timed; scale stays output-side',
                'variants':list(variants), 'refine':args.refine}
    (args.output/'environment.json').write_text(json.dumps(metadata,indent=2))
    log = (args.output/'results.jsonl').open('w')
    for seed in [20261110,20261111]:
        torch.manual_seed(seed)
        m,n,k = 56,12288,2048
        a = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        scales = w.float().abs().amax(1).clamp_min(1e-8)/127
        q = (w.float()/scales[:,None]).round().clamp(-127,127).to(torch.int8)
        reference = (a.float()@q.float().T)*scales
        original_reference = a.float()@w.float().T
        weights = {name:q for name in variants}
        jobs = [('original_bf16',None,lambda:flag_gems.mm(a,w.t())),
                ('current_dense',(64,64,64,8,4),lambda:backend._configured_dense(a,q,scales,(64,64,64,8,4)))]
        artifacts = set()
        def invoke(weight,cfg,name):
            bm,bn,bk,warps,stages = cfg[:5]
            pipeline,scenario,loads = cfg[5:] if len(cfg)>5 else ('basic','',-1)
            out = torch.empty((m,n),device=a.device,dtype=a.dtype)
            compiled = variants[name].run(a,weight,scales,out,m,n,k,*a.stride(),*weight.stride(),*out.stride(),
                BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,EVEN_M=m%bm==0,EVEN_N=n%bn==0,EVEN_K=k%bk==0,
                num_warps=warps,num_stages=stages,pipeline=pipeline,scenario=scenario,pipeline_load_num=loads,
                grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),),warmup=False)
            key = name+'_'+'_'.join(map(str,cfg))
            if key not in artifacts:
                artifacts.add(key)
                info = {attr:getattr(compiled,attr,None) for attr in ['n_regs','n_spills']}
                info['metadata'] = str(getattr(compiled,'metadata',None))
                (args.output/(key+'.json')).write_text(json.dumps(info,indent=2))
                for kind,source in getattr(compiled,'asm',{}).items():
                    if isinstance(source,str): (args.output/(key+'.'+kind)).write_text(source)
            return out
        for name,weight in weights.items():
            for cfg in ([(64,64,64,8,4)] if name=='old_prefetch' else configs):
                jobs.append((name,cfg,lambda weight=weight,cfg=cfg,name=name:invoke(weight,cfg,name)))
        valid=[]
        for name,cfg,fn in jobs:
            row={'seed':seed,'m':m,'n':n,'k':k,'path':name,'config':cfg}
            try:
                y=fn();torch.cuda.synchronize()
                ref=original_reference if name=='original_bf16' else reference
                row['relative_l2']=float((y.float()-ref).norm()/ref.norm())
                assert row['relative_l2']<0.005
                row['screen_ms']=triton.testing.do_bench(fn,warmup=20,rep=80)
                valid.append((row,fn))
            except Exception as exc: row['failure']=repr(exc)
            log.write(json.dumps(row)+'\n');log.flush()
            print(row,flush=True)
        samples={id(row):[] for row,fn in valid}
        for repeat in range(5):
            order=list(valid);random.Random(seed+repeat).shuffle(order)
            for row,fn in order: samples[id(row)].append(triton.testing.do_bench(fn,warmup=20,rep=100))
        for row,fn in valid:
            row['rounds_ms']=samples[id(row)];row['median_ms']=statistics.median(row['rounds_ms'])
            try:
                stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph,stream=stream): y=fn()
                graph.replay();torch.cuda.synchronize()
                ref=original_reference if row['path']=='original_bf16' else reference
                row['graph_relative_l2']=float((y.float()-ref).norm()/ref.norm())
                assert row['graph_relative_l2']<0.005
                row['graph_ms']=triton.testing.do_bench(graph.replay,warmup=20,rep=100)
            except Exception as exc:row['graph_failure']=repr(exc)
            with (args.output/'finalists.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (args.output/'complete.txt').write_text('complete; all candidates, including failures, retained')


if __name__=='__main__': main()
