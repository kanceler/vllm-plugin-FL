"""GateUp same-kernel dtype/layout diagnosis; no production dispatch changes."""
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
    p.add_argument('--prefetch', action='store_true')
    p.add_argument('--schedule-scan', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    kernel = backend.weight_int8pack_mm_dense.jit_function
    if args.prefetch:
        from metax_gateup_prefetch import gateup_prefetch
        kernel = gateup_prefetch
    configs = [(64,64,64,8,4), (64,64,128,4,2), (64,128,64,4,4),
               (32,64,128,4,2), (64,32,128,8,2), (64,128,128,4,4)]
    if args.schedule_scan:
        configs = [cfg+(pipeline,scenario,loads)
                   for cfg in [(64,64,64,8,4),(64,128,64,4,4),(64,128,128,4,4)]
                   for pipeline,scenario,loads in [('basic','fullstage',-1),('basic','dot_operands_out_loop',-1),
                       ('basic','cvt_shm_no_pad',-1),('basic','roll',-1),('basic','',2),
                       ('basic','unprefetch',-1),('cpasync-mixed','',-1)]]
    metadata = {'torch':torch.__version__, 'triton':triton.__version__,
                'backend':backend.__file__, 'bf16_provider':flag_gems.mm.__module__,
                'device':torch.cuda.get_device_name(0),
                'note':'BF16(Q) is diagnostic only; packed INT8 uses one-time transpose, scale stays output-side'}
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
        weights = {'int8_nt':q, 'bf16_q_nt':q.to(torch.bfloat16),
                   'int8_packed_kn':q.t().contiguous().t()}
        if args.prefetch or args.schedule_scan:
            weights = {'int8_prefetch' if args.prefetch else 'int8_schedule':q}
        jobs = [('original_bf16',None,lambda:flag_gems.mm(a,w.t()))]
        if args.prefetch or args.schedule_scan:
            jobs.append(('current_dense',(64,64,64,8,4),lambda:backend._configured_dense(a,q,scales,(64,64,64,8,4))))
        artifacts = set()
        def invoke(weight,cfg,name):
            bm,bn,bk,warps,stages = cfg[:5]
            pipeline,scenario,loads = cfg[5:] if len(cfg)>5 else ('basic','',-1)
            out = torch.empty((m,n),device=a.device,dtype=a.dtype)
            compiled = kernel.run(a,weight,scales,out,m,n,k,*a.stride(),*weight.stride(),*out.stride(),
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
            for cfg in configs:
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
