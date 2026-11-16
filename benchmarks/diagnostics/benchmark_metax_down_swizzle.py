"""Offline direct TTGIR shared swizzle ablation; never used by runtime dispatch."""
import argparse
import importlib
import json
from pathlib import Path
import random
import re
import statistics


def main():
    import torch
    import triton
    import triton.testing
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-ir',type=Path,required=True)
    p.add_argument('--variant', choices=['phase1','phase4','maxphase4','vec4','noswizzle','vector_int8'], required=True,
                   help='Run one factor per process, with both original controls.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    def emit(file,row):
        with (args.output/file).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    variants=[('ir_original',8,2,8),('phase1',8,1,8),('phase4',8,4,8),
              ('maxphase4',8,2,4),('vec4',4,2,8),('noswizzle',8,1,1),
              ('vector_int8',8,2,8)]
    variants=[v for v in variants if v[0] in ('ir_original',args.variant)]
    for seed in [20261118,20261119]:
        torch.manual_seed(seed)
        w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
        scales=w.float().abs().amax(1).clamp_min(1e-8)/127
        q=(w.float()/scales[:,None]).round().clamp(-127,127).to(torch.int8)
        for m in [56,64]:
            a=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
            ref=(a.float()@q.float().T)*scales
            text=(args.source_ir/f'm{m}-manual_original.ttgir').read_text()
            old='#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>'
            assert text.count(old)==1
            jobs=[({'name':'dispatch','m':m,'seed':seed},lambda:b._shape_dispatch(a,q,scales),None)]
            for name,vec,phase,maxphase in variants:
                row={'name':name,'m':m,'seed':seed,'shared_B':[vec,phase,maxphase]}
                try:
                    new=f'#shared1 = #triton_gpu.shared<{{vec = {vec}, perPhase = {phase}, maxPhase = {maxphase}, order = [0, 1], hasLeadingOffset = false}}>'
                    src=text.replace(old,new).replace('@partial_manual_original(',f'@partial_swizzle_{name}(')
                    path=args.output/f'm{m}-{name}.ttgir'
                    if not path.exists():path.write_text(src)
                    opts={'num_warps':4,'num_stages':1,'pipeline':'basic'}
                    if name=='vector_int8':
                        opts['extra_options']=' -mllvm -metaxgpu-sched-regpressure=false -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-sched-mma-maxnum=3 -mllvm -metaxgpu-mma-unroll-count=1 -mllvm -metaxgpu-slp-vectorize-i8=true '
                    compiled=triton.compile(str(path),options=opts)
                    run=compiled[(32,8,1)]
                    pp=torch.empty((8,m,2048),device='cuda',dtype=torch.float32)
                    def launch(out,run=run):return run(a,q,out,m,a.stride(0),q.stride(0))
                    def partial(launch=launch,pp=pp):return launch(pp)
                    def full(launch=launch):
                        work=torch.empty((8,m,2048),device='cuda',dtype=torch.float32)
                        y=torch.empty((m,2048),device='cuda',dtype=torch.bfloat16)
                        launch(work)
                        b.weight_int8pack_mm_splitk_reduce_scale.jit_function.run(work,scales,y,m,2048,SPLIT_K=8,BLOCK_SIZE=256,num_warps=4,grid=(triton.cdiv(m*2048,256),),warmup=False)
                        return y
                    partial();torch.cuda.synchronize()
                    row.update(n_regs=compiled.n_regs,n_spills=compiled.n_spills,metadata=str(compiled.metadata),signature=compiled.src.signature)
                    for kind,content in compiled.asm.items():
                        out=args.output/f'm{m}-{name}.{kind}'
                        if out.exists():continue
                        if isinstance(content,str):out.write_text(content)
                        elif isinstance(content,bytes):out.write_bytes(content)
                    jobs.append((row,full,partial))
                except Exception as exc:
                    row['failure']=repr(exc);emit('compile-failures.jsonl',row)
                    if 'illegal' in str(exc).lower():raise
            valid=[]
            for row,full,partial in jobs:
                try:
                    y=full();torch.cuda.synchronize()
                    row['l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert row['l2']<0.005
                    row['full_screen_ms']=triton.testing.do_bench(full,warmup=20,rep=90)
                    if partial:row['partial_ms']=triton.testing.do_bench(partial,warmup=20,rep=90)
                    valid.append((row,full))
                except Exception as exc:
                    row['failure']=repr(exc)
                    if 'illegal' in str(exc).lower():
                        emit('screen.jsonl',row)
                        raise
                emit('screen.jsonl',row)
            # Measure all factors, not only the fastest preliminary sample.
            times={id(row):[] for row,fn in valid}
            for repeat in range(7):
                order=list(valid);random.Random(seed+repeat).shuffle(order)
                for row,fn in order:times[id(row)].append(triton.testing.do_bench(fn,warmup=20,rep=100))
            graphs=[]
            for row,fn in valid:
                row['full_rounds_ms']=times[id(row)];row['full_median_ms']=statistics.median(times[id(row)])
                try:
                    stream=torch.cuda.Stream();g=torch.cuda.CUDAGraph();torch.cuda.synchronize()
                    with torch.cuda.graph(g,stream=stream):y=fn()
                    g.replay();torch.cuda.synchronize()
                    row['graph_l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert row['graph_l2']<0.005
                    graphs.append((row,g,y))
                except Exception as exc:row['graph_failure']=repr(exc)
            gt={id(row):[] for row,g,y in graphs}
            for repeat in range(5):
                order=list(graphs);random.Random(seed+100+repeat).shuffle(order)
                for row,g,y in order:gt[id(row)].append(triton.testing.do_bench(g.replay,warmup=20,rep=100))
            for row,fn in valid:
                if id(row) in gt:row['graph_rounds_ms']=gt[id(row)];row['graph_median_ms']=statistics.median(gt[id(row)])
                emit('finalists.jsonl',row)
    (args.output/'complete.txt').write_text('direct IR experiment complete; not runtime integration')


if __name__=='__main__':main()
