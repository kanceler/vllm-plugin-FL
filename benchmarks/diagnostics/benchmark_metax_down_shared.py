"""Single-factor Down partial layout/compiler experiment at the production tile."""
import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import statistics

# Only options verified in the installed MetaX compiler are used.
BASE_OPTIONS = ('-mllvm -metaxgpu-sched-regpressure=false '
                '-mllvm -metaxgpu-vectorize-slp=true '
                '-mllvm -metaxgpu-sched-mma-maxnum=3 '
                '-mllvm -metaxgpu-mma-unroll-count=1')
VARIANTS = [
    ('manual_original', {}, {}, False),
    ('shared_no_pad', {'scenario':'cvt_shm_no_pad'}, {}, False),
    ('dot_operands_out_loop', {'scenario':'dot_operands_out_loop'}, {}, False),
    ('shared_index', {'scenario':'smemIndexOpt'}, {}, False),
    ('shared_offset_cache', {'scenario':'smemOffsetCache'}, {}, False),
    ('merge_convert', {}, {'TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT':'1'}, False),
    ('separate_shared_layouts', {}, {'TRITON_DISABLE_MACA_MERGE_EQUAL_SHARED_LAYOUT':'1'}, False),
    ('disable_constancy_layout', {}, {'TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT':'1'}, False),
    ('vector_int8', {'extra_options':' '+BASE_OPTIONS+' -mllvm -metaxgpu-slp-vectorize-i8=true '}, {}, False),
    ('packed_kn', {}, {}, True),
]


def main():
    import torch
    import triton
    import triton.testing
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',default='20261116,20261117')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    source=inspect.getsource(b.weight_int8pack_mm_splitk_partial.jit_function.fn)
    source=source[source.index('def weight_int8pack_mm_splitk_partial'):]
    # Unique names isolate in-memory compiler caches from environment-only options.
    generated='import triton\nimport triton.language as tl\n\n'
    for name,_,_,_ in VARIANTS:
        generated+='@triton.jit\n'+source.replace('def weight_int8pack_mm_splitk_partial(',f'def partial_{name}(',1)+'\n'
    path=args.output/'partial_variants.py';path.write_text(generated)
    spec=importlib.util.spec_from_file_location('partial_variants',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controlled=set().union(*(set(env) for _,_,env,_ in VARIANTS))
    assert not any(os.environ.get(k) for k in controlled), 'inherited compiler override'
    def emit(name,row):
        with (args.output/name).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    (args.output/'environment.json').write_text(json.dumps({
        'torch':torch.__version__,'triton':triton.__version__,'backend':b.__file__,
        'backend_sha256':hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
        'variants':VARIANTS,'l2_threshold':0.005,
        'note':'fresh timings; JIT cache allowed; generated source names isolate compiler environment options'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        torch.manual_seed(seed)
        w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
        scales=w.float().abs().amax(1).clamp_min(1e-8)/127
        q=(w.float()/scales[:,None]).round().clamp(-127,127).to(torch.int8)
        # Diagnostic one-time INT8 repack; production packing/lifecycle not implemented.
        packed=q.T.contiguous().T
        for m in [56,64]:
            a=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
            ref=(a.float()@q.float().T)*scales
            current=[];original=b._configured_splitk
            def record(*xs,**kw):current.append(tuple(xs[3]));return original(*xs,**kw)
            b._configured_splitk=record
            try:b._shape_dispatch(a,q,scales)
            finally:b._configured_splitk=original
            assert current==[(64,64,128,4,1,8)],current
            jobs=[({'m':m,'seed':seed,'name':'dispatch','config':current[0]},lambda:b._shape_dispatch(a,q,scales),None)]
            for name,opts,env,repacked in VARIANTS:
                kernel=getattr(module,'partial_'+name);weight=packed if repacked else q
                pp=torch.empty((8,m,2048),device='cuda',dtype=torch.float32)
                def launch(out,kernel=kernel,weight=weight,opts=opts):
                    return kernel.run(a,weight,out,m,2048,6144,*a.stride(),weight.stride(1),weight.stride(0),
                        SPLIT_K=8,BLOCK_M=64,BLOCK_N=64,BLOCK_K=128,EVEN_M=m==64,
                        num_warps=4,num_stages=1,pipeline='basic',grid=(32,8),warmup=False,**opts)
                def partial(launch=launch,pp=pp):return launch(pp)
                def full(launch=launch):
                    work=torch.empty((8,m,2048),device='cuda',dtype=torch.float32)
                    y=torch.empty((m,2048),device='cuda',dtype=torch.bfloat16)
                    launch(work)
                    b.weight_int8pack_mm_splitk_reduce_scale.jit_function.run(work,scales,y,m,2048,
                        SPLIT_K=8,BLOCK_SIZE=256,num_warps=4,grid=(triton.cdiv(m*2048,256),),warmup=False)
                    return y
                row={'m':m,'seed':seed,'name':name,'options':opts,'compile_env':env,'weight_stride':list(weight.stride())}
                try:
                    os.environ.update(env)
                    compiled=partial();torch.cuda.synchronize()
                    row.update(n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),metadata=str(compiled.metadata))
                    if seed==int(args.seeds.split(',')[0]):
                        for kind,content in compiled.asm.items():
                            out=args.output/f'm{m}-{name}.{kind}'
                            if isinstance(content,str):out.write_text(content)
                            elif isinstance(content,bytes):out.write_bytes(content)
                    jobs.append((row,full,partial))
                except Exception as exc:
                    row['failure']=repr(exc);emit('compile-failures.jsonl',row)
                finally:
                    for key in env:os.environ.pop(key,None)
            valid=[]
            for row,full,partial in jobs:
                try:
                    y=full();torch.cuda.synchronize()
                    row['l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert row['l2']<0.005
                    row['full_screen_ms']=triton.testing.do_bench(full,warmup=20,rep=90)
                    if partial:row['partial_ms']=triton.testing.do_bench(partial,warmup=20,rep=90)
                    valid.append((row,full))
                except Exception as exc:row['failure']=repr(exc)
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
    (args.output/'complete.txt').write_text('single-factor screening complete; no runtime integration')


if __name__=='__main__':main()
