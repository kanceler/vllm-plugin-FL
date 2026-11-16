"""Audit current M56/M64 Down partial, without changing any runtime kernel."""
import argparse
import ctypes as C
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics


def enumerate_events():
    lib=C.CDLL('/opt/maca/lib/libmcpti.so')
    def check(code):
        if code: raise RuntimeError(f'MCPTI status {code}')
    n=C.c_uint32()
    check(lib.mcptiDeviceGetNumEventDomains(0,C.byref(n)))
    domains=(C.c_uint32*n.value)();size=C.c_size_t(C.sizeof(domains))
    check(lib.mcptiDeviceEnumEventDomains(0,C.byref(size),domains))
    rows=[]
    for domain in domains:
        name=C.create_string_buffer(1024);size=C.c_size_t(len(name))
        check(lib.mcptiEventDomainGetAttribute(domain,0,C.byref(size),name))
        count=C.c_uint32();check(lib.mcptiEventDomainGetNumEvents(domain,C.byref(count)))
        events=(C.c_uint32*count.value)();size=C.c_size_t(C.sizeof(events))
        check(lib.mcptiEventDomainEnumEvents(domain,C.byref(size),events))
        for event in events:
            text=C.create_string_buffer(4096);size=C.c_size_t(len(text))
            check(lib.mcptiEventGetAttribute(event,0,C.byref(size),text))
            rows.append({'domain':domain,'domain_name':name.value.decode(),'event':event,'name':text.value.decode()})
    return rows


def main():
    import torch
    import triton
    import triton.testing
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    torch.cuda.init();torch.manual_seed(20261115)
    props=torch.cuda.get_device_properties(0)
    env={'torch':torch.__version__,'triton':triton.__version__,'backend':b.__file__,
         'backend_sha256':hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
         'device':str(props),'properties':{x:getattr(props,x,None) for x in ['multi_processor_count','warp_size','max_threads_per_multi_processor','regs_per_multiprocessor','regs_per_block','shared_memory_per_block','shared_memory_per_multiprocessor','clock_rate','memory_clock_rate','memory_bus_width']}}
    (args.output/'environment.json').write_text(json.dumps(env,indent=2,default=str))
    try: events=enumerate_events();(args.output/'events.json').write_text(json.dumps(events,indent=2))
    except Exception as exc:(args.output/'events-error.txt').write_text(repr(exc))
    q=torch.randint(-127,128,(2048,6144),device='cuda',dtype=torch.int8)
    scale=torch.ones(2048,device='cuda',dtype=torch.float32)
    jobs={};storage={};metadata={}
    for m in [56,64]:
        a=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        seen=[];original=b._configured_splitk
        def record(*xs,**kw):seen.append(tuple(xs[3]));return original(*xs,**kw)
        b._configured_splitk=record
        try:b._shape_dispatch(a,q,scale)
        finally:b._configured_splitk=original
        assert len(seen)==1
        cfg=seen[0];bm,bn,bk,warps,stages,sk=cfg
        pp=torch.empty((sk,m,2048),device='cuda',dtype=torch.float32)
        def fn(a=a,pp=pp,m=m,cfg=cfg):
            bm,bn,bk,warps,stages,sk=cfg
            return b.weight_int8pack_mm_splitk_partial.jit_function.run(
                a,q,pp,m,2048,6144,*a.stride(),q.stride(1),q.stride(0),
                SPLIT_K=sk,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,EVEN_M=m%bm==0,
                num_warps=warps,num_stages=stages,pipeline='basic',
                grid=(triton.cdiv(m,bm)*(2048//bn),sk),warmup=False)
        compiled=fn();torch.cuda.synchronize()
        info={'m':m,'config':cfg,'n_regs':getattr(compiled,'n_regs',None),
              'n_spills':getattr(compiled,'n_spills',None),'metadata':str(compiled.metadata),
              'asm_keys':list(compiled.asm),'grid':[triton.cdiv(m,bm)*(2048//bn),sk],
              'logical_weight_bytes':q.numel(),'logical_A_bytes':a.numel()*2,
              'partial_write_bytes':pp.numel()*4,'matmul_flops':2*m*2048*6144}
        for kind,src in compiled.asm.items():
            path=args.output/f'm{m}.{kind}'
            if isinstance(src,str):path.write_text(src)
            elif isinstance(src,bytes):path.write_bytes(src)
        ref=a.float()@q.float().T
        err=float((pp.sum(0)-ref).norm()/ref.norm());assert err<0.005
        info['partial_sum_relative_l2']=err
        metadata[m]=info;jobs[m]=fn;storage[m]=(a,pp)
        (args.output/f'm{m}.json').write_text(json.dumps(info,indent=2))
    timings={m:[] for m in jobs}
    for repeat in range(7):
        order=list(jobs);random.Random(20261115+repeat).shuffle(order)
        for m in order:timings[m].append(triton.testing.do_bench(jobs[m],warmup=25,rep=120))
    for m,fn in jobs.items():
        g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();torch.cuda.synchronize()
        with torch.cuda.graph(g,stream=stream):fn()
        g.replay();torch.cuda.synchronize()
        graph=[triton.testing.do_bench(g.replay,warmup=25,rep=120) for _ in range(5)]
        row=dict(metadata[m]);row.update(partial_rounds_ms=timings[m],partial_median_ms=statistics.median(timings[m]),graph_rounds_ms=graph,graph_median_ms=statistics.median(graph))
        (args.output/f'm{m}-timing.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    (args.output/'complete.txt').write_text('compiler and timing audit complete; hardware counter collection is separate')


if __name__=='__main__':main()
