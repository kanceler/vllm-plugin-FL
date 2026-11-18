"""Paired graph microbenchmark; no model/performance acceptance claims."""
import hashlib
import json
import statistics
import sys
import types
from pathlib import Path
import torch
import validate_k_top4 as check
from vllm_fl.attention import v_only, k_top4
from vllm.triton_utils import triton


class Tap:
    def __init__(self,jit,label,root):
        self.jit,self.label,self.root=jit,label,root
        self.invoke=None;self.resource=None
    def __getitem__(self,grid):
        def launch(*a,**kw):
            compiled=self.jit[grid](*a,**kw)
            self.invoke=lambda:self.jit[grid](*a,**kw)
            if self.resource is None:
                self.resource={n:getattr(compiled,n,None) for n in ('n_regs','n_spills','n_max_threads')}
                self.resource['shared']=getattr(compiled.metadata,'shared',None)
                self.resource['launch']={n:kw.get(n) for n in ('BLOCK_M','TILE_SIZE','num_warps','num_stages','IS_3D','NUM_SEGMENTS_PER_SEQ')}
                self.resource['asm']={}
                for name,asm in compiled.asm.items():
                    data=asm.encode() if isinstance(asm,str) else bytes(asm)
                    self.resource['asm'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                    if isinstance(asm,str):
                        (self.root/(self.label+'.'+name+'.txt')).write_text(asm)
            return compiled
        return launch


def clone(base):
    fn=types.FunctionType(base.__code__,dict(base.__globals__),base.__name__,base.__defaults__,base.__closure__)
    fn.__kwdefaults__=base.__kwdefaults__
    return fn


def graph(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g


def main():
    root=Path(sys.argv[1]);root.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(20261115)
    cases=[(1,4096,1,True,False),(1,16384,1,True,False),(1,17408,1,True,False),
           (4,4097,1,False,False),(64,16384,1,False,False),(1,128,128,False,False),
           (1,4096,128,False,False)]
    rows=[]
    for ci,case in enumerate(cases):
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:captured.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(*case)
        finally:sys.setprofile(None)
        c=captured;batch,length,qlen,_,_=case
        ids=torch.cat([torch.arange(b*length+length-qlen,(b+1)*length,device='cuda') for b in range(batch)])
        wk,wv,ws=c['k'][ids],c['v'][ids],c['slots'][ids]
        base=clone(k_top4.launcher())
        early=clone(base);old=base.__globals__['kernel_unified_attention']
        source=old.src.replace('K_real = K_load.to(tl.float32) * ks[None, :]',
                               'K_real = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)')
        source=source.replace('val[None, :].to(tl.float32), K_real','val[None, :].to(Q.dtype), K_real')
        modified=triton.jit(old.fn);modified._unsafe_update_src(source)
        early.__globals__['kernel_unified_attention']=modified
        vr=torch.zeros((*c['cache'].shape[:3],388),dtype=torch.uint8,device='cuda')
        v_only.write(c['k'],c['v'],vr,c['slots'])
        vk,vv,vs=v_only.views(vr)
        variants={'k4_base':(base,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'k4_early_bf16':(early,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'v_only':(clone(v_only.launcher()),dict(c['args'],k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs),lambda:v_only.write(wk,wv,vr,ws))}
        graphs={};resources={}
        for name,(fn,args,write) in variants.items():
            d=root/f'case{ci}-{name}';d.mkdir()
            taps={}
            for label in ('kernel_unified_attention','reduce_segments'):
                tap=Tap(fn.__globals__[label],label,d);fn.__globals__[label]=tap;taps[label]=tap
            fn(**args);torch.cuda.synchronize()
            ref=c['expected'] if name!='v_only' else c['vonly_reference']
            err=check.error(args['out'],ref)
            # check.run changes Q for decode graph replay; recompute the reference.
            if qlen==1:
                ref=check.reference(c['q'],c['dqk'] if name!='v_only' else c['k'][:-1],c['dqv'],qlen,length,batch)
                err=check.error(args['out'],ref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(name,err)
            functions={'write_incremental':write,'attention':lambda f=fn,a=args:f(**a),
                       'full':lambda w=write,f=fn,a=args:(w(),f(**a))}
            functions.update({label:tap.invoke for label,tap in taps.items() if tap.invoke is not None})
            for label,operation in functions.items():graphs[(name,label)]=graph(operation)
            resources[name]=dict(error=err,kernels={label:tap.resource for label,tap in taps.items()})
        timing={str(key):[] for key in graphs}
        for rep in range(9):
            keys=list(graphs)
            if rep%2:keys.reverse()
            for key in keys:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graphs[key].replay()
                end.record();end.synchronize()
                timing[str(key)].append(start.elapsed_time(end)/30)
        row=dict(case=case,incremental_tokens=wk.shape[0],resources=resources,timings_ms=timing,
                 medians_ms={k:statistics.median(v) for k,v in timing.items()})
        rows.append(row)
        (root/'raw.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(case=case,medians_ms=row['medians_ms'])),flush=True)
    (root/'COMPLETE').write_text('7 paired cases; 9 timing repetitions; no production branch coverage claim\n')


if __name__=='__main__':main()
