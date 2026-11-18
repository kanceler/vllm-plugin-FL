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
    torch.manual_seed(20261116)
    cases=[(1,17,1,False,True),(2,4095,1,True,False),
           (16,16385,1,False,False),(4,257,17,False,False),
           (1,16384,1,True,False),(64,16384,1,False,False)]
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
        packed_source=source
        start=packed_source.index('        for protected in tl.static_range(4):')
        stop=packed_source.index('        K = K_real.to(Q.dtype)',start)
        packed_loads="""        packed_idx = tl.load((key_cache_ptr + meta_offset + 128).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0xffffffff)
        packed_v0 = tl.load((key_cache_ptr + meta_offset + 132).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        packed_v1 = tl.load((key_cache_ptr + meta_offset + 136).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
"""
        chain=packed_loads+"""        for protected in tl.static_range(4):
            idx = (packed_idx >> (protected * 8)) & 255
            vb = packed_v0 if protected < 2 else packed_v1
            valbits = ((vb >> ((protected % 2) * 16)) & 65535).to(tl.uint16)
            val = valbits.to(tl.bfloat16, bitcast=True)
            K_real = tl.where(offs_d[:, None] == idx[None, :], val[None, :], K_real)
"""
        merged=packed_loads+"""        restore_bits = tl.full([HEAD_SIZE_PADDED, TILE_SIZE], 0, tl.uint16)
        restore_mask = tl.full([HEAD_SIZE_PADDED, TILE_SIZE], False, tl.int1)
        for protected in tl.static_range(4):
            idx = (packed_idx >> (protected * 8)) & 255
            vb = packed_v0 if protected < 2 else packed_v1
            valbits = ((vb >> ((protected % 2) * 16)) & 65535).to(tl.uint16)
            match = offs_d[:, None] == idx[None, :]
            restore_bits = restore_bits | tl.where(match, valbits[None, :], 0).to(tl.uint16)
            restore_mask = restore_mask | match
        K_real = tl.where(restore_mask, restore_bits.to(tl.bfloat16, bitcast=True), K_real)
"""
        packed_variants={}
        for label,body in [('k4_packed_chain',chain),('k4_packed_merge',merged)]:
            src=packed_source[:start]+body+packed_source[stop:]
            jit=triton.jit(old.fn);jit._unsafe_update_src(src)
            fn=clone(base);fn.__globals__['kernel_unified_attention']=jit
            packed_variants[label]=fn
        vr=torch.zeros((*c['cache'].shape[:3],388),dtype=torch.uint8,device='cuda')
        v_only.write(c['k'],c['v'],vr,c['slots'])
        vk,vv,vs=v_only.views(vr)
        variants={'k4_base':(base,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'k4_early_bf16':(early,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'v_only':(clone(v_only.launcher()),dict(c['args'],k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs),lambda:v_only.write(wk,wv,vr,ws))}
        variants.update({name:(fn,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)) for name,fn in packed_variants.items()})
        variants={name:variants[name] for name in ('v_only','k4_packed_chain','k4_early_bf16')}
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
            preserved=args['out'].clone()
            functions={'write_incremental':write,'attention':lambda f=fn,a=args:f(**a),
                       'full':lambda w=write,f=fn,a=args:(w(),f(**a))}
            functions.update({label:tap.invoke for label,tap in taps.items() if tap.invoke is not None})
            for label,operation in functions.items():graphs[(name,label)]=graph(operation)
            graphs[(name,'full')].replay();torch.cuda.synchronize()
            replay_error=check.error(args['out'],ref)
            assert replay_error['max_abs']<=.08 and replay_error['nrmse']<=.02,(name,replay_error)
            assert torch.equal(args['out'],preserved), 'graph replay changed deterministic output'
            resources[name]=dict(error=err,replay_error=replay_error,kernels={label:tap.resource for label,tap in taps.items()})
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
    (root/'COMPLETE').write_text('6 heldout/repeat cases; 9 timing repetitions; no production branch coverage claim\n')


if __name__=='__main__':main()
