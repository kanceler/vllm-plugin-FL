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
from vllm.triton_utils import triton, tl


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
    torch.manual_seed(int(sys.argv[2]) if len(sys.argv)>2 else 20261117)
    cases=[(1,4096,1,True,False),(1,16384,1,True,False),
           (64,16384,1,False,False),(4,4097,1,False,False),
           (1,128,128,False,False),(1,4096,128,False,False)]
    rows=[]
    for ci,case in enumerate(cases):
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:captured.update(frame.f_locals)
        from vllm_fl.attention.tile_policy import _PREFILL_TILE
        _PREFILL_TILE["value"] = None
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
        src=packed_source[:start]+chain+packed_source[stop:]
        jit=triton.jit(old.fn);jit._unsafe_update_src(src)
        chain_fn=clone(base);chain_fn.__globals__['kernel_unified_attention']=jit
        packed_variants={'k4_packed_chain':chain_fn}
        vr=torch.zeros((*c['cache'].shape[:3],388),dtype=torch.uint8,device='cuda')
        v_only.write(c['k'],c['v'],vr,c['slots'])
        vk,vv,vs=v_only.views(vr)
        variants={'k4_base':(base,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'k4_early_bf16':(early,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)),
                  'v_only':(clone(v_only.launcher()),dict(c['args'],k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs),lambda:v_only.write(wk,wv,vr,ws))}
        variants.update({name:(fn,c['args'],lambda:k_top4.write(wk,wv,c['cache'],ws)) for name,fn in packed_variants.items()})
        variants={n:variants[n] for n in ('v_only','k4_packed_chain')}
        mode=sys.argv[3] if len(sys.argv)>3 else 'config'
        if mode=='write':
            writer_src=k_top4._write.src
            load_src=writer_src.replace('saved=tl.sum(tl.where(d==index,x,0.),0)',
                'saved=tl.load(K+t*kt+h*kh+index).to(tl.float32)')
            assert writer_src!=load_src
            for direct,ws_src in [(False,writer_src),(True,load_src)]:
                writer=triton.jit(k_top4._write.fn);writer._unsafe_update_src(ws_src)
                for nw in (1,2,4):
                    name=f'write_{"load" if direct else "sum"}_w{nw}'
                    def store(k,v,raw,slots,jit=writer,nw=nw):
                        kc,vc,vs=k_top4.views(raw)
                        jit[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
                            v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=nw)
                    # Check full packed cache, including negative slots, random page table and source strides.
                    scratch=torch.zeros_like(c['cache']);store(c['k'],c['v'],scratch,c['slots'])
                    expected_cache=torch.zeros_like(scratch);k_top4.write(c['k'],c['v'],expected_cache,c['slots'])
                    assert torch.equal(scratch,expected_cache),(name,'cache bytes differ')
                    variants[name]=(clone(chain_fn),c['args'],lambda st=store:st(wk,wv,c['cache'],ws))
        elif mode=='config':
            # Single-factor paired configurations, then limited segment exploration.
            settings=[(t,w,16) for t in (16,32,64) for w in (2,4)] if qlen==1 else [(t,w,16) for t in (32,64,128) for w in (8,16)]
            if case[3]:settings += [(16,2,n) for n in (8,32)]
            for tile,warps,segments in settings:
                for name in ('v_only','k4_packed_chain'):
                    fn,args,write=variants[name]; fn=clone(fn);args=dict(args)
                    original=fn.__globals__['_fl_prefill_tiles']
                    def tiles(*a,original=original,warps=warps,**kw):
                        bm,_,_=original(*a,**kw);return bm,warps,1
                    fn.__globals__['_fl_prefill_tiles']=tiles
                    fn.__globals__['_get_tile_size']=lambda *a,tile=tile,**kw:tile
                    if case[3]:
                        args.update(num_par_softmax_segments=segments,
                            softmax_segm_output=torch.empty((batch,16,segments,128),device='cuda'),
                            softmax_segm_max=torch.empty((batch,16,segments),device='cuda'),
                            softmax_segm_expsum=torch.empty((batch,16,segments),device='cuda'))
                    variants[f'{name}_t{tile}_w{warps}_s{segments}']=(fn,args,write)
        else:raise ValueError(mode)
        graphs={};resources={}
        for name,(fn,args,write) in variants.items():
            d=root/f'case{ci}-{name}';d.mkdir()
            taps={}
            for label in ('kernel_unified_attention','reduce_segments'):
                tap=Tap(fn.__globals__[label],label,d);fn.__globals__[label]=tap;taps[label]=tap
            try:
                fn(**args);torch.cuda.synchronize()
            except Exception as exc:
                (d/'FAILED.txt').write_text(repr(exc))
                print('FAILED',name,repr(exc),flush=True)
                continue
            ref=c['expected'] if not name.startswith('v_only') else c['vonly_reference']
            err=check.error(args['out'],ref)
            # check.run changes Q for decode graph replay; recompute the reference.
            if qlen==1:
                ref=check.reference(c['q'],c['dqk'] if not name.startswith('v_only') else c['k'][:-1],c['dqv'],qlen,length,batch)
                err=check.error(args['out'],ref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(name,err)
            total_ref=check.reference(c['q'],c['k'][:-1],c['v'][:-1],qlen,length,batch)
            total_error=check.error(args['out'],total_ref)
            assert total_error['max_abs']<=.20 and total_error['nrmse']<=.05,(name,total_error)
            preserved=args['out'].clone()
            functions={'write_incremental':write,'attention':lambda f=fn,a=args:f(**a),
                       'full':lambda w=write,f=fn,a=args:(w(),f(**a))}
            functions.update({label:tap.invoke for label,tap in taps.items() if tap.invoke is not None})
            for label,operation in functions.items():graphs[(name,label)]=graph(operation)
            graphs[(name,'full')].replay();torch.cuda.synchronize()
            replay_error=check.error(args['out'],ref)
            assert replay_error['max_abs']<=.08 and replay_error['nrmse']<=.02,(name,replay_error)
            assert torch.equal(args['out'],preserved), 'graph replay changed deterministic output'
            resources[name]=dict(error=err,total_error=total_error,replay_error=replay_error,kernels={label:tap.resource for label,tap in taps.items()})
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
    (root/'COMPLETE').write_text('6 tuning cases; 9 timing repetitions; no production branch coverage claim\n')


if __name__=='__main__':main()
