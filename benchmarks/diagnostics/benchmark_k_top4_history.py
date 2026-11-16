"""Same input/shape launcher history reproduction; independent diagnostics only."""
import json
import statistics
import sys
from pathlib import Path
import torch
import benchmark_k_top4_tuning as bench
from vllm_fl.attention import k_top4, v_only
from vllm_fl.attention.tile_policy import _PREFILL_TILE
from vllm.triton_utils import triton


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261118)
    captured={}
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is bench.check.run.__code__:captured.update(frame.f_locals)
    _PREFILL_TILE['value']=None
    sys.setprofile(profile)
    try:bench.check.run(4,4097,1,False,False)
    finally:sys.setprofile(None)
    c=captured
    base=bench.clone(k_top4.launcher());old=base.__globals__['kernel_unified_attention']
    src=old.src.replace('K_real = K_load.to(tl.float32) * ks[None, :]',
        'K_real = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)')
    start=src.index('        for protected in tl.static_range(4):');stop=src.index('        K = K_real.to(Q.dtype)',start)
    body='''        packed_idx = tl.load((key_cache_ptr + meta_offset + 128).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0xffffffff)
        packed_v0 = tl.load((key_cache_ptr + meta_offset + 132).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        packed_v1 = tl.load((key_cache_ptr + meta_offset + 136).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        for protected in tl.static_range(4):
            idx = (packed_idx >> (protected * 8)) & 255
            vb = packed_v0 if protected < 2 else packed_v1
            bits = ((vb >> ((protected % 2) * 16)) & 65535).to(tl.uint16)
            val = bits.to(tl.bfloat16, bitcast=True)
            K_real = tl.where(offs_d[:, None] == idx[None, :], val[None, :], K_real)
'''
    jit=triton.jit(old.fn);jit._unsafe_update_src(src[:start]+body+src[stop:]);base.__globals__['kernel_unified_attention']=jit
    vr=torch.zeros((*c['cache'].shape[:3],388),device='cuda',dtype=torch.uint8)
    v_only.write(c['k'],c['v'],vr,c['slots']);vk,vv,vs=v_only.views(vr)
    variants={'chain':(base,c['args']), 'v_only':(bench.clone(v_only.launcher()),dict(c['args'],k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs))}
    rows=[]
    for phase in ('fresh','after_prefill','per_call_reset'):
        if phase=='fresh':_PREFILL_TILE['value']=None
        else:bench.check.run(1,128,128,False,False)
        state_before=_PREFILL_TILE['value'];graphs={};resources={}
        for name,(original,args) in variants.items():
            fn=bench.clone(original)
            if phase=='per_call_reset':
                derive=fn.__globals__['_fl_prefill_tiles']
                def reset(*a,derive=derive,**kw):
                    _PREFILL_TILE['value']=None
                    return derive(*a,**kw)
                fn.__globals__['_fl_prefill_tiles']=reset
            d=root/f'{phase}-{name}';d.mkdir()
            tap=bench.Tap(fn.__globals__['kernel_unified_attention'],'attention',d)
            fn.__globals__['kernel_unified_attention']=tap
            fn(**args);torch.cuda.synchronize()
            ref=bench.check.reference(c['q'],c['dqk'] if name=='chain' else c['k'][:-1],c['dqv'],1,4097,4)
            err=bench.check.error(args['out'],ref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(phase,name,err)
            before=args['out'].clone();graphs[name]=bench.graph(lambda f=fn,a=args:f(**a))
            graphs[name].replay();torch.cuda.synchronize();assert torch.equal(before,args['out'])
            resources[name]=dict(kernel=tap.resource,error=err)
        timings={n:[] for n in graphs}
        for rep in range(9):
            for n in (list(graphs) if rep%2==0 else list(graphs)[::-1]):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(30):graphs[n].replay()
                b.record();b.synchronize();timings[n].append(a.elapsed_time(b)/30)
        row=dict(phase=phase,state_before=state_before,resources=resources,timings_ms=timings,medians_ms={n:statistics.median(t) for n,t in timings.items()})
        rows.append(row);(root/'raw.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    (root/'COMPLETE').write_text('Same decode input/shape; attention-only history reproduction, 3 phases\n')


if __name__=='__main__':main()
