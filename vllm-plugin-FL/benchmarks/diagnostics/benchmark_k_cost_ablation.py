"""Decode read-cost ablation. Fixed-position variant is a diagnostic, not dynamic top4."""
import json,statistics,sys
from pathlib import Path
import torch
import validate_k_top4 as check
import k_top4_frozen as frozen
from vllm.triton_utils import triton
from benchmark_k_top4_selected_confirm import graph


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(int(sys.argv[2]));rows=[]
    manifest=json.loads((Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json').read_text())
    for batch,length in [(1,16384),(16,16384),(56,16384),(64,16384),(65,16384)]:
        seg=batch<=64;c={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:c.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(batch,length,1,seg)
        finally:sys.setprofile(None)
        matches=[r for r in manifest['records'] if r['case'][0]==batch and r['case'][2]==1 and r['case'][3]==seg]
        cfg=max(matches,key=lambda r:r['case'][1])['config'] if matches else frozen.baseline_config(1,seg)
        original= frozen.packed_kernel();source=original.src
        begin=source.index('        ks = tl.load(');end=source.index('        K = K_real.to(Q.dtype)',begin)+len('        K = K_real.to(Q.dtype)')
        prefix=source[:begin];suffix=source[end:]
        scale_line=source[begin:source.index('        K_real =',begin)]
        graphset={};errors={}
        slots=c['slots'][:-1];packed=c['kc'][slots//16,slots%16].contiguous()
        scales=packed[...,140:144].contiguous().view(torch.float32).squeeze(-1)
        body=packed[...,:128].float();saved=packed[...,132:140].contiguous().view(torch.bfloat16).float()
        ids=torch.arange(batch,device='cuda')*length+length-1
        wk,wv,ws=c['k'][ids],c['v'][ids],c['slots'][ids]
        for name in ('convert','scale','dynamic4','fixed4','vonly'):
            fn=frozen.launcher(cfg,'v_only' if name=='vonly' else 'k');args=dict(c['args'])
            if seg:
                s=cfg['segments'];args.update(seq_threshold_3D=64,num_par_softmax_segments=s,
                    softmax_segm_output=torch.empty((batch,16,s,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,s),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,s),device='cuda'))
            if name=='convert':replacement='        K = K_load.to(Q.dtype)';dk=body
            elif name=='scale':replacement=scale_line+'        K = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)';dk=(body*scales[...,None]).bfloat16().float()
            elif name=='fixed4':
                replacement=scale_line+'''        K_real = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)
        for protected in tl.static_range(4):
            val = tl.load((key_cache_ptr + meta_offset + 132 + 2 * protected).to(tl.pointer_type(tl.bfloat16)), mask=tile_mask, other=0.)
            K_real = tl.where(offs_d[:, None] == protected, val[None, :], K_real)
        K = K_real'''
                dk=(body*scales[...,None]).bfloat16().float();dk[...,:4]=saved
            elif name=='dynamic4':replacement=None;dk=c['dqk']
            else:
                from vllm_fl.attention import v_only
                raw=torch.zeros((*c['cache'].shape[:3],388),device='cuda',dtype=torch.uint8)
                v_only.write(c['k'],c['v'],raw,c['slots']);kc,vc,vs=v_only.views(raw)
                args.update(k=kc,v=vc,k_scale_cache=vs,v_scale_cache=vs);dk=c['k'][:-1]
            if name not in ('dynamic4','vonly'):
                jit=triton.jit(original.fn);jit._unsafe_update_src(prefix+replacement+suffix);fn.__globals__['kernel_unified_attention']=jit
            ref=check.reference(c['q'],dk,c['dqv'],1,length,batch)
            fn(**args);torch.cuda.synchronize();err=check.error(args['out'],ref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(name,err)
            before=args['out'].clone();errors[name]=err
            store=(lambda r=raw:v_only.write(wk,wv,r,ws)) if name=='vonly' else (lambda:frozen.write(wk,wv,c['cache'],ws))
            graphset[name+'_attention']=graph(lambda f=fn,a=args:f(**a))
            graphset[name+'_full']=graph(lambda f=fn,a=args,w=store:(w(),f(**a)))
            graphset[name+'_full'].replay();torch.cuda.synchronize();assert torch.equal(before,args['out'])
        timings={k:[] for k in graphset}
        for repeat in range(9):
            keys=list(graphset)
            if repeat%2:keys.reverse()
            for k in keys:
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(5):graphset[k].replay()
                b.record();b.synchronize();timings[k].append(a.elapsed_time(b)/5)
        row=dict(batch=batch,length=length,config=cfg,errors=errors,timings_ms=timings,medians_ms={k:statistics.median(v) for k,v in timings.items()},note='convert/scale/fixed4 change semantics: cost diagnostics only; full uses common top4 writer for these variants')
        rows.append(row);(root/'raw.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row['medians_ms']),flush=True)
    (root/'COMPLETE').write_text('5 shapes; 5 variants; attention and full; 9 paired rounds\n')

if __name__=='__main__':main()
