"""Finite complete-call layout search; mixed lengths and independent confirmation."""
import hashlib,json,os,statistics,sys
from pathlib import Path
import torch
import validate_k_top4 as check
import k_top4_frozen as frozen
import k_top4_layout_candidates as candidate
from vllm_fl.attention import v_only
from benchmark_k_top4_selected_confirm import Tap,graph


def metadata(raw,layout):
    b,s,h,_=raw.shape;stride=raw.stride(0);off=raw.storage_offset()
    if layout=='separate_meta':return torch.as_strided(raw,(b,s,h,16),(stride,h*16,16,1),off+s*h*128)
    record=160 if layout=='record160' else 144
    return torch.as_strided(raw,(b,s,h,16),(stride,h*record,record,1),off+128)


def configurations(batch,seg):
    base=frozen.baseline_config(1,seg);configs=[base]
    if seg:
        for tile,warps,segments in [(16,2,32),(32,2,8),(32,2,16),(32,2,32),(32,4,16),(64,4,8),(64,4,16)]:
            configs.append(dict(base,tile=tile,warps=warps,segments=segments))
    else:
        for tile,warps in [(32,4),(64,4),(64,8),(128,8)]:configs.append(dict(base,tile=tile,warps=warps))
    return list({tuple(sorted(x.items())):x for x in configs}.values())


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);seed=int(sys.argv[2]);mode=sys.argv[3]
    torch.manual_seed(seed)
    if mode=='search':cases=[(1,16384),(16,16384),(56,16384),(64,16384),(65,16384)]
    elif mode=='confirm':cases=[(1,129),(1,17408),(8,4096),(16,16384),(56,16384),(64,16384),(65,16384)]
    else:raise ValueError(mode)
    selections=json.loads(Path(sys.argv[4]).read_text()) if mode=='confirm' else None
    manifest=json.loads((Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json').read_text())
    (root/'environment.json').write_text(json.dumps(dict(seed=seed,mode=mode,identity=frozen.identity(),python=sys.executable,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()),indent=2))
    rows=[]
    for ci,(batch,length) in enumerate(cases):
        seg=batch<=64;c={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:c.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(batch,length,1,seg,length==129)
        finally:sys.setprofile(None)
        # Mixed lengths, including a half-length request. Tensors contain full
        # cache history; sequence metadata bounds the actually attended prefix.
        lengths=[max(1,length-[0,17,length//2][i%3]) for i in range(batch)]
        lengths_tensor=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        ids=torch.tensor([i*length+n-1 for i,n in enumerate(lengths)],device='cuda')
        wk,wv,ws=c['k'][ids],c['v'][ids],c['slots'][ids]
        q=c['q'];args_base=dict(c['args'],seqused_k=lengths_tensor)
        def reference(dk,dv):
            return torch.cat([check.reference(q[i:i+1],dk[i*length:i*length+n],dv[i*length:i*length+n],1,n,1) for i,n in enumerate(lengths)])
        ref=reference(c['dqk'],c['dqv']);vref=reference(c['k'][:-1],c['dqv']);orig=reference(c['k'][:-1],c['v'][:-1])
        matches=[r for r in manifest['records'] if r['case'][0]==batch and r['case'][2]==1 and r['case'][3]==seg]
        canonical=frozen.baseline_config(1,seg);current=max(matches,key=lambda r:r['case'][1])['config'] if matches else canonical
        caches={}
        for layout in ('record144','record160','separate_meta'):
            raw=torch.zeros((*c['cache'].shape[:3],candidate.width(layout)),device='cuda',dtype=torch.uint8)
            before=raw[-1].clone();candidate.write(c['k'],c['v'],raw,c['slots'],layout)
            assert torch.equal(before,raw[-1]),(layout,'negative slot')
            kc,vc,sc=candidate.views(raw,layout);oldk,oldv,olds=c['kc'],c['vc'],c['vs']
            assert torch.equal(kc,oldk[...,:128]),(layout,'K body')
            assert torch.equal(vc,oldv) and torch.equal(sc,olds),(layout,'V/scales')
            assert torch.equal(metadata(raw,layout),metadata(c['cache'],'record144')),(layout,'metadata')
            caches[layout]=(raw,kc,vc,sc)
        vr=torch.zeros((*c['cache'].shape[:3],388),device='cuda',dtype=torch.uint8)
        v_only.write(c['k'],c['v'],vr,c['slots']);vk,vv,vsc=v_only.views(vr)
        configs=configurations(batch,seg);plans=[]
        if mode=='search':
            for layout in caches:
                for restore in ('chain','or_tree'):
                    for cfg in configs:plans.append((layout,restore,cfg))
            for cfg in configs:
                for w in (1,4):plans.append(('vonly',str(w),cfg))
        else:
            chosen=selections.get(str(batch))
            if chosen:
                for family in ('K','V'):
                    for record in chosen[family]:plans.append((record['layout'],record['restore'],record['config']))
            else:
                # Unsearched batches remain explicit validation of canonical,
                # rather than claiming policy generalization or optimality.
                plans += [(l,r,cfg) for l in caches for r in ('chain','or_tree') for cfg in [canonical]]
                plans += [('vonly',str(w),canonical) for w in (1,4)]
        plans += [('record144','chain',current),('record144','chain',canonical)]
        plans=list({json.dumps(p,sort_keys=True):p for p in plans}.values())
        graphs={};resources={};entries={};failures=[]
        for pi,(layout,restore,cfg) in enumerate(plans):
            label=f'p{pi}-{layout}-{restore}';d=root/f'case{ci}-{label}';d.mkdir()
            args=dict(args_base);s=cfg['segments']
            if seg:args.update(seq_threshold_3D=64,num_par_softmax_segments=s,softmax_segm_output=torch.empty((batch,16,s,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,s),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,s),device='cuda'))
            if layout=='vonly':
                fn=frozen.launcher(cfg,'v_only');args.update(k=vk,v=vv,k_scale_cache=vsc,v_scale_cache=vsc)
                def store(w=int(restore)):
                    return v_only._write[(ws.numel(),wk.shape[1])](wk,wv,vk,vv,vsc,ws,wk.stride(0),wk.stride(1),wv.stride(0),wv.stride(1),vk.stride(0),vv.stride(0),vsc.stride(0),wk.shape[1],vr.shape[1],num_warps=w)
                expected=vref
            else:
                raw,kc,vc,sc=caches[layout];args.update(k=kc,v=vc,k_scale_cache=sc,v_scale_cache=sc)
                fn=candidate.launcher(cfg,layout,restore)
                def store(raw=raw,layout=layout):return candidate.write(wk,wv,raw,ws,layout)
                expected=ref
            tap=Tap(fn.__globals__['kernel_unified_attention'],'attention',d);fn.__globals__['kernel_unified_attention']=tap
            try:
                compiled_writer=store();fn(**args);torch.cuda.synchronize()
            except Exception as exc:
                (d/'FAILED.txt').write_text(repr(exc));failures.append(dict(layout=layout,restore=restore,config=cfg,error=repr(exc)));print('FAILED',label,repr(exc),flush=True);continue
            err=check.error(args['out'],expected);total=check.error(args['out'],orig)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(label,err)
            assert total['max_abs']<=.20 and total['nrmse']<=.05,(label,total)
            if layout!='vonly':
                control=frozen.launcher(cfg);ca=dict(args,k=c['kc'],v=c['vc'],k_scale_cache=c['vs'],v_scale_cache=c['vs'],out=torch.empty_like(args['out']))
                control(**ca);torch.cuda.synchronize();assert torch.equal(args['out'],ca['out']),(label,'not bitwise equivalent to packed chain')
            before=args['out'].clone()
            full=lambda f=fn,a=args,w=store:(w(),f(**a))
            graphs[label]=graph(full);graphs[label].replay();torch.cuda.synchronize();assert torch.equal(before,args['out']),(label,'graph')
            resources[label]=dict(implementation_error=err,total_error=total,attention=tap.resource,writer={k:getattr(compiled_writer,k,None) for k in ('n_regs','n_spills')})
            entries[label]=dict(layout=layout,restore=restore,config=cfg,cache_bytes_per_token_head=388 if layout=='vonly' else candidate.width(layout))
        timing={k:[] for k in graphs}
        for repeat in range(9):
            names=list(graphs)
            if repeat%2:names.reverse()
            for name in names:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(5):graphs[name].replay()
                end.record();end.synchronize();timing[name].append(start.elapsed_time(end)/5)
        row=dict(batch=batch,length=length,sequence_lengths=lengths,current_config=current,canonical_config=canonical,entries=entries,resources=resources,failures=failures,timings_ms=timing,medians_ms={k:statistics.median(v) for k,v in timing.items()})
        rows.append(row);(root/'raw.json').write_text(json.dumps(rows,indent=2))
        best={fam:min((k for k in timing if (entries[k]['layout']=='vonly')==(fam=='V')),key=lambda k:row['medians_ms'][k]) for fam in ('K','V')}
        print(json.dumps(dict(batch=batch,length=length,best={f:dict(row['entries'][k],ms=row['medians_ms'][k]) for f,k in best.items()})),flush=True)
    (root/'COMPLETE').write_text(f'{mode}: {len(cases)} cases, full writer+Attention+reduce, mixed lengths, independent refs, bitwise K equivalence\n')

if __name__=='__main__':main()
