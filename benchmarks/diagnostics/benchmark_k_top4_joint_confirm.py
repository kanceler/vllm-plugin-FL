"""Joint cache-page, head-order, metadata-load and launch search.
Primary V-only control is the original untouched source/default launcher.
"""
import hashlib,importlib.util,json,os,statistics,sys
from pathlib import Path
import torch
import validate_k_top4 as check
import k_top4_frozen as frozen
import k_top4_joint_candidates as joint
from benchmark_k_top4_selected_confirm import Tap,graph
from vllm_fl.attention import v_only,tile_policy


def configs(seg):
    base=frozen.baseline_config(1,seg)
    if seg:return [dict(base,tile=t,warps=w,segments=s,stages=p) for t,w,s,p in [(32,2,8,1),(32,2,16,1),(32,2,32,1),(32,2,16,2)]]
    return [dict(base,tile=t,warps=w,stages=p) for t,w,p in [(32,4,1),(64,4,1),(64,8,1),(64,4,2)]]


def load_original(path):
    spec=importlib.util.spec_from_file_location('frozen_original_vonly',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!='0f08a4dfe8a04565c5853c407a5b79744296f08b221eba773cdd7f36c8583923':raise ValueError('Original V-only source fingerprint mismatch')
    return module


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);seed=int(sys.argv[2]);mode=sys.argv[3];torch.manual_seed(seed)
    original=load_original(sys.argv[4]);selection=json.loads(Path(sys.argv[5]).read_text()) if mode!='search' else None
    cases=[(1,16384),(56,16384),(64,16384),(80,16384)] if mode=='search' else [(b,n) for b,n in [(1,129),(1,17408),(2,4096),(4,16384),(8,4096),(16,4096),(24,16384),(32,4096),(40,16384),(48,16384),(56,4096),(56,16384),(64,4096),(64,16384),(72,16384),(80,16384)]]
    (root/'environment.json').write_text(json.dumps(dict(seed=seed,mode=mode,python=sys.executable,identity=frozen.identity(),baseline_path=str(Path(sys.argv[4]).resolve()),baseline_sha256=hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),note='original code/default launch; fresh and inherited Prefill tile state recorded separately; not model benchmark'),indent=2))
    rows=[]
    for ci,(batch,length) in enumerate(cases):
        seg=batch<=64;c={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:c.update(frame.f_locals)
        tile_policy._PREFILL_TILE['value']=None
        sys.setprofile(profile)
        try:check.run(batch,length,1,seg,length==129)
        finally:sys.setprofile(None)
        seqs=[max(1,length-[0,17,length//2][i%3]) for i in range(batch)]
        lens=torch.tensor(seqs,device='cuda',dtype=torch.int32)
        ids=torch.tensor([i*length+n-1 for i,n in enumerate(seqs)],device='cuda');wk,wv=c['k'][ids],c['v'][ids]
        def reference(dk,dv):return torch.cat([check.reference(c['q'][i:i+1],dk[i*length:i*length+n],dv[i*length:i*length+n],1,n,1) for i,n in enumerate(seqs)])
        ref=reference(c['dqk'],c['dqv']);vref=reference(c['k'][:-1],c['dqv']);bfref=reference(c['k'][:-1],c['v'][:-1])
        plans=[]
        if mode=='search':
            for page in (16,32,64):
                for head in (False,True):
                    for vector in (False,True):
                        for cfg in configs(seg):plans.append(dict(page=page,head_major=head,vector_meta=vector,config=cfg,writer_warps=1))
        else:
            # Global page/order/load choices are frozen from search; launch
            # parameters are explicitly screened for each validation batch.
            for chosen in selection['layouts']:
                for cfg in configs(seg):
                    for w in (1,2,4):plans.append(dict(chosen,config=cfg,writer_warps=w))
        graphs={};entries={};resources={};fails=[];buffers={};numpages=(length+15)//16
        for plan in plans:
            page=plan['page'];head=plan['head_major'];key=(page,head)
            if key in buffers:continue
            pages=(length+page-1)//page;num=batch*pages+2
            raw=torch.zeros((num,page,2,276),device='cuda',dtype=torch.uint8)
            table=torch.randperm(num-2,device='cuda').reshape(batch,pages).to(torch.int32)
            pos=torch.arange(length,device='cuda');slots=(table[:,pos//page].long()*page+pos%page).flatten();slots=torch.cat((slots,torch.tensor([-1],device='cuda')))
            untouched=raw[-1].clone();joint.write(c['k'],c['v'],raw,slots,head)
            kc,vc,vs=joint.views(raw,head);sl=slots[:-1];meta=joint.metadata(raw,head)[sl//page,sl%page].contiguous()
            old=c['kc'][c['slots'][:-1]//16,c['slots'][:-1]%16].contiguous()
            assert torch.equal(kc[sl//page,sl%page],old[...,:128]) and torch.equal(meta,old.view(torch.uint8)[...,128:144]),(plan,'K record')
            assert torch.equal(vc[sl//page,sl%page],c['vc'][c['slots'][:-1]//16,c['slots'][:-1]%16]),'V body'
            assert torch.equal(vs[sl//page,sl%page],c['vs'][c['slots'][:-1]//16,c['slots'][:-1]%16]),'V scales'
            assert torch.equal(raw[-1],untouched),'negative slot'
            buffers[key]=(raw,kc,vc,vs,table,slots[ids])
        def args_for(cfg,**kw):
            a=dict(c['args'],seqused_k=lens,**kw)
            if seg:
                s=cfg['segments'];a.update(seq_threshold_3D=64,num_par_softmax_segments=s,softmax_segm_output=torch.empty((batch,16,s,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,s),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,s),device='cuda'))
            return a
        for pi,plan in enumerate(plans):
            name=f'joint{pi}';d=root/f'case{ci}-{name}';d.mkdir();cfg=plan['config'];raw,kc,vc,vs,table,slots=buffers[plan['page'],plan['head_major']]
            args=args_for(cfg,k=kc,v=vc,k_scale_cache=vs,v_scale_cache=vs,block_table=table)
            fn=joint.launcher(cfg,plan['head_major'],plan['vector_meta'])
            def store(raw=raw,slots=slots,head=plan['head_major'],w=plan['writer_warps']):return joint.write(wk,wv,raw,slots,head,w)
            tap=Tap(fn.__globals__['kernel_unified_attention'],'attention',d);fn.__globals__['kernel_unified_attention']=tap
            try:
                writer=store();fn(**args);torch.cuda.synchronize()
            except Exception as exc:
                (d/'FAILED.txt').write_text(repr(exc));fails.append(dict(plan=plan,error=repr(exc)));print('FAILED',batch,plan,repr(exc),flush=True);continue
            err=check.error(args['out'],ref);total=check.error(args['out'],bfref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(plan,err)
            assert total['max_abs']<=.20 and total['nrmse']<=.05,(plan,total)
            control=frozen.launcher(cfg);ca=args_for(cfg,out=torch.empty_like(args['out']));control(**ca);torch.cuda.synchronize()
            # Different page sizes change segment partitions at tails. Exact
            # cache bytes and independent numerical refs are primary evidence.
            equivalent=check.error(args['out'],ca['out']);assert equivalent['max_abs']<=.08 and equivalent['nrmse']<=.02
            before=args['out'].clone();graphs[name]=graph(lambda f=fn,a=args,w=store:(w(),f(**a)))
            graphs[name].replay();torch.cuda.synchronize();assert torch.equal(before,args['out'])
            entries[name]=plan;resources[name]=dict(error=err,total_error=total,control_error=equivalent,kernel=tap.resource,writer={k:getattr(writer,k,None) for k in ('n_regs','n_spills')})
        # Current deployed K control, measured directly on its record144 cache.
        prior_manifest=json.loads((Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json').read_text())
        hits=[r for r in prior_manifest['records'] if r['case'][0]==batch and r['case'][2]==1 and r['case'][3]==seg]
        prior_cfg=max(hits,key=lambda r:r['case'][1])['config'] if hits else frozen.baseline_config(1,seg)
        fn=frozen.launcher(prior_cfg);args=args_for(prior_cfg);prior_slots=c['slots'][ids]
        graphs['current_K_service']=graph(lambda f=fn,a=args:(frozen.write(wk,wv,c['cache'],prior_slots),f(**a)))
        entries['current_K_service']=dict(current_K_service=True,config=prior_cfg,page=16)
        resources['current_K_service']=dict(note='record144 packed chain/load1 current service policy')
        # Freeze the original source and writer4/default launch. Record both
        # fresh and inherited Prefill state, never replace them by tuned V.
        vr=torch.zeros((*c['cache'].shape[:3],388),device='cuda',dtype=torch.uint8);original.write(c['k'],c['v'],vr,c['slots']);vk,vv,vs=original.views(vr);vslots=c['slots'][ids]
        for state in ('fresh','after_prefill128'):
            name='original_vonly_'+state;d=root/f'case{ci}-{name}';d.mkdir()
            fn=frozen.clone(original.launcher());cfg=frozen.baseline_config(1,seg);args=args_for(cfg,k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs)
            tile_policy._PREFILL_TILE['value']=None if state=='fresh' else 128
            tap=Tap(fn.__globals__['kernel_unified_attention'],'attention',d);fn.__globals__['kernel_unified_attention']=tap
            def store():return original.write(wk,wv,vr,vslots)
            fn(**args);torch.cuda.synchronize();err=check.error(args['out'],vref);assert err['max_abs']<=.08 and err['nrmse']<=.02
            graphs[name]=graph(lambda f=fn,a=args,w=store:(w(),f(**a)))
            entries[name]=dict(original_source=True,prefill_history=state,page=16,writer_warps=4);resources[name]=dict(error=err,kernel=tap.resource)
        tile_policy._PREFILL_TILE['value']=None
        timing={n:[] for n in graphs}
        for rep in range(9):
            keys=list(graphs)
            if rep%2:keys.reverse()
            for n in keys:
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(5):graphs[n].replay()
                b.record();b.synchronize();timing[n].append(a.elapsed_time(b)/5)
        row=dict(batch=batch,length=length,sequence_lengths=seqs,entries=entries,resources=resources,failures=fails,timings_ms=timing,medians_ms={n:statistics.median(t) for n,t in timing.items()})
        rows.append(row);(root/'raw.json').write_text(json.dumps(rows,indent=2));best=min((n for n in timing if n.startswith('joint')),key=row['medians_ms'].get)
        print(json.dumps(dict(batch=batch,length=length,best=entries[best],ms=row['medians_ms'][best],original_vonly={n:row['medians_ms'][n] for n in timing if n.startswith('original')})),flush=True)
    (root/'COMPLETE').write_text(f'{mode} {len(rows)} cases; original V-only unchanged source/defaults; full incremental call\n')

if __name__=='__main__':main()
