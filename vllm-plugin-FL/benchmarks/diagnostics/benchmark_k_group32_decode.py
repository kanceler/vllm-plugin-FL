"""Paired complete-call Decode screening. Quantization failures remain explicit."""
import json,statistics,sys,os
from pathlib import Path
import torch
import validate_k_top4 as check
import k_group32_candidate as group
import k_top4_frozen as frozen
from vllm_fl.attention import v_only
from benchmark_k_top4_selected_confirm import Tap,graph


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    seed=int(sys.argv[2]);torch.manual_seed(seed)
    manifest=json.loads((Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json').read_text())
    (root/'environment.json').write_text(json.dumps(dict(identity=frozen.identity(),seed=seed,python=sys.executable,source=group.__file__,flags={k:os.environ.get(k) for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_INT8_KV_REPAIR','VLLM_FL_V_ONLY')}),indent=2))
    rows=[]
    for ci,(batch,length,seg,unusual) in enumerate([(1,129,True,True),(1,16384,True,False),(16,16384,True,False),(56,16384,True,False),(64,16384,True,False),(65,16384,False,False)]):
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:captured.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(batch,length,1,seg,unusual)
        finally:sys.setprofile(None)
        c=captured
        raw=torch.zeros_like(c['cache']);untouched=raw[-1].clone()
        group.write(c['k'],c['v'],raw,c['slots']);assert torch.equal(raw[-1],untouched)
        kc,vc,vs=group.views(raw);slots=c['slots'][:-1]
        packed=kc[slots//16,slots%16].contiguous()
        scales=packed[...,128:144].contiguous().view(torch.float32)
        x=c['k'][:-1].float().reshape(-1,2,4,32)
        expected_scales=(x.abs().amax(-1)/127).clamp_min(1e-6)
        torch.testing.assert_close(scales,expected_scales,rtol=1e-5,atol=1e-8)
        z=x/expected_scales[...,None]
        quant=torch.where(z>=0,z+.5,z-.5).clamp(-128,127).to(torch.int8).reshape(-1,2,128)
        assert torch.equal(packed[...,:128],quant),'integer rounding'
        dqk=(packed[...,:128].float().reshape(-1,2,4,32)*scales[...,None]).reshape(-1,2,128)
        dqv=vc[slots//16,slots%16].float()*vs[slots//16,slots%16][...,None]
        refs={'group':check.reference(c['q'],dqk,dqv,1,length,batch),
              'top4':check.reference(c['q'],c['dqk'],c['dqv'],1,length,batch),
              'vonly':check.reference(c['q'],c['k'][:-1],c['dqv'],1,length,batch)}
        original=check.reference(c['q'],c['k'][:-1],c['v'][:-1],1,length,batch)
        vr=torch.zeros((*raw.shape[:3],388),device='cuda',dtype=torch.uint8)
        v_only.write(c['k'],c['v'],vr,c['slots']);vk,vv,vsc=v_only.views(vr)
        ids=torch.arange(batch,device='cuda')*length+length-1
        wk,wv,ws=c['k'][ids],c['v'][ids],c['slots'][ids]
        base=frozen.baseline_config(1,seg)
        matched=[r for r in manifest['records'] if r['case'][0]==batch and r['case'][2]==1 and r['case'][3]==seg]
        selected=max(matched,key=lambda r:r['case'][1])['config'] if matched else base
        configs=[base,selected]
        # Give group32 and V-only the same finite exploration opportunity.
        for tile,warps in ((16,2),(32,2),(32,4),(64,4)):
            for segments in ((8,16,32) if seg else (1,)):
                configs.append(dict(base,tile=tile,warps=warps,segments=segments))
        configs=list({tuple(sorted(cfg.items())):cfg for cfg in configs}.values())
        graphs={};resources={};candidates={}
        for family in ('top4','group','vonly'):
            for j,cfg in enumerate(configs):
                # Top4 control includes the original and selected configurations.
                if family=='top4' and cfg not in (base,selected):continue
                name=f'{family}_{j}';d=root/f'case{ci}-{name}';d.mkdir()
                args=dict(c['args']);s=cfg['segments']
                if seg:args.update(num_par_softmax_segments=s,seq_threshold_3D=64,
                    softmax_segm_output=torch.empty((batch,16,s,128),device='cuda'),
                    softmax_segm_max=torch.empty((batch,16,s),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,s),device='cuda'))
                if family=='group':
                    fn=group.launcher(cfg);args.update(k=kc,v=vc,k_scale_cache=vs,v_scale_cache=vs)
                    store=lambda:group.write(wk,wv,raw,ws)
                elif family=='top4':fn=frozen.launcher(cfg);store=lambda:frozen.write(wk,wv,c['cache'],ws)
                else:
                    fn=frozen.launcher(cfg,'v_only');args.update(k=vk,v=vv,k_scale_cache=vsc,v_scale_cache=vsc)
                    store=lambda:v_only.write(wk,wv,vr,ws)
                tap=Tap(fn.__globals__['kernel_unified_attention'],'attention',d);fn.__globals__['kernel_unified_attention']=tap
                try:fn(**args);torch.cuda.synchronize()
                except Exception as exc:
                    (d/'FAILED.txt').write_text(repr(exc));continue
                err=check.error(args['out'],refs[family]);total=check.error(args['out'],original)
                assert err['max_abs']<=.08 and err['nrmse']<=.02,(name,err)
                quant_pass=total['max_abs']<=.20 and total['nrmse']<=.05
                before=args['out'].clone()
                graphs[name]=graph(lambda f=fn,a=args,w=store:(w(),f(**a)))
                graphs[name].replay();torch.cuda.synchronize();assert torch.equal(before,args['out'])
                resources[name]=dict(implementation_error=err,total_error=total,quantization_pass=quant_pass,kernel=tap.resource)
                candidates[name]=dict(family=family,config=cfg,writer='load_w1' if family!='vonly' else 'original_w4')
        timing={name:[] for name in graphs}
        for repeat in range(9):
            names=list(graphs)
            if repeat%2:names.reverse()
            for name in names:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(5):graphs[name].replay()
                end.record();end.synchronize();timing[name].append(start.elapsed_time(end)/5)
        row=dict(case=[batch,length,1,seg,unusual],candidates=candidates,resources=resources,timings_ms=timing,medians_ms={n:statistics.median(t) for n,t in timing.items()})
        rows.append(row);(root/'raw.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(case=row['case'],medians=row['medians_ms'])),flush=True)
    (root/'COMPLETE').write_text('6 shapes, paired full write+attention+reduce, 9 rounds; quantization failures explicit\n')

if __name__=='__main__':main()
