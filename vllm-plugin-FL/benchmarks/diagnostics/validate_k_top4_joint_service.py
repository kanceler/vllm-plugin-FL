"""Frozen joint service validation and paired full-call timing; no model accuracy claim."""
import copy,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import validate_k_top4 as check
import k_top4_frozen as frozen
from benchmark_k_top4_selected_confirm import graph
from vllm_fl.attention import k_top4_joint_kernel as kernel,tile_policy
from vllm_fl.attention.k_top4_joint_policy import policy,choose,read_policy,verify_bindings,POLICY_PATH
from vllm_fl.attention.k_top4_joint_service import KTop4JointImpl,dispatch


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);torch.manual_seed(20261206)
    data=policy();verify_bindings(data)
    for kind in ('source','provider','identity','layout','record'):
        bad=copy.deepcopy(data)
        if kind=='source':bad['source_bindings']['vllm_fl/platform.py']='0'*64
        elif kind=='provider':bad['provider_bindings']['vllm.v1.attention.ops.triton_unified_attention']='0'*64
        elif kind=='identity':bad['identity']['torch']='invalid'
        elif kind=='layout':bad['layout']['page_size']=16
        else:bad['records'].append(bad['records'][0])
        p=root/f'bad-{kind}.json';p.write_text(json.dumps(bad))
        try:verify_bindings(read_policy(p))
        except ValueError:pass
        else:raise AssertionError(f'{kind} manifest guard accepted invalid policy')
    spec=importlib.util.spec_from_file_location('original_vonly',sys.argv[2]);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    assert hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()==data['original_vonly_sha256']
    old_manifest=json.loads((Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json').read_text())
    impl=KTop4JointImpl(16,128,128**-.5,2,None,None,'auto')
    cases=[(1,129,1,64),(1,17408,1,64),(2,4096,1,64),(4,16384,1,64),
           (8,4096,1,64),(16,4096,1,64),(16,16384,1,64),(24,16384,1,64),
           (32,4096,1,64),(40,16384,1,64),(48,16384,1,64),(56,4096,1,64),
           (56,16384,1,64),(64,4096,1,64),(64,16384,1,64),(72,16384,1,64),(80,16384,1,64),
           (7,257,1,64),(56,257,1,48),(4,257,17,64),(1,128,128,64),(4,2048,512,64)]
    rows=[]
    for ci,(batch,length,qlen,threshold) in enumerate(cases):
        seg=qlen==1 and batch<=threshold;c={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:c.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(batch,length,qlen,seg,length==129)
        finally:sys.setprofile(None)
        page=32;pages=(length+31)//32;num=batch*pages+2
        raw=torch.zeros((num,32,2,276),device='cuda',dtype=torch.uint8)
        table=torch.randperm(num-2,device='cuda').reshape(batch,pages).to(torch.int32)
        pos=torch.arange(length,device='cuda');slots=(table[:,pos//32].long()*32+pos%32).flatten();slots=torch.cat((slots,torch.tensor([-1],device='cuda')))
        saved=raw[-1].clone();impl.do_kv_cache_update(None,c['k'],c['v'],raw,slots);assert torch.equal(raw[-1],saved)
        kcache,vcache,scales=impl.cache_views(raw);old=c['kc'][c['slots'][:-1]//16,c['slots'][:-1]%16].contiguous()
        sl=slots[:-1];newbody=kcache[sl//32,sl%32];newmeta=torch.as_strided(raw,(num,32,2,16),(raw.stride(0),32,16,1),32*2*128)[sl//32,sl%32]
        assert torch.equal(newbody,old[...,:128]) and torch.equal(newmeta,old.view(torch.uint8)[...,128:144]),'K cache byte contract'
        assert torch.equal(vcache[sl//32,sl%32],c['vc'][c['slots'][:-1]//16,c['slots'][:-1]%16])
        assert torch.equal(scales[sl//32,sl%32],c['vs'][c['slots'][:-1]//16,c['slots'][:-1]%16])
        cached=impl.cache_views(raw);assert impl.cache_views(raw) is cached,'view cache not reused'
        # Force an actual different cache object; equal shape alone is not a safe identity.
        scratch=torch.zeros_like(raw);impl.cache_views(scratch);assert impl.cache_views(raw)[0].data_ptr()==kcache.data_ptr();del scratch
        lengths=[max(qlen,length-[0,17,length//2][i%3]) for i in range(batch)]
        lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        m=SimpleNamespace(num_actual_tokens=batch*qlen,query_start_loc=torch.arange(batch+1,device='cuda',dtype=torch.int32)*qlen,max_query_len=qlen,max_seq_len=length,seq_lens=lens,causal=True,use_cascade=False,block_table=table,seq_threshold_3D=threshold,num_par_softmax_segments=32,softmax_segm_output=torch.empty((max(batch,threshold),16,32,128),device='cuda'),softmax_segm_max=torch.empty((max(batch,threshold),16,32),device='cuda'),softmax_segm_expsum=torch.empty((max(batch,threshold),16,32),device='cuda'),mm_prefix_range_tensor=None)
        output=torch.full((batch*qlen+3,16,128),-3.,device='cuda',dtype=torch.bfloat16)
        def reference(dk,dv):return torch.cat([check.reference(c['q'][i*qlen:(i+1)*qlen],dk[i*length:i*length+n],dv[i*length:i*length+n],qlen,n,1) for i,n in enumerate(lengths)])
        ref=reference(c['dqk'],c['dqv']);total=reference(c['k'][:-1],c['v'][:-1]);vref=reference(c['k'][:-1],c['dqv'])
        ids=torch.cat([torch.arange(i*length+n-qlen,i*length+n,device='cuda') for i,n in enumerate(lengths)])
        wk,wv,ws=c['k'][ids],c['v'][ids],slots[ids]
        def attention():impl.forward(None,c['q'],None,None,raw,m,output)
        def full():impl.do_kv_cache_update(None,wk,wv,raw,ws);attention()
        for _ in range(3):full()
        torch.cuda.synchronize();err=check.error(output[:batch*qlen],ref);qe=check.error(output[:batch*qlen],total)
        assert err['max_abs']<=.08 and err['nrmse']<=.02,(ci,err)
        assert qe['max_abs']<=.20 and qe['nrmse']<=.05,(ci,qe)
        assert (output[batch*qlen:]==-3).all(),'output padding overwritten'
        expected=output.clone();g=graph(full);g.replay();torch.cuda.synchronize();assert torch.equal(expected,output)
        growth_error=None
        if qlen==1:
            remembered=lens.clone();lens.fill_(1)
            grow=graph(full)
            cached_info=choose.cache_info();lens.copy_(remembered);grow.replay();torch.cuda.synchronize()
            assert choose.cache_info()==cached_info,'Python dispatch ran during graph replay'
            growth_error=check.error(output[:batch],ref);assert growth_error['max_abs']<=.08 and growth_error['nrmse']<=.02
        # Validate workspace insufficiency fails before an unsafe launch.
        rejected=False
        if seg:
            saved_buffer=m.softmax_segm_output;m.softmax_segm_output=saved_buffer.view(-1)[:1]
            try:attention()
            except ValueError:rejected=True
            finally:m.softmax_segm_output=saved_buffer
            assert rejected,'undersized workspace accepted'
        # Same Q/data/valid lengths, old page16 K service policy.
        old_hits=[r for r in old_manifest['records'] if r['case'][0]==batch and r['case'][2]==qlen and r['case'][3]==seg]
        oldcfg=max(old_hits,key=lambda r:r['case'][1])['config'] if old_hits else frozen.baseline_config(qlen,seg)
        oldfn=frozen.launcher(oldcfg);oldargs=dict(c['args'],seqused_k=lens,out=torch.empty_like(c['args']['out']))
        if seg:
            s=oldcfg['segments'];oldargs.update(seq_threshold_3D=threshold,num_par_softmax_segments=s,softmax_segm_output=torch.empty((batch,16,s,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,s),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,s),device='cuda'))
        else:oldargs['seq_threshold_3D']=threshold
        oldslots=c['slots'][ids]
        oldfull=lambda:(frozen.write(wk,wv,c['cache'],oldslots),oldfn(**oldargs))
        oldfull();torch.cuda.synchronize();oe=check.error(oldargs['out'],ref);assert oe['max_abs']<=.08 and oe['nrmse']<=.02
        graphs={'joint_service':g,'old_K_service':graph(oldfull)}
        vr=torch.zeros((*c['cache'].shape[:3],388),device='cuda',dtype=torch.uint8);original.write(c['k'],c['v'],vr,c['slots']);vk,vv,vs=original.views(vr)
        for state in ('fresh','after_prefill128'):
            vfn=frozen.clone(original.launcher());va=dict(oldargs,k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs,out=torch.empty_like(oldargs['out']))
            # Original service segment count is 16, independent of old K's selected segments.
            if seg:
                va.update(num_par_softmax_segments=16,softmax_segm_output=torch.empty((batch,16,16,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,16),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,16),device='cuda'))
            tile_policy._PREFILL_TILE['value']=None if state=='fresh' else 128
            vfull=lambda f=vfn,a=va:(original.write(wk,wv,vr,oldslots),f(**a))
            vfull();torch.cuda.synchronize();ve=check.error(va['out'],vref);assert ve['max_abs']<=.08 and ve['nrmse']<=.02
            graphs['original_V_'+state]=graph(vfull)
        tile_policy._PREFILL_TILE['value']=None
        timings={n:[] for n in graphs}
        for rep in range(9):
            names=list(graphs)
            if rep%2:names.reverse()
            for name in names:
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(5):graphs[name].replay()
                b.record();b.synchronize();timings[name].append(a.elapsed_time(b)/5)
        items,w,origin=choose(batch,qlen,seg)
        rows.append(dict(case=[batch,length,qlen,threshold],sequence_lengths=lengths,config=dict(items),writer_warps=w,selection_origin=origin,error=err,total_error=qe,growth_error=growth_error,workspace_guard=rejected,timings_ms=timings,medians_ms={n:statistics.median(v) for n,v in timings.items()}))
        (root/'raw.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
    (root/'COMPLETE').write_text('22 frozen service cases, growth replay, mixed lengths, padding, bytes, workspace and 5 manifest guards\n')

if __name__=='__main__':main()
