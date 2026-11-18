"""Expanded fixed-strategy static candidate coverage, not a model benchmark."""
import k_static_protect_fold_candidate as candidate
import k_top4_layout_candidates as layout
import copy,hashlib,os,importlib.util,json,statistics,sys
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
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);torch.manual_seed(int(sys.argv[3]))
    data=policy();verify_bindings(data)
    (root/'environment.json').write_text(json.dumps(dict(seed=int(sys.argv[3]),python=sys.executable,identity=frozen.identity(),candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_sha256=hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest() if len(sys.argv)>4 else None,flags={n:os.environ.get(n) for n in ('VLLM_PLUGINS','VLLM_FL_INT8_KV_REPAIR','VLLM_FL_K_TOP4_JOINT','VLLM_FL_K_TOP4','VLLM_FL_V_ONLY')},candidate_import=str(Path(candidate.__file__).resolve()),model='/workspace/MiniCPM5-2B geometry only; no model loaded',synthetic_static_scale=.0625,synthetic_protected_channels=[[0,3,17,65],[0,9,37,93]],timing='nine alternating-order rounds, five graph replays per round; includes Q transform + incremental KV write + Attention/reduce'),indent=2))
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
    capture_sizes=[1,2,4,8]+list(range(16,257,8))
    cases=[(b,n,1,64) for b in capture_sizes for n in (257,4096)]
    cases += [(b,16384,1,64) for b in capture_sizes if b<=64]
    cases += [(1,17408,1,64),(7,257,1,64),(7,4096,1,64),(56,257,1,48),
              (4,257,17,64),(1,128,128,64),(4,2048,512,64)]
    (root/'coverage.json').write_text(json.dumps(dict(capture_sizes=capture_sizes,cases=cases,limit='all 35 inferred actual Decode sizes through256; >64 supplementary to official concurrency64; representative Prefill only'),indent=2))
    rows=[]
    def run_case(ci,case):
        batch,length,qlen,threshold=case
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
        # Synthetic unit/performance parameters, NOT a model calibration.
        # Shared across all cases, independent of the generated tensor extrema.
        static_scale=torch.full((2,128),.0625,device='cuda')
        perm,sc=candidate.parameters([[0,3,17,65],[0,9,37,93]],static_scale)
        sr=torch.zeros_like(raw);untouched=sr[-1].clone()
        candidate.write(c['k'],c['v'],sr,slots,perm,sc)
        assert torch.equal(sr[-1],untouched),'static negative slot'
        sk,sv,ss=layout.views(sr,'separate_meta');qq=torch.empty_like(c['q'])
        candidate.process_q(c['q'],qq,perm,sc)
        # Independent PyTorch reconstruction checks actual writer bytes.
        original_k=c['k'].float().gather(2,perm.long()[None].expand(c['k'].shape[0],-1,-1))
        z=original_k/sc[None]
        qi=torch.where(z>=0,z+.5,z-.5).clamp(-128,127).to(torch.int8)
        qi[...,:4]=0
        assert torch.equal(sk[sl//32,sl%32],qi[:-1]),'static integer bytes'
        meta=torch.as_strided(sr,(num,32,2,16),(sr.stride(0),32,16,1),32*256)[sl//32,sl%32].contiguous()
        saved=meta[...,4:12].contiguous().view(torch.bfloat16)
        assert torch.equal(saved,original_k[:-1,:,:4].to(torch.bfloat16)),'static protected values'
        assert torch.equal(sv[sl//32,sl%32],vcache[sl//32,sl%32]),'V byte equality'
        assert torch.equal(ss[sl//32,sl%32],scales[sl//32,sl%32]),'V scale equality'
        dk=qi[:-1].float();dk[...,:4]=saved.float()
        expected_q=c['q'].float().reshape(-1,2,8,128).gather(3,perm.long()[None,:,None,:].expand(c['q'].shape[0],2,8,128))
        eff=sc.clone();eff[:,:4]=1
        expected_q=(expected_q*eff[None,:,None,:]).to(torch.bfloat16).reshape_as(qq)
        assert torch.equal(qq,expected_q),'Q transform'
        static_ref=torch.cat([check.reference(expected_q[i*qlen:(i+1)*qlen],dk[i*length:i*length+n],c['dqv'][i*length:i*length+n],qlen,n,1) for i,n in enumerate(lengths)])
        cfg0=dict(choose(batch,qlen,seg)[0])
        # Reuse prior fixed candidate policy only where the branch matches.
        # All missing Decode/Prefill keys use the frozen joint canonical config;
        # this run measures coverage, without per-case re-optimization.
        records=json.loads(Path(sys.argv[4]).read_text())['records']
        matched=[r['config'] for r in records if r['batch']==batch and qlen==1 and (r['config']['segments']>1)==seg]
        cfgs=[matched[0] if matched else cfg0]
        candidate_errors={};candidate_configs={}
        for idx,cfg in enumerate(cfgs):
            name=f'static_{idx}';fn=candidate.launcher(cfg)
            ca=dict(c['args'],q=qq,k=sk,v=sv,k_scale_cache=ss,v_scale_cache=ss,block_table=table,seqused_k=lens,out=torch.empty_like(c['q']),seq_threshold_3D=threshold)
            if seg:
                ns=cfg['segments'];ca.update(num_par_softmax_segments=ns,softmax_segm_output=torch.empty((batch,16,ns,128),device='cuda'),softmax_segm_max=torch.empty((batch,16,ns),device='cuda'),softmax_segm_expsum=torch.empty((batch,16,ns),device='cuda'))
            def static_full(f=fn,a=ca):
                candidate.process_q(c['q'],qq,perm,sc)
                candidate.write(wk,wv,sr,ws,perm,sc)
                f(**a)
            static_full();torch.cuda.synchronize()
            se=check.error(ca['out'],static_ref);te=check.error(ca['out'],total)
            assert se['max_abs']<=.08 and se['nrmse']<=.02,(name,se)
            assert te['max_abs']<=.20 and te['nrmse']<=.05,(name,te)
            expected=ca['out'].clone();graphs[name]=graph(static_full)
            graphs[name].replay();torch.cuda.synchronize();assert torch.equal(ca['out'],expected)
            if seg:
                lens.fill_(1);grow=graph(static_full);lens.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32));grow.replay();torch.cuda.synchronize()
                assert torch.equal(ca['out'],expected),'static growth replay'
            candidate_errors[name]=dict(kernel_error=se,total_error=te)
            candidate_configs[name]=cfg
        timings={n:[] for n in graphs}
        for rep in range(9):
            names=list(graphs)
            if rep%2:names.reverse()
            for name in names:
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(5):graphs[name].replay()
                b.record();b.synchronize();timings[name].append(a.elapsed_time(b)/5)
        items,w,origin=choose(batch,qlen,seg)
        return dict(case=[batch,length,qlen,threshold],candidate_errors=candidate_errors,candidate_configs=candidate_configs,sequence_lengths=lengths,config=dict(items),writer_warps=w,selection_origin=origin,error=err,total_error=qe,growth_error=growth_error,workspace_guard=rejected,timings_ms=timings,medians_ms={n:statistics.median(v) for n,v in timings.items()})
    for ci,case in enumerate(cases):
        row=run_case(ci,case);rows.append(row)
        (root/'raw.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(progress=len(rows),total=len(cases),case=case,medians_ms=row['medians_ms'])),flush=True)
    (root/'COMPLETE').write_text(f'{len(rows)} expanded fixed-strategy cases; no model precision or throughput claim\n')

if __name__=='__main__':main()
