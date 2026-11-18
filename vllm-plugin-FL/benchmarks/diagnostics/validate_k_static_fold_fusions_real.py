"""GPU replay of prior CPU-screening tensors/parameters, not new blind data."""
import hashlib,json,sys
from pathlib import Path
import torch
import validate_k_top4 as check
import k_static_fold_fusions as fusion
import k_static_protect_fold_candidate as candidate
import k_top4_layout_candidates as layout
import k_top4_frozen as frozen
from vllm.v1.kv_cache_interface import KVQuantMode


def main():
    source,calfile,target=map(Path,sys.argv[1:4]);target.mkdir(exist_ok=False)
    calibration={(r['heldout'],r['layer']):r for r in json.loads(calfile.read_text())}
    entries=json.loads((source/'manifest.json').read_text());rows=[]
    for entry in entries:
        file=source/entry['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['sha256']
        x=torch.load(file,map_location='cpu',weights_only=True)
        if 'mask' in x:raise ValueError('Explicit extra masks require separate validation')
        q=x['q'].transpose(0,1).contiguous().cuda();k=x['k'].transpose(0,1).contiguous().cuda();v=x['v'].transpose(0,1).contiguous().cuda()
        n=k.shape[0];qlen=q.shape[0];assert torch.equal(x['positions'].cpu(),torch.arange(n-qlen,n))
        cal=calibration[x['case'].split('-')[0],x['layer']]
        scale=torch.tensor(cal['scale'],device='cuda')*2
        perm,sc=candidate.parameters(cal['fixed_channels'],scale)
        pages=(n+31)//32;raw=torch.zeros((pages+1,32,2,276),device='cuda',dtype=torch.uint8)
        slots=torch.arange(n,device='cuda');table=torch.arange(pages,device='cuda',dtype=torch.int32)[None]
        candidate.write(k,v,raw,slots,perm,sc);kc,vc,vs=layout.views(raw,'separate_meta')
        qq=torch.empty_like(q);candidate.process_q(q,qq,perm,sc);out=torch.empty_like(q)
        cfg=frozen.baseline_config(qlen,False);fn=candidate.launcher(cfg)
        args=dict(q=qq,k=kc,v=vc,out=out,cu_seqlens_q=torch.tensor([0,qlen],device='cuda',dtype=torch.int32),max_seqlen_q=qlen,seqused_k=torch.tensor([n],device='cuda',dtype=torch.int32),max_seqlen_k=n,softmax_scale=x['scaling'],causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,k_scale_cache=vs,v_scale_cache=vs,use_td=False,seq_threshold_3D=0)
        fn(**args);torch.cuda.synchronize()
        # Decode bytes independently. Compare kernel to BF16 operands and FP32
        # reference Attention, then evaluate total error against BF16 Q/K/V.
        body=kc[slots//32,slots%32].float()
        meta=torch.as_strided(raw,(pages+1,32,2,16),(raw.stride(0),32,16,1),32*256)[slots//32,slots%32].contiguous()
        body[...,:4]=meta[...,4:12].contiguous().view(torch.bfloat16).float()
        dv=(vc[slots//32,slots%32].float()*vs[slots//32,slots%32][...,None]).to(torch.bfloat16)
        reference=check.reference(qq,body,dv,qlen,n,1)
        baseline=check.reference(q,k,v,qlen,n,1)
        baseline_out=out.clone();expected_q=qq.clone();fusion_errors={}
        for mode,ww in [('q_in_attention',1),('q_in_write',1),('q_in_write',4),('q_in_write_heads',1)]:
            if mode=='q_in_attention':
                call=fusion.attention_launcher(cfg,perm,sc);a=dict(args,q=q)
            else:
                fusion.write_q(q,qq,k[-qlen:],v[-qlen:],raw,slots[-qlen:],perm,sc,ww,mode=='q_in_write_heads')
                assert torch.equal(qq,expected_q),'real fused Q transform'
                call=fn;a=args
            call(**a);torch.cuda.synchronize()
            ce=check.error(out,baseline_out)
            assert ce['max_abs']<=.08 and ce['nrmse']<=.02,(mode,ce)
            fusion_errors[f'{mode}_w{ww}']=dict(error=ce,bitwise_equal=torch.equal(out,baseline_out))
        ke=check.error(out,reference);te=check.error(out,baseline)
        assert ke['max_abs']<=.08 and ke['nrmse']<=.02,(entry['file'],ke)
        passed=te['max_abs']<=.20 and te['nrmse']<=.05
        row=dict(file=entry['file'],fusion_errors=fusion_errors,kernel_error=ke,total_error=te,passed=passed)
        rows.append(row)
        with (target/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    (target/'summary.json').write_text(json.dumps(dict(count=len(rows),passed=sum(r['passed'] for r in rows),max_abs=max(r['total_error']['max_abs'] for r in rows),max_nrmse=max(r['total_error']['nrmse'] for r in rows),calibration_sha256=hashlib.sha256(calfile.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),fusion_all_bitwise_equal=all(e['bitwise_equal'] for r in rows for e in r['fusion_errors'].values()),note='same old exploratory data, frozen per-domain fold parameters; not model accuracy'),indent=2))
    (target/'COMPLETE').write_text('90 prior screening snapshots replayed with static protection and folded Q\n')


if __name__=='__main__':main()
