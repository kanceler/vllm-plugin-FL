"""CPU-only, leave-one-domain-out screening of static post-RoPE K channels.

Calibration: other three domains, 512-token Prefill snapshots only.
Validation: held-out domain, every archived length and Prefill/Decode snapshot.
No evaluation-set calibration, CUDA, model execution, or performance claims.
"""
import hashlib,json,sys,os,time
from pathlib import Path
import torch


def error(actual,ref):
    delta=actual.float()-ref.float()
    return dict(max_abs=delta.abs().max().item(),nrmse=(delta.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-8)).item())


def round_int(x):
    return torch.where(x>=0,x+.5,x-.5).clamp(-128,127).to(torch.int8)


def bf(x):return x.to(torch.bfloat16).float()


def dynamic(x,fixed=None,top4=False):
    protected=torch.zeros_like(x,dtype=torch.bool)
    if top4:protected.scatter_(-1,torch.argsort(x.abs(),dim=-1,descending=True,stable=True)[...,:4],True)
    elif fixed is not None:protected.scatter_(-1,fixed[:,None,:].expand(x.shape[0],x.shape[1],4),True)
    scale=x.masked_fill(protected,0).abs().amax(-1,keepdim=True).div(127).clamp_min(1e-6)
    qi=round_int(x/scale).masked_fill(protected,0)
    reconstructed=bf(qi.float()*scale)
    return torch.where(protected,x,reconstructed)


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source,target=map(Path,sys.argv[1:3]);target.mkdir(parents=True,exist_ok=False)
    assert (source/'COMPLETE').exists()
    entries=json.loads((source/'manifest.json').read_text())
    def read(e):
        p=source/e['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],p
        x=torch.load(p,map_location='cpu',weights_only=True)
        assert all(not x[k].is_cuda for k in ('q','k','v'))
        return x
    # Validate every source file before any selection or computation.
    for e in entries:read(e)
    domains=('zh','en','code','math');layers=(0,10,20,30,41)
    calibration={};channel_rows=[]
    for heldout in domains:
        for layer in layers:
            selected=[e for e in entries if any(e['file']==f'{d}-512-step0-layer{layer}.pt' for d in domains if d!=heldout)]
            assert len(selected)==3
            data=[read(e)['k'].float() for e in selected];k=torch.cat(data,dim=1)
            indices=torch.argsort(k.abs(),dim=-1,descending=True,stable=True)[...,:4]
            counts=torch.zeros((2,128),dtype=torch.int64)
            counts.scatter_add_(1,indices.reshape(2,-1),torch.ones_like(indices.reshape(2,-1)))
            fixed=torch.argsort(counts,dim=-1,descending=True,stable=True)[:,:4]
            scale=(k.abs().amax(1)/127).clamp_min(1e-6)
            calibration[heldout,layer]=(scale,fixed)
            channel_rows.append(dict(heldout=heldout,layer=layer,files=[e['file'] for e in selected],scale=scale.tolist(),fixed_channels=fixed.tolist(),calibration_top4_coverage=counts.gather(1,fixed).sum(1).div(k.shape[1]*4).tolist()))
    (target/'calibration.json').write_text(json.dumps(channel_rows,indent=2))
    rows=[];stability=[]
    for e in entries:
        x=read(e);domain=x['case'].split('-')[0];layer=x['layer']
        q,k,v=[x[n].float() for n in ('q','k','v')];scale,fixed=calibration[domain,layer]
        repeat=q.shape[0]//k.shape[0]
        valid=torch.arange(k.shape[1])[None,:]<=x['positions'][:,None]
        def evaluate(qq,kk,vv,round_prob=True):
            logits=(qq@kk.repeat_interleave(repeat,0).transpose(-1,-2))*x['scaling']
            scores=logits.masked_fill(~valid,float('-inf'))
            if 'mask' in x:
                mask=x['mask'];scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else scores+mask.float()
            prob=scores.softmax(-1)
            return logits,prob,(bf(prob) if round_prob else prob)@vv.repeat_interleave(repeat,0)
        baseline=evaluate(q,k,v)
        fp32_baseline=evaluate(q,k,v,False)
        dv=dynamic(v)
        topidx=torch.argsort(k.abs(),dim=-1,descending=True,stable=True)[...,:4]
        matches=(topidx[...,None]==fixed[:,None,None,:]).any(-1)
        mass=k.square();protected_mass=mass.gather(-1,fixed[:,None,:].expand(2,k.shape[1],4)).sum(-1)/mass.sum(-1).clamp_min(1e-12)
        stability.append(dict(file=e['file'],domain=domain,layer=layer,step=x['step'],
            fixed_channels=fixed.tolist(),top4_coverage_per_head=matches.float().mean((1,2)).tolist(),
            all_four_match_fraction_per_head=matches.all(-1).float().mean(1).tolist(),
            protected_energy_fraction_per_head=protected_mass.mean(1).tolist(),
            position_quarters=[matches[:,i*k.shape[1]//4:(i+1)*k.shape[1]//4].float().mean((1,2)).tolist() for i in range(4)]))
        variants=[('v_only',q,k,dv,None),('dynamic_top4',q,dynamic(k,top4=True),dv,None),
                  ('static_protect4_dynamic_scale',q,dynamic(k,fixed=fixed),dv,None)]
        for margin in (1.,1.25,2.):
            sc=scale*margin;z=k/sc[:,None,:];qi=round_int(z).float()
            sat=((z>127)|(z< -128)).float().mean().item()
            dequant=bf(qi*sc[:,None,:])
            qscaled=bf(q*sc.repeat_interleave(repeat,0)[:,None,:])
            info=dict(margin=margin,saturation_fraction=sat,qscaled_max_abs=qscaled.abs().max().item())
            variants.append((f'static_channel_dequant_m{margin}',q,dequant,dv,info))
            variants.append((f'static_channel_fold_q_m{margin}',qscaled,qi,dv,info))
            # Protected coordinates retain their original Q and BF16 K values.
            # Only unprotected coordinates use the static scale folded into Q.
            mask=torch.zeros_like(scale,dtype=torch.bool).scatter_(1,fixed,True)
            effective_scale=torch.where(mask,torch.ones_like(sc),sc)
            mixed_k=torch.where(mask[:,None,:],k,qi)
            mixed_q=bf(q*effective_scale.repeat_interleave(repeat,0)[:,None,:])
            protected_dequant=torch.where(mask[:,None,:],k,dequant)
            combined_info=dict(margin=margin,saturation_fraction=((z>127)|(z< -128)).masked_select(~mask[:,None,:].expand_as(z)).float().mean().item())
            variants.append((f'protected_channel_dequant_m{margin}',q,protected_dequant,dv,combined_info))
            variants.append((f'protected_channel_fold_q_m{margin}',mixed_q,mixed_k,dv,combined_info))
        actuals={}
        for name,qq,kk,vv,info in variants:
            actual=evaluate(qq,kk,vv);actuals[name]=actual
            metrics={n:error(a,b) for n,a,b in zip(('logits','probability','output'),actual,baseline)}
            output=metrics['output'];row=dict(file=e['file'],heldout_domain=domain,layer=layer,step=x['step'],method=name,metrics=metrics,fp32_output_error=error(actual[2],fp32_baseline[2]),passed=output['max_abs']<=.20 and output['nrmse']<=.05,info=info)
            if 'fold_q' in name:
                dequant=actuals[name.replace('fold_q','dequant')]
                row['fold_vs_dequant_output_error']=error(actual[2],dequant[2])
            rows.append(row)
        with (target/'raw.jsonl').open('a') as f:
            for row in rows[-len(variants):]:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(file=e['file'],passed={r['method']:r['passed'] for r in rows[-len(variants):]})),flush=True)
    (target/'channel_stability.json').write_text(json.dumps(stability,indent=2))
    summary=[]
    for name in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==name]
        summary.append(dict(method=name,count=len(subset),passed=sum(r['passed'] for r in subset),max_abs=max(r['metrics']['output']['max_abs'] for r in subset),max_nrmse=max(r['metrics']['output']['nrmse'] for r in subset),by_domain={d:dict(count=sum(r['heldout_domain']==d for r in subset),passed=sum(r['passed'] and r['heldout_domain']==d for r in subset)) for d in domains},max_saturation=max((r['info'] or {}).get('saturation_fraction',0) for r in subset)))
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    (target/'metadata.json').write_text(json.dumps(dict(device='cpu',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,threads=torch.get_num_threads(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),thresholds=dict(max_abs=.20,nrmse=.05),precision_model='FP32 matmul accumulation on BF16 operands, full softmax then BF16 probabilities; not online-softmax kernel emulation',split='leave-one-text-domain-out; other domains 512 Prefill only for calibration; repeated prose snapshots not independent'),indent=2))
    (target/'COMPLETE').write_text(f'{len(entries)} snapshots x {len(variants)} variants; exploratory reuse of prior screening data; CPU only; no model accuracy or speed claims\n')

if __name__=='__main__':
    with torch.inference_mode():main()
