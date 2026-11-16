"""CPU-only numerical study; not kernel or model acceptance evidence."""
import json
from pathlib import Path
import sys
import torch


METHODS = [('head128',128,0), ('group64',64,0), ('group32',32,0),
           ('protect1',128,1), ('protect2',128,2), ('protect4',128,4),
           ('group32_protect1',32,1), ('group32_protect2',32,2),
           ('group32_protect4',32,4)]


def quant(x, group, protect=0):
    x = x.float()
    original = x
    if protect:
        # Protection count is per entire token/head, not per quantization group.
        index = (x.abs().argmax(-1, keepdim=True) if protect == 1
                 else x.abs().topk(int(protect), dim=-1).indices)
        saved = x.gather(-1, index)
        x = x.scatter(-1, index, 0.)
    grouped = x.reshape(*x.shape[:-1], -1, group)
    scale = (grouped.abs().amax(-1, keepdim=True)/127).clamp_min(1e-6)
    raw = grouped/scale
    rounded = (raw + torch.where(raw >= 0, .5, -.5)).trunc().clamp(-128,127)
    result = (rounded*scale).reshape_as(original)
    if protect:
        result.scatter_(-1, index, saved)
    return result


def attention(q,k,v):
    k = k.repeat_interleave(8,1).transpose(0,1)
    v = v.repeat_interleave(8,1).transpose(0,1)
    q = q.transpose(0,1)
    return ((q @ k.transpose(-1,-2)/128**.5).softmax(-1) @ v).transpose(0,1)


def main():
    torch.set_num_threads(2)
    target = Path(sys.argv[1]); target.mkdir(parents=True, exist_ok=False)
    rows=[]
    # Independent CPU-generated inputs: does not reproduce earlier GPU RNG.
    for seed in range(8):
        for length in (17,257):
            for count, amplitude in ((0,0),(1,10),(1,30),(4,30)):
                torch.manual_seed(20261113+seed)
                k=torch.randn(length,2,128).bfloat16().float()
                v=torch.randn_like(k).bfloat16().float()
                q=torch.randn(1,16,128).bfloat16().float()
                if count:
                    dims=torch.randperm(128)[:count]
                    k[1,:,dims]=amplitude; v[1,:,dims]=-amplitude
                    k[0].zero_();v[0].zero_()
                base=attention(q,k,v)
                for name,group,protect in METHODS:
                    dk,dv=quant(k,group,protect),quant(v,group,protect)
                    for side in ('K','V','KV'):
                        out=attention(q,dk if 'K' in side else k,dv if 'V' in side else v)
                        diff=out-base
                        maximum=diff.abs().max().item()
                        nrmse=(diff.square().mean().sqrt()/base.square().mean().sqrt().clamp_min(1e-8)).item()
                        rows.append(dict(seed=seed,length=length,outliers=count,amplitude=amplitude,method=name,side=side,max_abs=maximum,nrmse=nrmse,passed=maximum<=.20 and nrmse<=.05))
    (target/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=[]
    for name, _, _ in METHODS:
        for side in ('K','V','KV'):
            subset=[r for r in rows if r['method']==name and r['side']==side]
            summary.append(dict(method=name,side=side,total=len(subset),passed=sum(r['passed'] for r in subset),max_abs=max(r['max_abs'] for r in subset),max_nrmse=max(r['nrmse'] for r in subset)))
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
