"""CPU paired screening of captured post-RoPE Q and cached K/V."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from study_int8_kv_granularity import quant
from study_k_top4_extended import error


def main():
    torch.set_num_threads(2)
    source, target = map(Path, sys.argv[1:3])
    assert (source/'COMPLETE').exists(), 'Incomplete capture'
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for entry in json.loads((source/'manifest.json').read_text()):
        file = source/entry['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256']
        x = torch.load(file, map_location='cpu', weights_only=True)
        q, k, v = [x[t].float() for t in ('q','k','v')]
        repeat = q.shape[0]//k.shape[0]
        valid = torch.arange(k.shape[-2])[None,:] <= x['positions'][:,None]
        def evaluate(kk,vv):
            logits = q @ kk.repeat_interleave(repeat,0).transpose(-1,-2) * x['scaling']
            scores = logits.masked_fill(~valid, float('-inf'))
            if 'mask' in x:
                mask=x['mask']
                scores = scores.masked_fill(~mask, float('-inf')) if mask.dtype==torch.bool else scores+mask.float()
            prob=scores.softmax(-1)
            return logits,prob,prob @ vv.repeat_interleave(repeat,0)
        base=evaluate(k,v)
        for method,kk,vv in [('v_only',k,quant(v,128)),
                             ('k4_v0',quant(k,128,4),quant(v,128)),
                             ('k4_v4',quant(k,128,4),quant(v,128,4)),
                             ('k_group32_top4_v0',quant(k,32,4),quant(v,128))]:
            actual=evaluate(kk,vv)
            metrics={name:error(a,b) for name,a,b in zip(('logits','probability','output'),actual,base)}
            metrics['k']=error(kk,k);metrics['v']=error(vv,v)
            from_k=(actual[1]-base[1]) @ v.repeat_interleave(repeat,0)
            from_v=actual[1] @ (vv-v).repeat_interleave(repeat,0)
            metrics.update(k_output_component_max=from_k.abs().max().item(),
                v_output_component_max=from_v.abs().max().item(),
                probability_tv_max=((actual[1]-base[1]).abs().sum(-1)*.5).max().item())
            rows.append(dict(file=file.name,case=x['case'],layer=x['layer'],step=x['step'],method=method,
                metrics=metrics,passed=metrics['output']['max_abs']<=.20 and metrics['output']['nrmse']<=.05))
        print(file.name,flush=True)
    (target/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,total=len(subset),passed=sum(r['passed'] for r in subset),
            max_abs=max(r['metrics']['output']['max_abs'] for r in subset),
            max_nrmse=max(r['metrics']['output']['nrmse'] for r in subset)))
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    (target/'metadata.json').write_text(json.dumps(dict(torch=torch.__version__,device='cpu',
        capture_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),thresholds=dict(max_abs=.20,nrmse=.05)),indent=2))


if __name__=='__main__':
    main()
