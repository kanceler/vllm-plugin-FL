"""Standalone BF16 HF diagnostic, no production plugin changes or performance claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

CALIBRATION_TEXTS = {
 'zh_a':'博物馆整理藏品时需要记录材料、年代、修复经历和存放条件。请说明志愿者如何核对目录，遇到缺失记录时如何标记不确定性，并安排后续检查。',
 'zh_b':'社区准备修建一条步行路线，沿线有公园、车站和居民区。讨论路标、排水、无障碍通行与日常养护如何协调，以及怎样保存公开反馈。',
 'en_a':'A field station keeps records of soil moisture and wind. Describe a careful procedure for instrument maintenance, checking timestamps, reporting missing observations, and explaining the limits of a seasonal comparison. ',
 'en_b':'An archive receives letters from several decades. Explain how to label folders, preserve the original order, record uncertain dates, and help readers distinguish a transcription from an interpretation. ',
 'code_a':'Review a small program that merges sorted lists: compare the first unused entries, append the smaller entry, then append the remaining suffix. Discuss ties, empty lists, invariants, memory use and tests without relying on a particular library. ',
 'code_b':'Describe a queue implementation with a circular buffer. Explain the roles of head, tail and capacity, how to distinguish full from empty, and what should happen when validation rejects an input. ',
 'symbol_a':'Discuss the area of a parallelogram symbolically in terms of its base and perpendicular height. Explain why a slanted side is not generally the height and how a geometric dissection preserves area. ',
 'symbol_b':'Explain why a finite sum can be grouped into consecutive blocks without changing its value. State assumptions about indices and illustrate the reasoning using letters rather than a fixed numeric exercise. ',
}
VALIDATION_TEXTS = {
 'hold_zh':'自然保护区需要给巡护员制定雨季记录流程。请比较纸质表格和电子表格，分析信息延迟、设备故障和复核责任，提出一种透明且容易维护的安排。',
 'hold_en':'A neighborhood workshop lends tools to residents. Propose a system for checking equipment, arranging returns, handling conflicting reports, and publishing maintenance notes with clear distinctions between observations and assumptions. ',
 'hold_code':'Explain a parser that reads a stream of bracket tokens using a stack. Discuss unmatched openings, unmatched closings, empty input and how to document the position of an error. ',
 'hold_symbol':'Describe a symbolic argument for the volume of a prism from parallel cross sections, carefully distinguishing a length, an area and a volume, and explain which assumptions make the formula applicable. ',
}



def main():
    root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True)
    (root/'gpu-before.txt').write_text(gpu)
    (root/'processes-before.txt').write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))
    if 'No running processes found' not in gpu:
        raise RuntimeError('GPU occupied; no model loaded')
    torch.set_num_threads(2)
    torch.manual_seed(20261113)
    model_path = '/workspace/MiniCPM5-2B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    sdpa=ALL_ATTENTION_FUNCTIONS['sdpa'];context={};stats={};validation=[]
    for layer in range(42):stats[layer]=dict(maximum=torch.zeros(2,128),counts=torch.zeros(2,128,dtype=torch.int64),tokens=0)
    def round_int(x):return torch.where(x>=0,x+.5,x-.5).clamp(-128,127).to(torch.int8)
    def metric(a,b):
        z=a.float()-b.float();return dict(max_abs=z.abs().max().item(),nrmse=(z.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8)).item())
    def capture(module,query,key,value,attention_mask,**kwargs):
        layer=module.layer_idx
        if context['phase']=='calibration':
            # Only new tokens count on Decode; never duplicate the whole prefix.
            kk=key[0] if context['step']==0 else key[0,:,-1:]
            cpu=kk.detach().float().cpu();entry=stats[layer]
            entry['maximum']=torch.maximum(entry['maximum'],cpu.abs().amax(1))
            idx=torch.argsort(cpu.abs(),dim=-1,descending=True,stable=True)[...,:4].reshape(2,-1)
            entry['counts'].scatter_add_(1,idx,torch.ones_like(idx));entry['tokens']+=cpu.shape[1]
        elif context['step'] in (0,8):
            # Held-out diagnostics: BF16 trajectory, no quantization feedback.
            count=min(4,query.shape[-2]);q=query[0,:,-count:];k=key[0];v=value[0]
            scale=params[layer]['scale_gpu'];perm=params[layer]['perm_gpu'];rep=8
            kp=k.gather(-1,perm[:,None,:].expand_as(k).long())
            qi=round_int(kp.float()/scale[:,None,:]).float();qi[...,:4]=kp[...,:4].float()
            eff=scale.clone();eff[:,:4]=1
            qp=q.gather(-1,perm.repeat_interleave(rep,0)[:,None,:].expand_as(q).long())
            qp=(qp.float()*eff.repeat_interleave(rep,0)[:,None,:]).to(torch.bfloat16).float()
            sv=(v.float().abs().amax(-1,keepdim=True)/127).clamp_min(1e-6)
            dv=(round_int(v.float()/sv).float()*sv).to(torch.bfloat16).float()
            pos=torch.arange(k.shape[1]-count,k.shape[1],device=q.device)
            valid=torch.arange(k.shape[1],device=q.device)[None,:]<=pos[:,None]
            def evaluate(qq,kk,vv):
                score=(qq.float()@kk.float().repeat_interleave(rep,0).transpose(-1,-2))*kwargs['scaling']
                score=score.masked_fill(~valid,float('-inf'))
                if attention_mask is not None:
                    mask=attention_mask[0,:,-count:]
                    score=score.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else score+mask.float()
                return score.softmax(-1).to(torch.bfloat16).float()@vv.float().repeat_interleave(rep,0)
            err=metric(evaluate(qp,qi,dv),evaluate(q,k,v))
            row=dict(case=context['case'],step=context['step'],layer=layer,metrics=err,passed=err['max_abs']<=.20 and err['nrmse']<=.05)
            validation.append(row)
            with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return sdpa(module,query,key,value,attention_mask,**kwargs)
    ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    config=Path(model_path,'config.json');assert model.config.num_hidden_layers==42
    params={}
    def run(text,length,name):
        tokens=tokenizer.encode(text,add_special_tokens=False);ids=(tokens*(length//len(tokens)+1))[:length]
        (root/(name+'-tokens.json')).write_text(json.dumps(ids));context.update(case=name,step=0)
        output=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True,return_dict=True)
        for step in range(1,9):
            nxt=model.lm_head(output.last_hidden_state[:,-1:]).argmax(-1);context['step']=step
            output=model.model(input_ids=nxt,past_key_values=output.past_key_values,use_cache=True,return_dict=True)
        del output
        print(json.dumps(dict(phase=context['phase'],case=name,finished=True)),flush=True)
    with torch.inference_mode():
        context['phase']='calibration'
        cases=[(name,length) for name in CALIBRATION_TEXTS for length in (512,4096)]+[('en_a',16384)]
        for name,length in cases:run(CALIBRATION_TEXTS[name],length,f'cal-{name}-{length}')
        records=[]
        for layer in range(42):
            entry=stats[layer];fixed=torch.argsort(entry['counts'],dim=-1,descending=True,stable=True)[:,:4]
            perm=torch.tensor([x+[i for i in range(128) if i not in x] for x in fixed.tolist()],dtype=torch.int32)
            scale=(entry['maximum']/127).clamp_min(1e-6)*2
            packed=scale.gather(1,perm.long()).contiguous()
            params[layer]=dict(perm_gpu=perm.cuda(),scale_gpu=packed.cuda())
            records.append(dict(layer=layer,protected_channels=fixed.tolist(),permutation=perm.tolist(),scale=packed.tolist(),tokens=entry['tokens']))
        weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(model_path).glob('*.safetensors')}
        calibration=dict(version=1,model=model_path,model_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),weights_sha256=weights,margin=2,format='permuted per-channel FP32 scale; first4 BF16 protected',layers=records,calibration_texts=CALIBRATION_TEXTS,calibration_cases=cases,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,transformers=transformers.__version__,limitation='independent synthetic prose, BF16 HF trajectory; model accuracy must be tested separately')
        (root/'calibration.json').write_text(json.dumps(calibration,indent=2))
        context['phase']='validation'
        cases=[(name,length) for name in VALIDATION_TEXTS for length in (512,4096)]+[('hold_en',16384)]
        for name,length in cases:run(VALIDATION_TEXTS[name],length,f'val-{name}-{length}')
    assert len(validation)==9*42*2
    summary=dict(count=len(validation),passed=sum(x['passed'] for x in validation),max_abs=max(x['metrics']['max_abs'] for x in validation),max_nrmse=max(x['metrics']['nrmse'] for x in validation),validation_texts=VALIDATION_TEXTS,calibration_sha256=hashlib.sha256((root/'calibration.json').read_bytes()).hexdigest())
    (root/'validation-summary.json').write_text(json.dumps(summary,indent=2))
    (root/'COMPLETE').write_text('42-layer independent calibration and held-out numeric screening completed; no official accuracy claim\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
