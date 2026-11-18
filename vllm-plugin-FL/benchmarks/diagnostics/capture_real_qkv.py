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

TEXTS = {
 'zh': '城市图书馆正在规划新阅览室。请比较自然采光、通风、安静程度与日常维护成本，并说明如何记录读者反馈。不同区域需要兼顾儿童与成年人的阅读需求。',
 'en': 'A coastal town maintains a public weather archive. Explain how volunteers can record observations, check missing entries, compare seasonal patterns, and communicate uncertainty without confusing measurements with predictions. ',
 'code': 'Review this Python routine and explain its boundary cases: def rolling_sum(values, width):\n    return [sum(values[i:i+width]) for i in range(max(0, len(values)-width+1))]\nDiscuss empty inputs, negative values, and validation of width. ',
 'math': 'Consider a rectangular garden with a fixed perimeter. Explain symbolically how its area depends on its width, why the square maximizes the area, and how the argument changes when one side borders a wall. Do not assume numeric measurements. ',
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
    sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']
    context = {}
    records = []

    def capture(module, query, key, value, attention_mask, **kwargs):
        if module.layer_idx in (0, 10, 20, 30, 41) and context['step'] in (0, 4):
            if attention_mask is not None and attention_mask.ndim != 4:
                raise RuntimeError('Unsupported attention mask')
            count = min(4, query.shape[-2])
            positions = torch.arange(key.shape[-2]-count, key.shape[-2])
            name = f"{context['case']}-step{context['step']}-layer{module.layer_idx}.pt"
            payload = dict(q=query[0, :, -count:].detach().cpu().contiguous(),
                           k=key[0].detach().cpu().contiguous(),
                           v=value[0].detach().cpu().contiguous(),
                           positions=positions, scaling=float(kwargs['scaling']),
                           layer=module.layer_idx, case=context['case'], step=context['step'])
            if attention_mask is not None:
                payload['mask'] = attention_mask[0, :, -count:].detach().cpu().contiguous()
            torch.save(payload, root/name)
            records.append(dict(file=name, q=list(payload['q'].shape), k=list(payload['k'].shape),
                                sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()))
            (root/'manifest.json').write_text(json.dumps(records, indent=2))
        return sdpa(module, query, key, value, attention_mask, **kwargs)

    # Only this short-lived process receives a wrapped existing SDPA interface.
    ALL_ATTENTION_FUNCTIONS.register('sdpa', capture)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    (root/'metadata.json').write_text(json.dumps(dict(torch=torch.__version__,
        transformers=transformers.__version__, device=torch.cuda.get_device_name(),
        model=model_path, config_sha256=hashlib.sha256(Path(model_path,'config.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layers=[0,10,20,30,41], prompt_source=TEXTS,
        limitation='Repeated independent prose; HF BF16 trajectory, not production vLLM replay'),indent=2))
    cases = [(kind, length) for kind in TEXTS for length in (512,4096)] + [('en',16384)]
    with torch.inference_mode():
        for kind, length in cases:
            tokens = tokenizer.encode(TEXTS[kind], add_special_tokens=False)
            ids = (tokens * (length//len(tokens)+1))[:length]
            case = f'{kind}-{length}'
            (root/(case+'-tokens.json')).write_text(json.dumps(ids))
            context.update(case=case, step=0)
            output = model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True,return_dict=True)
            for step in range(1,5):
                next_token = model.lm_head(output.last_hidden_state[:,-1:]).argmax(-1)
                context['step'] = step
                output = model.model(input_ids=next_token,past_key_values=output.past_key_values,
                                     use_cache=True,return_dict=True)
            del output
            print('completed',case,flush=True)
    if len(records)!=90:
        raise RuntimeError(f'Unexpected capture coverage {len(records)}')
    (root/'COMPLETE').write_text('90 snapshots, 9 cases, 5 layers, prefill and decode step4\n')


if __name__=='__main__':
    main()
