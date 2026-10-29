import torch,triton.testing,importlib
from statistics import median
from pathlib import Path
from benchmark_flaggems_w8a16_full import split,check
from benchmark_flaggems_w8a16 import load_helper,dump
ref=load_helper('reference',Path('.'));deq=load_helper('dequant',Path('.'))
dense=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
torch.manual_seed(271828);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
x=torch.randn(64,6144,device='cuda',dtype=torch.bfloat16)
e=(torch.nn.functional.linear(x.float(),q.float())*s).to(torch.bfloat16)
configs={'old_s4':(16,64,128,4,2,4),'old_s8':(16,64,128,4,2,8),'new_s8':(64,64,128,4,2,8)}
paths={name:(lambda c=c:split(x,q,s,c)) for name,c in configs.items()}
errors={name:check(fn(),e) for name,fn in paths.items()}
paths['bf16']=lambda:dense.mm(x,w.t())
paths['separate']=lambda:dense.mm(x,deq.dequantize_weight(q,s).t())
for fn in paths.values():fn()
torch.cuda.synchronize();raw=[];names=list(paths)
for rep in range(3):
 for name in names[rep:]+names[:rep]:
  ms=triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
  raw.append(dict(round=rep+1,path=name,ms=ms));dump(Path('results'),'confirm_m64_raw.json',raw)
dump(Path('results'),'confirm_m64.json',dict(seed=271828,configs=configs,errors=errors,timings_ms={n:median([r['ms'] for r in raw if r['path']==n]) for n in names}))
