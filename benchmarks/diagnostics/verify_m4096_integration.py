from pathlib import Path
import argparse,torch,triton,triton.testing,importlib,json,random,statistics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);r=p.parse_args().output;r.mkdir()
import flag_gems
from vllm_fl.dispatch.backends.vendor.metax.nt_gemm import apply_c500_nt_gemm,_C500_NT_TILES
assert len(_C500_NT_TILES)==6 and apply_c500_nt_gemm(torch.device('cuda:0'))
mm=importlib.import_module(flag_gems.mm.__module__)
b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
torch.backends.cuda.matmul.allow_tf32=False
def emit(row):
 with (r/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
for seed in [20261208,20261209]:
 torch.manual_seed(seed)
 for m in [2048,4096]:
  for n,k in [(12288,2048),(2560,2048),(2048,2048),(2048,6144)]:
   a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
   if k==6144:
    scale=w.float().abs().amax(1).clamp_min(1e-8)/127;q=(w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
    ref=(a.float()@q.float().T)*scale
    def new():return b._shape_dispatch(a,q,scale)
   else:
    ref=a.float()@w.float().T
    def new():return flag_gems.mm(a,w.t())
   y=new();torch.cuda.synchronize();l2=float((y.float()-ref).norm()/ref.norm());assert l2<.005
   g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();torch.cuda.synchronize()
   with torch.cuda.graph(g,stream=stream):z=new()
   g.replay();torch.cuda.synchronize();gl2=float((z.float()-ref).norm()/ref.norm());assert gl2<.005
   row=dict(seed=seed,shape=[m,n,k],l2=l2,graph_l2=gl2,tile=_C500_NT_TILES.get((m,n,k)),quant=k==6144)
   if m==4096 and k!=6144:
    wrapped=mm.general_mm_nt
    def old():
     mm.general_mm_nt=wrapped.__wrapped__
     try:return flag_gems.mm(a,w.t())
     finally:mm.general_mm_nt=wrapped
    old();torch.cuda.synchronize()
    vals={'new':[],'old':[]}
    for rep in range(7):
     jobs=[('new',new),('old',old)];random.Random(seed+rep).shuffle(jobs)
     for name,fn in jobs:vals[name].append(triton.testing.do_bench(fn,warmup=20,rep=100))
    row['rounds_ms']=vals;row['medians_ms']={name:statistics.median(v) for name,v in vals.items()}
   emit(row)
# Device-only 003 prefix construction versus the original mathematical result.
lengths=torch.tensor([1,7,64,4096],device='cuda',dtype=torch.int32)
prefix=torch.cat([lengths.new_zeros(1),lengths]).cumsum(0,dtype=torch.int32)
assert prefix.cpu().tolist()==[0,1,8,72,4168]
(r/'complete.txt').write_text('6 native mappings and Down old/new shapes validated; model validation still required')
