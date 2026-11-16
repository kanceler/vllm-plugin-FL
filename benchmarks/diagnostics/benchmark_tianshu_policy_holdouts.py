"""Graph-only paired holdouts for fixed runtime policies; no candidate search."""
import importlib
import json
import os
import random
import sys
from pathlib import Path

for group in ('DOWN','GATE_UP','QKV','O'):
    os.environ['VLLM_FL_TIANSHU_W8A16_'+group]='1'
import torch
import triton
import triton.language as tl
from vllm_fl.quantization.w8a16.policy import choose,policy_matches,load_policy
from vllm_fl.quantization.w8a16.reference import quantize_weight
from vllm_fl.quantization.w8a16.separate import separate_linear

out=Path(sys.argv[1]); out.mkdir(exist_ok=False)
summary=json.loads(Path(sys.argv[2]).read_text())
assert policy_matches(0)
provider=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
kernel=provider.mm_kernel
while not isinstance(kernel,triton.runtime.JITFunction):kernel=kernel.fn
torch.backends.cuda.matmul.allow_tf32=False
rng=random.Random(20261117)

for group,(name,n,k) in enumerate((('down',2048,6144),('gate_up',12288,2048),
                                  ('qkv',2560,2048),('o',2048,2048))):
    torch.manual_seed(20261117+group)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    q,s=quantize_weight(w)
    ms={2,4,8,16,32,96,192,384,768,1536,3072}
    for left,right in load_policy()['groups'][name]['transition_intervals']:
        ms.update((left+1,right-1))
    for m in sorted(ms):
        a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        candidates=[r for r in summary if r['projection']==name and r['algorithm']=='native']
        baseline=min(candidates,key=lambda r:(abs(r['m']-m),r['graph']['median_us']))
        cfg=baseline['config']
        def native():
            bm,bn,bk,warps,stages=cfg
            y=torch.empty(m,n,device=a.device,dtype=a.dtype)
            kernel[(triton.cdiv(m,bm)*triton.cdiv(n,bn),1)](
                a,w.T,y,m,n,k,k,1,1,k,n,1,acc_dtype=tl.float32,input_precision=None,
                fp8_fast_accum=True,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=8,
                SPLIT_K=1,EVEN_K=k%bk==0,AB_DTYPE=tl.bfloat16,UPGRADE=False,
                UPGRADE_A_OFFS=m*k>2**31-1,UPGRADE_B_OFFS=n*k>2**31-1,
                UPGRADE_C_OFFS=m*n>2**31-1,num_warps=warps,num_stages=stages)
            return y
        functions=[]
        for label,fn in [('runtime_quant',lambda:separate_linear(a,q,s,None,group)),('bf16_control',native)]:
            y=fn(); torch.cuda.synchronize()
            ref=(a.float()@w.float().T if label=='bf16_control' else (a.float()@q.float().T)*s)
            err=y.float()-ref
            l2=(err.norm()/ref.norm().clamp_min(1e-12)).item()
            mx=(err.abs().max()/ref.abs().max().clamp_min(1e-12)).item()
            assert torch.isfinite(y).all() and l2<=.005 and mx<=.01,(name,m,label,l2,mx)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):yg=fn()
            graph.replay();torch.cuda.synchronize()
            functions.append((label,graph,yg,l2,mx))
        for repeat in range(3):
            seq=functions.copy();rng.shuffle(seq)
            for label,graph,output,l2,mx in seq:
                timing=triton.testing.do_bench(graph.replay,warmup=5,rep=20,quantiles=[.5])
                us=float(timing[0] if isinstance(timing,(tuple,list)) else timing)*1000
                row=dict(group=name,m=m,repeat=repeat,label=label,graph_us=us,l2=l2,normalized_max=mx,
                         choice=choose(group,m,n,k,0),bf16_config=cfg,
                         bf16_source_m=baseline['m'],selection='fixed policy vs nearest measured native config; no search')
                with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('HOLDOUT_DONE',name,m,flush=True)
(out/'COMPLETE').write_text('done\n')
