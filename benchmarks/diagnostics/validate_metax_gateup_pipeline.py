"""Tail, stride and graph checks for the experimental raw-weight lookahead."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    import triton
    from metax_gateup_pipeline import gateup_depth1_raw
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=int, nargs=5, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bm,bn,bk,warps,stages = args.config
    torch.manual_seed(20261112)
    shapes = [(1,17,1),(55,129,63),(56,127,65),(57,257,129),
              (64,12288,2048),(65,257,193),(128,12288,2048),(2048,257,193)]
    with (args.output/'validation.jsonl').open('w') as f:
        for m,n,k in shapes:
            for strided in [False,True]:
                a = torch.randn(m,k*(2 if strided else 1),device='cuda',dtype=torch.bfloat16)
                if strided: a = a[:,::2]
                q = torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
                s = torch.rand(n,device='cuda',dtype=torch.float32)*0.05
                s[0] = 0
                ref = (a.float()@q.float().T)*s
                def run():
                    out = torch.empty((m,n),device=a.device,dtype=a.dtype)
                    gateup_depth1_raw.run(a,q,s,out,m,n,k,*a.stride(),*q.stride(),*out.stride(),
                        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,EVEN_M=m%bm==0,EVEN_N=n%bn==0,EVEN_K=k%bk==0,
                        num_warps=warps,num_stages=stages,pipeline='basic',
                        grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),),warmup=False)
                    return out
                row={'shape':[m,n,k],'strided':strided,'config':args.config}
                try:
                    y = run();torch.cuda.synchronize()
                    row['relative_l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                    assert row['relative_l2']<0.005
                    assert torch.isfinite(y).all() and (y[:,0]==0).all()
                    stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): z=run()
                    graph.replay();torch.cuda.synchronize()
                    row['graph_l2']=float((z.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                    assert row['graph_l2']<0.005
                    row['passed']=True
                except Exception as exc:row['failure']=repr(exc)
                f.write(json.dumps(row)+'\n');f.flush();print(row,flush=True)
    (args.output/'complete.txt').write_text('finished; inspect failures')


if __name__=='__main__': main()
