"""Bounded M56 Dense pipeline experiment; no production dispatch updates."""
import argparse
import importlib
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton
    import triton.testing
    import flag_gems
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    b = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    from vllm_fl.dispatch.backends.vendor.metax.nt_gemm import apply_c500_nt_gemm
    assert apply_c500_nt_gemm(torch.device('cuda:0'))
    configs = [(32,64,128,4,2),(64,32,128,8,2),(64,64,64,8,2),
               (64,64,64,8,4),(64,128,64,4,2),(64,128,64,8,2),
               (64,128,64,4,4),(64,64,128,4,2),(64,64,128,4,4),
               (32,128,64,4,2),(32,128,128,4,3),(64,128,128,4,4)]
    log = (args.output/'timings.jsonl').open('w')
    for n in [12288,2560,2048]:
        for seed in [20261110,20261111]:
            torch.manual_seed(seed)
            a=torch.randn(56,2048,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,2048,device='cuda',dtype=torch.bfloat16)
            s=w.float().abs().amax(1).clamp_min(1e-8)/127
            q=(w.float()/s[:,None]).round().clamp(-127,127).to(torch.int8)
            reference=(a.float()@q.float().T)*s
            jobs=[('bf16',None,None,lambda:flag_gems.mm(a,w.t())),
                  ('dispatch',None,None,lambda:b._shape_dispatch(a,q,s)),
                  ('separate',None,None,lambda:b._large_m_separate(a,q,s)),
                  ('historical_splitk',(64,64,128,4,1,16),'basic',lambda:b.run_w8a16_algorithm(a,q,s,'splitk',(64,64,128,4,1,16)))]
            for cfg in configs:
                for pipeline in ['basic','cpasync']:
                    jobs.append(('dense',cfg,pipeline,lambda cfg=cfg,pipeline=pipeline:b._configured_dense(a,q,s,cfg,pipeline=pipeline)))
            valid=[]
            for name,cfg,pipeline,fn in jobs:
                row={'m':56,'n':n,'k':2048,'seed':seed,'algorithm':name,'config':cfg,'pipeline':pipeline}
                try:
                    y=fn();torch.cuda.synchronize()
                    ref=a.float()@w.float().T if name=='bf16' else reference
                    row['relative_l2']=float((y.float()-ref).norm()/ref.norm())
                    assert row['relative_l2']<0.005
                    row['screen_ms']=triton.testing.do_bench(fn,warmup=20,rep=70)
                    valid.append((row,fn))
                except Exception as e:
                    row['failure']=repr(e)
                log.write(json.dumps(row)+'\n');log.flush()
            finalists=sorted(valid,key=lambda z:z[0]['screen_ms'])[:6]
            finalists += [x for x in valid if x[0]['algorithm']!='dense' and x not in finalists]
            samples={id(row):[] for row,_ in finalists}
            for repeat in range(5):
                order=list(finalists);random.Random(repeat).shuffle(order)
                for row,fn in order:samples[id(row)].append(triton.testing.do_bench(fn,warmup=30,rep=150))
            for row,fn in finalists:
                row['final_ms']=samples[id(row)];row['median_ms']=statistics.median(row['final_ms'])
                try:
                    graph=torch.cuda.CUDAGraph()
                    for _ in range(3):fn()
                    torch.cuda.synchronize()
                    # MACA requires capture on a non-default stream.
                    stream=torch.cuda.Stream()
                    with torch.cuda.graph(graph, stream=stream):output=fn()
                    graph.replay();torch.cuda.synchronize()
                    ref=a.float()@w.float().T if row['algorithm']=='bf16' else reference
                    row['graph_relative_l2']=float((output.float()-ref).norm()/ref.norm());assert row['graph_relative_l2']<0.005
                    row['graph_ms']=triton.testing.do_bench(graph.replay,warmup=30,rep=150)
                except Exception as e:row['graph_failure']=repr(e)
                with (args.output/'finalists.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(n,seed,[(row['algorithm'],row['pipeline'],row['median_ms']) for row,_ in finalists],flush=True)
    (args.output/'complete.txt').write_text('complete; inspect failures and both seeds before selecting')


if __name__=='__main__':
    main()
