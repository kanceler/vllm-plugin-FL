"""Independent acceptance of config-only winners through the production interface."""
import argparse
import importlib
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton.testing
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    configs={56:(64,64,128,4,1,16),112:(64,128,64,4,1,16)}
    def emit(name,row):
        with (args.output/name).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    passed=True
    for seed in [20261113,20261114]:
        torch.manual_seed(seed)
        w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
        scale=w.float().abs().amax(1).clamp_min(1e-8)/127
        q=(w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        # Exhaustive integer M coverage in the small interval; includes held-out tails.
        for m in range(1,114):
            a=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
            ref=(a.float()@q.float().T)*scale
            fn=(lambda: b._configured_splitk(a,q,scale,configs[m])) if m in configs else (lambda:b._shape_dispatch(a,q,scale))
            y=fn();torch.cuda.synchronize()
            l2=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert l2<0.005
            emit('correctness.jsonl',{'m':m,'seed':seed,'l2':l2,'candidate':m in configs})
            if m not in configs:continue
            jobs={'baseline':lambda:b._shape_dispatch(a,q,scale),
                  'candidate':lambda:b._configured_splitk(a,q,scale,configs[m])}
            graphs={};outputs={}
            for name,f in jobs.items():
                f();torch.cuda.synchronize();g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
                with torch.cuda.graph(g,stream=stream): out=f()
                g.replay();torch.cuda.synchronize()
                err=float((out.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert err<0.005
                graphs[name]=g;outputs[name]=out
            # Alternate randomized order every round in both eager and graph timing.
            eager={x:[] for x in jobs};graph={x:[] for x in jobs}
            for repeat in range(7):
                order=list(jobs);random.Random(seed+repeat).shuffle(order)
                for name in order:
                    eager[name].append(triton.testing.do_bench(jobs[name],warmup=25,rep=120))
                    graph[name].append(triton.testing.do_bench(graphs[name].replay,warmup=25,rep=120))
            ee={x:statistics.median(v) for x,v in eager.items()};gg={x:statistics.median(v) for x,v in graph.items()}
            gains={'eager':1-ee['candidate']/ee['baseline'],'graph':1-gg['candidate']/gg['baseline']}
            ok=all(x>0.01 for x in gains.values());passed=passed and ok
            emit('timing.jsonl',{'m':m,'seed':seed,'config':configs[m],
                 'eager_ms':eager,'graph_ms':graph,'medians_eager_ms':ee,
                 'medians_graph_ms':gg,'latency_reduction':gains,'accepted':ok})
    (args.output/'complete.json').write_text(json.dumps({'all_accepted':passed,
        'threshold_l2':0.005,'scope':'M1..113 two seeds; only M56/M112 changed; no model-level claim'}))


if __name__=='__main__':main()
