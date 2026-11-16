"""Four same-input entry points, hot graph replay vs original cache-clearing bench."""
import hashlib,importlib,inspect,json,os,random,statistics,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
(root/'gpu-before.txt').write_text(gpu);assert 'No running processes found' in gpu
import torch
import triton
import triton.language as tl
import triton.testing
from vllm_fl.quantization.w8a16 import policy
from vllm_fl.quantization.w8a16.reference import quantize_weight
from tianshu_muxi_splitk import run as old_splitk
assert policy.policy_matches(0)
provider=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
kernel=provider.mm_kernel
while not isinstance(kernel,triton.runtime.JITFunction):kernel=kernel.fn
testing_source=inspect.getsource(triton.testing.do_bench)
(root/'do-bench-source.txt').write_text(testing_source)
(root/'identity.json').write_text(json.dumps(dict(provider=provider.__file__,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
    old_splitk_source=inspect.getfile(old_splitk),do_bench_module=inspect.getfile(triton.testing.do_bench),
    clear_cache_in_do_bench='clear_cache' in testing_source,db=os.environ.get('FLAGGEMS_DB_URL'),
    note='GPU cache clearing is diagnostic-only, paired for every path; persistent compile caches retained'),indent=2))
torch.backends.cuda.matmul.allow_tf32=False
rng=random.Random(20261113)
for seed in (20261114,20261115):
    torch.manual_seed(seed);m,n,k=64,2048,6144
    a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    q,s=quantize_weight(w)
    choice=policy.choose(0,m,n,k,0)
    def fixed():
        y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        kernel[(triton.cdiv(m,64)*triton.cdiv(n,64),1)](a,w.T,y,m,n,k,k,1,1,k,n,1,
            acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
            BLOCK_M=64,BLOCK_N=64,BLOCK_K=128,GROUP_M=8,SPLIT_K=1,EVEN_K=True,
            AB_DTYPE=tl.bfloat16,UPGRADE=False,UPGRADE_A_OFFS=False,UPGRADE_B_OFFS=False,
            UPGRADE_C_OFFS=False,num_warps=16,num_stages=2)
        return y
    fns=dict(historical_fixed_bf16=fixed,current_public_bf16=lambda:provider.mm(a,w.T),
             historical_splitk=lambda:old_splitk(a,q,s,(64,64,64,8,1,4),('vector',256)),
             current_policy=lambda:policy.dispatch(a,q,s,0))
    refs=dict(bf16=a.float()@w.float().T,quantized=(a.float()@q.float().T)*s)
    graphs={};errors={}
    for name,fn in fns.items():
        for _ in range(3):fn()
        y=fn();ref=refs['bf16' if 'bf16' in name else 'quantized']
        err=((y.float()-ref).norm()/ref.norm()).item();assert err<=.005,(name,err);errors[name]=err
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):output=fn()
        g.replay();torch.cuda.synchronize();graphs[name]=g
    times={mode:{name:[] for name in fns} for mode in ('hot','cache_cleared')}
    for repeat in range(9):
        order=list(graphs);rng.shuffle(order)
        for name in order:
            # Rewarm this path immediately before timing to avoid the previous
            # path's working set determining the nominal hot condition.
            for _ in range(3):graphs[name].replay()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):graphs[name].replay()
            end.record();end.synchronize();times['hot'][name].append(start.elapsed_time(end)/10)
        order=list(graphs);rng.shuffle(order)
        for name in order:
            timing=triton.testing.do_bench(graphs[name].replay,warmup=5,rep=30,quantiles=[.5])
            times['cache_cleared'][name].append(timing[0] if isinstance(timing,(list,tuple)) else timing)
    record=dict(seed=seed,shape=[m,n,k],choice=choice,errors=errors,round_ms=times,
                median_ms={mode:{name:statistics.median(values) for name,values in paths.items()} for mode,paths in times.items()})
    with (root/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(record['median_ms']),flush=True)
    del graphs
(root/'COMPLETE').write_text('Two-seed same-process cache-condition comparison complete\n')
