"""Service-vs-frozen operator checks; synthetic .0625 tables only in this fixture."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import validate_k_top4 as check
import k_static_fold_frozen as frozen
import k_static_protect_fold_candidate as original
from benchmark_k_top4_selected_confirm import graph
from vllm_fl.attention.k_static_fold_service import KStaticFoldImpl,KStaticFoldBackend
from vllm_fl.attention.k_static_fold_policy import choose
from vllm_fl.attention import k_static_fold_kernel as kernel


def main():
    root=Path(sys.argv[1]);root.mkdir(exist_ok=False);torch.manual_seed(20261218)
    assert KStaticFoldBackend.forward_includes_kv_cache_update is True
    impl=KStaticFoldImpl(16,128,128**-.5,2,None,None,'auto')
    results=[]
    cases=[(1,257,1),(56,4096,1),(64,16384,1),(72,4096,1),(80,257,1),
           (4,257,17),(1,128,128),(4,2048,512)]
    for b,n,ql in cases:
        c={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:c.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(b,n,ql,ql==1 and b<=64)
        finally:sys.setprofile(None)
        pages=(n+31)//32;raw=torch.zeros((b*pages+2,32,2,276),device='cuda',dtype=torch.uint8)
        table=torch.randperm(b*pages,device='cuda').reshape(b,pages).to(torch.int32)
        pos=torch.arange(n,device='cuda');slots=(table[:,pos//32].long()*32+pos%32).flatten()
        slots=torch.cat((slots,torch.tensor([-1],device='cuda')))
        perm,sc=original.parameters([[0,3,17,65],[0,9,37,93]],torch.full((2,128),.0625,device='cuda'))
        # Test arithmetic/call plumbing against the frozen prototype. Official
        # service startup below never uses this synthetic parameter override.
        impl._layer_index=0;impl._perm=perm;impl._scale=sc
        kernel.write(c['k'],c['v'],raw,slots,perm,sc)
        ids=torch.cat([torch.arange(i*n+n-ql,i*n+n,device='cuda') for i in range(b)])
        wk,wv,ws=c['k'][ids],c['v'][ids],slots[ids]
        qo=torch.empty_like(c['q']);output=torch.empty_like(c['q']);control=torch.empty_like(c['q'])
        lens=torch.full((b,),n,device='cuda',dtype=torch.int32)
        m=SimpleNamespace(num_actual_tokens=b*ql,max_query_len=ql,max_seq_len=n,
            query_start_loc=torch.arange(b+1,device='cuda',dtype=torch.int32)*ql,
            seq_lens=lens,slot_mapping=ws,block_table=table,causal=True,use_cascade=False,
            static_q_workspace=qo,softmax_segm_output=torch.empty((80,16,32,128),device='cuda'),
            softmax_segm_max=torch.empty((80,16,32),device='cuda'),softmax_segm_expsum=torch.empty((80,16,32),device='cuda'),mm_prefix_range_tensor=None)
        layer=SimpleNamespace(layer_name='model.layers.0.self_attn.attn')
        args=dict(c['args'],out=control,block_table=table,seqused_k=lens)
        bound=frozen.make_call(b,ql,64,c['q'],wk,wv,raw,ws,perm,sc,args)
        bound();impl.forward(layer,c['q'],wk,wv,raw,m,output);torch.cuda.synchronize()
        err=check.error(output,control)
        assert err['max_abs']<=.08 and err['nrmse']<=.02,err
        g=graph(lambda:impl.forward(layer,c['q'],wk,wv,raw,m,output))
        g.replay();torch.cuda.synchronize();assert torch.equal(output,control)
        if ql==1:
            lens.fill_(1);grow=graph(lambda:impl.forward(layer,c['q'],wk,wv,raw,m,output))
            lens.fill_(n);grow.replay();torch.cuda.synchronize();assert torch.equal(output,control)
        results.append(dict(case=[b,n,ql],error=err,bitwise_equal=torch.equal(output,control)))
        (root/'raw.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results[-1]),flush=True)
    (root/'COMPLETE').write_text('8 production entrypoint vs frozen prototype cases; no actual model accuracy claim\n')


if __name__=='__main__':main()
