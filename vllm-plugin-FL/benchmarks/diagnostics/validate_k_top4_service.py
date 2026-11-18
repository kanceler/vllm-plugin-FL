"""Model adapter gate: shuffled pages, mixed lengths, padding and graph replay."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import validate_k_top4 as check
from vllm_fl.attention import k_top4
from vllm_fl.attention.k_top4_service import KTop4Impl, select_config


def main():
    root=Path(sys.argv[1]); root.mkdir(exist_ok=False)
    torch.manual_seed(20261128)
    impl=KTop4Impl(16,128,128**-.5,2,None,None,'auto')
    rows=[]
    for batch,seqlen,qlen,seg in [(1,129,1,True),(8,257,1,True),(16,513,1,True),(65,129,1,False),(4,257,17,False)]:
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__: captured.update(frame.f_locals)
        sys.setprofile(profile)
        try: check.run(batch,seqlen,qlen,seg)
        finally: sys.setprofile(None)
        c=captured
        expected=torch.zeros_like(c['cache']); k_top4.write(c['k'],c['v'],expected,c['slots'])
        actual=torch.zeros_like(expected)
        impl.do_kv_cache_update(None,c['k'],c['v'],actual,c['slots'])
        assert torch.equal(expected,actual),'writer bytes'
        args=c['args']
        lengths=torch.arange(batch,device='cuda',dtype=torch.int32)%3
        lengths=seqlen-lengths
        m=SimpleNamespace(num_actual_tokens=batch*qlen,query_start_loc=args['cu_seqlens_q'],
            max_query_len=qlen,max_seq_len=seqlen,seq_lens=lengths,causal=True,use_cascade=False,
            block_table=c['table'],seq_threshold_3D=64 if seg else None,
            num_par_softmax_segments=32,
            softmax_segm_output=torch.empty((max(batch,64),16,32,128),device='cuda'),
            softmax_segm_max=torch.empty((max(batch,64),16,32),device='cuda'),
            softmax_segm_expsum=torch.empty((max(batch,64),16,32),device='cuda'),mm_prefix_range_tensor=None)
        output=torch.full((batch*qlen+3,16,128),-3.,device='cuda',dtype=torch.bfloat16)
        def call(): impl.forward(None,c['q'],None,None,actual,m,output)
        def reference():
            result=[]; originals=[]
            for i,n in enumerate(lengths.tolist()):
                q=c['q'][i*qlen:(i+1)*qlen]
                result.append(check.reference(q,c['dqk'][i*seqlen:i*seqlen+n],c['dqv'][i*seqlen:i*seqlen+n],qlen,n,1))
                originals.append(check.reference(q,c['k'][i*seqlen:i*seqlen+n],c['v'][i*seqlen:i*seqlen+n],qlen,n,1))
            return torch.cat(result),torch.cat(originals)
        for _ in range(3): call()
        torch.cuda.synchronize()
        ref,original=reference(); err=check.error(output[:batch*qlen],ref); total=check.error(output[:batch*qlen],original)
        assert err['max_abs']<=.08 and err['nrmse']<=.02,err
        assert total['max_abs']<=.20 and total['nrmse']<=.05,total
        assert (output[batch*qlen:]==-3).all(),'output padding overwritten'
        graph_error=None
        if qlen==1:
            # Match the service: capture at length 1, then replay mixed longer histories.
            saved=lengths.clone();lengths.fill_(1)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): call()
            lengths.copy_(saved); graph.replay();torch.cuda.synchronize()
            graph_error=check.error(output[:batch],ref)
            assert graph_error['max_abs']<=.08 and graph_error['nrmse']<=.02,graph_error
        rows.append(dict(case=[batch,seqlen,qlen,seg],config=select_config(batch,qlen,0 if qlen==1 else seqlen,seg),error=err,total_error=total,graph_error=graph_error))
        (root/'results.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
    (root/'COMPLETE').write_text('5 adapter cases passed\n')

if __name__=='__main__':main()
