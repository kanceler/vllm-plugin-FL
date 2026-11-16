"""GPU adapter, cache, graph and stale-manifest validation; no timing claims."""
import copy
import json
import sys
from pathlib import Path
import torch
import validate_k_top4 as check
from benchmark_k_top4_tuning import Tap
import k_top4_frozen_v2 as frozen
from vllm_fl.attention import k_top4


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    path=Path(__file__).parent/'configs/k_top4_biv150_frozen_v2.json'
    manifest=frozen.load_manifest(path)
    for kind in ('identity','candidate','provider','writer'):
        bad=copy.deepcopy(manifest)
        if kind=='identity':bad['identity']['torch']='invalid-test-version'
        elif kind=='candidate':bad['candidate_bindings']['k_top4_frozen.py']='0'*64
        elif kind=='writer':bad['records'][0]['writer']='invalid-test-writer'
        else:bad['source_bindings']['vllm.v1.attention.ops.triton_unified_attention']='0'*64
        target=root/f'bad-{kind}.json';target.write_text(json.dumps(bad))
        try:frozen.load_manifest(target)
        except ValueError:pass
        else:raise AssertionError(f'{kind} guard accepted stale manifest')
    from vllm_fl.attention.tile_policy import _PREFILL_TILE
    torch.manual_seed(20261127)
    rows=[]
    for case in [(16,4096,1,True,False),(65,16384,1,False,False),(4,2048,512,False,False),(1,17,1,False,True)]:
        _PREFILL_TILE["value"]=None
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:captured.update(frame.f_locals)
        sys.setprofile(profile)
        try:check.run(*case)
        finally:sys.setprofile(None)
        c=captured;b,n,q,seg,_=case
        record=next(r for r in manifest['records'] if r['case']==list(case))
        cfg=record['config'];args=dict(c['args'])
        if seg:
            s=cfg['segments'];args.update(num_par_softmax_segments=s,
                softmax_segm_output=torch.empty((b,16,s,128),device='cuda'),
                softmax_segm_max=torch.empty((b,16,s),device='cuda'),
                softmax_segm_expsum=torch.empty((b,16,s),device='cuda'))
        ref=check.reference(c['q'],c['dqk'],c['dqv'],q,n,b)
        total=check.reference(c['q'],c['k'][:-1],c['v'][:-1],q,n,b)
        expected=torch.zeros_like(c['cache']);k_top4.write(c['k'],c['v'],expected,c['slots'])
        for writer in ('load_w1','load_w2','load_w4','sum_w1','sum_w4','direct_load_w1'):
            directory=root/('case'+str(len(rows)));directory.mkdir()
            actual=torch.zeros_like(expected)
            compiled=frozen.write(c['k'],c['v'],actual,c['slots'],writer)
            assert torch.equal(actual,expected),(case,writer,'cache')
            del actual
            fn=frozen.launcher(cfg)
            tap=Tap(fn.__globals__['kernel_unified_attention'],'attention',directory)
            fn.__globals__['kernel_unified_attention']=tap
            # Poison legacy Prefill state: the explicit frozen launcher must ignore it.
            from vllm_fl.attention.tile_policy import _PREFILL_TILE
            _PREFILL_TILE['value']=128
            def full():
                frozen.write(c['k'],c['v'],c['cache'],c['slots'],writer)
                fn(**args)
            for _ in range(3):full()
            torch.cuda.synchronize();before=args['out'].clone()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):full()
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(before,args['out']),(case,writer,'graph')
            err=check.error(args['out'],ref);qe=check.error(args['out'],total)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(case,writer,err)
            assert qe['max_abs']<=.20 and qe['nrmse']<=.05,(case,writer,qe)
            assert tap.resource['launch']['TILE_SIZE']==cfg['tile']
            rows.append(dict(case=case,writer=writer,error=err,total_error=qe,kernel=tap.resource,
                writer_resources={k:getattr(compiled,k,None) for k in ('n_regs','n_spills')}))
            (root/'raw.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(case=case,passed=True)),flush=True)
    (root/'COMPLETE').write_text('4 cases x 6 writer spellings; graph/cache/history checks; 4 manifest guards\n')


if __name__=='__main__':main()
