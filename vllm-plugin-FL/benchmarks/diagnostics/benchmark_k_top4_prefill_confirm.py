"""Paired graph microbenchmark; no model/performance acceptance claims."""
import hashlib
import json
import statistics
import sys
import types
from pathlib import Path
import torch
import validate_k_top4 as check
from vllm_fl.attention import v_only, k_top4
from vllm.triton_utils import triton, tl
import k_top4_frozen as frozen


class Tap:
    def __init__(self,jit,label,root):
        self.jit,self.label,self.root=jit,label,root
        self.invoke=None;self.resource=None
    def __getitem__(self,grid):
        def launch(*a,**kw):
            compiled=self.jit[grid](*a,**kw)
            self.invoke=lambda:self.jit[grid](*a,**kw)
            if self.resource is None:
                self.resource={n:getattr(compiled,n,None) for n in ('n_regs','n_spills','n_max_threads')}
                self.resource['shared']=getattr(compiled.metadata,'shared',None)
                self.resource['launch']={n:kw.get(n) for n in ('BLOCK_M','TILE_SIZE','num_warps','num_stages','IS_3D','NUM_SEGMENTS_PER_SEQ')}
                self.resource['asm']={}
                for name,asm in compiled.asm.items():
                    data=asm.encode() if isinstance(asm,str) else bytes(asm)
                    self.resource['asm'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                    if isinstance(asm,str):
                        (self.root/(self.label+'.'+name+'.txt')).write_text(asm)
            return compiled
        return launch


def clone(base):
    fn=types.FunctionType(base.__code__,dict(base.__globals__),base.__name__,base.__defaults__,base.__closure__)
    fn.__kwdefaults__=base.__kwdefaults__
    return fn


def graph(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g


def main():
    root=Path(sys.argv[1]);root.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(int(sys.argv[2]) if len(sys.argv)>2 else 20261117)
    manifest=frozen.load_manifest(Path(__file__).parent/'configs/k_top4_biv150_frozen_v1.json')
    group=sys.argv[3] if len(sys.argv)>3 else 'freeze'
    if group=='freeze':cases=[tuple(r['case']) for r in manifest['records']]
    elif group=='decode':cases=[(b,n,1,b<=64,False) for b,n in [(16,4096),(16,16384),(32,4096),(32,16384),(63,16384),(64,16384),(65,16384),(80,16384)]]
    elif group=='prefill':cases=[(4,2048,512,False,False),(16,4096,1024,False,False),(64,2048,2048,False,False),(32,16384,8192,False,False)]
    else:raise ValueError(group)
    if len(sys.argv)>4:cases=[cases[int(sys.argv[4])]]
    import os
    (root/'environment.json').write_text(json.dumps(dict(identity=frozen.identity(),python=sys.executable,
        torch_path=torch.__file__,triton_path=triton.__file__,plugin_path=k_top4.__file__,
        flags={k:os.environ.get(k) for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_INT8_KV_REPAIR','VLLM_FL_V_ONLY','LD_LIBRARY_PATH','MACA_PATH')}),indent=2))
    rows=[]
    for ci,case in enumerate(cases):
        print(json.dumps(dict(stage="case_begin",case=case)),flush=True)
        captured={}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is check.run.__code__:captured.update(frame.f_locals)
        from vllm_fl.attention.tile_policy import _PREFILL_TILE
        _PREFILL_TILE["value"] = None
        sys.setprofile(profile)
        try:check.run(*case)
        finally:sys.setprofile(None)
        c=captured;batch,length,qlen,_,_=case
        ids=torch.cat([torch.arange(b*length+length-qlen,(b+1)*length,device='cuda') for b in range(batch)])
        wk,wv,ws=c['k'][ids],c['v'][ids],c['slots'][ids]
        config=frozen.baseline_config(qlen,case[3])
        record=next((r for r in manifest['records'] if r['case']==list(case)),None)
        selected=record['config'] if record else config
        vr=torch.zeros((*c['cache'].shape[:3],388),dtype=torch.uint8,device='cuda')
        v_only.write(c['k'],c['v'],vr,c['slots']);vk,vv,vs=v_only.views(vr)
        variants={}
        choices=[('baseline',config),('frozen',selected)] if group=='freeze' else [('baseline',config)]
        if group=='decode':choices=[('service16',config),('explore32',dict(config,segments=32))] if case[3] else [('service2d',config),('explore2d',dict(config,tile=64,warps=4))]
        for label,cfg in choices:
            args=dict(c['args'])
            if case[3]:
                segments=cfg['segments'];args.update(seq_threshold_3D=64 if group=='decode' else batch+1,
                    num_par_softmax_segments=segments,
                    softmax_segm_output=torch.empty((batch,16,segments,128),device='cuda'),
                    softmax_segm_max=torch.empty((batch,16,segments),device='cuda'),
                    softmax_segm_expsum=torch.empty((batch,16,segments),device='cuda'))
            variants['k_'+label]=(frozen.launcher(cfg),args,lambda:k_top4.write(wk,wv,c['cache'],ws))
            variants['load_'+label]=(frozen.launcher(cfg),args,lambda:frozen.write(wk,wv,c['cache'],ws))
            va=dict(args,k=vk,v=vv,k_scale_cache=vs,v_scale_cache=vs)
            variants['v_only_'+label]=(frozen.launcher(cfg,'v_only'),va,lambda:v_only.write(wk,wv,vr,ws))
        # Additional matched-warp V-only control; original V-only remains intact.
        if group=='prefill':
            def v_write_one(k,v,raw,slots):
                kc,vc,vs=v_only.views(raw)
                v_only._write[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
                    v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=1)
            vcheck=torch.zeros_like(vr);v_write_one(c['k'],c['v'],vcheck,c['slots'])
            assert torch.equal(vcheck,vr),'V-only 1-warp control cache differs'
            del vcheck
            variants['v_only_writer1']=(frozen.launcher(config,'v_only'),va,lambda:v_write_one(wk,wv,vr,ws))
        # Check frozen writer against the original for the complete cache.
        scratch=torch.zeros_like(c['cache']);frozen.write(c['k'],c['v'],scratch,c['slots'])
        expected_cache=torch.zeros_like(scratch);k_top4.write(c['k'],c['v'],expected_cache,c['slots'])
        assert torch.equal(scratch,expected_cache),'Frozen cache bytes differ'
        del scratch,expected_cache
        replays=1 if batch*qlen>=32768 else 30
        references=({False:c['expected'],True:c['vonly_reference']} if qlen>1 else
                    {False:check.reference(c['q'],c['dqk'],c['dqv'],qlen,length,batch),
                     True:check.reference(c['q'],c['k'][:-1],c['dqv'],qlen,length,batch)})
        total_ref=c['original'] if qlen>1 else check.reference(c['q'],c['k'][:-1],c['v'][:-1],qlen,length,batch)
        graphs={};resources={}
        for name,(fn,args,write) in variants.items():
            d=root/f'case{ci}-{name}';d.mkdir()
            taps={}
            for label in ('kernel_unified_attention','reduce_segments'):
                tap=Tap(fn.__globals__[label],label,d);fn.__globals__[label]=tap;taps[label]=tap
            try:
                fn(**args);torch.cuda.synchronize()
            except Exception as exc:
                (d/'FAILED.txt').write_text(repr(exc))
                print('FAILED',name,repr(exc),flush=True)
                continue
            ref=references[name.startswith('v_only')]
            err=check.error(args['out'],ref)
            assert err['max_abs']<=.08 and err['nrmse']<=.02,(name,err)
            total_error=check.error(args['out'],total_ref)
            assert total_error['max_abs']<=.20 and total_error['nrmse']<=.05,(name,total_error)
            preserved=args['out'].clone()
            functions={'write_incremental':write,'attention':lambda f=fn,a=args:f(**a),
                       'full':lambda w=write,f=fn,a=args:(w(),f(**a))}
            functions.update({label:tap.invoke for label,tap in taps.items() if tap.invoke is not None})
            for label,operation in functions.items():graphs[(name,label)]=graph(operation)
            graphs[(name,'full')].replay();torch.cuda.synchronize()
            replay_error=check.error(args['out'],ref)
            assert replay_error['max_abs']<=.08 and replay_error['nrmse']<=.02,(name,replay_error)
            assert torch.equal(args['out'],preserved), 'graph replay changed deterministic output'
            resources[name]=dict(error=err,total_error=total_error,replay_error=replay_error,kernels={label:tap.resource for label,tap in taps.items()})
        timing={str(key):[] for key in graphs}
        for rep in range(9):
            keys=list(graphs)
            if rep%2:keys.reverse()
            for key in keys:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):graphs[key].replay()
                end.record();end.synchronize()
                timing[str(key)].append(start.elapsed_time(end)/replays)
        row=dict(case=case,replays_per_round=replays,manifest_hit=record is not None,baseline_config=config,selected_config=selected,incremental_tokens=wk.shape[0],resources=resources,timings_ms=timing,
                 medians_ms={k:statistics.median(v) for k,v in timing.items()})
        rows.append(row)
        (root/'raw.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(case=case,medians_ms=row['medians_ms'])),flush=True)
    (root/'COMPLETE').write_text(f'{len(cases)} {group} cases; 9 timing repetitions; no production coverage claim\n')


if __name__=='__main__':main()
