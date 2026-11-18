"""Synthetic framework/graph checks; not model accuracy or TP collective tests."""
import json
import os
import sys
from pathlib import Path

mask=sys.argv[1]
assert len(mask)==4 and set(mask)<=set('01')
names=('DOWN','GATE_UP','QKV','O')
for name,value in zip(names,mask):
    os.environ['VLLM_FL_TIANSHU_W8A16_'+name]=value

import torch
from vllm.model_executor.layers import linear as upstream
from vllm_fl.quantization.w8a16.linear import install_tianshu_separate
from vllm_fl.quantization.w8a16.policy import choose, policy_matches, load_policy
from vllm_fl.quantization.w8a16.reference import quantize_weight
from vllm_fl.quantization.w8a16.separate import separate_linear

torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(20261118)
assert policy_matches(0), 'manifest/device/provider/kernel mismatch'
upstream.UnquantizedLinearMethod.process_weights_after_loading=lambda self,layer: None
assert install_tianshu_separate() == ('1' in mask)
assert install_tianshu_separate() == ('1' in mask)
out=Path(sys.argv[2]); out.mkdir(exist_ok=False)

def check(y,ref):
    error=y.float()-ref.float()
    l2=(error.norm()/ref.float().norm().clamp_min(1e-12)).item()
    mx=(error.abs().max()/ref.float().abs().max().clamp_min(1e-12)).item()
    assert torch.isfinite(y).all() and l2<=.005 and mx<=.01,(l2,mx)
    return dict(l2=l2,normalized_max=mx)

def emit(row):
    with (out/'checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def reference(a,q,s,group,bias=None):
    flat=a.reshape(-1,a.shape[-1])
    value=flat.float()@q.float().T
    if choose(group,flat.shape[0],q.shape[0],q.shape[1],0)['algorithm']=='separate_generic':
        # Match the explicitly distinct BF16-intermediate fallback contract.
        value=value.to(torch.bfloat16).float()
    result=(value*s).to(torch.bfloat16)
    if bias is not None:result=(result.float()+bias.float()).to(torch.bfloat16)
    return result.reshape(*a.shape[:-1],q.shape[0])

# Actual class family selection, including packed QKV and Gate/Up classes.
for layer_cls in (upstream.RowParallelLinear,upstream.ColumnParallelLinear,
                  upstream.MergedColumnParallelLinear,upstream.QKVParallelLinear):
    for group,name in enumerate(('down_proj','gate_up_proj','qkv_proj','o_proj','lm_head','embed_tokens')):
        layer=layer_cls.__new__(layer_cls); torch.nn.Module.__init__(layer)
        layer.prefix='model.layers.0.'+name
        layer.weight=torch.nn.Parameter(torch.randn(17,33,device='cuda',dtype=torch.bfloat16),requires_grad=False)
        method=upstream.UnquantizedLinearMethod(); layer.quant_method=method
        method.process_weights_after_loading(layer)
        enabled=group<4 and mask[group]=='1'
        assert (layer.weight.dtype==torch.int8)==enabled,(layer_cls,name)
        if enabled:
            a=torch.randn(3,33,device='cuda',dtype=torch.bfloat16)
            bias=torch.randn(17,device='cuda',dtype=torch.bfloat16)
            ref=reference(a,layer.weight,layer.w8a16_scale,group,bias)
            check(layer.quant_method.apply(layer,a,bias),ref)
            weight=layer.weight; method.process_weights_after_loading(layer)
            assert layer.weight is weight
        emit(dict(stage='layer_selection',layer=layer_cls.__name__,projection=name,quantized=enabled))

if mask!='1111':
    (out/'COMPLETE').write_text('selection passed\n')
    sys.exit(0)

# Test exact measured points plus representative untouched M and boundary neighbors.
for group,(name,shape) in enumerate((('down',(2048,6144)),('gate_up',(12288,2048)),
                                     ('qkv',(2560,2048)),('o',(2048,2048)))):
    n,k=shape
    torch.manual_seed(20261116+group)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    q,s=quantize_weight(w)
    group_policy=load_policy()['groups'][name]
    points={1,2,4,8,16,32,64,96,192,384,768,1536,3072}
    for left,right in group_policy.get('transition_intervals',[]):
        points.update((left,max(1,left+1),max(1,right-1),right))
    for m in sorted(points):
        a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        ref=reference(a,q,s,group)
        y=separate_linear(a,q,s,None,group)
        row=dict(stage='runtime',group=name,m=m,choice=choose(group,m,n,k,0),**check(y,ref))
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):yg=separate_linear(a,q,s,None,group)
        a.mul_(.5);g.replay();torch.cuda.synchronize()
        row['updated_graph']=check(yg,reference(a,q,s,group))
        emit(row)
        del g,yg
    # Non-contiguous offset/3-D activation and bias are handled inside the op.
    a=torch.randn(2,3,k+1,device='cuda',dtype=torch.bfloat16)[...,1:].transpose(0,1)
    bias=torch.randn(n,device='cuda',dtype=torch.bfloat16)
    ref=reference(a,q,s,group,bias)
    emit(dict(stage='noncontiguous_bias_3d',group=name,**check(separate_linear(a,q,s,bias,group),ref)))

# Opaque registration/fake schema with the extra group argument must survive fullgraph.
q,s=quantize_weight(torch.randn(17,33,device='cuda',dtype=torch.bfloat16))
compiled=torch.compile(separate_linear,backend='inductor',fullgraph=True,dynamic=True)
for group in range(4):
    for m in (2,5):
        a=torch.randn(m,33,device='cuda',dtype=torch.bfloat16)
        assert torch.equal(compiled(a,q,s,None,group),separate_linear(a,q,s,None,group))
assert separate_linear(a[:0],q,s,None,0).shape==(0,17)
try:
    separate_linear(a.float(),q,s,None,0)
    raise AssertionError('unsupported dtype accepted')
except ValueError:
    pass
emit(dict(stage='compile_dynamic_empty_dtype',passed=True))
(out/'COMPLETE').write_text('passed\n')
