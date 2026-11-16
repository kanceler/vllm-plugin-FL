"""Adopted BI-V150 operator snapshot. No model registration or runtime search.

make_call prepares a fixed joint Q/K/V call before graph capture. The caller
owns independently calibrated, immutable per-layer permutation/scale tables.
The research calibration in this workspace is not a 42-layer model artifact.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).parent / 'frozen/k_static_fold_biv150_v1'
MANIFEST_SHA256 = '81ba27289fd11205a6b200f96c74c958c41379a960b44042fd825190bec04f13'


def read_manifest():
    raw=(BUNDLE/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:
        raise ValueError('Frozen manifest checksum mismatch')
    data=json.loads(raw)
    keys=[(r['batch'],r['qlen'],r['input_threshold']) for r in data['records']]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate frozen configuration key')
    return data


def verify_sources(data=None):
    data=read_manifest() if data is None else data
    for path,expected in data['source_bindings'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen source mismatch: {path}')
        snapshot=BUNDLE/Path(path).name
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen snapshot mismatch: {path}')
    return data


def select(batch,qlen,input_threshold=64):
    for record in read_manifest()['records']:
        if (record['batch'],record['qlen'],record['input_threshold'])==(batch,qlen,input_threshold):
            return record
    raise ValueError('Unmeasured frozen operator key; no implicit quantization fallback')


def verify_environment(data):
    import torch
    from vllm.triton_utils import triton
    actual=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__)
    if actual!=data['identity']:
        raise ValueError('Frozen device/software mismatch')
    for name,expected in data['provider_bindings'].items():
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen provider mismatch: {name}')


def _module(name):
    directory=str(Path(__file__).parent)
    sys.path.insert(0,directory)
    try:
        module=importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve()!=Path(directory)/f'{name}.py':
        raise ValueError(f'Unexpected diagnostic import: {name}')
    return module


class FrozenCall:
    def __call__(self):
        result=self.fusion.write_q(self.q,self.q_workspace,self.k,self.v,self.cache,
            self.slots,self.perm,self.scale,self.record['writer_warps'],
            self.record['mode']=='q_in_write_heads')
        self.attention(**self.args)
        return result


def make_call(batch,qlen,input_threshold,q,k,v,cache,slots,perm,scale,args,q_workspace=None):
    """Validate and allocate once; returned call performs no GPU allocation.

    args contains the ordinary Attention metadata and output tensor. Incremental
    K/V, Q and slot_mapping must have the same token ordering, including pads.
    Unknown keys fail explicitly in this diagnostic API; model routing is intact.
    """
    import torch
    data=verify_sources();verify_environment(data)
    record=select(batch,qlen,input_threshold);cfg=record['config']
    if (q.shape!=(batch*qlen,16,128) or k.shape!=(batch*qlen,2,128)
        or v.shape!=k.shape or slots.numel()!=q.shape[0]
        or any(x.dtype!=torch.bfloat16 for x in (q,k,v))):
        raise ValueError('Expected aligned BF16 16Q/2KV/D128 incremental tensors')
    if cache.dtype!=torch.uint8 or cache.shape[1:]!=(32,2,276) or not cache.is_contiguous():
        raise ValueError('Expected frozen page32/276B contiguous cache')
    if (perm.shape!=(2,128) or perm.dtype!=torch.int32 or scale.shape!=(2,128)
        or scale.dtype!=torch.float32 or not perm.is_contiguous() or not scale.is_contiguous()):
        raise ValueError('Invalid immutable parameter table contract')
    if any(x.device!=q.device for x in (k,v,cache,slots,perm,scale)):
        raise ValueError('Frozen call tensors must share a device')
    if not torch.equal(perm.sort(-1).values,torch.arange(128,device=q.device,dtype=torch.int32)[None].expand(2,-1)):
        raise ValueError('Each head must provide a true channel permutation')
    if not bool((torch.isfinite(scale)&(scale>0)).all()):
        raise ValueError('Static scales must be finite and positive')
    for x in (q,k,v):
        if x.stride(-1)!=1:raise ValueError('Contiguous channel dimension required')
    fusion=_module('k_static_fold_fusions');layout=_module('k_top4_layout_candidates')
    for name in ('k_static_protect_fold_candidate','k_top4_frozen','k_static_fold_fusion_signature'):_module(name)
    from vllm.v1.kv_cache_interface import KVQuantMode
    if args.get('kv_quant_mode')!=KVQuantMode.INT8_PER_TOKEN_HEAD:
        raise ValueError('Frozen cache requires INT8_PER_TOKEN_HEAD dispatch')
    if args.get('max_seqlen_q')!=qlen or args['cu_seqlens_q'].numel()!=batch+1:
        raise ValueError('Attention metadata does not match frozen uniform Q layout')
    kc,vc,vs=layout.views(cache,'separate_meta')
    qo=torch.empty(q.shape,device=q.device,dtype=q.dtype) if q_workspace is None else q_workspace
    if qo.shape!=q.shape or qo.dtype!=q.dtype or qo.device!=q.device or not qo.is_contiguous():
        raise ValueError('Invalid Q workspace')
    if any(qo.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() for x in (q,k,v,cache,args['out'])):
        raise ValueError('Q workspace must not alias inputs, cache or output')
    perm=perm.clone();scale=scale.clone()
    a=dict(args,q=qo,k=kc,v=vc,k_scale_cache=vs,v_scale_cache=vs,
        seq_threshold_3D=max(input_threshold,batch) if cfg['segments']>1 else input_threshold)
    if cfg['segments']>1:
        ns=cfg['segments']
        a.update(num_par_softmax_segments=ns,
            softmax_segm_output=torch.empty((batch,16,ns,128),device=q.device,dtype=torch.float32),
            softmax_segm_max=torch.empty((batch,16,ns),device=q.device,dtype=torch.float32),
            softmax_segm_expsum=torch.empty((batch,16,ns),device=q.device,dtype=torch.float32))
    result=FrozenCall()
    for name,value in dict(q=q,q_workspace=qo,k=k,v=v,cache=cache,slots=slots,
        perm=perm,scale=scale,args=a,record=record,fusion=fusion,
        attention=fusion.prior.launcher(cfg)).items():
        setattr(result,name,value)
    return result


if __name__=='__main__':
    data=verify_sources()
    print(json.dumps(dict(status=data['status'],verified_keys=len(data['records']),
        model_integration=False,production_calibration=False)))
