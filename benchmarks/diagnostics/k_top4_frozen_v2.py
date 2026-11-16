"""Diagnostic configuration v2 adapter; preserves the immutable v1 candidate.

Manifest records are explicit experimental choices, not a service dispatcher.
The caller owns segment workspace and validates the recorded tensor contract.
"""
import hashlib
import importlib
from pathlib import Path
import torch
try:
    from . import k_top4_frozen as v1
except ImportError:
    import k_top4_frozen as v1
from vllm_fl.attention import k_top4

launcher=v1.launcher
baseline_config=v1.baseline_config


def write(k,v,raw,slots,writer='load_w1'):
    """Launch exactly the measured writer; no search or allocation."""
    if writer=='direct_load_w1':writer='load_w1'  # v1 manifest spelling
    choices={'load_w1':(False,1),'load_w2':(False,2),'load_w4':(False,4),
             'sum_w1':(True,1),'sum_w4':(True,4)}
    if writer not in choices:raise ValueError(f'Unknown writer {writer}')
    if writer=='load_w1':return v1.write(k,v,raw,slots)
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128:
        raise ValueError('Matching BF16 D128 K/V required')
    if k.stride(-1)!=1 or v.stride(-1)!=1 or slots.numel()!=k.shape[0]:
        raise ValueError('Unsupported strides or slot count')
    use_sum,warps=choices[writer]
    jit=k_top4._write if use_sum else v1.writer_kernel()
    kc,vc,vs=k_top4.views(raw)
    return jit[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=warps)


def load_manifest(path):
    """Reject stale kernel, provider and software bindings explicitly."""
    manifest=v1.load_manifest(path)
    for filename,expected in manifest['candidate_bindings'].items():
        actual=hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Frozen candidate mismatch: {filename}')
    for module_name,expected in manifest['source_bindings'].items():
        module=importlib.import_module(module_name)
        actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Frozen provider mismatch: {module_name}')
    allowed={'direct_load_w1','load_w1','load_w2','load_w4','sum_w1','sum_w4'}
    for record in manifest['records']:
        if record['writer'] not in allowed:raise ValueError(f"Unknown manifest writer: {record['writer']}")
    return manifest
