"""Immutable offline policy selection; no per-step KV-length decisions or search."""
import hashlib
import importlib
import json
from functools import lru_cache
from pathlib import Path

POLICY_PATH = Path(__file__).parents[2] / 'benchmarks/diagnostics/configs/k_top4_joint_integration_v1.json'


def read_policy(path=POLICY_PATH):
    data=json.loads(Path(path).read_text())
    if data['layout'] != dict(page_size=32,head_major=False,vector_meta=False,bytes_per_token_head=276):
        raise ValueError('Unsupported frozen joint cache layout')
    seen=set()
    for record in data['records']:
        key=(record['batch'],record['segmented']);cfg=record['config']
        if key in seen:raise ValueError('Duplicate joint policy record')
        seen.add(key)
        if (cfg['block_m']!=16 or cfg['tile'] not in (16,32,64,128)
            or cfg['warps'] not in (2,4,8) or cfg['stages'] not in (1,2)
            or cfg['segments'] not in (1,8,16,32)
            or record['writer_warps'] not in (1,2,4)
            or (not record['segmented'] and cfg['segments']!=1)):
            raise ValueError('Invalid joint launch record')
    return data


@lru_cache(None)
def policy():return read_policy()


def verify_bindings(data):
    root=Path(__file__).parents[2]
    for name,expected in data['source_bindings'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Joint source binding mismatch: {name}')
    for module_name,expected in data['provider_bindings'].items():
        module=importlib.import_module(module_name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Joint provider binding mismatch: {module_name}')
    import torch
    from vllm.triton_utils import triton
    actual=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__)
    if actual!=data['identity']:raise ValueError('Joint software/device identity mismatch')


@lru_cache(None)
def choose(batch,qlen,segmented):
    if qlen==1:
        for record in policy()['records']:
            if (record['batch'],record['segmented'])==(batch,segmented):
                return tuple(sorted(record['config'].items())),record['writer_warps'],'offline_joint_frozen'
    from benchmarks.diagnostics.k_top4_frozen import baseline_config
    cfg=baseline_config(qlen,segmented)
    return tuple(sorted(cfg.items())),1,'canonical_fallback'
