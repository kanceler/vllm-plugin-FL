"""Offline static K service parameters; immutable during the model lifetime."""
import hashlib
import importlib
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT=Path(__file__).parents[2]
POLICY_PATH=ROOT/'vllm_fl/attention/configs/k_static_fold_biv150_service_v1.json'
CALIBRATION_PATH=ROOT/'vllm_fl/attention/configs/k_static_fold_minicpm5_calibration_v1.json'


@lru_cache(None)
def policy():return json.loads(POLICY_PATH.read_text())


@lru_cache(None)
def calibration():
    data=json.loads(CALIBRATION_PATH.read_text())
    if [r['layer'] for r in data['layers']]!=list(range(42)):
        raise ValueError('Static K calibration must cover exactly all42 layers')
    if data['margin']!=2 or not data['weights_sha256']:
        raise ValueError('Static calibration must bind margin2 and model weights')
    for row in data['layers']:
        if len(row['permutation'])!=2 or len(row['scale'])!=2:
            raise ValueError('Static calibration requires two KV heads')
        for perm,scale,fixed in zip(row['permutation'],row['scale'],row['protected_channels']):
            if sorted(perm)!=list(range(128)) or perm[:4]!=fixed or len(scale)!=128:
                raise ValueError('Invalid calibrated permutation/scale shape')
            if not all(math.isfinite(x) and x>0 for x in scale):
                raise ValueError('Invalid calibrated static scale')
    return data


@lru_cache(None)
def verify():
    data=policy()
    for path,expected in data['source_bindings'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Static service source mismatch: {path}')
    for name,expected in data['provider_bindings'].items():
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Static provider mismatch: {name}')
    if hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()!=data['calibration_sha256']:
        raise ValueError('Static calibration fingerprint mismatch')
    import torch
    from vllm.triton_utils import triton
    if dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__)!=data['identity']:
        raise ValueError('Static service device/software mismatch')
    cal=calibration();model=Path('/workspace/MiniCPM5-2B')
    if hashlib.sha256((model/'config.json').read_bytes()).hexdigest()!=cal['model_config_sha256']:
        raise ValueError('Static model configuration mismatch')
    for name,expected in cal['weights_sha256'].items():
        if hashlib.sha256((model/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Static model weight mismatch: {name}')
    return True


@lru_cache(None)
def choose(batch,qlen):
    for r in policy()['records']:
        if (r['batch'],r['qlen'])==(batch,qlen):
            return tuple(sorted(r['config'].items())),r['mode'],r['writer_warps'],'offline_frozen'
    from vllm_fl.attention.k_static_fold_launch import baseline_config
    cfg=baseline_config(qlen,qlen==1 and batch<=80)
    return tuple(sorted(cfg.items())),'q_in_write_heads',1,'canonical_same_static_format'
