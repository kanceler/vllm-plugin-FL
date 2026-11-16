"""Independent load-time switches and version-bound offline dispatch choices."""
import hashlib
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path

GROUPS = ('down', 'gate_up', 'qkv', 'o')
SWITCHES = tuple('VLLM_FL_TIANSHU_W8A16_'+g.upper() for g in GROUPS)
PROJECTIONS = {'down_proj': 0, 'gate_up_proj': 1, 'qkv_proj': 2, 'o_proj': 3}


def quantization_flags():
    values = tuple(os.environ.get(name, '0') for name in SWITCHES)
    if any(value not in ('0','1') for value in values):
        raise ValueError('W8A16 projection switches require 0 or 1; restart after changing')
    return tuple(value == '1' for value in values)


@lru_cache(maxsize=1)
def load_policy():
    return json.loads(Path(__file__).with_name('dispatch_manifest.json').read_text())


@lru_cache(maxsize=None)
def policy_matches(device_index):
    import torch
    import triton
    policy = load_policy()
    provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    hashes = policy.get('kernel_hashes', {})
    if any(hashes.get(name) != hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
           for name in ('backend.py','splitk.py','gemv.py','hotspot.py')):
        return False
    return (policy.get('torch') == torch.__version__ and policy.get('triton') == triton.__version__
            and policy.get('device') == torch.cuda.get_device_name(device_index)
            and policy.get('provider_sha256') == hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest())


def choose(projection_id, m, n, k, device_index):
    if projection_id not in range(len(GROUPS)):
        raise ValueError('unknown W8A16 projection id')
    if policy_matches(device_index):
        group = load_policy().get('groups', {}).get(GROUPS[projection_id], {})
        if group.get('shape') == [n,k]:
            point = group.get('points', {}).get(str(m))
            if point is not None:
                return point
            for entry in group.get('ranges', []):
                if entry['min_m'] <= m <= entry['max_m']:
                    return entry['choice']
    # INT8 weights remain INT8 persistently. This is not original-BF16 fallback.
    return dict(algorithm='separate_generic', config=[], layout='nk', convert=[],
                source='legal_transient_BF16_conversion_fallback')


def dispatch(a, q, scale, projection_id):
    choice = choose(projection_id, a.shape[0], q.shape[0], q.shape[1], a.device.index)
    algorithm = choice['algorithm']
    if algorithm == 'separate_generic':
        # Version fallback uses the public BF16 MM, not the optional fused epilogue.
        # Its BF16 intermediate rounding is distinct from direct FP32 scaling.
        import torch
        provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
        converted = q.to(torch.bfloat16)
        intermediate = provider.mm(a, converted.T)
        return (intermediate.float()*scale).to(torch.bfloat16)
    if algorithm == 'separate_hotspot':
        from .hotspot import run
        return run(a,q,scale,choice)
    if algorithm == 'splitk_muxi':
        if projection_id == 1:
            raise ValueError('Gate/Up Split-K is excluded from this policy')
        from .splitk import run
        return run(a,q,scale,tuple(choice['config']),tuple(choice['convert']))
    if algorithm == 'gemv':
        from .gemv import gemv
        return gemv(a,q,scale,tuple(choice['config']))
    if algorithm not in ('fused_native','dense','separate'):
        raise ValueError('unsupported offline dispatch algorithm')
    from .backend import run
    return run(a,q,scale,algorithm,tuple(choice['config']),
               tuple(choice.get('convert',(2048,8))),choice.get('layout','nk'))
