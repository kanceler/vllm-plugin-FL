"""Fail before model timing when the pinned provider or dispatch is unavailable."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import torch
import triton
from vllm_fl.quantization.w8a16 import policy

provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
record = dict(provider=provider.__file__, provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
              torch=torch.__version__, triton=triton.__version__, device=torch.cuda.get_device_name(0),
              pythonpath=os.environ.get('PYTHONPATH'), matches=policy.policy_matches(0), cases=[])
out = Path(sys.argv[1])
out.write_text(json.dumps(record, indent=2))
assert record['matches'], 'Pinned GEMM policy did not match; refusing model timing'
torch.manual_seed(20261116)
q = torch.randint(-127, 128, (2048, 6144), device='cuda', dtype=torch.int8)
scale = torch.full((2048,), .01, device='cuda', dtype=torch.float32)
for m in (1, 16, 32, 56, 64, 128, 256, 2048, 4096):
    choice = policy.choose(0, m, 2048, 6144, 0)
    assert choice['algorithm'] != 'separate_generic', (m, choice)
    a = torch.randn((m, 6144), device='cuda', dtype=torch.bfloat16)
    y = policy.dispatch(a, q, scale, 0)
    ref = ((a.float() @ q.float().T) * scale).to(torch.bfloat16)
    error = ((y.float()-ref.float()).norm()/ref.float().norm()).item()
    assert error <= .005, (m, error)
    torch.cuda.synchronize()
    record['cases'].append(dict(m=m, choice=choice, relative_l2=error))
    out.write_text(json.dumps(record, indent=2))
record['complete'] = True
out.write_text(json.dumps(record, indent=2))
print('PINNED_PROVIDER_AND_NINE_DOWN_CHOICES_PASS', flush=True)
