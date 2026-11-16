# SPDX-License-Identifier: Apache-2.0
"""Bind compiled model caches to the custom quantization implementation."""
import hashlib
from pathlib import Path


def install_cache_identity():
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationConfig
    digest = hashlib.sha256()
    for name in ('reference.py', 'separate.py', 'linear.py', 'cache.py', 'backend.py',
                 'policy.py', 'splitk.py', 'gemv.py', 'hotspot.py', 'dispatch_manifest.json'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    from .policy import quantization_flags
    digest.update(repr(quantization_flags()).encode())
    import torch
    import triton
    digest.update((torch.__version__ + ':' + triton.__version__).encode())
    import importlib
    provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    digest.update(Path(provider.__file__).read_bytes())
    identity = 'tianshu_w8a16_four_projection_v3:' + digest.hexdigest()
    for cls in (CompilationConfig, VllmConfig):
        if getattr(cls, '_fl_tianshu_quant_cache', False):
            continue
        original = cls.compute_hash

        def compute_hash(self, _original=original):
            return hashlib.sha256((_original(self) + identity).encode()).hexdigest()

        cls.compute_hash = compute_hash
        cls._fl_tianshu_quant_cache = True
