"""Select existing final quantization implementations by the active vendor."""
import hashlib
import os
from pathlib import Path


def _install_identity():
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationConfig
    import torch
    import triton
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(b"fl-quantization-integration-v1")
    manifest = root / "quantization-integration-manifest.json"
    import json
    data = json.loads(manifest.read_text())
    for name, expected in data["files"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Integrated quantization source mismatch: " + name)
    digest.update(manifest.read_bytes())
    for name in ("integration.py",):
        digest.update(Path(__file__).with_name(name).read_bytes())
    flags = sorted((k, v) for k, v in os.environ.items()
                   if k.startswith(("VLLM_FL_W8A16", "VLLM_FL_TIANSHU_W8A16",
                                    "VLLM_FL_K_", "VLLM_FL_V_ONLY", "VLLM_FL_INT8_KV",
                                    "VLLM_FL_BIV150")))
    digest.update(repr(flags).encode())
    digest.update((torch.__version__ + triton.__version__).encode())
    identity = digest.hexdigest()
    for cls in (VllmConfig, CompilationConfig):
        if getattr(cls, "_fl_quantization_integration_identity", False):
            continue
        original = cls.compute_hash
        def compute_hash(self, _original=original):
            return hashlib.sha256((_original(self) + identity).encode()).hexdigest()
        cls.compute_hash = compute_hash
        cls._fl_quantization_integration_identity = True


def install_quantization():
    from flag_gems import runtime
    vendor = runtime.device.vendor_name
    if vendor not in ("metax", "iluvatar"):
        return False
    _install_identity()
    if vendor == "metax":
        from .metax_w8a16.linear import install_w8a16_down
        return install_w8a16_down()
    from .w8a16.linear import install_tianshu_separate
    return install_tianshu_separate()
