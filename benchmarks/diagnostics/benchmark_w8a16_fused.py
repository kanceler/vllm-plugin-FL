# SPDX-License-Identifier: Apache-2.0
"""Full-path diagnostic; never a replacement for the official model benchmark."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
import triton.testing

from vllm_fl.quantization.w8a16.reference import quantize_weight, dequantize_reference
from vllm_fl.quantization.w8a16.dequant import dequantize_weight
from vllm_fl.quantization.w8a16.fused import fused_linear, _fused_kernel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Refuse to overwrite any previous diagnostic run.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    n, k = 2048, 6144
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .02
    q, scale = quantize_weight(w)
    rounded = dequantize_reference(q, scale)
    torch.testing.assert_close(dequantize_weight(q, scale), rounded, rtol=0, atol=0)
    metadata = dict(torch=torch.__version__, triton=triton.__version__, seed=42,
                    n=n, k=k, rounds=3, warmup_ms=100, rep_ms=300,
                    statistic='median of three per-round GPU latency medians',
                    timing='triton.testing.do_bench; eager full paths including allocation',
                    preprocessing='load-time weight quantization excluded',
                    device=torch.cuda.get_device_name())
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    rows, raw = [], []
    for m in (1, 16, 32, 64, 128, 256, 512, 1024, 2048):
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        expected = F.linear(x, rounded)
        separate = F.linear(x, dequantize_weight(q, scale))
        torch.testing.assert_close(separate, expected, rtol=0, atol=0)
        actual = fused_linear(x, q, scale)
        rel = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert rel < .005, (m, rel)
        paths = {'bf16': lambda: F.linear(x, w),
                 'separate_triton': lambda: F.linear(x, dequantize_weight(q, scale)),
                 'fused': lambda: fused_linear(x, q, scale)}
        names = list(paths)
        samples = {name: [] for name in names}
        # Rotate order to avoid giving one path the same position every round.
        for repeat in range(3):
            for name in names[repeat:] + names[:repeat]:
                ms = triton.testing.do_bench(paths[name], warmup=100, rep=300,
                                             return_mode='median')
                samples[name].append(ms)
                raw.append(dict(m=m, round=repeat + 1, path=name, median_ms=ms))
                (out / 'raw_timings.json').write_text(json.dumps(raw, indent=2))
        row = dict(m=m, n=n, k=k, relative_l2_error=rel,
                   separate_exact_match=True,
                   **{name + '_ms': median(values) for name, values in samples.items()})
        rows.append(row)
        (out / 'microbenchmark.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    compiled = _fused_kernel.warmup(
        x, q, scale, actual, actual, m, n, k, k, False, 32, 64, 64,
        grid=(triton.cdiv(m, 32) * triton.cdiv(n, 64),),
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    for kind, assembly in compiled.asm.items():
        if isinstance(assembly, str):
            (out / ('kernel.' + kind)).write_text(assembly)


if __name__ == '__main__':
    main()
