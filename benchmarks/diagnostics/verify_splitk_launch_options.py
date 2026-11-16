"""Check explicit Split-K launch options on a MetaX GPU before retuning."""

import importlib
import json

import torch


def main():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.weight_int8pack_mm"
    )
    m, n, k = 3, 2048, 2048
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.ones(n, device="cuda", dtype=torch.float32) / 127
    partials = torch.empty((2, m, n), device="cuda", dtype=torch.float32)
    jit = backend.weight_int8pack_mm_splitk_partial.jit_function
    observations = []
    for warps, stages in ((2, 1), (8, 3), (2, 1)):
        compiled = jit.run(
            x, q, partials, m, n, k, x.stride(0), x.stride(1),
            q.stride(1), q.stride(0), SPLIT_K=2, BLOCK_M=16,
            BLOCK_N=64, BLOCK_K=128, EVEN_M=False, num_warps=warps,
            num_stages=stages, pipeline="basic", grid=(32, 2), warmup=False,
        )
        torch.cuda.synchronize()
        metadata = compiled.metadata
        actual = (metadata.num_warps, metadata.num_stages)
        if actual != (warps, stages):
            raise AssertionError(f"requested {(warps, stages)}, actual {actual}")
        observations.append({"requested": [warps, stages],
                             "actual": list(actual),
                             "kernel_id": id(compiled)})
    if observations[0]["kernel_id"] == observations[1]["kernel_id"]:
        raise AssertionError("different launch options shared one compiled kernel")
    expected = (x.float() @ q.float().T * scales).to(torch.bfloat16)
    output = backend.run_w8a16_algorithm(
        x, q, scales, "splitk", (16, 64, 128, 8, 3, 2)
    )
    rel_l2 = ((output.float() - expected.float()).norm()
              / expected.float().norm().clamp_min(1e-12)).item()
    if rel_l2 > 0.003:
        raise AssertionError(f"relative L2 error {rel_l2}")
    print(json.dumps({"observations": observations,
                      "relative_l2_error": rel_l2}, indent=2))


if __name__ == "__main__":
    main()
