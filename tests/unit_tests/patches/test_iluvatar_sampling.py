"""GPU semantics and fallbacks for the BI-V150 sampling launch policy."""

import pytest
import torch

from vllm_fl.patches import iluvatar_sampling as policy


def test_disabled_does_not_install(monkeypatch):
    monkeypatch.delenv("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", raising=False)
    assert policy.install_row_parallel_sampling() is False


def test_enabled_install_is_idempotent_and_cpu_falls_back(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    calls = []
    sentinel = object()
    monkeypatch.setenv("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", "1")
    monkeypatch.setattr(
        sampler,
        "apply_top_k_top_p_triton",
        lambda *args: calls.append(args) or sentinel,
    )
    assert policy.install_row_parallel_sampling()
    wrapper = sampler.apply_top_k_top_p_triton
    assert policy.install_row_parallel_sampling()
    assert sampler.apply_top_k_top_p_triton is wrapper
    assert wrapper(torch.empty(17, 130560), None, torch.ones(17), -100) is sentinel
    assert calls[0][-1] == -100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires BI-V150")
@pytest.mark.parametrize("batch", [17, 31, 64])
@pytest.mark.parametrize(
    "mode", ["p09", "p099", "p1", "k32", "kp", "ties", "masked", "all_masked"]
)
def test_same_kernel_preserves_filter_bits(batch, mode):
    if "bi-v150" not in torch.cuda.get_device_name().lower().split():
        pytest.skip("validated device only")
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    torch.manual_seed(41)
    x = torch.randn(batch, 130576, device="cuda")[:, :130560]
    p = torch.full((batch,), 0.99, device="cuda")
    k = None
    if mode == "p09":
        p.fill_(0.9)
    if mode == "p1":
        p.fill_(1)
    if mode in ("k32", "kp"):
        k = torch.full((batch,), 32, dtype=torch.int32, device="cuda")
        if mode == "k32":
            p = None
    if mode == "ties":
        x.round_()
    if mode == "masked":
        x[:, :10000] = -float("inf")
    if mode == "all_masked":
        x.fill_(-float("inf"))
    a = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype).copy_(
        x
    )
    b = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype).copy_(
        x
    )
    apply_top_k_top_p_triton(a, k, p)
    policy.apply_row_parallel(b, k, p)
    assert torch.equal(a, b)


def test_unknown_kernel_abi_keeps_original(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.sample.ops import (
        topk_topp_sampler as sampler,
        topk_topp_triton as upstream,
    )

    original = lambda *args: None
    monkeypatch.setenv("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", "1")
    monkeypatch.setattr(sampler, "apply_top_k_top_p_triton", original)
    monkeypatch.setattr(upstream, "_topk_topp_kernel", SimpleNamespace(arg_names=[]))
    assert policy.install_row_parallel_sampling() is False
    assert sampler.apply_top_k_top_p_triton is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((16, 130560), torch.float32),
        ((17, 64), torch.float32),
        ((17, 130560), torch.bfloat16),
    ],
)
def test_unvalidated_shapes_fall_back(monkeypatch, shape, dtype):
    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    sentinel = object()
    monkeypatch.setenv("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", "1")
    monkeypatch.setattr(sampler, "apply_top_k_top_p_triton", lambda *args: sentinel)
    monkeypatch.setattr(policy, "_validated_device", lambda _: True)
    assert policy.install_row_parallel_sampling()
    assert (
        sampler.apply_top_k_top_p_triton(
            torch.empty(shape, device="cuda", dtype=dtype),
            None,
            torch.ones(shape[0], device="cuda"),
        )
        is sentinel
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "reason", ["device", "filter_shape", "filter_device", "vocab_stride"]
)
def test_device_and_layout_guards(monkeypatch, reason):
    from vllm.v1.sample.ops import topk_topp_sampler as sampler

    sentinel = object()
    monkeypatch.setenv("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", "1")
    monkeypatch.setattr(sampler, "apply_top_k_top_p_triton", lambda *args: sentinel)
    monkeypatch.setattr(policy, "_validated_device", lambda _: reason != "device")
    x = torch.empty(17, 130560, device="cuda")
    p = torch.ones(17, device="cuda")
    if reason == "filter_shape":
        p = p[:16]
    if reason == "filter_device":
        p = p.cpu()
    if reason == "vocab_stride":
        x = torch.empty(17, 261120, device="cuda")[:, ::2]
    assert policy.install_row_parallel_sampling()
    assert sampler.apply_top_k_top_p_triton(x, None, p) is sentinel
