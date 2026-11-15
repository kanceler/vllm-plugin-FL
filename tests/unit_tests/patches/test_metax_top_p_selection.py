"""Run with pytest --noconftest on a C500; no model or server is loaded."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def selection():
    if not torch.cuda.is_available():
        pytest.skip("C500 is required")
    if torch.cuda.get_device_name().strip().lower() != "metax c500":
        pytest.skip("C500 is required")
    path = (
        Path(__file__).parents[3]
        / "vllm_fl/dispatch/backends/vendor/metax/patches/top_p_selection.py"
    )
    spec = importlib.util.spec_from_file_location("c500_selection_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["fp32", "bf16", "uniform", "twenty_ties", "one_peak"])
def test_independent_fp64_mass_and_stable_ties(selection, kind):
    torch.manual_seed(131113)
    x = torch.randn(8, 130560, device="cuda") * 3
    if kind == "bf16":
        x = x.bfloat16().float()
    elif kind == "uniform":
        x.zero_()
    elif kind in ["twenty_ties", "one_peak"]:
        x.fill_(-100.0)
        x[:, : 20 if kind == "twenty_ties" else 1] = 0.0
    p = torch.tensor([0.1, 0.5, 0.9, 0.95, 0.999, 0.95, 0.5, 0.9], device="cuda")
    out = selection.try_apply(x, None, p)
    assert out is not None
    values, indices = x.cpu().double().sort(dim=-1, stable=True)
    weights = (values - values[:, -1:]).exp()
    kept = torch.isfinite(out).cpu().gather(1, indices)
    removed = (weights * ~kept).sum(-1)
    target = (1.0 - p.cpu().double()) * weights.sum(-1)
    next_weight = torch.where(kept, weights, torch.inf).amin(-1)
    tolerance = 64 * torch.finfo(torch.float64).eps * weights.sum(-1)
    assert bool((removed <= target + tolerance).all())
    assert bool(
        ((removed + next_weight > target - tolerance) | (kept.sum(-1) == 1)).all()
    )
    assert not bool((kept[:, :-1] & ~kept[:, 1:]).any())


def test_feature_and_value_fallbacks(selection):
    x = torch.ones(8, 130560, device="cuda")
    p = torch.full((8,), 0.95, device="cuda")
    assert (
        selection.try_apply(x, torch.ones(8, device="cuda", dtype=torch.int32), p)
        is None
    )
    assert selection.try_apply(x.half(), None, p) is None
    assert selection.try_apply(x.T, None, p) is None
    assert selection.try_apply(x[:, :1024].contiguous(), None, p) is None
    for value in [0.0, 1.0, -1.0, float("nan")]:
        q = p.clone()
        q[0] = value
        assert selection.try_apply(x, None, q) is None
    for value in [float("inf"), float("nan"), -0.0]:
        z = x.clone()
        z[0, 0] = value
        assert selection.try_apply(z, None, p) is None
    z = x.clone()
    z[0].fill_(-float("inf"))
    assert selection.try_apply(z, None, p) is None


def test_tail_reuse_returns_independent_logits(selection):
    x = torch.randn(64, 130560, device="cuda")
    p = torch.full((64,), 0.95, device="cuda")
    first = selection.try_apply(x, None, p)
    copy = first.clone()
    tail = selection.try_apply(x[:8].contiguous() + 1, None, p[:8])
    assert tail.shape == (8, 130560)
    assert torch.equal(first, copy)
    assert first.data_ptr() != tail.data_ptr()


def test_sampling_preserves_rng_with_same_filtered_logits(selection):
    # The optimization produces a filter only and must never draw random noise.
    x = torch.randn(8, 130560, device="cuda")
    p = torch.full((8,), 0.95, device="cuda")
    before = torch.cuda.get_rng_state()
    selection.try_apply(x, None, p)
    assert torch.equal(before, torch.cuda.get_rng_state())
