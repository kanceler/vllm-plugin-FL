import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.attention import mixed_policy
from vllm_fl.attention.mixed_policy import dispatch_mixed, make_plan


def metadata_for(query_lengths, scratch=64, **overrides):
    starts = torch.tensor([0] + query_lengths, dtype=torch.int32).cumsum(0).int()
    metadata = SimpleNamespace(
        max_query_len=max(query_lengths),
        use_cascade=False,
        causal=True,
        mm_prefix_range_tensor=None,
        query_start_loc=starts,
        seq_threshold_3D=64,
        softmax_segm_output=torch.empty(scratch, 1),
    )
    for key, value in overrides.items():
        setattr(metadata, key, value)
    common = SimpleNamespace(
        query_start_loc_cpu=starts.clone(), num_reqs=len(query_lengths)
    )
    return metadata, common


@pytest.mark.parametrize(
    "lengths", [[1] * 8, [64, 128], [1, 1, 32], [1] * 8 + [32], [32, 1, 1, 1, 1]]
)
def test_nonmixed_or_small_prefix_keeps_original_path(lengths):
    metadata, common = metadata_for(lengths)
    assert make_plan(metadata, common) is None


@pytest.mark.parametrize(
    "overrides", [{"causal": False}, {"use_cascade": True}, {"scratch": 3}]
)
def test_unsupported_metadata_keeps_original_path(overrides):
    metadata, common = metadata_for([1] * 16 + [8], **overrides)
    assert make_plan(metadata, common) is None


def test_split_preserves_row_coverage_positions_and_cache_views():
    metadata, common = metadata_for([1] * 16 + [8, 3])
    plan = make_plan(metadata, common)
    assert plan is not None
    q = torch.randn(27, 2, 8)
    output = torch.full_like(q, float("nan"))
    lengths = torch.arange(18) + 100
    table = torch.arange(18 * 8).reshape(18, 8)
    calls = []

    def attention(**kwargs):
        calls.append(kwargs)
        kwargs["out"].copy_(kwargs["q"] * 2)

    kwargs = dict(
        q=q,
        out=output,
        seqused_k=lengths,
        block_table=table,
        cu_seqlens_q=metadata.query_start_loc,
        max_seqlen_q=8,
        causal=True,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    dispatch_mixed(attention, plan, kwargs)
    assert len(calls) == 2
    assert torch.equal(output, q * 2)
    assert calls[0]["max_seqlen_q"] == 1
    assert calls[0]["cu_seqlens_q"].tolist() == list(range(17))
    assert calls[1]["cu_seqlens_q"].tolist() == [0, 8, 11]
    assert torch.equal(calls[0]["seqused_k"], lengths[:16])
    assert torch.equal(calls[1]["block_table"], table[16:])
    assert metadata.query_start_loc.tolist() == list(range(17)) + [24, 27]


@pytest.mark.parametrize(
    "extra", [{"qq_bias": torch.ones(1)}, {"use_td": True}, {"kv_quant_mode": 2}]
)
def test_unvalidated_features_fall_back_to_one_launch(extra):
    metadata, common = metadata_for([1] * 16 + [8])
    plan = make_plan(metadata, common)
    calls = []
    kwargs = dict(
        cu_seqlens_q=metadata.query_start_loc, max_seqlen_q=8, causal=True, **extra
    )
    dispatch_mixed(lambda **kw: calls.append(kw), plan, kwargs)
    assert len(calls) == 1
    assert calls[0]["cu_seqlens_q"] is metadata.query_start_loc


def install_backend():
    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata

    class Implementation:
        def forward(self, *args, **kwargs):
            return None

    def attention(**kwargs):
        return None

    return SimpleNamespace(
        TritonAttentionMetadataBuilder=Builder,
        TritonAttentionImpl=Implementation,
        unified_attention=attention,
    )


@pytest.mark.parametrize(
    "vendor, device_name, enabled",
    [
        ("iluvatar", "Iluvatar BI-V150", True),
        ("iluvatar", "BI-V150", True),
        ("iluvatar", "Iluvatar BI-V100", False),
        ("iluvatar", "Iluvatar BI-V150S", False),
        ("nvidia", "Iluvatar BI-V150", False),
        ("metax", "MetaX C500", False),
        ("mthreads", "MUSA", False),
        (None, "unknown", False),
    ],
)
def test_install_only_on_validated_device(monkeypatch, vendor, device_name, enabled):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setenv("VLLM_FL_MIXED_ATTN_SPLIT", "1")
    monkeypatch.setattr(mixed_policy, "_installed", False)
    probes = []

    def name():
        probes.append(True)
        return device_name

    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    backend = install_backend()
    original = (
        backend.TritonAttentionMetadataBuilder.build,
        backend.TritonAttentionImpl.forward,
        backend.unified_attention,
    )
    assert mixed_policy.install_mixed_policy(backend) is enabled
    hooks = (
        backend.TritonAttentionMetadataBuilder.build,
        backend.TritonAttentionImpl.forward,
        backend.unified_attention,
    )
    assert all(new is not old for new, old in zip(hooks, original)) is enabled
    assert bool(probes) is (vendor == "iluvatar")
    if enabled:
        assert mixed_policy.install_mixed_policy(backend)
        assert backend.TritonAttentionMetadataBuilder.build is hooks[0]
        assert backend.TritonAttentionImpl.forward is hooks[1]
        assert backend.unified_attention is hooks[2]
    else:
        assert hooks == original


def test_disable_flag_skips_install_before_device_probe(monkeypatch):
    monkeypatch.setenv("VLLM_FL_MIXED_ATTN_SPLIT", "0")
    monkeypatch.setattr(mixed_policy, "_installed", False)

    def unexpected_probe():
        pytest.fail("A disabled policy must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_probe)
    assert not mixed_policy.install_mixed_policy(SimpleNamespace())


def test_device_probe_failure_leaves_backend_untouched(monkeypatch):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(vendor_name="iluvatar")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setenv("VLLM_FL_MIXED_ATTN_SPLIT", "1")
    monkeypatch.setattr(mixed_policy, "_installed", False)

    def unavailable():
        raise RuntimeError("device is unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_name", unavailable)
    backend = install_backend()
    original = backend.unified_attention
    assert not mixed_policy.install_mixed_policy(backend)
    assert backend.unified_attention is original
