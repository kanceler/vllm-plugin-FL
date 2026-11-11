from types import SimpleNamespace

import pytest
import torch

from vllm_fl.attention import decode_index_policy as policy


@pytest.mark.parametrize("starts", [[0, 1], [0, 1, 2, 3], [0, 1, 2, 2, 2]])
def test_dense_prefix_accepts_tail_padding(starts):
    assert policy.dense_single_query_prefix(starts)


@pytest.mark.parametrize(
    "starts", [[], [0], [1, 2], [0, 0], [0, 2], [0, 1, 1, 2], [0, 1, 0]]
)
def test_holes_multitoken_and_empty_prefix_fall_back(starts):
    assert not policy.dense_single_query_prefix(starts)


def test_metadata_reuses_cpu_prefix_and_mixed_plan():
    starts = torch.tensor([0, 1, 2, 2])
    target = object()
    metadata = SimpleNamespace(
        causal=True,
        use_cascade=False,
        mm_prefix_range_tensor=None,
        max_query_len=1,
        query_start_loc=target,
    )
    common = SimpleNamespace(query_start_loc_cpu=starts, num_reqs=3)
    assert policy.qualify_metadata(metadata, common) is target
    common.query_start_loc_cpu = torch.tensor([0, 1, 1, 2])
    assert policy.qualify_metadata(metadata, common) is None
    metadata._fl_mixed_plan = SimpleNamespace(decode_starts=target)
    metadata.max_query_len = 2048
    assert policy.qualify_metadata(metadata, common) is target
    metadata.use_cascade = True
    assert policy.qualify_metadata(metadata, common) is None


def test_disabled_does_not_install(monkeypatch):
    monkeypatch.setattr(policy, "_installed", False)
    monkeypatch.delenv("VLLM_FL_BIV150_DECODE_DIRECT_INDEX", raising=False)
    assert policy.install_decode_index_policy(None) is False


def test_unknown_upstream_source_is_rejected():
    fake = SimpleNamespace(
        kernel_unified_attention=SimpleNamespace(fn=test_disabled_does_not_install)
    )
    with pytest.raises(ValueError, match="unsupported"):
        policy.build_specialized_head(fake)


def test_positional_and_ineligible_launches_keep_original():
    calls = []

    class Original:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))

    dispatch = policy.DecodeIndexKernel(Original(), None)
    dispatch[(3, 2, 16)]("positional")
    dispatch[(3, 2, 16)](IS_3D=False)
    assert len(calls) == 2 and calls[0][1] == ("positional",)


def eligible_kwargs():
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), dtype=torch.bfloat16, shape=(32, 16, 128)
    )
    values = dict(
        IS_3D=True,
        num_query_heads=16,
        num_queries_per_kv=8,
        HEAD_SIZE=128,
        HEAD_SIZE_PADDED=128,
        BLOCK_SIZE=16,
        BLOCK_M=16,
        BLOCK_Q=2,
        TILE_SIZE=16,
        NUM_SEGMENTS_PER_SEQ=16,
        num_warps=2,
        num_stages=1,
        KV_QUANT_MODE=0,
        USE_CAUSAL=True,
        num_seqs=32,
    )
    values.update(
        query_ptr=tensor,
        key_cache_ptr=tensor,
        value_cache_ptr=tensor,
        output_ptr=tensor,
        query_start_len_ptr=object(),
    )
    return values


@pytest.mark.parametrize(
    "overrides",
    [
        {"IS_3D": False},
        {"HEAD_SIZE": 64},
        {"num_warps": 4},
        {"KV_QUANT_MODE": 1},
        {"USE_FP8": True},
        {"USE_MM_PREFIX": True},
        {"SLIDING_WINDOW": 16},
        {"CHUNK_LOOKBACK": 0},
    ],
)
def test_unvalidated_features_fall_back(overrides):
    kwargs = eligible_kwargs()
    metadata = SimpleNamespace(_fl_decode_index_starts=kwargs["query_start_len_ptr"])
    dispatch = policy.DecodeIndexKernel(None, None, lambda: {"layer": metadata})
    assert dispatch.eligible(kwargs)
    kwargs.update(overrides)
    assert not dispatch.eligible(kwargs)


def test_metadata_identity_and_diagnostic_fallback():
    kwargs = eligible_kwargs()
    metadata = SimpleNamespace(_fl_decode_index_starts=object())
    dispatch = policy.DecodeIndexKernel(None, None, lambda: {"layer": metadata})
    assert not dispatch.eligible(kwargs)
    metadata._fl_decode_index_starts = kwargs["query_start_len_ptr"]
    assert dispatch.eligible(kwargs)
    dispatch.force_original = True
    assert not dispatch.eligible(kwargs)
