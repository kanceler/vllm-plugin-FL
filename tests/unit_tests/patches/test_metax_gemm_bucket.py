# SPDX-License-Identifier: Apache-2.0
"""C500 checks with an isolated FLAGGEMS_DB_URL; pytest --noconftest."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def bucket():
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name().lower() != "metax c500"
    ):
        pytest.skip("C500 is required")
    path = (
        Path(__file__).parents[3]
        / "vllm_fl/dispatch/backends/vendor/metax/patches/gemm_bucket.py"
    )
    spec = importlib.util.spec_from_file_location("c500_gemm_bucket_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepared(bucket):
    mm = bucket.active_module()
    original = mm.mm_kernel_nt.fn
    old_strategy = list(original.strategy)
    old_table = original.config_table_name
    kernel, tuner = bucket._make_kernel(mm)
    assert tuner.config_table_name != old_table
    assert tuner.benchmark_table_name == original.benchmark_table_name
    assert list(original.strategy) == old_strategy
    assert original.config_table_name == old_table
    assert tuner.fn is original.fn and tuner.configs == original.configs

    model = torch.nn.Module()
    generator = torch.Generator(device="cuda").manual_seed(151116)
    for i, (n, k) in enumerate(sorted(bucket._PAIRS)):
        weight = (
            torch.randn(
                (n, k), device="cuda", dtype=torch.bfloat16, generator=generator
            )
            / k**0.5
        )
        model.register_parameter(
            f"weight{i}", torch.nn.Parameter(weight, requires_grad=False)
        )
    state = bucket.BucketState(mm, kernel, tuner, torch.cuda.current_device())
    before = [w.clone() for w in model.parameters()]
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    assert not state.ready
    assert state.warmup(model)
    assert tuner._last_benchmark_args is None
    assert tuner._last_benchmark_meta is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    return state, dict(zip(sorted(bucket._PAIRS), model.parameters())), generator


def assert_reference(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.isfinite(actual).all()
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative.item() < 3e-4


@pytest.mark.parametrize(
    "m",
    [
        513,
        1007,
        1008,
        1023,
        1024,
        1025,
        1535,
        1536,
        1554,
        1567,
        1584,
        1599,
        1600,
        1664,
        1886,
        1909,
        1983,
        2032,
        2047,
        2048,
        2049,
        3072,
        4080,
        4095,
        4096,
    ],
)
def test_real_shapes_and_no_runtime_search(bucket, prepared, monkeypatch, m):
    state, weights, generator = prepared
    entry = importlib.import_module("flag_gems.utils.libentry")

    def unexpected(*a, **kw):
        raise AssertionError("runtime candidate benchmark after bucket warmup")

    monkeypatch.setattr(entry.LibTuner, "_bench", unexpected)
    for (n, k), weight in weights.items():
        x = torch.randn(
            (m, k), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        assert state.eligible(x, weight.t(), out, m, n, k)
        assert state.dispatch(x, weight.t(), out, m, n, k) is out
        assert state.tuner._last_benchmark_args is None
        assert_reference(out, torch.mm(x, weight.t()))


def test_tail_does_not_write_bucket_padding(bucket, prepared):
    state, weights, generator = prepared
    m, n, k = 1909, 2048, 2048
    weight = weights[n, k]
    x = torch.randn(
        (bucket.bucket_m(m) + 16, k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    storage = torch.full(
        (bucket.bucket_m(m) + 16, n), 17, device="cuda", dtype=torch.bfloat16
    )
    state.dispatch(x[:m], weight.t(), storage[:m], m, n, k)
    assert_reference(storage[:m], torch.mm(x[:m], weight.t()))
    assert (storage[m:] == 17).all()


def test_canonical_shape_preserves_normal_launch(prepared, monkeypatch):
    state, weights, _ = prepared
    m, n, k = 2048, 2048, 2048
    x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    sentinel = object()
    calls = []

    def original(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(state, "original", original)
    assert state.dispatch(x, weights[n, k].t(), out, m, n, k) is sentinel
    assert len(calls) == 1


def test_registered_gems_mm_and_mm_out_route_to_bucket(prepared, monkeypatch):
    import flag_gems

    state, weights, generator = prepared
    m, n, k = 1903, 2048, 2048
    weight = weights[n, k]
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(x, weight.t())
    entry = importlib.import_module("flag_gems.utils.libentry")
    calls = []
    launch = state.launch

    def tracked(*args):
        calls.append(args[3:])
        return launch(*args)

    def unexpected(*a, **kw):
        raise AssertionError("registered GEMM bypassed the warmed bucket")

    monkeypatch.setattr(state, "launch", tracked)
    monkeypatch.setattr(state.module, "general_mm_nt", state.dispatch)
    monkeypatch.setattr(entry.LibTuner, "_bench", unexpected)
    with flag_gems.use_gems():
        actual = torch.mm(x, weight.t())
        torch.mm(x, weight.t(), out=out)
    assert calls == [(m, n, k), (m, n, k)]
    assert_reference(actual, reference)
    assert_reference(out, reference)


def test_new_shape_capture_and_changed_inputs(prepared, monkeypatch):
    state, weights, generator = prepared
    entry = importlib.import_module("flag_gems.utils.libentry")
    m, n, k = 1907, 2048, 2048
    weight = weights[n, k].detach().clone()
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def unexpected(*a, **kw):
        raise AssertionError("candidate benchmark during graph capture/replay")

    monkeypatch.setattr(entry.LibTuner, "_bench", unexpected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.dispatch(x, weight.t(), out, m, n, k)
    x.mul_(0.75)
    weight.add_(0.001)
    before = torch.cuda.get_rng_state()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(before, torch.cuda.get_rng_state())
    assert_reference(out, torch.mm(x, weight.t()))


@pytest.mark.parametrize(
    "case", ["not_ready", "small", "large", "fp32", "nn_layout", "unsupported_pair"]
)
def test_unsupported_uses_original(prepared, monkeypatch, case):
    state, weights, _ = prepared
    m, n, k = 513, 2048, 2048
    if case == "small":
        m = 64
    if case == "large":
        m = 4097
    if case == "unsupported_pair":
        n = k = 8
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    x = torch.empty((m, k), device="cuda", dtype=dtype)
    w = torch.empty((n, k), device="cuda", dtype=dtype)
    b = w.t().contiguous() if case == "nn_layout" else w.t()
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    if case == "not_ready":
        monkeypatch.setattr(state, "ready", False)
    sentinel = object()
    calls = []

    def original(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(state, "original", original)
    assert state.dispatch(x, b, out, m, n, k) is sentinel
    assert len(calls) == 1 and calls[0][0] is x


def test_config_scope_and_buckets(bucket):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization=None,
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"],
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=42,
                num_attention_heads=16,
                num_key_value_heads=2,
                vocab_size=130560,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        quant_config=None,
        lora_config=None,
    )
    assert bucket._supported_config(config)
    config.parallel_config.tensor_parallel_size = 2
    assert not bucket._supported_config(config)
    config.parallel_config.tensor_parallel_size = 1
    config.model_config.hf_config.intermediate_size = 8192
    assert not bucket._supported_config(config)
    assert [
        bucket.bucket_m(m) for m in [513, 1024, 1025, 1886, 2032, 2048, 2049, 4096]
    ] == [1024, 1024, 2048, 2048, 2048, 2048, 4096, 4096]
    assert bucket.configuration_m(1567, 2048, 6144) == 1600
    assert bucket.configuration_m(1567, 12288, 2048) == 2048
