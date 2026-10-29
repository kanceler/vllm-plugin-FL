# SPDX-License-Identifier: Apache-2.0
"""Run with pytest in the target torch/vLLM environment."""

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.quantization.w8a16.reference import (
    dequantize_reference, quantize_weight,
)


def test_zero_channels_and_symmetric_endpoints():
    w = torch.tensor([[0, 0, 0], [-127, 0, 127]], dtype=torch.bfloat16)
    q, s = quantize_weight(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert s.tolist() == [1, 1]
    assert q.tolist() == [[0, 0, 0], [-127, 0, 127]]
    assert torch.equal(dequantize_reference(q, s), w)


def test_round_to_even_and_noncontiguous_weight():
    w = torch.tensor([[127, .5, 1.5, 2.5, -.5, -1.5]], dtype=torch.bfloat16)
    w = w.t().contiguous().t()
    q, _ = quantize_weight(w)
    assert q.tolist() == [[127, 0, 2, 2, 0, -2]]
    assert q.is_contiguous()


def test_nonfinite_rejected_before_weight_replacement():
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_weight(torch.tensor([[float("nan")]], dtype=torch.bfloat16))


def test_row_parallel_local_shards_have_explicit_local_scales():
    # Row-parallel shards split K. Their channel scales may differ; each
    # partial product must use its own scale before the upstream reduction.
    w = torch.tensor([[127, 0, 254, 0]], dtype=torch.bfloat16)
    a, sa = quantize_weight(w[:, :2])
    b, sb = quantize_weight(w[:, 2:])
    assert sa.item() == 1 and sb.item() == 2
    restored = torch.cat([dequantize_reference(a, sa),
                          dequantize_reference(b, sb)], dim=1)
    assert torch.equal(restored, w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("n,k", [(17, 33), (2048, 6144)])
def test_gpu_dequant_and_bf16_linear(n, k):
    from vllm_fl.quantization.w8a16.dequant import dequantize_weight
    generator = torch.Generator(device="cuda").manual_seed(42)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    q, s = quantize_weight(w)
    expected = dequantize_reference(q, s)
    actual = dequantize_weight(q, s)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=generator)
    torch.testing.assert_close(F.linear(x, actual, bias), F.linear(x, expected, bias),
                               rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_cuda_graph_replays_dequantized_weight():
    from vllm_fl.quantization.w8a16.dequant import dequantize_weight
    q = torch.ones(17, 33, dtype=torch.int8, device="cuda")
    s = torch.ones(17, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dequantize_weight(q, s)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dequantize_weight(q, s)
    s.fill_(2)
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(result, 2), rtol=0, atol=0)


def test_install_preserves_other_linear_methods(monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.quantization.w8a16 import linear
    from vllm_fl import utils
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(UnquantizedLinearMethod, linear._MARKER, False, raising=False)
    monkeypatch.setattr(UnquantizedLinearMethod, "process_weights_after_loading",
                        lambda self, layer: calls.append(layer))
    monkeypatch.setattr(UnquantizedLinearMethod, "apply",
                        lambda self, layer, x, bias=None: "original")
    # Non-target layers must keep the exact original apply: optional-attribute
    # guards in a global wrapper broke full-model TorchDynamo compilation.
    original_apply = UnquantizedLinearMethod.apply
    assert linear.install_w8a16_down()
    assert UnquantizedLinearMethod.apply is original_apply
    wrapped = UnquantizedLinearMethod.apply
    assert linear.install_w8a16_down()
    assert UnquantizedLinearMethod.apply is wrapped
    layer = SimpleNamespace(prefix="model.layers.0.o_proj")
    method = UnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    assert calls == [layer]
    assert method.apply(layer, None) == "original"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "mode",
    ["separate", "legacy_fused", "int8pack_scale_after_dot"],
)
def test_down_adapter_uses_completed_local_shard(monkeypatch, mode):
    from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
    from vllm_fl.quantization.w8a16 import linear
    from vllm_fl import utils

    monkeypatch.setenv(
        "VLLM_FL_W8A16_DOWN_FUSED", "1" if mode == "legacy_fused" else "0"
    )
    monkeypatch.setenv(
        "VLLM_FL_W8A16_DOWN_INT8PACK",
        "1" if mode == "int8pack_scale_after_dot" else "0",
    )
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(UnquantizedLinearMethod, linear._MARKER, False, raising=False)
    monkeypatch.setattr(UnquantizedLinearMethod, "process_weights_after_loading",
                        lambda self, layer: None)
    monkeypatch.setattr(UnquantizedLinearMethod, "apply",
                        lambda *args, **kwargs: pytest.fail("active Down fell through"))
    linear.install_w8a16_down()
    # Test the adapter in isolation; distributed RowParallelLinear forward is
    # deliberately untouched and still needs a target-environment smoke test.
    layer = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.mlp.down_proj"
    layer.tp_size = 2
    weight = torch.randn(17, 33, device="cuda", dtype=torch.bfloat16)
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    method = UnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    first = layer.weight
    method.process_weights_after_loading(layer)
    assert layer.weight is first
    assert first.dtype == torch.int8 and layer.w8a16_scale.dtype == torch.float32
    x = torch.randn(3, 33, device="cuda", dtype=torch.bfloat16)
    if mode == "int8pack_scale_after_dot":
        expected = (
            F.linear(x.float(), first.float()) * layer.w8a16_scale
        ).to(torch.bfloat16)
    else:
        expected = F.linear(x, dequantize_reference(first, layer.w8a16_scale))
    torch.testing.assert_close(layer.quant_method.apply(layer, x), expected,
                               rtol=.02 if mode != "separate" else 0,
                               atol=.125 if mode != "separate" else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_int8pack_uses_scale_after_dot_and_preserves_dynamic_m():
    from vllm_fl.quantization.w8a16.int8pack import (
        int8pack_linear,
        supports_int8pack,
    )

    q = torch.full((64, 1024), 127, device="cuda", dtype=torch.int8)
    scale = torch.full((64,), 1.003 / 127, device="cuda", dtype=torch.float32)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def forward(x):
        assert supports_int8pack(x, q, scale)
        return int8pack_linear(x, q, scale)

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    for m in (16, 32, 64, 128):
        x = torch.zeros(m, 1024, device="cuda", dtype=torch.bfloat16)
        x[:, 0] = 1.5
        torch._dynamo.mark_dynamic(x, 0, min=1, max=2048)
        actual = compiled(x)
        expected = (F.linear(x.float(), q.float()) * scale).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)

        # The small-M fused epilogue scales FP32 accumulators. Large-M MM
        # first emits BF16, so this one-term example may coincide with the
        # old result; verify its explicit intermediate-rounding contract.
        rounded_weight = F.linear(x, dequantize_reference(q, scale))
        if m < 128:
            assert not torch.equal(actual, rounded_weight)
        else:
            rounded_dot = F.linear(x, q.to(torch.bfloat16))
            expected_large = (rounded_dot.float() * scale).to(torch.bfloat16)
            torch.testing.assert_close(actual, expected_large, rtol=0, atol=0)
    assert len(graphs) == 1, "runtime M dispatch must stay inside the opaque op"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_int8pack_large_separate_dispatch_and_graph():
    import importlib
    from vllm_fl.quantization.w8a16.int8pack import int8pack_linear
    backend = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.weight_int8pack_mm")
    metax = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    gen = torch.Generator(device="cuda").manual_seed(20261029)
    q = torch.randint(-127, 128, (64, 1024), device="cuda",
                      dtype=torch.int8, generator=gen)
    s = torch.rand(64, device="cuda", generator=gen) * .02 + .001
    x = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16, generator=gen)
    assert backend._can_use_large_m_separate(x, q, s)
    assert not backend._can_use_small_m_splitk(x, q, s)

    def expected():
        return (metax.mm(x, q.to(torch.bfloat16).t()).float() * s).to(torch.bfloat16)

    actual = int8pack_linear(x, q, s)
    torch.testing.assert_close(actual, expected(), rtol=0, atol=0)
    ref = (F.linear(x.float(), q.float()) * s).to(torch.bfloat16)
    error = (actual.float() - ref.float()).norm() / ref.float().norm()
    assert error.item() < .005
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        int8pack_linear(x, q, s)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = int8pack_linear(x, q, s)
    s.mul_(1.25)
    graph.replay()
    torch.testing.assert_close(replayed, expected(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("m,n,k", [(1, 17, 33), (3, 17, 33), (64, 129, 257),
                                  (1, 2048, 6144), (64, 2048, 6144),
                                  (512, 2048, 6144)])
def test_fused_rounds_weight_before_dot(m, n, k):
    from vllm_fl.quantization.w8a16.fused import fused_linear
    gen = torch.Generator(device="cuda").manual_seed(73)
    q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=gen)
    s = torch.rand(n, device="cuda", generator=gen) * .03 + .001
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    actual = fused_linear(x, q, s)
    # Compare with exactly the rounded weights used by the separate path.
    expected = F.linear(x, dequantize_reference(q, s))
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error.item() < .005
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fused_bf16_rounding_boundary_and_graph():
    from vllm_fl.quantization.w8a16.fused import fused_linear
    # One nonzero term makes GEMM accumulation exact: any difference here
    # exposes removal of the BF16 rounding before dot (scale-after-dot).
    x = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1.5
    q = torch.full((17, 64), 127, device="cuda", dtype=torch.int8)
    s = torch.full((17,), 1.003 / 127, device="cuda", dtype=torch.float32)
    bias = torch.ones(17, device="cuda", dtype=torch.bfloat16) / 8
    expected = F.linear(x, dequantize_reference(q, s), bias)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_linear(x, q, s, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_linear(x, q, s, bias)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    s.mul_(1.25)
    graph.replay()
    torch.testing.assert_close(actual, F.linear(x, dequantize_reference(q, s), bias),
                               rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_preserves_dynamic_token_dimension(has_bias):
    from vllm_fl.quantization.w8a16.fused import fused_linear, supports_fused

    q = torch.randint(-127, 128, (17, 33), device="cuda", dtype=torch.int8)
    scale = torch.full((17,), .01, device="cuda", dtype=torch.float32)
    bias = torch.ones(17, device="cuda", dtype=torch.bfloat16) if has_bias else None
    rounded = dequantize_reference(q, scale)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def forward(x):
        # Exercise the same capability guard used by the Down adapter.
        if not supports_fused(x, q, scale, bias):
            raise ValueError("unexpected unsupported test input")
        return fused_linear(x, q, scale, bias)

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    for m in (2048, 16, 33, 64):
        x = torch.randn(m, 33, device="cuda", dtype=torch.bfloat16)
        # Strict dynamic marking reproduces vLLM's startup guard failure;
        # dynamic=True alone can silently specialize and would miss this bug.
        torch._dynamo.mark_dynamic(x, 0, min=2, max=2048)
        actual = compiled(x)
        torch.testing.assert_close(actual, F.linear(x, rounded, bias),
                                   rtol=.02, atol=.125)
    assert len(graphs) == 1, "token counts must share one dynamic model graph"
