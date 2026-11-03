"""Guard regressions for the C500 NT override; no GPU execution required."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture
def nt():
    path = Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/vendor/metax/nt_gemm.py"
    spec = importlib.util.spec_from_file_location("metax_nt_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["shape", "dtype", "layout", "device"])
def test_non_target_calls_original(nt, change):
    device = torch.device("cuda:0")
    def tensor(stride):
        return SimpleNamespace(device=device, dtype=torch.bfloat16, stride=lambda: stride)
    a, b, c = tensor((2048, 1)), tensor((1, 2048)), tensor((2048, 1))
    shape = (2048, 2048, 2048)
    if change == "shape":
        shape = (64, 2048, 2048)
    elif change == "dtype":
        b.dtype = torch.float16
    elif change == "layout":
        b.stride = lambda: (2048, 1)
    else:
        a.device = torch.device("cuda:1")
    calls = []
    def original(*args):
        calls.append(args)
        return "original-result"
    dispatch = nt._make_nt_dispatch(original, {(2048, 2048, 2048): object()}, device)
    assert dispatch(a, b, c, *shape) == "original-result"
    assert calls == [(a, b, c, *shape)]


def test_target_preserves_out_and_grouping(nt, monkeypatch):
    device = torch.device("cuda:0")
    def tensor(stride):
        return SimpleNamespace(device=device, dtype=torch.bfloat16, stride=lambda: stride)
    a, b, c = tensor((2048, 1)), tensor((1, 2048)), tensor((2048, 1))
    calls = []
    class Entry:
        def __getitem__(self, grid):
            assert grid({"BLOCK_M": 128, "BLOCK_N": 128}) == (256,)
            return lambda *args, **kwargs: calls.append((args, kwargs))
    def original(*args):
        pytest.fail("Target unexpectedly fell back")
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    dispatch = nt._make_nt_dispatch(original, {(2048, 2048, 2048): Entry()}, device)
    assert dispatch(a, b, c, 2048, 2048, 2048) is c
    assert calls == [((a, b, c, 2048, 2048, 2048), {"GROUP_M": 8})]


def test_other_card_does_not_install(nt, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "MetaX C550")
    assert nt.apply_c500_nt_gemm(torch.device("cuda:0")) is False


def test_installs_in_registered_module_not_fullpath_copy(nt, monkeypatch):
    import sys
    class Tuner:
        fn = object()
    class Entry:
        def __init__(self, fn):
            self.fn = fn
    fake_gems = SimpleNamespace(mm=SimpleNamespace(__module__="_metax.ops.mm"))
    fake_lib = SimpleNamespace(LibEntry=Entry, LibTuner=Tuner,
                              libtuner=lambda **kwargs: lambda fn: Tuner())
    monkeypatch.setitem(sys.modules, "flag_gems", fake_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.utils.libentry", fake_lib)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "MetaX C500")
    original = lambda *args: None
    active = SimpleNamespace(general_mm_nt=original, mm_kernel_nt=Entry(Tuner()))
    imports = []
    def get_module(name):
        imports.append(name)
        assert name == "_metax.ops.mm"
        return active
    monkeypatch.setattr(nt.importlib, "import_module", get_module)
    assert nt.apply_c500_nt_gemm(torch.device("cuda:0"))
    assert imports == ["_metax.ops.mm"]
    assert active.general_mm_nt is not original
    patched = active.general_mm_nt
    assert nt.apply_c500_nt_gemm(torch.device("cuda:0"))
    assert active.general_mm_nt is patched
