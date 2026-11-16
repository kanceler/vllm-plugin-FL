"""C500 NT GEMM configuration buckets, with unchanged arithmetic and shapes.

Only the configuration-selection key is rounded. The existing MetaX kernel,
heuristics, pruning and default search policy receive the real M/N/K. A private
best-config namespace prevents representative choices from changing the normal
exact-shape dispatcher. Per-candidate measurements retain exact shape keys.
"""

import importlib
import inspect
import logging
import os
import sys
import time

import torch
import triton

logger = logging.getLogger(__name__)
_PAIRS = frozenset(((2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)))
_REPRESENTATIVES = (1024, 2048, 4096)
_NAMESPACE = "_fl_c500_mbucket_v3"
_state = None


def bucket_m(m):
    return 1 << (m - 1).bit_length()


def configuration_m(m, n, k):
    # The down projection changes its preferred tile in this interval. Use a
    # closer representative there while keeping the other projections sparse.
    if (n, k) == (2048, 6144) and 1536 < m <= 1664:
        return 1600
    return bucket_m(m)


def _supported_config(config):
    model = config.model_config
    hf = model.hf_config
    parallel = config.parallel_config
    return (
        getattr(hf, "architectures", None) == ["LlamaForCausalLM"]
        and getattr(hf, "hidden_size", None) == 2048
        and getattr(hf, "intermediate_size", None) == 6144
        and getattr(hf, "num_hidden_layers", None) == 42
        and getattr(hf, "num_attention_heads", None) == 16
        and getattr(hf, "num_key_value_heads", None) == 2
        and getattr(hf, "vocab_size", None) == 130560
        and model.dtype == torch.bfloat16
        and model.quantization is None
        and config.quant_config is None
        and config.lora_config is None
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1"
    )


class BucketState:
    def __init__(self, module, kernel, tuner, device):
        self.module = module
        self.kernel = kernel
        self.tuner = tuner
        self.device = device
        self.original = module.general_mm_nt
        self.ready = False
        self.warmup_seconds = None

    def eligible(self, a, b, c, m, n, k):
        return (
            self.ready
            and 512 < m <= 4096
            and (n, k) in _PAIRS
            and a.device.type == "cuda"
            and a.device.index == self.device
            and b.device == a.device == c.device
            and a.dtype == b.dtype == c.dtype == torch.bfloat16
            and a.shape == (m, k)
            and b.shape == (k, n)
            and c.shape == (m, n)
            and a.stride() == (k, 1)
            and b.stride() == (1, k)
            and c.stride() == (n, 1)
        )

    def launch(self, a, b, c, m, n, k):
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )
        with self.module.torch_device_fn.device(a.device):
            self.kernel[grid](a, b, c, m, n, k, GROUP_M=8)
        return c

    def dispatch(self, a, b, c, m, n, k):
        # The dominant decode and full-chunk shapes retain their original
        # launch path. Representative measurements are already shared through
        # the exact-shape BenchmarkCache if this is their first normal launch.
        if m <= 512 or m in _REPRESENTATIVES:
            return self.original(a, b, c, m, n, k)
        if self.eligible(a, b, c, m, n, k):
            return self.launch(a, b, c, m, n, k)
        return self.original(a, b, c, m, n, k)

    def warmup(self, model):
        if self.ready:
            return True
        weights = {}
        for _, parameter in model.named_parameters():
            if parameter.ndim != 2:
                continue
            pair = tuple(parameter.shape)
            if (
                pair in _PAIRS
                and parameter.dtype == torch.bfloat16
                and parameter.device.type == "cuda"
                and parameter.device.index == self.device
                and parameter.is_contiguous()
            ):
                weights.setdefault(pair, parameter.detach())
        if weights.keys() != _PAIRS:
            logger.warning("C500 GEMM buckets inactive: expected NT weights missing")
            return False

        start = time.monotonic()
        generator = torch.Generator(device=f"cuda:{self.device}").manual_seed(0)
        with torch.cuda.device(self.device), torch.inference_mode():
            for (n, k), weight in sorted(weights.items()):
                representatives = _REPRESENTATIVES
                if (n, k) == (2048, 6144):
                    representatives = (1024, 1600, 2048, 4096)
                for representative in representatives:
                    variants = (representative, representative - 16, representative - 1)
                    if representative not in _REPRESENTATIVES:
                        # The extra bucket spans one 128-row output tile. Tune
                        # a masked midpoint, then load both masked and aligned
                        # launch variants without repeating configuration search.
                        variants = (
                            representative - 1,
                            representative,
                            representative - 16,
                            representative + 64,
                        )
                    allocation_rows = max(variants)
                    a = torch.randn(
                        (allocation_rows, k),
                        device=weight.device,
                        dtype=weight.dtype,
                        generator=generator,
                    )
                    c = torch.empty(
                        (allocation_rows, n),
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    for m in variants:
                        self.launch(a[:m], weight.t(), c[:m], m, n, k)
            torch.cuda.synchronize(self.device)
        self.warmup_seconds = time.monotonic() - start
        self.ready = True
        logger.info(
            "C500 NT GEMM buckets ready: M>512 through 4096, representatives=%s, "
            "down also uses 1600, 4 projections, real shapes retained, "
            "warmup=%.2fs, cache=%s",
            _REPRESENTATIVES,
            self.warmup_seconds,
            self.tuner.config_table_name,
        )
        return True


def _make_kernel(module):
    entry = importlib.import_module("flag_gems.utils.libentry")
    runtime = importlib.import_module("flag_gems.runtime")
    original = module.mm_kernel_nt.fn
    if (
        not isinstance(original, entry.LibTuner)
        or list(original.keys) != ["M", "N", "K"]
        or original._flagtune_mode is not runtime.TuningMode.DEFAULT
        or runtime.resolve_tuning_mode("mm", supports_cost_model=False)
        is not runtime.TuningMode.DEFAULT
    ):
        raise ValueError("unsupported NT tuner or non-default FlagTune mode")

    class BucketPolicy(type(original)):
        def _make_config_table_name(self):
            return super()._make_config_table_name() + _NAMESPACE

        def get_key(self, args):
            key = super().get_key(args)
            return (configuration_m(args["M"], args["N"], args["K"]),) + key[1:]

        def run(self, *args, **kwargs):
            try:
                return super().run(*args, **kwargs)
            finally:
                # This private tuner never uses the interactive benchmark API.
                # Do not retain warmup/forward activations after a launch.
                self._last_benchmark_args = None
                self._last_benchmark_meta = None

    # Reuse the immutable heuristics/JIT chain, with independent tuner and
    # LibEntry launch caches. Do not modify the existing kernel or tuner.
    tuner = entry.libtuner(
        configs=original.configs,
        key=["M", "N", "K"],
        strategy=[bucket_m, "default", "default"],
        prune_configs_by={"early_config_prune": module._prune_mm_dense_configs_nt},
        policy=BucketPolicy,
    )(original.fn)
    if tuner._benchmark_protocol != original._benchmark_protocol:
        raise ValueError("NT benchmark protocol differs")
    return entry.libentry()(tuner), tuner


def active_module():
    """Resolve the globals used by the registered vendor mm/mm_out callables.

    FlagGems imports its vendor as ``_metax``; importing the same file under
    ``flag_gems.runtime.backend`` creates an independent module and tuner.
    """
    import flag_gems

    functions = [inspect.unwrap(getattr(flag_gems, name)) for name in ("mm", "mm_out")]
    namespace = functions[0].__globals__
    module = sys.modules.get(namespace["__name__"])
    if (
        module is None
        or module.__dict__ is not namespace
        or functions[1].__globals__ is not namespace
        or not module.__file__.replace("\\", "/").endswith("/_metax/ops/mm.py")
    ):
        raise ValueError("registered mm/mm_out do not share the MetaX NT module")
    return module


def install(config):
    """Install before model profiling; routing stays original until warmed."""
    global _state

    def inactive():
        if _state is not None:
            _state.ready = False
        return None

    if os.environ.get("VLLM_FL_C500_GEMM_BUCKET", "1") != "1":
        return inactive()
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return inactive()
    if not _supported_config(config):
        return inactive()
    device = torch.cuda.current_device()
    if torch.cuda.get_device_name(device).strip().lower() != "metax c500":
        return inactive()
    if _state is not None:
        return _state if _state.device == device else inactive()
    try:
        module = active_module()
        kernel, tuner = _make_kernel(module)
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        logger.warning("C500 GEMM buckets unavailable: %s", error)
        return None
    state = BucketState(module, kernel, tuner, device)
    module.general_mm_nt = state.dispatch
    _state = state
    return state


def warmup_model(state, model):
    if state is None:
        return
    try:
        state.warmup(model)
    except RuntimeError:
        state.ready = False
        logger.exception("C500 GEMM bucket warmup failed; using original NT path")
