# SPDX-License-Identifier: Apache-2.0
"""Install Down-only load-time quantization without replacing TP forward logic."""

import os

import torch

from .reference import dequantize_reference, quantize_weight

from vllm.logger import init_logger

logger = init_logger("vllm.fl.w8a16")
_MARKER = "_fl_w8a16_down_installed"


def install_w8a16_down() -> bool:
    if os.environ.get("VLLM_FL_W8A16_DOWN", "1").lower() in ("0", "false", "no"):
        print("[FL] w8a16 down disabled by env; using upstream defaults.")
        return False
    from vllm.model_executor.layers.linear import (
        RowParallelLinear, UnquantizedLinearMethod,
    )
    from vllm_fl.utils import is_oot_enabled

    if not is_oot_enabled():
        return False
    if getattr(UnquantizedLinearMethod, _MARKER, False):
        return True
    int8pack_enabled = os.environ.get(
        "VLLM_FL_W8A16_DOWN_INT8PACK", "0").lower() in ("1", "true", "yes")
    legacy_fused_enabled = os.environ.get(
        "VLLM_FL_W8A16_DOWN_FUSED", "0").lower() in ("1", "true", "yes")
    if int8pack_enabled:
        # Register the opaque scale-after-dot op before Dynamo traces forward.
        from .int8pack import int8pack_linear, supports_int8pack
    elif legacy_fused_enabled:
        # Register the custom op before Dynamo starts tracing model forward.
        from .fused import fused_linear, supports_fused

    original_process = UnquantizedLinearMethod.process_weights_after_loading

    def process(self, layer):
        # Quantize only completed local TP shards, leaving loaders and all
        # RowParallelLinear communication/bias handling intact.
        if getattr(layer, "_fl_w8a16_ready", False):
            return
        selected = (isinstance(layer, RowParallelLinear)
                    and getattr(layer, "prefix", "").split(".")[-1] == "down_proj"
                    and type(self) is UnquantizedLinearMethod)
        if not selected:
            if getattr(layer, "prefix", "").split(".")[-1] == "down_proj":
                logger.warning("W8A16 fallback prefix=%s layer_type=%s method_type=%s "
                               "reason=unsupported_linear_method",
                               layer.prefix, type(layer).__name__, type(self).__name__)
            return original_process(self, layer)
        weight = layer.weight
        reason = None
        if weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
            reason = "requires CUDA-alike BF16 weight"
        elif weight.ndim != 2 or min(weight.shape) == 0:
            reason = "requires nonempty 2D weight"
        if reason is None:
            try:
                q, scale = quantize_weight(weight)
            except ValueError as exc:
                reason = str(exc)
        if reason is not None:
            logger.warning("W8A16 fallback prefix=%s reason=%s", layer.prefix, reason)
            return original_process(self, layer)
        # Keep the canonical weight name for shape/device inspection; loading
        # has finished, so the loader-specific Parameter is no longer needed.
        layer.weight = torch.nn.Parameter(q, requires_grad=False)
        layer.register_buffer("w8a16_scale", scale)
        layer._fl_w8a16_ready = True
        layer.quant_method = W8A16DownLinearMethod()
        logger.info("W8A16 enabled prefix=%s local_shape=%s tp_size=%s "
                    "weight=int8 scale=float32 mode=%s",
                    layer.prefix, tuple(q.shape), getattr(layer, "tp_size", None),
                    ("int8pack_scale_after_dot" if int8pack_enabled else
                     "legacy_fused_scale_before_dot" if legacy_fused_enabled else
                     "separate_dequant"))

    class W8A16DownLinearMethod(UnquantizedLinearMethod):
        """Only installed on a successfully quantized, fully loaded Down."""

        def apply(self, layer, x, bias=None):
            if x.dtype != torch.bfloat16 or x.device != layer.weight.device:
                # No original BF16 copy is retained after quantization.
                raise ValueError("W8A16 requires BF16 input on weight device")
            from vllm.model_executor.layers import linear as upstream
            if int8pack_enabled and not upstream.envs.VLLM_BATCH_INVARIANT:
                if supports_int8pack(x, layer.weight, layer.w8a16_scale, bias):
                    return int8pack_linear(x, layer.weight, layer.w8a16_scale, bias)
                logger.warning_once(
                    "W8A16 int8pack fallback: unsupported input/layout/bias; "
                    "using separate BF16 dequantization")
            elif legacy_fused_enabled and not upstream.envs.VLLM_BATCH_INVARIANT:
                if supports_fused(x, layer.weight, layer.w8a16_scale, bias):
                    return fused_linear(x, layer.weight, layer.w8a16_scale, bias)
                logger.warning_once("W8A16 fused fallback: unsupported input/layout; "
                                    "using separate BF16 dequantization")
            weight = dequantize_reference(layer.weight, layer.w8a16_scale)
            # Preserve the original BF16 GEMM selector and invariant branch.
            if (upstream.envs.VLLM_BATCH_INVARIANT
                    and upstream.current_platform.is_cuda_alike()):
                return upstream.linear_batch_invariant(x, weight, bias)
            return upstream.dispatch_unquantized_gemm()(layer, x, weight, bias)

    UnquantizedLinearMethod.process_weights_after_loading = process
    setattr(UnquantizedLinearMethod, _MARKER, True)
    return True
