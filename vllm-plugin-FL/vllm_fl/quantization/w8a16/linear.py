# SPDX-License-Identifier: Apache-2.0
"""Opt-in local-shard quantization preserving vLLM TP forward and bias ordering."""
import torch
from vllm.logger import init_logger

from .reference import quantize_weight
from .policy import PROJECTIONS, quantization_flags

logger = init_logger(__name__)
_MARKER = "_fl_tianshu_w8a16_separate_installed"


def install_tianshu_separate():
    flags = quantization_flags()
    if not any(flags):
        return False
    from flag_gems import runtime
    from vllm.model_executor.layers import linear as upstream
    from vllm_fl.utils import is_oot_enabled
    if runtime.device.vendor_name != "iluvatar" or not is_oot_enabled():
        return False
    if upstream.envs.VLLM_BATCH_INVARIANT:
        logger.warning("Separate disabled: batch invariant path is unvalidated")
        return False
    cls = upstream.UnquantizedLinearMethod
    if getattr(cls, _MARKER, False):
        return True
    from .cache import install_cache_identity
    install_cache_identity()
    from .separate import separate_linear
    original = cls.process_weights_after_loading

    class SeparateLinearMethod(cls):
        def apply(self, layer, x, bias=None):
            # Framework forward retains gather/reduce and skip_bias_add behavior.
            return separate_linear(x, layer.weight, layer.w8a16_scale, bias,
                                   layer._fl_tianshu_projection_id)

    def process(self, layer):
        if getattr(layer, "_fl_tianshu_separate_ready", False):
            return
        original(self, layer)
        projection = getattr(layer, "prefix", "").split(".")[-1]
        projection_id = PROJECTIONS.get(projection)
        eligible = (type(self) is cls and projection_id is not None and flags[projection_id]
                    and isinstance(layer, (upstream.ColumnParallelLinear,
                                           upstream.RowParallelLinear)))
        if not eligible:
            return
        weight = layer.weight
        if weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
            logger.warning("Separate fallback prefix=%s reason=weight_dtype_or_device", layer.prefix)
            return
        try:
            q, scale = quantize_weight(weight)
        except ValueError as exc:
            logger.warning("Separate fallback prefix=%s reason=%s", layer.prefix, exc)
            return
        layer.weight = torch.nn.Parameter(q, requires_grad=False)
        layer.register_buffer("w8a16_scale", scale)
        layer._fl_tianshu_separate_ready = True
        layer._fl_tianshu_projection_id = projection_id
        layer.quant_method = SeparateLinearMethod()
        print(f"W8A16 enabled projection={projection} prefix={layer.prefix} local_shape={tuple(q.shape)}", flush=True)
        logger.info("Separate enabled prefix=%s local_shape=%s tp_size=%s weight=int8 "
                    "scale=float32 activation=bf16 mode=offline_dispatch",
                    layer.prefix, tuple(q.shape), getattr(layer, "tp_size", None))

    cls.process_weights_after_loading = process
    setattr(cls, _MARKER, True)
    return True
