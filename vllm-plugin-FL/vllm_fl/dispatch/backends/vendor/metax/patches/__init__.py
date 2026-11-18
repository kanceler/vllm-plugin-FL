from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
def install_model_patches():
    """Install model-dependent patches after Linear has finished importing."""
    import importlib
    import sys
    linear = sys.modules.get("vllm.model_executor.layers.linear")
    if (linear is None or not hasattr(linear, "UnquantizedLinearMethod")
            or getattr(getattr(linear, "__spec__", None), "_initializing", False)):
        return False
    importlib.import_module(__name__ + ".gdn_linear_attn")
    return True


# Platform discovery can occur inside Linear's own import. Keep low-level
# patches available immediately, then retry model registration at plugin load.
install_model_patches()

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False
