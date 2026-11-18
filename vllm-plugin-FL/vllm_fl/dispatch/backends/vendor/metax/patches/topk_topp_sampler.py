# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# The upstream Triton topk_topp kernel fails to compile on this MetaX stack.
# C500 has a guarded mass-selection implementation; all other cases retain
# the existing PyTorch implementation.

import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    from .top_p_selection import try_apply

    selected = try_apply(logits, k, p)
    if selected is not None:
        return selected
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
