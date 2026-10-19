# Copyright (c) 2025 BAAI. All rights reserved.

import os
from typing import Any, Callable

from vllm_fl.utils import use_flaggems


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("true", "1")


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_FL_PREFER_ENABLED": lambda: (
        os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    ),
    "FLAGGEMS_ENABLE_OPLIST_PATH": lambda: os.environ.get(
        "FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"
    ),
    "USE_FLAGGEMS": use_flaggems,
    # --- Prefill chunk 预算自适应，见 改动记录/002-framework-prefill自适应chunk预算调度.md ---
    # 总开关。关掉即完全退化为 vLLM 原生行为（基线）。
    "VLLM_FL_ADAPTIVE_CHUNK_ENABLED": lambda: _env_bool(
        "VLLM_FL_ADAPTIVE_CHUNK_ENABLED", "True"
    ),
    # 单步 token 预算的上界（同时是模型 runner 缓冲区的大小）。
    # 若设成等于 max_num_batched_tokens（即 2048），本功能自动失效 → 等价基线。
    #
    # 默认 4096：这是【与 attention 逐层同步消除配套】后重扫出的最优值。
    # 同步存在时扫出的最优曾是 8192，但同步消除后瓶颈迁移，8192 会使
    # 4K 场景 TTFT 升至基线的 1.23 倍（违反约束），4096 则四项指标全面占优。
    # 两者必须配套使用，详见技术报告第三章。
    "VLLM_FL_ADAPTIVE_CHUNK_MAX": lambda: _env_int(
        "VLLM_FL_ADAPTIVE_CHUNK_MAX", "4096"
    ),
    # 活跃 decode 请求数门限：不超过它才用上界预算。
    #   -1（默认）= 不设门限，始终用上界。
    #                实测：门限=0 与门限=-1 的吞吐/TTFT 差异在噪声内，
    #                因为 decode 几乎全程存在，门限形同虚设。
    #   >=0        = 仅在 decode 数不超过该值时放大（留给其他负载形态的旋钮）。
    "VLLM_FL_ADAPTIVE_CHUNK_DECODE_GATE": lambda: _env_int(
        "VLLM_FL_ADAPTIVE_CHUNK_DECODE_GATE", "-1"
    ),
    # 只有「长 prompt」才把预算抬到上界。0 = 自动（4 × 基线预算）。
    #
    # 为什么要这个门限：抬高 chunk 会让**每个调度步变长**（实测 4k 场景
    # Mean ITL 59.6→71.3 ms）。这对 prefill 主导的负载有利（少切几块、
    # q_len 更大、attention 效率更高），但对 decode 主导的负载有害
    # （步变长 = decode 出 token 变慢）。
    # 实测：16k 场景（prompt 16384）抬高后 +11.6%；4k 场景（prompt 4096）
    # 抬高后 total tokens/s −16.0%、TTFT +8.6%，两条硬约束都踩。
    # 判据用「prompt 长度 ≥ 4×基线预算」——即该 prompt 会被切成 4 块以上，
    # 这才值得为它增大步长。详见 改动记录/014。
    "VLLM_FL_ADAPTIVE_CHUNK_LONG_PROMPT": lambda: _env_int(
        "VLLM_FL_ADAPTIVE_CHUNK_LONG_PROMPT", "0"
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in fl_vllm_environment_variables:
        return fl_vllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(fl_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in fl_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
