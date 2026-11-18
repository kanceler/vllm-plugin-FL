# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

"""Prefill 压力自适应的 chunk 预算调度器。

## 问题

基线 ``max_num_batched_tokens=2048`` 是每个调度步的硬 token 预算上限。
``Scheduler.schedule()`` 顺序取用这个预算（深度优先），所以单次 prefill 的
q_len 被钉死在 2048，Prefill Attention 的算力利用率只有 32%，而整序列可达
45.6%。实测把 chunk 提到 8192：端到端 +14.1%，TTFT -17.8%。

详见 ``改动记录/测试总结与优化方向.md``。

## 为什么不能只改启动参数

1. 赛题要求 ``vllm serve`` 参数与基线一致；
2. 即使能改，它也是个静态值，无法同时满足「prefill 效率」与
   「decode 响应性」这两个互相冲突的目标。

## 为什么必须同时抬高 max_num_batched_tokens

不能「参数保持 2048、调度器内部按 8192 调度」：

    vllm/v1/worker/gpu_model_runner.py:469
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
    vllm/v1/worker/gpu_model_runner.py:5741
        assert num_tokens <= self.max_num_tokens

调度步的总 token 数一旦超过该值，就会撞断言 / 溢出固定长度的缓冲区。
所以必须：

* 把 ``max_num_batched_tokens`` 抬到上界 —— 缓冲区按上界分配；
* 把 ``max_num_scheduled_tokens`` 设为基线值 —— 作为运行期的默认预算。

这正是在 vLLM 里 ``max_num_scheduled_tokens`` 这个字段存在的意义
（其文档：「通常等于 max_num_batched_tokens，但可以更小」）。
``VllmConfig.__post_init__`` 在平台钩子之后会重算 compile ranges
（见 ``vllm/config/vllm.py:1350-1357``），因此这一改动在 vLLM 支持的契约内。

## 策略

单个旋钮：活跃 decode 请求数门限 ``N``。

    N < 0（默认）  →  不设门限，始终用上界
    n_decode <= N  →  用上界预算（最大化 q_len，换 prefill 效率）
    n_decode >  N  →  回落到基线预算（保护 decode 的 token 间隔）

**为什么默认不设门限**：实测（见改动文档 E1/E2 两组实验）表明，
本负载下 decode 几乎全程存在，``N=0`` 与 ``N=-1`` 的吞吐/TTFT 差异在噪声内；
而 ITL 实测显示，大预算**改善了**每请求 ITL 中位数（43.5→41.5ms）和 P95
（43.9→41.7ms），只让全局 ITL 尾部变差（P95 253.9→603.8ms，尖峰≈一个 chunk
的步耗时，是 chunked prefill 的固有结构）。也就是说，门限想保护的东西
（典型 ITL）本来就没有变差。故默认不设门限，旋钮保留给其他负载形态。
"""

import importlib
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# scheduler_cls 以「模块路径字符串」注册，避免在 platform 配置钩子里
# 提前拉起 vLLM 调度器模块（那时 vllm.config 仍在构造中，有循环导入风险）。
SCHEDULER_CLS = "vllm_fl.worker.adaptive_sched.AdaptiveSchedulerFL"
ASYNC_SCHEDULER_CLS = "vllm_fl.worker.adaptive_sched.AdaptiveAsyncSchedulerFL"


class AdaptiveChunkBudgetMixin:
    """按活跃 decode 请求数自适应单步 prefill chunk 预算。

    可同时组合进 ``Scheduler`` 与 ``AsyncScheduler``：二者的 ``schedule()``
    是同一个实现（``AsyncScheduler`` 并未覆写它）。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        from vllm_fl import envs

        self._fl_adaptive_enabled = envs.VLLM_FL_ADAPTIVE_CHUNK_ENABLED
        self._fl_chunk_cap = min(
            envs.VLLM_FL_ADAPTIVE_CHUNK_MAX,
            self.scheduler_config.max_num_batched_tokens,
        )
        self._fl_decode_gate = envs.VLLM_FL_ADAPTIVE_CHUNK_DECODE_GATE
        # 只有 prompt 长度 ≥ 这个门限，才把单步预算抬到 cap。
        # 0（默认）= 自动取 4 × 基线预算（基线 2048 ⇒ 8192）。
        _lp = envs.VLLM_FL_ADAPTIVE_CHUNK_LONG_PROMPT
        self._fl_long_prompt = (
            _lp if _lp > 0 else 4 * self.max_num_scheduled_tokens
        )

        if self._fl_adaptive_enabled:
            logger.info(
                "FL adaptive chunk budget: base=%d, cap=%d, decode_gate=%d "
                "(-1 = no gate), long_prompt_threshold=%d "
                "(only prompts >= this use the cap).",
                self.max_num_scheduled_tokens,
                self._fl_chunk_cap,
                self._fl_decode_gate,
                self._fl_long_prompt,
            )

    def _fl_num_decoding(self) -> int:
        """统计已跑完 prompt、正在 decode 的请求数。

        语义与 vLLM 自身的 ``Request.is_prefill_chunk`` 判定一致
        （``scheduler.py``：``num_computed_tokens < num_tokens``），
        但这里在 ``schedule()`` 入口求值——那个属性是每步末尾才刷新的，
        在 ``schedule()`` 里读到的是上一拍的快照。
        """
        count = 0
        for request in self.running:
            if request.num_computed_tokens >= request.num_prompt_tokens:
                count += 1
        return count

    def _fl_max_prefilling_prompt_len(self) -> int:
        """当前「正在预填」或「还在排队」的请求里，最长的 prompt 长度。

        用 ``num_prompt_tokens``（整段 prompt 的长度）而不是剩余量，
        因为它不随 chunk 推进而变化，判据稳定。
        """
        best = 0
        for request in self.running:
            if request.num_computed_tokens < request.num_prompt_tokens:
                if request.num_prompt_tokens > best:
                    best = request.num_prompt_tokens
        for request in self.waiting:
            if request.num_prompt_tokens > best:
                best = request.num_prompt_tokens
        return best

    def _fl_adaptive_budget(self, base: int) -> int:
        if not self._fl_adaptive_enabled:
            return base
        cap = self._fl_chunk_cap
        if cap <= base:
            return base
        gate = self._fl_decode_gate
        if gate >= 0 and self._fl_num_decoding() > gate:
            return base
        # 只有长 prompt 才值得付出「步变长」的代价换 prefill 效率。
        # 短 prompt（会被切成很少几块）抬高预算只会拖慢 decode。
        if self._fl_max_prefilling_prompt_len() < self._fl_long_prompt:
            return base
        return cap

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        base = self.max_num_scheduled_tokens
        budget = self._fl_adaptive_budget(base)
        if budget == base:
            return super().schedule(*args, **kwargs)

        # Scheduler.schedule() 只在入口读一次 self.max_num_scheduled_tokens
        # 作为本步预算，所以临时替换这一个属性就够了，不必重写整个调度循环
        # ——也就不会随 vLLM 版本漂移。
        self.max_num_scheduled_tokens = budget
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_scheduled_tokens = base


# 类名 → (基类所在模块, 基类名)。延迟到真正取值时才 import。
_LAZY_CLASS_BASES: dict[str, tuple[str, str]] = {
    "AdaptiveSchedulerFL": ("vllm.v1.core.sched.scheduler", "Scheduler"),
    "AdaptiveAsyncSchedulerFL": (
        "vllm.v1.core.sched.async_scheduler",
        "AsyncScheduler",
    ),
}
_LAZY_CLASS_CACHE: dict[str, type] = {}


def __getattr__(name: str) -> Any:
    """PEP 562：按需构造两个调度器子类。

    ``scheduler_cls`` 是字符串，vLLM 在**调度器实例化时**才通过
    ``resolve_obj_by_qualname()`` -> ``getattr(module, name)`` 取这两个类。
    那一刻 ``VllmConfig`` 早已构造完毕，再 import vLLM 的调度器模块是安全的。
    """
    bases = _LAZY_CLASS_BASES.get(name)
    if bases is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name not in _LAZY_CLASS_CACHE:
        module_path, base_name = bases
        base = getattr(importlib.import_module(module_path), base_name)
        doc = (
            "非 async 调度路径（本项目默认路径：未传 --async-scheduling）。"
            if base_name == "Scheduler"
            else "async 调度路径。"
        )
        _LAZY_CLASS_CACHE[name] = type(
            name,
            (AdaptiveChunkBudgetMixin, base),
            {"__doc__": doc, "__module__": __name__},
        )
    return _LAZY_CLASS_CACHE[name]



def _also_compile_for(budget: int) -> None:
    """让 torch.compile 的编译区间额外覆盖 ``budget``。

    抬高 ``max_num_batched_tokens`` 会连带改变编译区间上限
    （``vllm/config/vllm.py`` 的 ``_set_compile_ranges`` 直接拿它当上界）。
    绝大多数调度步仍落在基线预算内，若只为更大的上界编译，
    短步会拿到对更大形状特化的 kernel——实测这一项会让 4k 场景慢约 7%。

    这里把基线预算补进编译区间：短步用为它特化的那份，长步用为大上界特化的那份。
    """

    try:
        from vllm.config.vllm import VllmConfig
    except Exception:  # noqa: BLE001
        return
    if getattr(VllmConfig, "_fl_compile_ranges_patched", False):
        return
    original = VllmConfig._set_compile_ranges

    def _patched(self) -> None:
        original(self)
        cc = self.compilation_config
        endpoints = getattr(cc, "compile_ranges_endpoints", None)
        if endpoints and budget not in endpoints:
            cc.compile_ranges_endpoints = sorted(set(list(endpoints) + [budget]))
            logger.info(
                "FL compile ranges extended to %s (baseline budget %d kept).",
                cc.compile_ranges_endpoints,
                budget,
            )

    VllmConfig._set_compile_ranges = _patched
    VllmConfig._fl_compile_ranges_patched = True


def apply_adaptive_chunk_config(vllm_config: Any) -> None:
    """把 prefill 预算与缓冲区上界解耦，并挂上自适应调度器。

    在 ``PlatformFL.check_and_update_config`` 中调用（引擎核心进程，
    ``VllmConfig.__post_init__`` 期间），早于调度器实例化。
    """
    from vllm_fl import envs

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        return
    if not envs.VLLM_FL_ADAPTIVE_CHUNK_ENABLED:
        logger.info("FL adaptive chunk budget disabled by env; using stock scheduler.")
        return

    base = scheduler_config.max_num_batched_tokens
    # max_model_len 在 ModelConfig 上，不在 SchedulerConfig 上；且 model_config
    # 在测试路径下可能为 None。
    model_config = getattr(vllm_config, "model_config", None)
    cap = envs.VLLM_FL_ADAPTIVE_CHUNK_MAX
    if model_config is not None:
        cap = min(cap, model_config.max_model_len)
    if cap <= base:
        logger.info("FL adaptive chunk budget inactive: cap=%d <= base=%d.", cap, base)
        return

    # 缓冲区按上界分配，运行期预算从基线值起步。
    scheduler_config.max_num_batched_tokens = cap
    scheduler_config.max_num_scheduled_tokens = base
    # 这两项在 SchedulerConfig.__post_init__ 里由 max_num_batched_tokens 派生；
    # 我们是在其后修改的，需要同步（仅多模态场景会用到）。
    scheduler_config.max_num_encoder_input_tokens = cap
    scheduler_config.encoder_cache_size = cap

    # 缓冲区变大不该拖慢"仍在基线预算内"的步：让编译区间把基线也覆盖进来。
    _also_compile_for(base)

    if scheduler_config.scheduler_cls is None:
        scheduler_config.scheduler_cls = (
            ASYNC_SCHEDULER_CLS
            if scheduler_config.async_scheduling
            else SCHEDULER_CLS
        )
    elif scheduler_config.scheduler_cls not in (SCHEDULER_CLS, ASYNC_SCHEDULER_CLS):
        logger.warning(
            "scheduler_cls is already set to %s; FL adaptive chunk budget will "
            "size buffers to %d but the stock scheduling policy still applies.",
            scheduler_config.scheduler_cls,
            cap,
        )

    logger.info(
        "FL adaptive chunk budget enabled: buffer cap=%d (was %d), "
        "runtime budget=%d, decode_gate=%d.",
        cap,
        base,
        base,
        envs.VLLM_FL_ADAPTIVE_CHUNK_DECODE_GATE,
    )
