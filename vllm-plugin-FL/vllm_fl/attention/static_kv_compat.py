"""CPU planning and launch guards for page32 static KV + BI-V150 #011/#013.

No GPU scalar reads. Classification happens in MetadataBuilder, before layer
execution/capture. A graph replay must preserve the runner's dense-prefix ABI.
"""
from dataclasses import dataclass
import os


def feature_enabled(name):
    return os.environ.get(name, "1").lower() not in ("0", "false", "no")


@dataclass(frozen=True)
class PrefixLayout:
    decode_rows: int
    all_decode: bool
    valid: bool


def classify_prefix(starts, actual_tokens):
    if len(starts) < 2 or starts[0] != 0 or starts[-1] != actual_tokens:
        return PrefixLayout(0, False, False)
    lengths = [right - left for left, right in zip(starts, starts[1:])]
    if any(length < 0 for length in lengths):
        return PrefixLayout(0, False, False)
    padding = False
    for length in lengths:
        if length == 0:
            padding = True
        elif padding:
            # Internal holes are not the runner's verified trailing-padding ABI.
            return PrefixLayout(0, False, False)
    decodes = 0
    for length in lengths:
        if length != 1:
            break
        decodes += 1
    pure = decodes > 0 and all(length == 0 for length in lengths[decodes:])
    return PrefixLayout(decodes, pure, True)


@dataclass(frozen=True)
class StaticMixedPlan:
    num_decodes: int
    decode_starts: object
    prefill_starts: object
    source_starts: object
    leading_prefill_tokens: int = 0


@dataclass(frozen=True)
class GatherMixedPlan(StaticMixedPlan):
    prefill_row: int = 0
    prefill_begin: int = 0
    prefill_end: int = 0
    decode_rows: object = None
    decode_tokens: object = None


def attach_plan(metadata, common):
    """Use the existing CPU prefix; create/reuse rebased GPU prefixes once/step."""
    metadata._fl_static_mixed_plan = None
    metadata._fl_static_decode_starts = None
    cpu = getattr(common, "query_start_loc_cpu", None)
    if cpu is None or cpu.device.type != "cpu":
        return metadata
    if (metadata.causal is not True or metadata.use_cascade
            or metadata.mm_prefix_range_tensor is not None):
        return metadata
    starts = cpu[:common.num_reqs + 1].tolist()
    layout = classify_prefix(starts, metadata.num_actual_tokens)
    if not layout.valid:
        return metadata
    if layout.all_decode:
        metadata._fl_static_decode_starts = metadata.query_start_loc
        return metadata
    if not feature_enabled("VLLM_FL_MIXED_ATTN_SPLIT"):
        return metadata
    d = layout.decode_rows
    extended_split = os.environ.get(
        "VLLM_FL_BIV150_MIXED_EXTENDED_SPLIT", "0"
    ).lower() in ("1", "true")
    prefill_tokens = starts[1]
    trailing_decodes = common.num_reqs - 1
    if (
        extended_split
        and prefill_tokens > 1
        and 16 <= trailing_decodes <= 80
        and trailing_decodes <= metadata.softmax_segm_output.shape[0]
        and all(right - left == 1 for left, right in zip(starts[1:], starts[2:]))
    ):
        gpu_starts = metadata.query_start_loc
        metadata._fl_static_mixed_plan = StaticMixedPlan(
            trailing_decodes,
            gpu_starts[1:common.num_reqs + 1] - prefill_tokens,
            gpu_starts[:2],
            gpu_starts,
            prefill_tokens,
        )
        return metadata
    if (
        extended_split
        and 0 < d < 16
        and 16 <= trailing_decodes <= 80
        and trailing_decodes <= metadata.softmax_segm_output.shape[0]
        and starts[d + 1] - starts[d] >= 1024
        and all(right - left == 1 for left, right in zip(starts[d + 1:], starts[d + 2:]))
    ):
        import torch

        gpu_starts = metadata.query_start_loc
        begin, end = starts[d:d + 2]
        decode_starts = torch.arange(
            trailing_decodes + 1, dtype=gpu_starts.dtype, device=gpu_starts.device
        )
        # The first d Decode requests are also the first d Query tokens.
        compact = decode_starts[:-1].long()
        after_prefill = compact >= d
        metadata._fl_static_mixed_plan = GatherMixedPlan(
            num_decodes=trailing_decodes,
            decode_starts=decode_starts,
            prefill_starts=gpu_starts[d:d + 2] - begin,
            source_starts=gpu_starts,
            prefill_row=d,
            prefill_begin=begin,
            prefill_end=end,
            decode_rows=compact + after_prefill,
            decode_tokens=compact + after_prefill * (end - begin),
        )
        return metadata
    if (not 16 <= d < common.num_reqs or d > 80
            or d > metadata.softmax_segm_output.shape[0]):
        return metadata
    existing = getattr(metadata, "_fl_mixed_plan", None)
    if (existing is not None and existing.source_starts is metadata.query_start_loc
            and existing.num_decodes == d):
        plan = existing
    else:
        plan = StaticMixedPlan(d, metadata.query_start_loc[:d + 1],
                               metadata.query_start_loc[d:] - d,
                               metadata.query_start_loc)
    metadata._fl_static_mixed_plan = plan
    return metadata


def split_attention_args(args, plan):
    """Split only query/sequence views, never the physical cache or token scales.

    The caller has already written the complete incremental KV set exactly once.
    Returning None keeps the original static-KV attention invocation intact.
    """
    if (plan is None or args.get("cu_seqlens_q") is not plan.source_starts
            or args.get("max_seqlen_q", 1) <= 1
            or args.get("causal") is not True
            or args.get("mm_prefix_range") is not None
            or any(args.get(k) is not None for k in ("alibi_slopes", "sinks", "qq_bias"))
            or args.get("use_td", False)
            or args.get("window_size", (-1, -1)) != (-1, -1)
            or args.get("softcap", 0) not in (None, 0)):
        return None
    d = plan.num_decodes
    decode, prefill = dict(args), dict(args)
    if isinstance(plan, GatherMixedPlan):
        start, end = plan.prefill_begin, plan.prefill_end
        row = plan.prefill_row
        decode["q"] = args["q"].index_select(0, plan.decode_tokens)
        decode["out"] = decode["q"].new_empty(decode["q"].shape)
        prefill["q"], prefill["out"] = args["q"][start:end], args["out"][start:end]
        for name in ("seqused_k", "block_table"):
            decode[name] = args[name].index_select(0, plan.decode_rows)
            prefill[name] = args[name][row:row + 1]
    else:
        prefill_tokens = getattr(plan, "leading_prefill_tokens", 0)
        for name in ("q", "out", "seqused_k", "block_table"):
            if prefill_tokens:
                prefix = prefill_tokens if name in ("q", "out") else 1
                decode[name] = args[name][prefix:prefix + d]
                prefill[name] = args[name][:prefix]
            else:
                decode[name] = args[name][:d]
                prefill[name] = args[name][d:]
    decode["cu_seqlens_q"] = plan.decode_starts
    decode["max_seqlen_q"] = 1
    prefill["cu_seqlens_q"] = plan.prefill_starts
    return decode, prefill


ORIGINAL_LOOKUP = """    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )"""
DIRECT_LOOKUP = """    # Qualified static-KV dense single-query prefix and trailing padding.
    seq_idx = q_block_global_idx
    q_block_local_idx = 0
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_batch_in_all_start_index
    seq_len = tl.load(seq_lens_ptr + seq_idx)"""


def specialize_source(source):
    """Transform the FINAL static arithmetic, not the original BF16 source."""
    if source.count(ORIGINAL_LOOKUP) != 1:
        raise ValueError("Unsupported static-KV direct-index lookup source")
    return source.replace(ORIGINAL_LOOKUP, DIRECT_LOOKUP, 1)


def compact_grid(kwargs, grid, *, bf16, int8, kv_mode):
    expected = dict(IS_3D=True, num_query_heads=16, num_queries_per_kv=8,
                    HEAD_SIZE=128, HEAD_SIZE_PADDED=128, BLOCK_SIZE=32,
                    BLOCK_M=16, BLOCK_Q=2, KV_QUANT_MODE=kv_mode, USE_CAUSAL=True)
    if any(kwargs.get(key) != value for key, value in expected.items()):
        return None
    if any(kwargs.get(k, False) for k in (
        "USE_ALIBI_SLOPES", "USE_ALIBI_SQRT", "USE_QQ_BIAS", "USE_SOFTCAP",
        "USE_SINKS", "USE_PER_SEQ_CAUSAL", "USE_MM_PREFIX", "USE_TD", "USE_FP8",
    )):
        return None
    if kwargs.get("SLIDING_WINDOW", 0) > 0 or kwargs.get("CHUNK_LOOKBACK", -1) >= 0:
        return None
    n = kwargs.get("num_seqs", 0)
    if not 1 <= n <= 80 or kwargs.get("NUM_SEGMENTS_PER_SEQ") not in (8, 16, 32):
        return None
    q = kwargs["query_ptr"]
    if q.device.type != "cuda" or q.dtype != bf16 or not 1 <= q.shape[0] <= n:
        return None
    if (kwargs["output_ptr"].dtype != bf16
            or kwargs["key_cache_ptr"].dtype != int8
            or kwargs["value_cache_ptr"].dtype != int8):
        return None
    if not isinstance(grid, tuple) or len(grid) != 3 or grid[1:] != (2, kwargs["NUM_SEGMENTS_PER_SEQ"]):
        return None
    return (n, *grid[1:])


class StaticDecodeKernel:
    """Only installed on a launcher explicitly qualified by static metadata."""
    def __init__(self, original, specialized, *, bf16, int8, kv_mode):
        self.original, self.specialized = original, specialized
        self.types = dict(bf16=bf16, int8=int8, kv_mode=kv_mode)
        self.reported = False

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            compact = None if args else compact_grid(kwargs, grid, **self.types)
            if compact is None:
                return self.original[grid](*args, **kwargs)
            if not self.reported:
                print(f"[FL] Static KV compact Decode active: grid={compact}; capture/eager evidence only", flush=True)
                self.reported = True
            return self.specialized[compact](**kwargs)
        return launch
