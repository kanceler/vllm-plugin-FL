"""C500 top-p mass selection without sorting or changing RNG.

Use all 32 bits of the FP32 logits as ordered keys and FP64 unnormalized
probability mass. This avoids the BF16-key assumption and the FP32-CDF
boundary error of the exploratory prototype. Near the cutoff, FP32 sort/
softmax/scan can still select a different last token; correctness is checked
against an independent FP64 mass reference, not bitwise FP32 scan results.

Unsupported shapes, top-k, capture, and non-finite/invalid rows retain the
existing MetaX PyTorch path. Scratch is owned per device and stream; returned
logits have independent storage, including when processed logits are requested.
"""

import logging
import os
import threading
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

logger = logging.getLogger(__name__)
_enabled = os.environ.get("VLLM_FL_C500_TOP_P_SELECTION", "1") == "1"
_workspaces = OrderedDict()
_cache_lock = threading.Lock()
_device_support = {}
_logged = False


@triton.jit
def key(x):
    u = x.to(tl.uint32, bitcast=True)
    return tl.where((u & 0x80000000) != 0, ~u, u ^ 0x80000000).to(tl.int64)


@triton.jit
def max_parts(X, M, F, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    i = s * B + tl.arange(0, B)
    x = tl.load(X + r * V + i, i < V, -float("inf"))
    # Negative zero has a distinct ordered bit key but compares equal to +0.
    # Preserve the old implementation for rows containing this rare ambiguity.
    negative_zero = (x == 0.0) & ((x.to(tl.uint32, bitcast=True) & 0x80000000) != 0)
    bad = (i < V) & ((x != x) | (x == float("inf")) | negative_zero)
    tl.store(M + r * S + s, tl.max(x, 0))
    tl.store(F + r * S + s, tl.sum(bad.to(tl.int32), 0))


@triton.jit
def max_rows(M, F, R, G, P, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    m = tl.max(tl.load(M + r * S + i, i < S, -float("inf")), 0)
    f = tl.sum(tl.load(F + r * S + i, i < S, 0), 0)
    p = tl.load(P + r)
    tl.store(R + r, m)
    tl.store(
        G + r,
        ((f > 0) | (m == -float("inf")) | (p != p) | (p < 1.0e-6) | (p >= 1.0)).to(
            tl.int32
        ),
    )


@triton.jit
def weights(X, R, W, T, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    i = s * B + tl.arange(0, B)
    x = tl.load(X + r * V + i, i < V, -float("inf")).to(tl.float64)
    m = tl.load(R + r).to(tl.float64)
    w = libdevice.exp(x - m)
    w = tl.where(i < V, w, 0.0)
    tl.store(W + r * V + i, w, i < V)
    tl.store(T + r * S + s, tl.sum(w, 0))


@triton.jit
def totals(T, P, A, L, H, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    t = tl.sum(tl.load(T + r * S + i, i < S, 0.0).to(tl.float64), 0)
    p = tl.load(P + r).to(tl.float64)
    tl.store(A + r, (1.0 - p) * t)
    tl.store(L + r, 0)
    tl.store(H + r, 0xFFFFFFFF)


@triton.jit
def mass_parts(X, W, L, H, M, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    i = s * B + tl.arange(0, B)
    x = tl.load(X + r * V + i, i < V, 0.0)
    w = tl.load(W + r * V + i, i < V, 0.0)
    mid = (tl.load(L + r) + tl.load(H + r)) // 2
    tl.store(M + r * S + s, tl.sum(tl.where((i < V) & (key(x) <= mid), w, 0.0), 0))


@triton.jit
def move(M, L, H, A, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    m = tl.sum(tl.load(M + r * S + i, i < S, 0.0).to(tl.float64), 0)
    lo = tl.load(L + r)
    hi = tl.load(H + r)
    mid = (lo + hi) // 2
    greater = m > tl.load(A + r)
    tl.store(L + r, tl.where(greater, lo, mid + 1))
    tl.store(H + r, tl.where(greater, mid, hi))


@triton.jit
def boundary(X, W, L, M, C, Z, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    i = s * B + tl.arange(0, B)
    x = tl.load(X + r * V + i, i < V, 0.0)
    w = tl.load(W + r * V + i, i < V, 0.0)
    k = key(x)
    cut = tl.load(L + r)
    eq = (i < V) & (k == cut)
    tl.store(M + r * S + s, tl.sum(tl.where((i < V) & (k < cut), w, 0.0), 0))
    tl.store(C + r * S + s, tl.sum(eq.to(tl.int32), 0))
    tl.store(Z + r * S + s, tl.max(tl.where(eq, w, 0.0), 0))


@triton.jit
def count(M, Z, A, N, S: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    m = tl.sum(tl.load(M + r * S + i, i < S, 0.0).to(tl.float64), 0)
    w = tl.max(tl.load(Z + r * S + i, i < S, 0.0).to(tl.float64), 0)
    n = libdevice.floor((tl.load(A + r) - m) / w).to(tl.int32)
    tl.store(N + r, tl.maximum(n, 0))


@triton.jit
def mask(
    X, L, C, N, O, V: tl.constexpr, S: tl.constexpr, B: tl.constexpr, R: tl.constexpr
):
    r = tl.program_id(0)
    s = tl.program_id(1)
    i = s * B + tl.arange(0, B)
    x = tl.load(X + r * V + i, i < V, 0.0)
    k = key(x)
    cut = tl.load(L + r)
    eq = (i < V) & (k == cut)
    j = tl.arange(0, R)
    prev = tl.sum(tl.load(C + r * S + j, (j < S) & (j < s), 0), 0)
    rank = prev + tl.cumsum(eq.to(tl.int32), 0)
    drop = (k < cut) | (eq & (rank <= tl.load(N + r)))
    tl.store(O + r * V + i, tl.where(drop, -float("inf"), x), i < V)


class Selection:
    def __init__(self, x, p):
        self.x = x
        self.p = p
        self.b, self.v = x.shape
        self.s = triton.cdiv(self.v, 4096)
        opts = {"device": x.device}
        self.w = torch.empty_like(x, dtype=torch.float64)
        self.m = torch.empty(self.b, self.s, dtype=torch.float64, **opts)
        self.z = torch.empty_like(self.m)
        self.c = torch.empty(self.b, self.s, dtype=torch.int32, **opts)
        self.flags = torch.empty_like(self.c)
        self.bad = torch.empty(self.b, dtype=torch.int32, **opts)
        self.rowmax = torch.empty(self.b, dtype=torch.float32, **opts)
        self.a = torch.empty(self.b, dtype=torch.float64, **opts)
        self.lo = torch.empty(self.b, dtype=torch.int64, **opts)
        self.hi = torch.empty_like(self.lo)
        self.n = torch.empty(self.b, dtype=torch.int32, **opts)
        self.lock = threading.Lock()

    def __call__(self):
        b, v, s = self.b, self.v, self.s
        rb = triton.next_power_of_2(s)
        max_parts[(b, s)](self.x, self.m, self.flags, v, s, 4096)
        max_rows[(b,)](self.m, self.flags, self.rowmax, self.bad, self.p, v, s, rb)
        if self.bad[:b].any().item():
            return None
        weights[(b, s)](self.x, self.rowmax, self.w, self.m, v, s, 4096)
        totals[(b,)](self.m, self.p, self.a, self.lo, self.hi, s, rb)
        for _ in range(32):
            mass_parts[(b, s)](self.x, self.w, self.lo, self.hi, self.m, v, s, 4096)
            move[(b,)](self.m, self.lo, self.hi, self.a, s, rb)
        boundary[(b, s)](self.x, self.w, self.lo, self.m, self.c, self.z, v, s, 4096)
        count[(b,)](self.m, self.z, self.a, self.n, s, rb)
        out = torch.empty_like(self.x)
        mask[(b, s)](self.x, self.lo, self.c, self.n, out, v, s, 4096, rb)
        return out


def _eligible(logits, k, p):
    return (
        _enabled
        and k is None
        and p is not None
        and logits.device.type == "cuda"
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 1 <= logits.shape[0] <= 64
        and logits.shape[1] == 130560
        and logits.is_contiguous()
        and p.device == logits.device
        and p.dtype == torch.float32
        and p.ndim == 1
        and p.shape[0] == logits.shape[0]
        and p.is_contiguous()
    )


def try_apply(logits, k, p):
    """Return filtered logits, or None for the unchanged reference fallback."""
    global _logged
    if not _eligible(logits, k, p):
        return None
    device = logits.device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        index = device.index
        with _cache_lock:
            if index not in _device_support:
                _device_support[index] = (
                    torch.cuda.get_device_name(device).strip().lower() == "metax c500"
                )
            supported = _device_support[index]
        if not supported:
            return None
        stream = torch.cuda.current_stream(device).cuda_stream
        cache_key = (index, stream)
        with _cache_lock:
            workspace = _workspaces.get(cache_key)
            if workspace is None:
                # A single capacity handles every active B without retaining one
                # full-vocabulary workspace for each tail batch size.
                capacity = torch.empty((64, 130560), dtype=torch.float32, device=device)
                workspace = Selection(capacity, p)
                workspace.x = None
                del capacity
                _workspaces[cache_key] = workspace
                if len(_workspaces) > 4:
                    _workspaces.popitem(last=False)
            else:
                _workspaces.move_to_end(cache_key)
        with workspace.lock:
            workspace.x = logits
            workspace.p = p
            workspace.b = logits.shape[0]
            try:
                result = workspace()
            finally:
                workspace.x = None
                workspace.p = None
        if result is not None and not _logged:
            logger.info(
                "C500 FP64 top-p mass selection active; FP32 keys, original RNG, no vocabulary sort"
            )
            _logged = True
        return result
