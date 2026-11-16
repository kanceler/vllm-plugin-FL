"""Diagnostic only: validate installed vLLM INT8 KV, without changing kernels.

Thresholds fixed before GPU execution: kernel NRMSE <= .02 and max absolute
error <= .08; quantized-vs-BF16-reference NRMSE <= .05 and max <= .20.
These synthetic screening limits do not replace model accuracy evaluation.
"""
import json
import os
from pathlib import Path
import sys

import torch

# Load the final baseline's compatibility patches before importing Triton ops.
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar  # noqa: F401
from vllm_fl.attention.tile_policy import install_tile_policy

install_tile_policy()
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant as write_cache,
)
from vllm.v1.attention.backends import triton_attn
from vllm.v1.kv_cache_interface import KVQuantMode


def error(x, y):
    delta = x.float() - y.float()
    return dict(max_abs=delta.abs().max().item(),
                nrmse=(delta.square().mean().sqrt() /
                       y.float().square().mean().sqrt().clamp_min(1e-8)).item())


def reference(q, k, v, qlen, seqlen, batch):
    # Independent float32 causal GQA reference, chunked to bound workspace.
    result = []
    for b in range(batch):
        kb = k[b*seqlen:(b+1)*seqlen].float().repeat_interleave(8, dim=1).transpose(0, 1)
        vb = v[b*seqlen:(b+1)*seqlen].float().repeat_interleave(8, dim=1).transpose(0, 1)
        for start in range(0, qlen, 32):
            stop = min(start + 32, qlen)
            qb = q[b*qlen+start:b*qlen+stop].float().transpose(0, 1)
            scores = (qb @ kb.transpose(1, 2)) / (128 ** .5)
            positions = torch.arange(start, stop, device=q.device) + seqlen - qlen
            scores.masked_fill_(torch.arange(seqlen, device=q.device)[None, None, :] > positions[None, :, None], -float('inf'))
            result.append((scores.softmax(-1) @ vb).transpose(0, 1))
    return torch.cat(result)


def run(batch, seqlen, qlen, segmented, unusual=False):
    device = 'cuda'
    block = 16
    pages = (seqlen + block - 1) // block
    num_blocks = batch * pages + 2
    # Physical pages are deliberately shuffled; scales share the actual inline
    # layout used by TritonAttentionImpl, including the four-byte scale suffix.
    v_only = False
    from vllm_fl.attention.k_top4 import views, write, launcher
    cache = torch.zeros((num_blocks,block,2,276), device=device, dtype=torch.uint8)
    kc,vc,vs=views(cache)
    ks=vs  # K scales live in inline metadata; launcher never loads this alias.
    store=lambda k,v,*unused: write(k,v,cache,slots)
    call=launcher()
    table = torch.randperm(num_blocks-2, device=device).reshape(batch, pages).to(torch.int32)
    pos = torch.arange(seqlen, device=device)
    slots = (table[:, pos // block].long()*block + pos % block).flatten()
    slots = torch.cat((slots, torch.tensor([-1], device=device)))
    # A token stride gap exercises packed/non-contiguous source tensors.
    k = torch.randn((batch*seqlen+1, 4, 128), device=device, dtype=torch.bfloat16)[:, ::2]
    v = torch.randn_like(k.contiguous())
    if unusual:
        k[0].zero_(); v[0].zero_()
        k[1, :, 0] = 30; v[1, :, 0] = -30
    untouched = cache[-1].clone()
    store(k, v, kc, vc, ks, vs, slots)
    torch.cuda.synchronize()
    assert torch.equal(cache[-1], untouched), 'negative slot modified unused page'
    selected_k = kc[slots[:-1] // block, slots[:-1] % block]
    selected_v = vc[slots[:-1] // block, slots[:-1] % block, :, :128]
    sk = ks[slots[:-1] // block, slots[:-1] % block]
    sv = vs[slots[:-1] // block, slots[:-1] % block]
    # Decode actual inline bytes independently of the production loader.
    packed=selected_k.contiguous().view(torch.uint8)
    indices=packed[...,128:132].long()
    saved=packed[...,132:140].contiguous().view(torch.bfloat16).float()
    sk=packed[...,140:144].contiguous().view(torch.float32).squeeze(-1)
    dqk=selected_k[...,:128].float()*sk[...,None]
    dqk.scatter_(-1,indices,saved)
    assert torch.equal(saved,k[:-1].float().gather(-1,indices)), 'protected values changed'
    expected_indices=torch.argsort(k[:-1].float().abs(),dim=-1,descending=True,stable=True)[...,:4]
    assert torch.equal(indices,expected_indices), 'top4 selection differs'
    remainder=k[:-1].float().scatter(-1,indices,0.)
    torch.testing.assert_close(sk,(remainder.abs().amax(-1)/127).clamp_min(1e-6),rtol=1e-5,atol=1e-8)
    dqv=selected_v.float()*sv[...,None]
    q = torch.randn((batch*qlen, 16, 128), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    args = dict(q=q, k=kc, v=vc, out=out,
                cu_seqlens_q=torch.arange(batch+1, device=device, dtype=torch.int32)*qlen,
                max_seqlen_q=qlen, seqused_k=torch.full((batch,), seqlen, device=device, dtype=torch.int32),
                max_seqlen_k=seqlen, softmax_scale=128**-.5, causal=True,
                window_size=(-1, -1), block_table=table, softcap=0.,
                q_descale=None, k_descale=None, v_descale=None,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                k_scale_cache=ks, v_scale_cache=vs, use_td=False)
    if segmented:
        args.update(seq_threshold_3D=batch+1, num_par_softmax_segments=16,
                    softmax_segm_output=torch.empty((batch,16,16,128),device=device),
                    softmax_segm_max=torch.empty((batch,16,16),device=device),
                    softmax_segm_expsum=torch.empty((batch,16,16),device=device))
    call(**args)
    torch.cuda.synchronize()
    expected = reference(q, dqk, dqv, qlen, seqlen, batch)
    original = reference(q, k[:-1], v[:-1], qlen, seqlen, batch)
    kernel_error, quant_error = error(out, expected), error(expected, original)
    vonly_reference=reference(q,k[:-1],dqv,qlen,seqlen,batch)
    incremental_error=error(expected,vonly_reference)
    print(json.dumps(dict(stage='before_gate', batch=batch, seqlen=seqlen,
                          qlen=qlen, segmented=segmented, unusual=unusual,
                          kernel_error=kernel_error, quant_error=quant_error)), flush=True)
    assert kernel_error['nrmse'] <= .02 and kernel_error['max_abs'] <= .08, kernel_error
    assert quant_error['nrmse'] <= .05 and quant_error['max_abs'] <= .20, quant_error
    graph_error = None
    if qlen == 1:
        # Warmup before graph capture, then replay the complete write+read path.
        for _ in range(3):
            store(k, v, kc, vc, ks, vs, slots)
            call(**args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            store(k, v, kc, vc, ks, vs, slots)
            call(**args)
        q.mul_(.9)
        graph.replay()
        torch.cuda.synchronize()
        graph_error = error(out, reference(q, dqk, dqv, qlen, seqlen, batch))
        assert graph_error['nrmse'] <= .02 and graph_error['max_abs'] <= .08
    timings = None
    if os.environ.get('K_TOP4_TIMING') == '1':
        def full():
            store(k,v,kc,vc,ks,vs,slots)
            call(**args)
        functions={'write':lambda:store(k,v,kc,vc,ks,vs,slots),
                   'attention_with_reduce':lambda:call(**args),'full':full}
        graphs={}
        for label,fn in functions.items():
            for _ in range(3):fn()
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            graphs[label]=g
        timings={label:[] for label in graphs}
        for repeat in range(7):
            order=list(graphs) if repeat%2==0 else list(reversed(graphs))
            for label in order:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graphs[label].replay()
                end.record();end.synchronize()
                timings[label].append(start.elapsed_time(end)/20)
    return dict(batch=batch, seqlen=seqlen, qlen=qlen, segmented=segmented,
                unusual=unusual, kernel_error=kernel_error, quant_error=quant_error,
                graph_error=graph_error,incremental_k_error=incremental_error,timings_ms=timings)


if __name__ == '__main__':
    torch.manual_seed(20261113)
    target = Path(sys.argv[1]); target.mkdir(parents=True, exist_ok=True)
    cases = [(1,16,1,False,False), (1,17,1,False,False), (1,17,1,False,True), (1,129,1,True,False),
             (4,257,17,False,False), (1,128,128,False,False),
             (1,4096,1,True,False), (1,16384,1,True,False),
             (64,16384,1,False,False), (1,16384,128,False,False), (1,17408,1,True,False), (4,4097,1,False,False)]
    failures = 0
    with (target/'operator.jsonl').open('x') as log:
        for case in cases:
            try:
                record = dict(passed=True, **run(*case))
            except AssertionError as exc:
                failures += 1
                record = dict(passed=False, case=case, error=str(exc))
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record)+'\n'); log.flush()
    if failures:
        (target/'OPERATOR_FAILED').write_text(f'{failures} cases failed; no model or performance acceptance.\n')
        raise SystemExit(1)
    (target/'OPERATOR_PASS').write_text('Synthetic screening passed; model accuracy remains required.\n')
