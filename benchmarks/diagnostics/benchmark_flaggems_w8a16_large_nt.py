# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: MetaX NT mm with weight-only channel scaling before output.
Derived from FlagGems _metax/ops/mm.py mm_kernel_nt; no production registration.
"""
import argparse
import importlib
from pathlib import Path
from statistics import median
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from benchmark_flaggems_w8a16 import load_helper, dump, relative, save_kernel

@triton.jit
def nt_scaled(
    A,
    B,
    C,
    Scales,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] + rbn[None, :] * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    channel_scales = tl.load(Scales + rn, mask=rn < N, other=0.0).to(tl.float32)
    result = (acc * channel_scales[None, :]).to(C.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])

CONFIGS = [('basic4',4,'basic'),('cpasync3',3,'cpasync')]


def call(kind, cfg, x, q, scale, generic, capture=False):
    label,stages,pipeline=cfg
    m,k=x.shape;n=q.shape[0]
    assert x.is_contiguous() and q.is_contiguous() and scale.is_contiguous()
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    grid=(triton.cdiv(m,64)*triton.cdiv(n,64),)
    with torch.cuda.device(x.device):
        if kind=='nt':
            compiled=nt_scaled[grid](x,q,y,scale,m,n,k,64,64,64,8,
                m%64==0,n%64==0,k%64==0,
                num_warps=4,num_stages=stages,pipeline=pipeline)
        else:
            compiled=generic[grid](x,q,scale,y,m,n,k,x.stride(0),x.stride(1),
                q.stride(0),q.stride(1),y.stride(0),y.stride(1),scale.stride(0),
                64,64,64,8,num_warps=4,num_stages=stages,pipeline=pipeline)
    return (y,compiled) if capture else y


def verify(actual,expected):
    err=relative(actual,expected)
    assert err<.005,err
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.125)
    return err


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--helpers-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    ref=load_helper('reference',args.helpers_dir);deq=load_helper('dequant',args.helpers_dir)
    weight=importlib.import_module('flag_gems.ops.weight_int8pack_mm')
    dense=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    generic=weight.weight_int8pack_mm_kernel.jit_function
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    dump(out,'metadata.json',dict(seed=20261028,m=[128,256,512,1024,2048],n=2048,k=6144,
        tile=[64,64,64],warps=4,configs=CONFIGS,torch=torch.__version__,triton=triton.__version__,
        source_nt=dense.__file__,source_generic=weight.__file__,rounds=3,warmup=100,rep=300,
        contract='BF16 activation, INT8 weights converted to BF16, FP32 dot then FP32 channel scale, BF16 output',
        scope='diagnostic NT source adaptation plus matched-config generic controls; full allocation-inclusive GPU event timings'))
    torch.manual_seed(20261028);correctness=[];available=[];failures=[]
    for kind in ('generic','nt'):
        for cfg in CONFIGS:
            label=kind+'_'+cfg[0]
            try:
                for m,n,k in ((3,129,257),(65,67,33)):
                    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02;w[0]=0
                    q,s=ref.quantize_weight(w);e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
                    y=call(kind,cfg,x,q,s,generic)
                    error=verify(y,e);assert torch.count_nonzero(y[:,0]).item()==0
                    correctness.append(dict(path=label,m=m,n=n,k=k,error=error,zero_channel=True))
                available.append((kind,cfg))
            except Exception as exc:
                failures.append(dict(path=label,error=repr(exc)));dump(out,'failures.json',failures)
                if type(exc).__name__ not in ('OutOfResources','CompilationError','AssertionError'):raise
            dump(out,'correctness.json',correctness)
    dump(out,'failures.json',failures)
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
    torch.testing.assert_close(deq.dequantize_weight(q,s),ref.dequantize_reference(q,s),rtol=0,atol=0)
    raw=[];summary=[]
    for m in (128,256,512,1024,2048):
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
        paths=dict(bf16=lambda:dense.mm(x,w.t()),
            separate=lambda:dense.mm(x,deq.dequantize_weight(q,s).t()),
            original=lambda:weight.weight_int8pack_mm(x,q,s))
        errors={'original':verify(paths['original'](),e)}
        for kind,cfg in available:
            label=kind+'_'+cfg[0]
            actual,compiled=call(kind,cfg,x,q,s,generic,True)
            errors[label]=verify(actual,e)
            save_kernel(out,f'm{m}-{label}',compiled,dict(BM=64,BN=64,BK=64,warps=4,stages=cfg[1],pipeline=cfg[2]))
            paths[label]=lambda kind=kind,cfg=cfg:call(kind,cfg,x,q,s,generic)
        for fn in paths.values():fn()
        torch.cuda.synchronize();names=list(paths);samples={n:[] for n in names}
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                ms=triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
                samples[name].append(ms);raw.append(dict(m=m,path=name,round=repeat+1,ms=ms))
                dump(out,'raw.json',raw)
        row=dict(m=m,errors=errors,timings_ms={n:median(v) for n,v in samples.items()})
        summary.append(row);dump(out,'summary.json',summary);print('FINAL',row,flush=True)

if __name__=='__main__':main()
