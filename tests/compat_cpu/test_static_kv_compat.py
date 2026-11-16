"""CPU-only compatibility gates. No torch/vLLM imports or GPU simulation claims."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace, FunctionType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat = load('static_kv_compat_cpu', 'vllm_fl/attention/static_kv_compat.py')
repair = load('static_kv_repair_cpu', 'vllm_fl/attention/int8_kv_repair.py')


class Tensor:
    def __init__(self, data, tail=(), dtype='bf16', device='cuda'):
        self.data, self.tail, self.dtype = list(data), tail, dtype
        self.device = SimpleNamespace(type=device)
        self.shape = (len(self.data), *tail)

    def __getitem__(self, index):
        return Tensor(self.data[index], self.tail, self.dtype, self.device.type)

    def __sub__(self, value):
        return Tensor([x-value for x in self.data], self.tail, self.dtype)

    def tolist(self):
        return self.data[:]

    def numel(self):
        total = len(self.data)
        for size in self.tail:
            total *= size
        return total


def metadata(lengths):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1]+n)
    m = SimpleNamespace(causal=True, use_cascade=False, mm_prefix_range_tensor=None,
                        num_actual_tokens=starts[-1], max_query_len=max(lengths),
                        query_start_loc=Tensor(starts),
                        softmax_segm_output=Tensor(range(80), (16,32,128)))
    common = SimpleNamespace(query_start_loc_cpu=Tensor(starts, device='cpu'), num_reqs=len(lengths))
    return m, common


class PrefixTests(unittest.TestCase):
    def test_pure_decode_and_tail_padding(self):
        for lengths in ([1], [1]*80, [1]*16+[0]*8):
            with self.subTest(lengths=lengths):
                m,c = metadata(lengths)
                compat.attach_plan(m,c)
                self.assertIs(m._fl_static_decode_starts,m.query_start_loc)
                self.assertIsNone(m._fl_static_mixed_plan)

    def test_holes_bad_counts_and_prefill_not_compact(self):
        for starts,n in [([0,1,1,2],2),([0,2],2),([1,2],2),([0,1],2),([0],0),([0,2,1],1),([0,0],0)]:
            with self.subTest(starts=starts):
                layout=compat.classify_prefix(starts,n)
                self.assertFalse(layout.all_decode)

    def test_mixed_prefixes_rebased_once(self):
        for d in (16,32,64,80):
            with self.subTest(d=d):
                m,c=metadata([1]*d+[7,3,0])
                compat.attach_plan(m,c);p=m._fl_static_mixed_plan
                self.assertEqual(p.num_decodes,d)
                self.assertEqual(p.decode_starts.data,list(range(d+1)))
                self.assertEqual(p.prefill_starts.data,[0,7,10,10])
                self.assertIs(p.source_starts,m.query_start_loc)

    def test_small_prefix_and_missing_cpu_fall_back(self):
        for d in (1,4,8,15,81):
            with self.subTest(d=d):
                m,c=metadata([1]*d+[7]);compat.attach_plan(m,c)
                self.assertIsNone(m._fl_static_mixed_plan)
        m,c=metadata([1]*16+[7]);c.query_start_loc_cpu=None
        compat.attach_plan(m,c);self.assertIsNone(m._fl_static_mixed_plan)

    def test_reuse_existing_011_gpu_prefixes(self):
        m,c=metadata([1]*16+[7]);existing=compat.StaticMixedPlan(16,object(),object(),m.query_start_loc)
        m._fl_mixed_plan=existing;compat.attach_plan(m,c)
        self.assertIs(m._fl_static_mixed_plan,existing)

    def test_disabled_mixed_and_special_features(self):
        with patch.dict(os.environ,{'VLLM_FL_MIXED_ATTN_SPLIT':'0'}):
            m,c=metadata([1]*16+[7]);compat.attach_plan(m,c)
            self.assertIsNone(m._fl_static_mixed_plan)
        for changes in ({'causal':False},{'use_cascade':True},{'mm_prefix_range_tensor':object()}):
            m,c=metadata([1]*16+[7]);m.__dict__.update(changes);compat.attach_plan(m,c)
            self.assertIsNone(m._fl_static_mixed_plan)

    def test_split_keeps_cache_and_scales_identity(self):
        m,c=metadata([1]*16+[7,3]);compat.attach_plan(m,c)
        args=dict(q=Tensor(range(26)),out=Tensor(range(26)),seqused_k=Tensor(range(18)),
                  block_table=Tensor(range(18)),k=object(),v=object(),
                  k_scale_cache=object(),v_scale_cache=object(),
                  cu_seqlens_q=m.query_start_loc,max_seqlen_q=7,causal=True)
        a,b=compat.split_attention_args(args,m._fl_static_mixed_plan)
        self.assertEqual(a['q'].data,list(range(16)))
        self.assertEqual(b['q'].data,list(range(16,26)))
        self.assertEqual(a['block_table'].data,list(range(16)))
        for name in ('k','v','k_scale_cache','v_scale_cache'):
            self.assertIs(a[name],args[name]);self.assertIs(b[name],args[name])
        self.assertEqual(a['max_seqlen_q'],1)
        self.assertEqual(args['max_seqlen_q'],7)
        for changes in ({'cu_seqlens_q':object()},{'causal':False},{'sinks':object()}, {'window_size':(0,0)}):
            self.assertIsNone(compat.split_attention_args(dict(args,**changes),m._fl_static_mixed_plan))


class CompactTests(unittest.TestCase):
    def kwargs(self,n=32,segments=8):
        q=Tensor(range(n),(16,128));k=Tensor(range(2),dtype='int8')
        return dict(IS_3D=True,num_query_heads=16,num_queries_per_kv=8,
                    HEAD_SIZE=128,HEAD_SIZE_PADDED=128,BLOCK_SIZE=32,
                    BLOCK_M=16,BLOCK_Q=2,KV_QUANT_MODE='int8_kv',USE_CAUSAL=True,
                    NUM_SEGMENTS_PER_SEQ=segments,num_seqs=n,query_ptr=q,
                    output_ptr=q,key_cache_ptr=k,value_cache_ptr=k)

    def grid(self,kwargs,grid):
        return compat.compact_grid(kwargs,grid,bf16='bf16',int8='int8',kv_mode='int8_kv')

    def test_all_verified_segment_counts_and_80_rows(self):
        for n in (1,16,32,56,64,72,80):
            for s in (8,16,32):
                self.assertEqual(self.grid(self.kwargs(n,s),(n+n//2,2,s)),(n,2,s))

    def test_layout_features_and_unsupported_shapes(self):
        k=self.kwargs()
        changes=[{'IS_3D':False},{'BLOCK_SIZE':16},{'KV_QUANT_MODE':0},
                 {'BLOCK_Q':1},{'NUM_SEGMENTS_PER_SEQ':7},{'num_seqs':81},
                 {'USE_TD':True},{'USE_SINKS':True},{'SLIDING_WINDOW':128},
                 {'CHUNK_LOOKBACK':0},{'HEAD_SIZE':64},
                 {'value_cache_ptr':Tensor([0],dtype='bf16')}]
        for change in changes:
            with self.subTest(change=change):self.assertIsNone(self.grid(dict(k,**change),(48,2,8)))
        self.assertIsNone(self.grid(k,(48,2,16)))
        self.assertIsNone(self.grid(k,lambda:None))
        k['query_ptr']=Tensor(range(16),(16,128))
        self.assertEqual(self.grid(k,(48,2,8)),(32,2,8))

    def test_final_numeric_source_is_preserved(self):
        original='def head():\n'+compat.ORIGINAL_LOOKUP+'\n    K_real = K_load.to(Q.dtype)\n    packed_v0 = protect\n'
        out=compat.specialize_source(original)
        self.assertIn('K_real = K_load.to(Q.dtype)',out)
        self.assertIn('packed_v0 = protect',out)
        self.assertIn('seq_idx = q_block_global_idx',out)
        self.assertEqual(out.replace(compat.DIRECT_LOOKUP,compat.ORIGINAL_LOOKUP),original)
        for bad in ('unknown',original+compat.ORIGINAL_LOOKUP):
            with self.assertRaises(ValueError):compat.specialize_source(bad)

    def test_guarded_head_fallback_and_compact_grid(self):
        calls=[]
        class Head:
            def __init__(self,name):self.name=name
            def __getitem__(self,grid):return lambda *a,**kw:calls.append((self.name,grid))
        dispatch=compat.StaticDecodeKernel(Head('original'),Head('direct'),bf16='bf16',int8='int8',kv_mode='int8_kv')
        dispatch[(48,2,8)](**self.kwargs())
        dispatch[(48,2,8)](**dict(self.kwargs(),BLOCK_SIZE=16))
        dispatch[(48,2,8)]('positional')
        self.assertEqual(calls,[('direct',(32,2,8)),('original',(48,2,8)),('original',(48,2,8))])


class IntegrationTests(unittest.TestCase):
    def test_actual_forward_writes_once_then_splits(self):
        tree=ast.parse((ROOT/'vllm_fl/attention/k_static_fold_service.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KStaticFoldImpl')
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        calls=[]
        ns=dict(torch=SimpleNamespace(bfloat16='bf16'),
                writer=SimpleNamespace(write_q=lambda *a:calls.append('write')),
                choose=lambda b,q:((), 'q_in_write_heads',1,'fixture'),
                KVQuantMode=SimpleNamespace(INT8_PER_TOKEN_HEAD='int8_kv'),
                split_attention_args=compat.split_attention_args,
                feature_enabled=compat.feature_enabled)
        exec(compile(ast.Module(body=[forward],type_ignores=[]),'<actual static forward>','exec'),ns)
        for mixed in (False,True):
            calls.clear();m,c=metadata([1]*16+[4]);m.max_seq_len=100
            m.slot_mapping=Tensor(range(20));m.static_q_workspace=Tensor(range(20),(16,128))
            m.block_table=Tensor(range(17));m.seq_lens=Tensor([100]*17)
            if mixed:compat.attach_plan(m,c)
            impl=SimpleNamespace(_parameters=lambda *a:None,_mixed_reported=False,
                _perm=None,_scale=None,cache_views=lambda r:(object(),object(),object()),
                scale=.1,sliding_window=(-1,-1),logits_soft_cap=0.,alibi_slopes=None,
                sinks=None,use_alibi_sqrt=False,chunk_lookback=-1,
                _attention=lambda a,m,direct=False:calls.append(('attn',a['q'].data,direct)))
            q=Tensor(range(20),(16,128));kv=Tensor(range(20),(2,128));out=Tensor(range(20),(16,128))
            result=ns['forward'](impl,object(),q,kv,kv,object(),m,out)
            self.assertIs(result,out);self.assertEqual(calls.count('write'),1)
            self.assertEqual(calls[0],'write')
            self.assertEqual(len(calls),3 if mixed else 2)
            if mixed:self.assertEqual(calls[1:], [('attn',list(range(16)),True),('attn',list(range(16,20)),False)])

    def test_original_bf16_entry_is_not_modified_by_repair_build(self):
        class Jit:
            def __init__(self,fn):self.fn=fn;self.src='L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
            def _unsafe_update_src(self,src):self.src=src
        old=Jit(lambda:None)
        ns={'kernel_unified_attention':old,'_fl_prefill_tiles':object()}
        original=FunctionType((lambda:None).__code__,ns)
        clone=repair._build_repaired(original,SimpleNamespace(jit=Jit))
        self.assertIs(original.__globals__['kernel_unified_attention'],old)
        fixed=clone.__globals__['kernel_unified_attention']
        self.assertIsNot(fixed,old);self.assertIn('if USE_SINKS else 0.0',fixed.src)

    def test_repair_build_then_bf16_wrap_then_publish(self):
        def mod(name,**kw):
            m=ModuleType(name);m.__dict__.update(kw);return m
        class Jit:
            def __init__(self,fn):self.fn=fn;self.src='L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
            def _unsafe_update_src(self,src):self.src=src
        calls=[];ns={'kernel_unified_attention':Jit(lambda:None),'_fl_prefill_tiles':None,'calls':calls}
        exec("def tuned(*args, **kwargs): calls.append('numeric')",ns);tuned=ns['tuned']
        backend=mod('vllm.v1.attention.backends.triton_attn',unified_attention=tuned,_fl_tuned_launcher=tuned)
        modules={
            'torch':mod('torch',bfloat16='bf16'), 'vllm':mod('vllm'),
            'vllm.triton_utils':mod('vllm.triton_utils',triton=SimpleNamespace(jit=Jit)),
            'vllm.v1':mod('vllm.v1'),'vllm.v1.attention':mod('vllm.v1.attention'),
            'vllm.v1.attention.backends':mod('vllm.v1.attention.backends',triton_attn=backend),
            'vllm.v1.attention.backends.triton_attn':backend,
            'vllm.v1.kv_cache_interface':mod('vllm.v1.kv_cache_interface',KVQuantMode=SimpleNamespace(INT8_PER_TOKEN_HEAD='int8_kv'))}
        with patch.dict(sys.modules,modules),patch.dict(os.environ,{'VLLM_FL_INT8_KV_REPAIR':'1'}):
            self.assertTrue(repair.install_int8_kv_repair(publish=False))
            def wrapped(**kw):calls.append('bf16_wrapped')
            backend.unified_attention=wrapped
            tuned.__globals__['kernel_unified_attention']=object() # #013 wrapper has no .src
            self.assertTrue(repair.install_int8_kv_repair())
            published=backend.unified_attention
            published(kv_quant_mode=0)
            published(kv_quant_mode='int8_kv',q=SimpleNamespace(dtype='bf16'))
            self.assertEqual(calls,['bf16_wrapped','numeric'])
            self.assertTrue(repair.install_int8_kv_repair())
            self.assertIs(backend.unified_attention,published)


if __name__=='__main__':
    unittest.main()
