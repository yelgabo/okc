"""Actual CPU Torch host-fallback and BF16 fixtures; no GPU performance evidence."""
import ast
import io
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from dl_fork_moe import HOST_BRIDGE


class FallbackHost(unittest.TestCase):
    def scope(self, *, suffix=True, kind='success', installed=True):
        calls=[]
        prop=SimpleNamespace(major=9,minor=0,multi_processor_count=132)
        def install(w,s): s['owned_moe']='owned' if installed else None;calls.append('install');return installed
        def prep(w,s): s['ma_w']=[];s['ma_scratch']=[];calls.append('pack')
        def run(*args):
            calls.append('run')
            if kind=='resource': raise RuntimeError('FORK_MOE_RESOURCE synthetic residency')
            if kind=='launch': raise RuntimeError('FORK_MOE_LAUNCH synthetic execution')
            self.assertTrue(bool((args[3][:,0]==0).all()))
            self.assertTrue(bool((args[3][:,1:]==-1).all()))
        def native(m,s):
            calls.append(('native',m))
            if kind=='compile': raise RuntimeError('synthetic nvcc failure')
            return SimpleNamespace(okc_run=run)
        def sync():
            calls.append('sync')
            if kind=='sync': raise RuntimeError('synthetic async CUDA fault')
        facade=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda d:prop,synchronize=sync),
            zeros=torch.zeros,full=torch.full,empty_like=torch.empty_like,
            bfloat16=torch.bfloat16,float32=torch.float32,int32=torch.int32)
        scope=dict(torch=facade,_fork_install_owned_moe=install,_fork_native_ma_prepare=prep,
            _fork_native_moe=native,_fork_owned_moe_for=lambda m,s:('owned',m),calls=calls)
        exec(HOST_BRIDGE,scope)
        def marlin(w,s):
            calls.append('prepare Marlin');s['_fork_marlin_adapter']=SimpleNamespace(okc_run=run)
        scope['_fork_prepare_marlin']=marlin
        st={'ma_sfx':('-DOKC_PRMTSC=1',) if suffix else ()}
        weights={'w13':torch.empty(1)}
        return scope,st,weights

    def test_outside_compression_domain_never_compiles_native(self):
        scope,st,w=self.scope(suffix=False)
        scope['_ma_prepare'](w,st)
        self.assertEqual(scope['calls'],['install','sync'])
        self.assertEqual(scope['_moe'](256,st),('owned',256))
        self.assertIsNone(scope['_moe']())

    def test_resource_and_compile_fallback_are_explicit(self):
        for kind in ('compile','resource'):
            scope,st,w=self.scope(kind=kind)
            scope['_ma_prepare'](w,st)
            self.assertFalse(st['_fork_native_eligible'])
            self.assertEqual(scope['_moe'](32,st),('owned',32))
            self.assertTrue(st['_fork_moe_status'].startswith('owned:'))

    def test_outside_owned_domain_prepares_and_warms_marlin(self):
        scope,st,w=self.scope(suffix=False,installed=False)
        scope['_ma_prepare'](w,st)
        self.assertFalse(st['_fork_native_eligible'])
        self.assertEqual(st['_fork_fallback_kind'],'Marlin')
        self.assertEqual(scope['calls'].count('run'),7)
        self.assertEqual(scope['calls'].count('sync'),8)
        self.assertFalse(any(isinstance(c,tuple) and c[0]=='native' for c in scope['calls']))
        self.assertIs(scope['_moe'](256,st),st['_fork_marlin_adapter'])

    def test_execution_errors_fatal_and_success_uses_native(self):
        for kind in ('launch','sync'):
            scope,st,w=self.scope(kind=kind)
            with self.assertRaises(RuntimeError): scope['_ma_prepare'](w,st)
        scope,st,w=self.scope()
        with contextlib.redirect_stdout(io.StringIO()): scope['_ma_prepare'](w,st)
        self.assertTrue(st['_fork_native_eligible'])
        self.assertEqual(scope['calls'].count('sync'),16)
        self.assertEqual(scope['calls'].count('run'),7)

    def test_exact_bf16_boundaries_distinguish_old_path(self):
        # Actual torch BF16 rounding: official cache1/cache2/y dtypes and gamma225.
        gen=torch.Generator().manual_seed(481)
        q=lambda v:v.to(torch.bfloat16).float()
        raw=torch.randn(4096,generator=gen).to(torch.bfloat16)
        effective=(raw.float()+1).to(torch.bfloat16)
        self.assertTrue(torch.equal(q(raw.float()+1),effective.float()))
        self.assertFalse(torch.equal(raw.float()+1,effective.float()))
        gate=torch.randn(4096,generator=gen)*3;up=torch.randn(4096,generator=gen)
        rounded=q(torch.nn.functional.silu(q(gate))*q(up))
        self.assertFalse(torch.equal(rounded,q(torch.nn.functional.silu(gate)*up)))
        gemm2=torch.randn(4096,generator=gen)
        product=q(gemm2)*effective.float()
        self.assertFalse(torch.equal(product,q(gemm2*effective.float())))

    def test_actual_marlin_adapter_body_dtype_alias_and_ids(self):
        # Real Torch CPU buffers with official-op stubs: ABI/lifetimes only,
        # not a mathematical or GPU-library capability test.
        calls=[]
        tree=ast.parse(HOST_BRIDGE)
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_ForkMarlinAdapter')
        for fn in klass.body:
            if isinstance(fn,ast.FunctionDef):
                fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        def align(ids,bm,experts):
            calls.append(('align',bm,experts));return ids,ids,torch.tensor(ids.numel())
        def gemm(x,out,*args,**kw):
            self.assertEqual(out.dtype,torch.bfloat16)
            self.assertFalse(kw['use_atomic_add'])
            self.assertTrue(kw['use_fp32_reduce'])
            calls.append(('gemm',kw['top_k'],tuple(out.shape)))
            return out.zero_()
        def activate(x,out,**kw):
            self.assertEqual(x.dtype,torch.bfloat16); self.assertEqual(out.dtype,torch.bfloat16)
            out.zero_()
        def finish(y,tw,**kw):
            self.assertEqual(kw['norm_weight'].dtype,torch.bfloat16)
            self.assertTrue(kw['weight_first']);self.assertEqual(kw['eps'],1e-8)
            self.assertEqual(kw['slot_expert_ids'].min().item(),-1)
            return torch.zeros((y.shape[0],y.shape[-1]),dtype=torch.bfloat16)
        scope=dict(torch=torch,scalar_types=SimpleNamespace(float4_e2m1f='fp4'),moe_align_block_size=align,
            moe_wna16_marlin_gemm=gemm,silu_and_mul=activate,fused_post_expert_norm_weighted_sum=finish)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'actual_marlin_adapter','exec'),scope)
        bank=dict.fromkeys(('w13','w13_s','w13_gs','w2','w2_s','w2_gs','workspace'))
        st={'_fork_marlin_bank':bank,'_fork_marlin_gamma':torch.ones(4096,dtype=torch.bfloat16)}
        adapter=scope['_ForkMarlinAdapter'](st)
        ids=torch.full((4,8),-1,dtype=torch.int32);ids[:,0]=0
        out=torch.empty((4,4096),dtype=torch.bfloat16)
        adapter.okc_run((),(),torch.zeros_like(out),ids,torch.full((4,8),.125),out,0,32,132)
        self.assertEqual(calls,[('align',8,256),('gemm',8,(32,8192)),('gemm',1,(32,4096))])
        self.assertEqual(out.count_nonzero().item(),0)


if __name__=='__main__': unittest.main()
