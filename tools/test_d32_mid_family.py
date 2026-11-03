"""Actual final host entry points and neighbor/cache/fallback contracts; CPU only."""
import ast
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import tempfile
import types
import unittest
import torch
import d32_mid_family as g
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings
from g_expert_g224_prefix import function_source


BASE=(g.ROOT/g.BASE).read_text()


class FakeTorch:
    def __init__(self,major=9):
        self.cuda=types.SimpleNamespace(synchronize=lambda:None,get_device_capability=lambda:(major,0),
            get_device_properties=lambda dev:types.SimpleNamespace(major=major,minor=0,multi_processor_count=132),
            device=lambda dev:contextlib.nullcontext())
    def __getattr__(self,name):return getattr(torch,name)


def environment(source,major=9,failure=None):
    e=bindings(source);e.update(flags(source));calls=[];runs=[]
    def loader(**kw):
        calls.append(kw);module=types.ModuleType(kw['name'])
        module.okc_set_pdl=lambda n:None
        def run(w,s,x,ids,tw,out,*args):
            runs.append((x.shape[0],kw['name'],ids.clone()))
            if failure and x.shape[0]==32:raise RuntimeError(failure)
            out.zero_()  # Explicit numerical/device boundary, not GPU evidence.
        module.okc_run=run;return module
    e.update(torch=FakeTorch(major),load_inline=loader,_MA_EXTS={},_MA_MAIN_NS=3,_MA_PDL=False)
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_RouterExt')
    exec(ast.get_source_segment(source,node),e)
    for name in ('_main_defines','_ma_ext','_fork_native_moe','_fork_build_tasks',
                 '_fork_install_fallback','_fork_moe_fallback','_ma_prepare','_moe'):
        exec(function_source(source,name),e)
    if '_d32_mid_attest' in source:exec(function_source(source,'_d32_mid_attest'),e)
    def pack(w,st):st['ma_w']=[torch.zeros(1)];st['ma_scratch']=[torch.zeros(1)]
    e['_fork_native_ma_prepare']=pack
    def marlin(w,st):
        obj=types.SimpleNamespace(okc_run=lambda w,s,x,ids,tw,out,*args:out.zero_())
        st['_fork_marlin_adapter']=obj
    e['_fork_prepare_marlin']=marlin
    return e,calls,runs


def generated():
    with tempfile.TemporaryDirectory() as d:return g.generate(Path(d))


def build(e,sc6=True,sfx=('-DOKC_PRMTSC=1',),concurrent=False):
    st=dict(ma_ext={},ma_sfx=sfx,ma_sc6=sc6);weights={'w13':torch.zeros(1)}
    tasks=e['_fork_build_tasks'](weights,st)
    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda f:f(),tasks))
    else:
        for f in tasks:f()
    return st,weights,tasks


class Mid32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.source,cls.manifest=generated()

    def test_exact_host_delta_and_unchanged_shaders_math_startup(self):
        old=ast.parse(BASE);new=ast.parse(self.source)
        changed={'_fork_build_tasks','_ma_prepare'}
        self.assertEqual(len(new.body),len(old.body)+3)
        for a,b in zip(old.body,new.body):
            if isinstance(a,ast.FunctionDef) and a.name in changed:continue
            if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MOE_MID_MAX_M' for t in a.targets):continue
            self.assertEqual(ast.dump(a),ast.dump(b))
        for name in ('_fork_native_moe','_moe','_ma_ext','_ma_make_scratch','_fork_native_ma_prepare','_release_expert_self_test','prepare'):
            self.assertEqual(function_source(BASE,name),function_source(self.source,name))
        self.assertTrue(self.manifest['exact_reversal'])
        self.assertFalse(self.manifest['shader_edits'])
        self.assertEqual(flags(self.source)['MOE_MID_MAX_M'],32)

    def test_actual_final_selector_neighbors_raw_sc6_and_other_five_loads(self):
        for sc6 in (False,True):
            e,calls,_=environment(self.source);st,w,_=build(e,sc6)
            old,oldcalls,_=environment(BASE);control,_,_=build(old,sc6)
            original={c['name']:c for c in oldcalls}
            for m in (1,8,16,128,256,33):
                x=e['_fork_native_moe'](m,st);y=old['_fork_native_moe'](m,control)
                self.assertEqual(x._module.__name__,y._module.__name__)
                self.assertEqual(next(c for c in calls if c['name']==x._module.__name__),original[y._module.__name__])
            for m in (17,24,31,32):
                self.assertIs(e['_fork_native_moe'](m,st),st['ma_ext']['mid'])
            self.assertIs(e['_fork_native_moe'](33,st),st['ma_ext']['main'])
            self.assertEqual(set(st['ma_ext']),{'tiny','tiny2','mid','main'})
            self.assertEqual(len(calls),4)
            self.assertFalse(any('-DOKC_B4FIN=1' in c['extra_cuda_cflags'] for c in calls))
            mid=next(c for c in calls if c['name']==st['ma_ext']['mid']._module.__name__)
            self.assertEqual(g.sha(mid['cuda_sources']),g.MID_NATIVE)
            self.assertIn('-DOKC_SPLIT=1',mid['extra_cuda_cflags'])
            self.assertNotIn('-DOKC_SC6W=1',mid['extra_cuda_cflags'])

    def test_actual_concurrent_tasks_and_preparation_warm_attestation(self):
        for sc6 in (False,True):
            e,calls,runs=environment(self.source);st,w,tasks=build(e,sc6,concurrent=True)
            self.assertEqual([f.__defaults__[0] for f in tasks],[1,8,32,128])
            self.assertEqual(len(calls),4);self.assertEqual(len(st['ma_ext']),4)
            out=io.StringIO()
            with contextlib.redirect_stdout(out):e['_ma_prepare'](w,st)
            self.assertEqual([r[0] for r in runs],[1,8,16,32,33,128,256])
            self.assertTrue(st['_fork_native_eligible'])
            line=next(l for l in out.getvalue().splitlines() if l.startswith('D32_MID_SELECTION '))
            report=ast.literal_eval(line[len('D32_MID_SELECTION '):])
            self.assertEqual(report['source_sha256'],g.MID_NATIVE)
            self.assertEqual(report['defines'],self.manifest['mid_defines'])
            self.assertIsNone(report['resource_query'])  # No fabricated GPU resources.
            self.assertEqual(report['module'],st['ma_ext']['mid']._module.__name__)
            self.assertTrue(all(int(ids[:,0].eq(0).sum())==ids.shape[0] for _,_,ids in runs))

    def test_wrong_keys_source_flags_module_and_runtime_failure_negatives(self):
        for kind in ('b4','missing','source','flags','module'):
            e,calls,runs=environment(self.source);st,w,_=build(e)
            if kind=='b4':st['ma_ext']['b4']=st['ma_ext']['mid']
            if kind=='missing':st['ma_ext'].pop('mid')
            if kind=='source':e['_D32_MID_SOURCE']='0'*64
            if kind=='flags':e['_D32_MID_EXPECTED_DEFINES']=()
            if kind=='module':st['ma_ext']['mid']=st['ma_ext']['tiny2']
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):e['_ma_prepare'](w,st)
            self.assertFalse(runs)  # Reject wrong ownership before native execution.
        e,_,_=environment(self.source,failure='CUDA math/execution injected');st,w,_=build(e)
        with self.assertRaisesRegex(RuntimeError,'CUDA math/execution'):e['_ma_prepare'](w,st)

    def test_real_fallback_entries_resource_failures_and_no_unconditional_build(self):
        for major,sfx in ((8,('-DOKC_PRMTSC=1',)),(9,())):
            e,calls,runs=environment(self.source,major);st,w,tasks=build(e,sfx=sfx)
            self.assertFalse(tasks)
            with contextlib.redirect_stdout(io.StringIO()):e['_ma_prepare'](w,st)
            self.assertFalse(st['_fork_native_eligible']);self.assertFalse(calls)
            self.assertIs(e['_moe'](32,st),st['_fork_marlin_adapter'])
            self.assertIsNone(e['_moe']())
        e,calls,runs=environment(self.source,failure='FORK_MOE_RESOURCE injected');st,w,_=build(e)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):e['_ma_prepare'](w,st)
        self.assertFalse(st['_fork_native_eligible'])
        self.assertNotIn('D32_MID_SELECTION',out.getvalue())
        self.assertIs(e['_moe'](33,st),st['_fork_marlin_adapter'])

    def test_legal_distinct_topk_worst_split_job_bound(self):
        # A realizable degree sequence attaining60 jobs atT32:28experts x9
        # entries plus4experts x1; greedily assign8distinct experts/token.
        remaining=[9]*28+[1]*4;routes=[]
        for token in range(32):
            chosen=sorted(range(32),key=lambda i:(remaining[i],-i),reverse=True)[:8]
            self.assertTrue(all(remaining[i]>0 for i in chosen))
            self.assertEqual(len(set(chosen)),8)
            for i in chosen:remaining[i]-=1
            routes.append(chosen)
        self.assertEqual(remaining,[0]*32)
        counts=[sum(i in row for row in routes) for i in range(32)]
        self.assertEqual(sum((c+7)//8 for c in counts),60)
        self.assertLess(60,88)
        for m in (17,24,31,32):self.assertLessEqual((m*8+32*7)//8,60)


if __name__=='__main__':unittest.main(verbosity=2)
