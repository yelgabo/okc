"""Actual generated host orchestration/validators on Torch CPU; CUDA/math stand-ins explicit."""
import ast, base64, contextlib, hashlib, importlib.util, io, json, os, random, sys, types, unittest, zlib
from pathlib import Path
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('OKC_REPO_ROOT',HERE.parent/'okc'))
MOE=Path(os.environ.get('OKC_MOE_WORKTREE',HERE.parent/'okc-today-moe'))
ART=Path(os.environ.get('SRM_ARTIFACT',MOE/'.build/today-moe/short-final-rmw/check.py'))
os.environ.setdefault('OKC_REPO_ROOT',str(ROOT))
sys.path.insert(0,str(MOE/'tools'))
import reset_host_facade_test as old
SOURCE=ART.read_text(encoding='utf8')

class G:
    active=None
    def __init__(self,fn=None):self.fn=fn;self.ops=[]
    def replay(self):
        if self.fn:self.fn()
        for op in self.ops:op()
class CUDA(old.Cuda):
    CUDAGraph=G
    @contextlib.contextmanager
    def graph(self,g):
        assert G.active is None
        G.active=g
        try:yield
        finally:G.active=None
class Native(old.Module):
    def okc_run(self,*args):
        fn=lambda:args[5].zero_()
        if G.active is not None:G.active.ops.append(fn)
        fn()

def fixture():
    ns=old.fixture('stage');ns['torch'].cuda=CUDA()
    modules={};production=[];stats=dict(captures=0,replays=0,oracle_checks=0,dispatches=[])
    def raw_native(n,s):
        key='tiny' if n<=4 else 'tiny2' if n<=8 else 'mid' if n<=16 else 'b4' if n<=32 else 'main'
        on=bool(s.get('_srm_on',False)) and key in ('tiny','tiny2');ident=(key,on)
        if ident not in modules:modules[ident]=old._RouterExt(Native(str(ident)))
        ext=modules[ident];s['ma_ext'][key]=ext
        flags=('-DOKC_SHORT_FINAL_RELAXED=1',) if on else ()
        if key=='b4':flags+=('-DOKC_B4FIN=1',)
        ns.setdefault('_WL_EXT_FLAGS',{})[id(ext._module)]=flags
        stats['dispatches'].append(ident)
        return ext
    def prep(w):
        s=old.state();production.append(s)
        for n in (1,8,16,32,128):raw_native(n,s)
        return s
    def inputs(b,l,c,seed):
        d=old.inputs(b,l,c,seed);n=b*l
        d['config'].update(page_size=1)
        d.update(seq_lens=torch.zeros(b,dtype=torch.int32),page_table=torch.arange(n,dtype=torch.int32).reshape(b,l),
                 k_scale=torch.ones(1),v_scale=torch.ones(1),k_cache=torch.zeros(n,2,64),v_cache=torch.zeros(n,2,64))
        return d
    def front(d,w,*unused):
        f=old.front(d,w);n=d['config']['tokens'];b=d['config']['batch'];l=d['config']['q_len']
        f.update(ref_h=d['stream']+d['attn_input'],ref_s=d['attn_input']*2,
                 k_new=d['attn_input'][:,0,None,None].expand(n,2,64).reshape(b,l,2,64).clone(),
                 v_new=d['stream'][:,0,None,None].expand(n,2,64).reshape(b,l,2,64).clone())
        return f
    def model(d,s):
        n=d['config']['tokens'];ns['_decode_attention7'](d,s,None);ns['_fork_native_moe'](n,s)
        f=ns['_ab_ref_front'](d,None)
        d['k_cache'].copy_(f['k_new'].reshape_as(d['k_cache']));d['v_cache'].copy_(f['v_new'].reshape_as(d['v_cache']))
        return f['ref_h'].clone(),f['ref_s'].clone(),f['biased'].topk(8,dim=-1).indices.to(torch.int32)
    ns.update(prepare=prep,kernel_fn=model,_fork_native_moe=raw_native,_decode_attention7=lambda *x:None,
              _ab_inputs=inputs,_ab_ref_front=front,N_EXPERTS=256,FP8=torch.float32)
    # Execute exact generated clone/prepare/policy/score/empty/normal-wrapper source.
    suffix=SOURCE[SOURCE.index('def _wl_storage('):]
    exec(suffix,ns)
    # Execute actual generated routing/finite/adaptive/cache validators separately;
    # replace only expensive reference arithmetic with the documented CPU model.
    tree=ast.parse(SOURCE)
    for name in ('_ab_local','_ab_check_routing','_ab_compare','_ab_validate'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(ast.get_source_segment(SOURCE,node),ns)
    ns.update(_ab_ref_back=lambda f,w,ids:(f['ref_h'],f['ref_s']),_ab_coeffs=lambda w:(0.,1.))
    validator=ns['_ab_validate']
    def counted_validator(*args):
        stats['oracle_checks']+=1
        return validator(*args)
    ns['_ab_validate']=counted_validator
    def capture(d,s):
        stats['captures']+=1
        out=ns['_wl_run'](d,s)
        def replay():
            stats['replays']+=1
            current=model(d,s)
            for a,b in zip(out,current):a.copy_(b)
        return G(replay),out
    ns['_wl_capture']=capture
    return ns,stats,production

WEIGHTS=lambda:dict(attn_residual_gate=torch.ones(1),ffn_residual_gate=torch.ones(1))
class Tests(unittest.TestCase):
    def call(self,ns):
        log=io.StringIO();before=torch.backends.cuda.matmul.allow_tf32
        ns['_AB_DEQ']['sentinel']=True
        fill=torch.Tensor.fill_
        def recorded_fill(t,v):
            if G.active is not None:G.active.ops.append(lambda:fill(t,v))
            return fill(t,v)
        try:
            with patch.object(torch.Tensor,'fill_',recorded_fill),contextlib.redirect_stdout(log):result=ns['prepare'](WEIGHTS())
        finally:
            self.assertEqual(torch.backends.cuda.matmul.allow_tf32,before);self.assertFalse(ns['_AB_DEQ']);self.assertIsNone(G.active)
        line=next(x for x in log.getvalue().splitlines() if x.startswith('SRM_REPORT '))
        report=json.loads(zlib.decompress(base64.b64decode(line.split(' ',1)[1])))
        self.assertLess(len(line.split(' ',1)[1]),11000)
        return result,report
    def test_whole_prepare_and_normal_entry(self):
        ns,stats,prod=fixture();result,r=self.call(ns)
        self.assertIs(result,prod[0]);self.assertFalse(result.get('_srm_on',False))
        self.assertEqual((len(r['rows']),r['math_checks'],r['fresh_final_checks'],r['empty_native_checks']),(36,144,144,32))
        self.assertEqual(stats['captures'],144);self.assertEqual(stats['oracle_checks'],576);self.assertFalse(r['decision']['gate'])
        self.assertEqual(set(r['resources']),{'tiny','tiny2','mid','b4','main'})
        self.assertEqual(r['seeds'],[993811,993817,993821])
        for n in (1,8,16,32,128,256):
            d=ns['_ab_inputs'](n,1,'sliding',1);ns['kernel_fn'](d,result)
            self.assertFalse(stats['dispatches'][-1][1])
        rng=random.Random(5)
        for row in r['rows']:row['times_us']=[rng.uniform(55,500) for _ in range(4)]
        blob=base64.b64encode(zlib.compress(json.dumps(r,separators=(',',':')).encode(),9))
        self.assertLess(len(blob),11000)
    def test_stale_outputs_and_cache(self):
        ns,_,_=fixture();oldcap=ns['_wl_capture']
        def stale(d,s):
            g,o=oldcap(d,s);g.fn=lambda:None;return g,o
        ns['_wl_capture']=stale
        with self.assertRaisesRegex(AssertionError,'WL_MATH'):self.call(ns)
        ns,_,_=fixture();cap=ns['_wl_capture']
        def badcache(d,s):
            g,o=cap(d,s);d['k_cache'].fill_(99);return g,o
        ns['_wl_capture']=badcache
        with self.assertRaisesRegex(AssertionError,'WL_MATH'):self.call(ns)
        ns,_,_=fixture()
        def empty_stale(self,*args):args[5].zero_() # eager succeeds, captured native op missing
        with patch.object(Native,'okc_run',empty_stale):
            with self.assertRaisesRegex(AssertionError,'SRM_EMPTY_GRAPH'):self.call(ns)
    def test_reference_and_output_nonfinite(self):
        ns,_,_=fixture();back=ns['_ab_ref_back']
        ns['_ab_ref_back']=lambda *a:(back(*a)[0]*float('nan'),back(*a)[1])
        with self.assertRaisesRegex(AssertionError,'SRM_ADAPTIVE_NONFINITE'):self.call(ns)
        ns,_,_=fixture();cap=ns['_wl_capture']
        def bad(d,s):
            g,o=cap(d,s);o[0].fill_(float('nan'));return g,o
        ns['_wl_capture']=bad
        with self.assertRaisesRegex(AssertionError,'WL_MATH'):self.call(ns)
    def test_owners_resource_flags_and_fatal_sync(self):
        ns,_,_=fixture();clone=ns['_wl_clone']
        def alias(s,ro,on):
            a=clone(s,ro,on);a['ma_scratch']=s['ma_scratch'];return a
        ns['_wl_clone']=alias
        with self.assertRaisesRegex(AssertionError,'overlap|owned'):self.call(ns)
        ns,_,_=fixture();raw=ns['_WL_NATIVE']
        def badflag(n,s):
            e=raw(n,s);ns['_WL_EXT_FLAGS'][id(e._module)]=();return e
        ns['_WL_NATIVE']=badflag
        with self.assertRaisesRegex(AssertionError,'short RMW flag drift|B4 drift|SRM_PREP_FLAGS'):self.call(ns)
        ns,_,_=fixture();ns['torch'].cuda.synchronize=lambda:(_ for _ in ()).throw(RuntimeError('fatal CUDA'))
        with self.assertRaisesRegex(RuntimeError,'fatal CUDA'):self.call(ns)
    def test_routing_resources_and_score_decisions(self):
        ns,_,_=fixture();cap=ns['_wl_capture']
        def routing(d,s):
            g,o=cap(d,s)
            if s['_srm_on']:o[2].copy_(o[2].flip(-1))
            return g,o
        ns['_wl_capture']=routing
        with self.assertRaisesRegex(AssertionError,'SRM_ROUTING_ARM_MISMATCH'):self.call(ns)
        ns,_,_=fixture()
        with patch.object(Native,'wl_resources',return_value=[96,0,231168,128,0,16]*2):
            with self.assertRaisesRegex(AssertionError,'SRM_RESOURCE'):self.call(ns)
        ns,_,_=fixture()
        rows=[dict(seed=s,tokens=t,context=c,times_us=[100.,100.,95.,95.])for s in ns['_WL_SEEDS']for t in ns['_WL_LOADS']for c in ('sliding','full')]
        self.assertTrue(ns['_wl_summary'](rows,'all')['gate'])
        rows[0]['times_us'][2]=110
        self.assertFalse(ns['_wl_summary'](rows,'all')['gate'])
        state=old.state();state.update(_srm_on=True,_wl_probe=True,_wl_hits=[])
        raw=ns['_WL_NATIVE']
        def wrong(n,s):
            e=raw(n,s);ns['_WL_EXT_FLAGS'][id(e._module)]=();return e
        ns['_WL_NATIVE']=wrong
        with self.assertRaisesRegex(AssertionError,'short RMW flag drift'):ns['_fork_native_moe'](1,state)
    def test_budget_and_timing_guards(self):
        ns,_,_=fixture();clock=iter((0,296));ns['time']=types.SimpleNamespace(monotonic=lambda:next(clock))
        with self.assertRaisesRegex(AssertionError,'SRM_INCOMPLETE'):self.call(ns)
        ns,stats,_=fixture();ns['time']=types.SimpleNamespace(monotonic=lambda:296 if stats['replays'] else 0)
        with self.assertRaisesRegex(AssertionError,'SRM_INCOMPLETE'):self.call(ns)
        self.assertGreater(stats['replays'],0) # actual inner fresh-loop guard, not just initial prepare
        ns,_,_=fixture()
        with patch.object(old.Event,'elapsed_time',return_value=float('nan')):
            with self.assertRaisesRegex(AssertionError,'SRM_BAD_TIMING'):self.call(ns)
        ns,_,_=fixture()
        rows=[dict(seed=s,tokens=t,context=c,times_us=[100.]*4)for s in ns['_WL_SEEDS']for t in ns['_WL_LOADS']for c in ('sliding','full')]
        rows[0]['times_us'][0]=0
        with self.assertRaisesRegex(AssertionError,'SRM_BAD_SUMMARY_TIMING'):ns['_wl_summary'](rows,'all')

if __name__=='__main__':unittest.main()
