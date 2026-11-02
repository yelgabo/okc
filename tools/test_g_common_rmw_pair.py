"""Generated paired entry points on CPU; native math/CUDA explicitly mocked."""
import ast,base64,contextlib,copy,inspect,io,json,math,os,types,unittest,zlib
from pathlib import Path
from unittest.mock import patch
import torch
from g_common_rmw_pair import build,function
from g_short_rmw_check import build as old_build

ROOT=Path(__file__).resolve().parents[1]
os.environ['OKC_REPO_ROOT']=str(ROOT)
os.environ['OKC_MOE_WORKTREE']=str(ROOT)
os.environ['SRM_ARTIFACT']=str(ROOT/'.build/rmw-pair-v4/check.py')
import proj_g_short_rmw_host_test as old
from test_g_common_release_host import Attention

SOURCE,MANIFEST=build(ROOT)
old.SOURCE=SOURCE


def fixture():
    # Retain existing four-owner arithmetic/CUDA stand-ins; execute the new
    # support as well as the frozen machinery and count all three capture calls.
    text=inspect.getsource(old.fixture)
    text=text.replace("suffix=SOURCE[SOURCE.index('def _wl_storage('):]",
                      "suffix=SOURCE[SOURCE.index('def _gqb_signature('):]")
    text=text.replace('ns.update(prepare=prep,',
        "ns.update(_GUARDED_IDS=set(),_ab_ref_back=lambda f,w,ids:(f['ref_h'],f['ref_s']),_att7=lambda:None,prepare=prep,")
    text=text.replace("d['config'].update(page_size=1)","d['config'].update(page_size=1,num_kv_heads=2,window=2048 if c=='sliding' else 0)")
    text=text.replace("out=ns['_wl_run'](d,s)","out=ns['_wl_run'](d,s)\n        out=ns['_wl_run'](d,s)\n        out=ns['_wl_run'](d,s)")
    env=dict(vars(old));env['SOURCE']=SOURCE;exec(text,env)
    ns,stats,production=env['fixture']()
    audit=Attention()
    def attention(d,*args):
        d['config'].setdefault('window',128 if d['config']['context']=='sliding' else 0)
        d['config'].setdefault('num_kv_heads',2)
        audit.run(d)
    ns['_att7']=lambda:audit;ns['_WL_ATTN']=attention
    ns['_wl_weight_signature']=lambda w:'a'*64
    ns['math']=math
    return ns,stats,production,audit


def native_run(self,*args):
    def op():
        ids=args[3];t=args[2].shape[0]
        local=ids[ids>=0].to(torch.int64)
        counts=torch.bincount(local,minlength=32)
        assert int(counts.max())<=t,"SRP_FIXTURE_ROUTE_CAPACITY"
        assert all(len(row[row>=0])==len(torch.unique(row[row>=0])) for row in ids),"SRP_FIXTURE_ROUTE_UNIQUE"
        args[5].zero_();args[5].add_(1 if bool((ids>=0).any()) else 0)
    if old.G.active is not None:old.G.active.ops.append(op)
    op()


class Pair(unittest.TestCase):
    def call(self,ns,inference=True):
        log=io.StringIO();fill=torch.Tensor.fill_
        def recorded(t,v):
            if old.G.active is not None:old.G.active.ops.append(lambda:fill(t,v))
            return fill(t,v)
        resource=[96,0,0,231168,288,1024,1,132,1,132]*2
        with patch.object(torch.Tensor,'fill_',recorded),patch.object(old.Native,'okc_run',native_run),patch.object(old.Native,'wl_resources',return_value=[96,0,0,231168,1,288]*2),patch.object(old.Native,'qual_native_resources',return_value=resource,create=True),contextlib.redirect_stdout(log),torch.inference_mode(inference):
            result=ns['prepare'](old.WEIGHTS())
        line=next(x for x in log.getvalue().splitlines() if x.startswith('SRM_COMMON_REPORT '))
        return result,json.loads(zlib.decompress(base64.b64decode(line.split(' ',1)[1])))

    def test_full_prepare_off_return_and_cleanup(self):
        for inference in (True,False):
            ns,stats,prod,audit=fixture();saved=tuple(ns[k] for k in ('_ma_ext','_fork_native_moe','_decode_attention7','_ab_ref_back'))
            result,r=self.call(ns,inference)
            self.assertIs(result,prod[0]);self.assertFalse(result.get('_srm_on',False))
            self.assertEqual(tuple(ns[k] for k in ('_ma_ext','_fork_native_moe','_decode_attention7','_ab_ref_back')),saved)
            self.assertFalse(audit.active);self.assertFalse(ns['_WL_EXT_FLAGS']);self.assertIsNone(ns['_SRP_CACHE'])
            self.assertEqual((len(r['rows']),r['math_checks'],r['fresh_final_checks'],r['empty_native_checks'],r['occupied_to_empty_checks']),(36,144,144,32,16))
            self.assertEqual(len(r['actual_attention']),432)
            self.assertEqual(r['reference_cache'],dict(hit=1008,miss=144,peak_entries=4))
            self.assertTrue(r['cleanup_restored']);self.assertFalse(r['decision']['gate'])
            self.assertEqual(stats['oracle_checks'],576)
            self.assertEqual(r['seeds'],[993811,993817,993821])
            self.assertLess(len(base64.b64encode(zlib.compress(json.dumps(r).encode(),9))),11000)

    def test_actual_entry_rejects_frozen_duplicate_fixture(self):
        # Execute the old fixture through the actual generated prepare entry.
        # Its first occupied T1 row violates the native per-expert capacity.
        frozen=(ROOT/'.build/rmw-pair-v3/check.py').read_text()
        ns,_,_,audit=fixture();saved=ns['_fork_native_moe']
        exec(function(frozen,'_srp_occupied_empty'),ns)
        with self.assertRaisesRegex(AssertionError,'SRP_FIXTURE_ROUTE_CAPACITY'):self.call(ns)
        self.assertIs(ns['_fork_native_moe'],saved);self.assertFalse(audit.active)
        self.assertIsNone(ns['_SRP_CACHE'])

    def test_actual_entry_rejects_duplicate_below_capacity(self):
        # T8 distributed IDs keep every expert count <=8 while one row repeats
        # an expert: the distinct-ID contract must independently reject it.
        occupied=function(SOURCE,'_srp_occupied_empty')
        occupied=occupied.replace('for t in (1,8):','for t in (8,):')
        occupied=occupied.replace('torch.arange(8,device=dev,dtype=torch.int32).expand(t,8).contiguous()',
            'torch.arange(t*8,device=dev,dtype=torch.int32).reshape(t,8)%32')
        occupied=occupied.replace('ids=occupied_ids.clone()',
            'ids=occupied_ids.clone();ids[0,1]=ids[0,0]')
        ns,_,_,audit=fixture();saved=ns['_fork_native_moe'];exec(occupied,ns)
        with self.assertRaisesRegex(AssertionError,'SRP_FIXTURE_ROUTE_UNIQUE'):self.call(ns)
        self.assertIs(ns['_fork_native_moe'],saved);self.assertFalse(audit.active)
        self.assertIsNone(ns['_SRP_CACHE'])

    def test_failure_restores_and_cannot_print_success(self):
        for target in ('_WL_PREPARE','_wl_check','_srm_resource_gate'):
            ns,_,_,audit=fixture();saved={k:ns[k] for k in ('_ma_ext','_fork_native_moe','_decode_attention7','_ab_ref_back')}
            ns[target]=lambda *a:(_ for _ in ()).throw(RuntimeError('specific failure'))
            with self.assertRaisesRegex(RuntimeError,'specific failure'):self.call(ns)
            self.assertTrue(all(ns[k] is v for k,v in saved.items()));self.assertFalse(audit.active)
            self.assertIsNone(ns['_SRP_CACHE']);self.assertFalse(ns['_WL_EXT_FLAGS'])

    def test_memo_owners_versions_and_inference_scope(self):
        ns,_,_,_=fixture();ns['_SRP_START']=__import__('time').monotonic()
        weights={};front={'x':torch.ones(2)};ids=torch.tensor([[0,1]])
        ns['_SRP_REFBACK']=lambda *a:(torch.ones(2),torch.ones(2))
        ns['_SRP_CACHE']={};ns['_srp_back'](front,weights,ids)
        ns['_srp_back'](front,weights,ids);self.assertEqual(ns['_SRP_CACHE_COUNTS']['hit'],1)
        with self.assertRaisesRegex(AssertionError,'REFERENCE_OWNER'):ns['_srp_back'](front,{},ids)
        entry=ns['_SRP_CACHE'][id(front)];entry['result'][0].add_(1)
        with self.assertRaisesRegex(AssertionError,'RESULT_MUTATED'):ns['_srp_back'](front,weights,ids)
        ns['_SRP_CACHE']={};ns['_srp_back'](front,weights,ids)
        ns['_SRP_CACHE'][id(front)]['ids'].add_(1)
        with self.assertRaisesRegex(AssertionError,'IDS_MUTATED'):ns['_srp_back'](front,weights,ids)
        ns['_SRP_CACHE']={};ns['_srp_back'](front,weights,ids);front['x'].add_(1)
        before=ns['_SRP_CACHE_COUNTS']['miss'];ns['_srp_back'](front,weights,ids)
        self.assertEqual(ns['_SRP_CACHE_COUNTS']['miss'],before+1)

    def test_actual_attention_failure_is_not_a_hit(self):
        ns,_,_,audit=fixture();audit.failure=True
        with self.assertRaisesRegex(RuntimeError,'native failure'):self.call(ns)
        self.assertFalse(audit.active);self.assertFalse(ns['_SRP_ACTUAL'])

    def test_budget_cleanup_is_unconditional(self):
        ns,_,_,_=fixture();clock=iter([0.,296.]+[296.]*10)
        ns['time']=types.SimpleNamespace(monotonic=lambda:next(clock))
        saved=ns['_ma_ext']
        with self.assertRaisesRegex(AssertionError,'295S'):self.call(ns)
        self.assertIs(ns['_ma_ext'],saved);self.assertIsNone(ns['_SRP_CACHE'])

    def test_source_delta_and_frozen_measurement(self):
        self.assertEqual((SOURCE,MANIFEST),build(ROOT))
        old_source,_=old_build(ROOT)
        for name in ('_wl_summary','_wl_check','_wl_copy','_wl_check_owners','_srm_resource_gate','_fork_native_moe'):
            # Last wrapper where product and observer share a name.
            funcs=lambda s:[ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name]
            self.assertEqual(funcs(SOURCE)[-1],funcs(old_source)[-1],name)
        self.assertIn('torch.inference_mode(False),torch.no_grad()',SOURCE)
        self.assertIn('for rep in range(24):',SOURCE);self.assertIn('256<<20',SOURCE)
        self.assertEqual(MANIFEST['control_sha256'],'4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280')
        self.assertLess(MANIFEST['bytes'],1_000_000)

    def test_actual_capsule_and_reader_negatives(self):
        import g_common_rmw_pair_read as reader
        from test_g_short_rmw_read import receipt
        ns,_,_,_=fixture();_,report=self.call(ns)
        def log(r):return 'SRM_COMMON_REPORT '+base64.b64encode(zlib.compress(json.dumps(r).encode(),9)).decode()+'\n'
        self.assertEqual(reader.MANIFEST,MANIFEST)
        result=reader.read(receipt(),'abc123',log(report))
        self.assertTrue(result['complete']);self.assertFalse(result['private_raw_gate'])
        reader.verify_source(ROOT/'.build/rmw-pair-v4/check.py')
        changes=[lambda r:r.update(cleanup_restored=False),lambda r:r.update(common_control=False),
            lambda r:r.update(occupied_to_empty_checks=15),lambda r:r['actual_attention'].pop(),
            lambda r:r['common_resources']['tiny'][0].__setitem__(8,0),
            lambda r:r['reference_cache'].update(miss=0),lambda r:r['rows'][0]['times_us'].__setitem__(2,float('nan')),
            lambda r:r['decision'].update(gate=True),lambda r:r['actual_attention'][0][2].__setitem__(24,0)]
        for change in changes:
            bad=copy.deepcopy(report);change(bad)
            with self.assertRaises((AssertionError,ValueError)):reader.capsule(log(bad))
        for key,value in (('id','wrong'),('mode','run'),('status','running'),('board','wrong')):
            bad=receipt();bad[key]=value
            with self.assertRaises(AssertionError):reader.read(bad,'abc123',log(report))
        bad=receipt();bad['result']['flags']=['guard']
        with self.assertRaises(AssertionError):reader.read(bad,'abc123',log(report))
        with self.assertRaises(AssertionError):reader.capsule(log(report)+log(report))
        with self.assertRaises(AssertionError):reader.verify_source(ROOT/'.build/rmw-v2/short-rmw.py')

    def test_owner_and_stale_output_negatives(self):
        ns,_,_,_=fixture();ns['_wl_clone']=lambda state,*args:state
        with self.assertRaises(AssertionError):self.call(ns)
        ns,_,_,_=fixture();capture=ns['_wl_capture']
        def stale(d,s):
            graph,out=capture(d,s);graph.fn=lambda:None;return graph,out
        ns['_wl_capture']=stale
        with self.assertRaisesRegex(AssertionError,'WL_MATH'):self.call(ns)
        ns,_,_,_=fixture();check=ns['_wl_check'];calls=[]
        def corrupt_after_hit(*args):
            calls.append(None)
            if len(calls)==2:args[2][0].add_(1000)
            return check(*args)
        ns['_wl_check']=corrupt_after_hit
        with self.assertRaisesRegex(AssertionError,'WL_MATH'):self.call(ns)
        self.assertGreater(ns['_SRP_CACHE_COUNTS']['hit'],0)

    def test_exact_common_functions_and_device_deltas(self):
        from g_integration_rmw_release import build as make_common
        from g_short_final_rmw import NEW,OLD
        from g_common_release import MOE_RESOURCE,patch_cuda,attention_exports
        from g_short_rmw_check import load
        original=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text()
        control,_=make_common(original,False)
        device=SOURCE.split('\n# PRIVATE CHECK ONLY: four-owner short-RMW diagnostic. NEVER SUBMIT.\n')[0]
        a=[n for n in ast.parse(control).body if isinstance(n,ast.FunctionDef)]
        b=[n for n in ast.parse(device).body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            if x.name not in ('_ma_ext','_fork_native_moe','_att7'):self.assertEqual(ast.dump(x),ast.dump(y),x.name)
        self.assertEqual(function(device,'_att7'),function(attention_exports(control),'_att7'))
        literal=lambda s,k:ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==k))
        resource=load(ROOT/'tools/moe_make_sc6_compare.py').RESOURCE.replace('sc6_resources','wl_resources')
        for key in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            expected=literal(control,key)
            if key!="MOE_CUDA_SRC":expected=expected.replace(OLD,NEW)
            self.assertEqual(literal(device,key),expected+resource+MOE_RESOURCE)
        self.assertEqual(literal(device,'ATT7_CUDA_SRC'),patch_cuda(literal(control,'ATT7_CUDA_SRC')))


if __name__=='__main__':unittest.main()
