"""CPU contracts only: short-path source scope, precision and interval accounting."""
import ast,unittest,os
from pathlib import Path
from dl_short_path_analyze import analyze
BODY=Path(__file__).with_name('dl_short_path_body.py').read_text(encoding='utf8')
ROOT=Path(os.environ.get('OKC_REVIEW_ROOT',Path(__file__).resolve().parents[1]))

def function(name,env=None):
    env={} if env is None else env
    n=next(n for n in ast.parse(BODY).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),'short','exec'),env);return env[name]

class Short(unittest.TestCase):
    def test_complete_same_stream_producer_chain(self):
        f=function('_sp_chain')
        e=[(0,1000,1,'qkv'),(1100,2100,1,'glue_cluster_boundary'),(2200,20100,1,'sk2_kernel_gle'),(20200,22200,1,'_sl_tile_route'),(22300,52300,1,'void okc::moe_fused'),(52400,58400,1,'unpool')]
        r=f(e,'fused');self.assertEqual(r['producer_span_us'],17.9)
        self.assertEqual(r['producer_to_route_gap_us'],.1);self.assertEqual(r['route_to_expert_gap_us'],.1)
        self.assertEqual(r['PR_start_to_expert_end_us'],50.1)
        self.assertNotEqual(r['graph_span_us'],r['activity_sum_not_saved_us'])
        with self.assertRaises(AssertionError):f(e,'split')
        bad=list(e);bad[4]=(22300,52300,2,bad[4][3])
        with self.assertRaises(AssertionError):f(bad,'fused')
    def test_precision_preserving_existing_serial_path(self):
        import torch
        calls=[]
        def fused(h,hr,t,st,pick):calls.append('fused');return (torch.empty(t,4096,dtype=torch.bfloat16),torch.empty(t,256)),-1
        def split(h,hr,t,st,pick):calls.append('split');self.assertEqual(hr.dtype,torch.float32);return fused(h,hr,t,st,pick)
        f=function('_pool_router_fp32',{'torch':torch,'_SP_POOL':fused,'_gle_prior_pool_router_fp32':split})
        for mode in ('fused','split'):
            st={'_sp_active':True,'_sp_mode':mode,'_sp_capture':{}}
            f(torch.empty(8,8192,dtype=torch.bfloat16),torch.empty(8,8192),8,st,'bfp')
            self.assertEqual(st['_sp_capture']['pooled'].dtype,torch.bfloat16)
            self.assertEqual(st['_sp_capture']['logits'].dtype,torch.float32)
        self.assertIn('split',calls)
    def test_actual_scoring_counterfactual(self):
        def report(k):
            rows=[]
            for seed in (1,2,3):
                for length in (1,8):
                    for context in ('sliding','full'):
                        for arm in ('fused1','fused2','split1','split2'):
                            rows.append({'seed':seed,'length':length,'context':context,'arm':arm,'median_us':100 if arm.startswith('fused') else k,'trace':{'chain':{}}})
            return {'rows':rows,'checks':96}
        self.assertTrue(analyze(report(96))['decision_for_further_candidate_gate'])
        self.assertFalse(analyze(report(100))['decision_for_further_candidate_gate'])
        self.assertFalse(analyze(report(104))['decision_for_further_candidate_gate'])
    def test_metadata_codec_full48row_budget(self):
        import importlib.util,random,statistics
        root=ROOT
        spec=importlib.util.spec_from_file_location('codec',root/'tools/dl_timeline_codec.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
        transport=function('_sp_transport');rng=random.Random(11891);rows=[]
        long='void sk2_kernel_gle4944bfa97e00<1,8,4,1,8,1,6,true,false,2,1,__nv_bfloat16,true>(const uint8_t*, const __nv_bfloat16*, float*, int*)'
        names=['qkv','attn','o','glue_cluster_boundary',long,'router_fp32_kernel','_sl_tile_route','void okc::moe_fused<false>(okc::Params)','unpool','_sl_cluster_finish']
        for i in range(48):
            samples=[rng.randrange(2000,4000)*.032+1.e-9*rng.random() for _ in range(20)]
            events=[];cursor=0
            for name in names:
                width=rng.randrange(1000,20000);events.append((cursor,cursor+width,1,name));cursor+=width+rng.randrange(100,500)
            trace=codec.summarize(events)
            trace['chain']={k:rng.random()*20 for k in ('producer_span_us','producer_to_route_gap_us','route_us','route_to_expert_gap_us','expert_us','PR_start_to_expert_end_us','graph_span_us','activity_sum_not_saved_us')}
            trace['chain']['producer_kernels']=[long,'router_fp32_kernel']
            rows.append({'arm':('fused1','fused2','split1','split2')[i%4],'mode':'fused' if i%4<2 else 'split','length':1 if i<24 else 8,'seed':401031+i%3,'context':'sliding' if i%2 else 'full',
                'times_us':samples,'median_us':statistics.median(samples),'trace':trace,'routing':{'local_start':96,'reference_group_slots':[4,7,11,13,6,5,7,11],'local_slots':13,'local_experts':7},'correct':True,'pool_choice':long})
        report={'schema':1,'rows':rows,'checks':96}
        with self.assertRaises(ValueError):codec.encode_report(report)
        compact=transport(report);blob=codec.encode_report(compact)
        decoded=codec.decode_report(blob)
        self.assertLessEqual(len(blob),11000)
        self.assertEqual(len(decoded['rows']),48)
        for original,got in zip(rows,decoded['rows']):
            self.assertEqual(original['median_us'],got['median_us'])
            self.assertEqual(len(got['times_us']),20)
            self.assertEqual(len(got['trace']['events']),10)
        print('SHORT_CODEC48_BYTES',len(blob))
    def test_reference_cache_bracket_and_primary_owner(self):
        import torch
        cache={'old':1};previous=torch.backends.cuda.matmul.allow_tf32
        def fail(w):
            self.assertEqual(cache,{})
            cache['new']=1
            raise RuntimeError('budget')
        f=function('prepare',{'torch':torch,'_AB_DEQ':cache,'_sp_prepare':fail})
        with self.assertRaises(RuntimeError):f({})
        self.assertEqual(cache,{})
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,previous)
        tree=ast.parse(BODY);prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sp_prepare')
        self.assertEqual(ast.unparse(prep.body[-1]),'return production')
    def test_every_reference_global_resolves_in_emitted_source(self):
        import builtins,dis,types
        from dl_short_path_trace import build
        root=ROOT
        out=build((root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'),root)
        bound=set(dir(builtins))
        def bindings(nodes):
            for n in nodes:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):bound.add(n.name)
                elif isinstance(n,(ast.Import,ast.ImportFrom)):
                    bound.update(a.asname or a.name.split('.')[0] for a in n.names)
                elif isinstance(n,(ast.Assign,ast.AnnAssign)):
                    for target in (n.targets if isinstance(n,ast.Assign) else [n.target]):bound.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name))
                elif isinstance(n,ast.If):bindings(n.body+n.orelse)
        bindings(ast.parse(out).body)
        names={'_ab_inputs','_ab_rms','_ab_rope','_ab_coeffs','_ab_attention','_ab_ref_front','_ab_deq','_ab_ref_back','_ab_local','_ab_check_routing','_ab_compare','_ab_validate'}
        code=compile(out,'emitted','exec');missing={}
        def loaded(c):
            result={i.argval for i in dis.get_instructions(c) if i.opname=='LOAD_GLOBAL'}
            for child in c.co_consts:
                if isinstance(child,types.CodeType):result|=loaded(child)
            return result
        for c in code.co_consts:
            if isinstance(c,types.CodeType) and c.co_name in names:
                diff=loaded(c)-bound
                if diff:missing[c.co_name]=sorted(diff)
        self.assertEqual(missing,{})
        self.assertIn('ROUTE_EPS',bound)

    def test_native_literals_and_other_functions_unchanged(self):
        from dl_short_path_trace import build
        root=ROOT
        src=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')
        out=build(src,root)
        a,b=ast.parse(src),ast.parse(out)
        def lits(t):return {ast.unparse(n.targets[0]):ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000}
        self.assertEqual(lits(a),{k:v for k,v in lits(b).items() if k in lits(a)})
        self.assertLess(len(out.encode()),1_000_000)
        self.assertIn("'natural routing drift; cannot attribute expert timing'",out)
        self.assertIn("'routing changed during timed replay'",out)

if __name__=='__main__':unittest.main()
