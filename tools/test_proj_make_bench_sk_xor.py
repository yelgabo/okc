"""CPU transport/source/host-scope gates; no H200 or CUDA performance claim."""
import ast,base64,copy,importlib.util,json,os,random,struct,types,unittest,zlib
from pathlib import Path
from unittest.mock import patch
from proj_make_bench_sk_xor import score_helper
from proj_make_bench_sk_xor_body import BODY
from proj_sk_xor_codec import encode_report,decode_report
ROOT=Path(__file__).resolve().parents[1]
READER=Path(os.environ.get('SK_XOR_SCORE_READER',ROOT/'tools/proj_sk_xor_read.py'))


def helper_ns():
    ns={'prepare':lambda w:None,'kernel_fn':lambda d,s:None}
    source=BODY.replace('@@PIN@@',repr({'cpu_fixture':True})).replace('@@MODEL@@',score_helper(READER.read_text(encoding='utf8')))
    code=Path(__file__).with_name('proj_sk_xor_codec.py').read_text(encoding='utf-8-sig')
    fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='encode_report')
    source=source.replace('@@CODEC@@',ast.get_source_segment(code,fn).replace('def encode_report(','def _sx_encode(',1))
    import statistics,time
    ns.update(statistics=statistics,time=time)
    exec(source,ns)
    return ns


def report_fixture():
    rng=random.Random(431);rows=[]
    target={('qkv_gate',8),('qkv_gate',16),('qkv_gate',32),('o',16),('o',32),('pooler',16),('pooler',32),('unpooler',8),('unpooler',16),('unpooler',32)}
    for seed in (903151,903157,903163):
        for mi,m in enumerate((1,8,16,32,128,256)):
            for li,layer in enumerate(('qkv_gate','o','pooler','unpooler','router')):
                yes=(layer,m) in target;values={}
                for arm in (('c0','c1','t0','t1','anchor') if yes else ('common',)):
                    # Realistic independent CUDA-event float32-ms jitter; no float rounding.
                    ticks=[int(rng.gauss(210+mi*30+li*11,9)) for _ in range(24)]
                    values[arm]=[1000*struct.unpack('<f',struct.pack('<f',t*.000032))[0] for t in ticks]
                rows.append({'seed':seed,'layer':layer,'m':m,'target':yes,'samples_us':values})
    return {'format':'PROJ_SK_XOR_V1','seeds':[903151,903157,903163],'rounds':24,'rows':rows,
            'pins':{'c2':'a'*64,'helper':'b'*64},'anchors':{str(i):{'code':'run','cid':i,'full':i+1,'vars':['base4','cid','diag','ext','full','k','n','out_dtype','pmax','stream','tab','tokr']} for i in range(10)},
            'resources':{str(i):{a:{'cfg':i,'module_name':'okc_owned_sx_'+('sx_a' if a.endswith('0') else 'sx_b'),
                'resources':[60+i,0,0,10000+i*1024,192,192,2], 'register_thread_product':(60+i)*192,
                'pack_metadata':{'n':8192,'k':8192,'pmax':i*32,'escapes':4000+i*30,'records':5000+i*31,'header_bytes':32000,'patch_bytes':10000,'rowmap_bytes':32768},
                'immutable_bytes':[11000000+i*3200,4096+i*512],
                'expected_layout':{'static':0,'dynamic':10000+i*1024,'threads':192,'headers_per_stage':64,'stage_stride':8192,'fixed':14000,'escape_warps':4}}
                for a in ('c0','c1','t0','t1')} for i in range(10)}}


def mutate_header(encoded,fn):
    b=zlib.decompress(base64.b85decode(encoded));n=struct.unpack_from('<I',b)[0];h=json.loads(b[4:4+n]);fn(h)
    j=json.dumps(h).encode();return base64.b85encode(zlib.compress(struct.pack('<I',len(j))+j+b[4+n:])).decode()


class Gates(unittest.TestCase):
    def test_lossless_full_payload(self):
        report=report_fixture();encoded=encode_report(report)
        self.assertLess(len(encoded),13000);self.assertEqual(decode_report(encoded),report)
        print('SX realistic90row payloadbytes',len(encoded))
    def test_bad_offsets_lengths_and_float(self):
        e=encode_report(report_fixture())
        for field,value in (('offset',-1),('count',23),('offset',10**9)):
            bad=mutate_header(e,lambda h:h['rows'][0]['samples_us']['common'].__setitem__(field,value))
            with self.assertRaises(AssertionError):decode_report(bad)
        bad=mutate_header(e,lambda h:h['rows'][1]['samples_us']['common'].__setitem__('offset',0))
        with self.assertRaises(AssertionError):decode_report(bad)
        report=report_fixture();report['rows'][0]['samples_us']['common'][0]=1.234567890123
        with self.assertRaises(AssertionError):encode_report(report)
    def test_embedding_matches_model(self):
        ns=helper_ns();spec=importlib.util.spec_from_file_location('score_reader',READER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=report_fixture()['rows'];self.assertEqual(ns['_sx_score_model'](rows),module.score_model(rows))
        self.assertEqual(ns['_SX_SEEDS'],module.SEEDS)
    def test_fail_closed_budget_and_source(self):
        ns=helper_ns()
        with self.assertRaises(AssertionError):ns['_sx_guard'](ns['time'].monotonic()-296)
        self.assertIn('len(packed)<13000',ast.get_source_segment(Path(__file__).with_name('proj_sk_xor_codec.py').read_text(encoding='utf-8-sig'),next(n for n in ast.parse(Path(__file__).with_name('proj_sk_xor_codec.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='encode_report')))
        self.assertNotIn('import reference',BODY);self.assertNotIn('subprocess',BODY.split('# Private')[0])
    def test_oracle_and_view_alias_reject(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch required')
        ns=helper_ns();ns['torch']=torch
        x=torch.zeros(2,65,dtype=torch.bfloat16);x[:,16]=8
        ref=torch.ones(2,4,dtype=torch.bfloat16)
        ns['_sx_check'](ref.clone(),ref,x)
        for corrupt in (torch.zeros_like(ref),torch.full_like(ref,float('nan'))):
            with self.assertRaises(AssertionError):ns['_sx_check'](corrupt,ref,x)
        base=torch.zeros(100)
        with self.assertRaises(AssertionError):ns['_sx_disjoint']({'a':[base[:10]],'b':[base[80:]]})
        ns['_sx_disjoint']({'a':[base[:10]],'b':[torch.zeros(10)]})
    def test_capture_replay_and_stream_joins(self):
        import contextlib
        ns=helper_ns();log=[]
        class Stream:
            def wait_stream(self,s):log.append('join')
        class Graph:
            def replay(self):log.append('replay')
        current=Stream()
        cuda=types.SimpleNamespace(Stream=lambda **kw:Stream(),current_stream=lambda:current,
            stream=lambda s:contextlib.nullcontext(),synchronize=lambda:log.append('sync'),
            CUDAGraph=Graph,graph=lambda g,stream:contextlib.nullcontext())
        ns['torch']=types.SimpleNamespace(cuda=cuda)
        g,out,s=ns['_sx_capture'](lambda x:log.append('run'),types.SimpleNamespace(device='cpu'))
        self.assertEqual(log.count('run'),3);self.assertEqual(log.count('join'),3)
        self.assertEqual(log.count('replay'),1);self.assertEqual(log[-1],'sync')
    def test_full_host_scope(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch required; run host-validate WSL interpreter')
        ns=helper_ns();ns['torch']=torch
        weights={name:torch.randn(4,65).to(torch.float32 if name=='router' else torch.bfloat16) for name in ns['_SX_LAYERS']}
        configs=[];indices={};metadata={}
        for (layer,m),name in ns['_SX_TARGET'].items():
            indices[name]=len(configs);configs.append((name,1,m,1,1,1,1,4,True,False,0,1,False,False));metadata[name]={'geom':{'NW':1}}
        ns['SK_INDEX']=indices;ns['SK_CONFIGS']=configs;ns['_XOR_CONFIG']=metadata
        def _sk_runner(ext,cfg_name,stream,tab,pmax,base4,n,k,out_dtype,diag):
            cid=indices[cfg_name];full=None;tokr=configs[cid][2]
            def run(x):
                _=(cid,full,tokr,stream,tab,pmax,diag)
                return (x.double()@ext.double().T).to(out_dtype)
            return run
        ns['_sk_runner']=_sk_runner
        anchor_diag=torch.zeros(4,dtype=torch.int32)
        state={'pick':{key:_sk_runner(weights[key[0]],name,object(),object(),1,0,4,65,torch.bfloat16,anchor_diag) for key,name in ns['_SX_TARGET'].items()}}
        ns['_SX_NORMAL_KERNEL']=lambda d,s:(d['x'].double()@weights[d['config']['layer']].double().T).to(torch.float32 if d['config']['layer']=='router' else torch.bfloat16)
        ns['xor_build']=lambda tag:types.SimpleNamespace(__name__=tag)
        def pack(name,w,v):return {'name':name,'variant':v,'immutable':[torch.ones(2),torch.ones(3)],'n':4,'k':65,'pmax':1}
        ns['xor_pack']=pack
        def runner(mod,name,w,v,nearby=False,packed=None):
            diag=torch.zeros(4,dtype=torch.int32)
            return (lambda x:(x.double()@w.double().T).to(torch.bfloat16)),dict(private_diag=diag,immutable=packed['immutable'],pack=packed,resources=[32,0,0,ns['_sx_layout'](name,v,1)['dynamic'],ns['_sx_layout'](name,v,1)['threads'],ns['_sx_layout'](name,v,1)['threads'],2],module_name=mod.__name__)
        ns['xor_runner']=runner
        class Graph:
            def __init__(self,fn,x):self.fn=fn;self.x=x;self.out=fn(x)
            def replay(self):self.out.copy_(self.fn(self.x))
        def capture(fn,x):g=Graph(fn,x);return g,g.out,object()
        ns['_sx_capture']=capture
        ns['_sx_cold']=lambda graphs,*args:{a:[1000*struct.unpack('<f',struct.pack('<f',.010 if a.startswith('t') else .012))[0]]*24 for a in graphs}
        printed=[]
        ns['print']=lambda *a,**kw:printed.append(' '.join(map(str,a)))
        original_picks=dict(state['pick']);original_weights={k:v.clone() for k,v in weights.items()}
        with patch.object(torch,'cuda',types.SimpleNamespace(synchronize=lambda:None)):
            ns['_sx_bench'](weights,state,ns['time'].monotonic())
        payload=next(x.split(' ',1)[1] for x in printed if x.startswith('SXREPORT1 '));result=decode_report(payload)
        self.assertEqual(result['checks'],dict(captured_placements=210,oracle_checks=840,fresh_replay_checks=420,post_timing_checks=210,onehot_nearby_checks=80,outlier_nearby_checks=80))
        self.assertTrue(result['complete']);self.assertTrue(result['raw_pass'])
        self.assertEqual(state['pick'],original_picks)
        self.assertTrue(all(torch.equal(v,original_weights[k]) for k,v in weights.items()))
        # Actual C2 watchdog status is part of qualification, even with valid output.
        anchor_diag[0]=1
        with patch.object(torch,'cuda',types.SimpleNamespace(synchronize=lambda:None)):
            with self.assertRaisesRegex(AssertionError,'device diagnostics'):ns['_sx_bench'](weights,state,ns['time'].monotonic())
        anchor_diag.zero_()
        # Unsupported actual tuner pick cannot silently qualify.
        state['pick'][('o',16)]=lambda x:x
        with patch.object(torch,'cuda',types.SimpleNamespace(synchronize=lambda:None)):
            with self.assertRaises(AssertionError):ns['_sx_bench'](weights,state,ns['time'].monotonic())

if __name__=='__main__':unittest.main()

