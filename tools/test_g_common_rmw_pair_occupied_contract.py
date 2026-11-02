"""CPU-only diagnosis of the frozen occupied fixture; no artifact or kernel edits."""
import ast,collections,contextlib,hashlib,io,json,pathlib,sys,time,types,unittest
from unittest.mock import patch
import torch
from g_common_rmw_pair_compile import capture

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.build/rmw-pair-v3/check.py'
TEXT=SOURCE.read_text()
PIN='c4a5ee29bcfeca248c0fee638ed103e9f211f7a8cc32b86bd3bb642676bf45e3'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
NODE=next(n for n in ast.parse(TEXT).body if isinstance(n,ast.FunctionDef) and n.name=='_srp_occupied_empty')
FROZEN=ast.get_source_segment(TEXT,NODE)
INIT="ids=torch.zeros((t,8),device=dev,dtype=torch.int32)"
LEGAL_INIT="occupied_ids=torch.arange(8,device=dev,dtype=torch.int32).expand(t,8).contiguous();ids=occupied_ids.clone()"
TRANSITION="ids.fill_(0 if occupied else -1)"
LEGAL_TRANSITION="ids.copy_(occupied_ids) if occupied else ids.fill_(-1)"
assert FROZEN.count(INIT)==FROZEN.count(TRANSITION)==1
# This in-memory proposal is never written to check.py or production support.
PROPOSED=FROZEN.replace(INIT,LEGAL_INIT).replace(TRANSITION,LEGAL_TRANSITION)
CALLS=capture(TEXT)
SELECTED={(1,False):CALLS[0],(8,False):CALLS[1],(1,True):CALLS[5],(8,True):CALLS[6]}
FILL=torch.Tensor.fill_
OBS=[]

class Graph:
    active=None
    def __init__(self):self.ops=[]
    def replay(self):
        for op in self.ops:op()

@contextlib.contextmanager
def recording(g):
    assert Graph.active is None
    Graph.active=g
    try:yield
    finally:Graph.active=None

def recorded_fill(t,v):
    if Graph.active is not None:Graph.active.ops.append(lambda:FILL(t,v))
    return FILL(t,v)

class ContractViolation(Exception):pass
class FatPredicate(Exception):pass

class Native:
    def __init__(self,t,arm,mode,seen):self.t=t;self.arm=arm;self.mode=mode;self.seen=seen
    def okc_run(self,ma_w,scratch,x,ids,weights,out,lo,nloc,nsm):
        def op():
            rows=ids.tolist();counts=collections.Counter(v-lo for row in rows for v in row if 0<=v-lo<nloc)
            unique=all(len([v for v in row if v>=0])==len(set(v for v in row if v>=0)) for row in rows)
            entry=dict(t=self.t,arm=self.arm,unique=unique,max_expert_entries=max(counts.values(),default=0),occupied=bool(counts))
            self.seen.append(entry)
            if self.mode=='contract' and not unique:raise ContractViolation((self.t,self.arm,'duplicate top-k IDs'))
            if self.mode=='native_predicate' and entry['max_expert_entries']>8:raise FatPredicate((self.t,self.arm,'njF>0 under tiny/split0'))
            FILL(out,1 if counts else 0)
        if Graph.active is not None:Graph.active.ops.append(op)
        op()

def exercise(text,mode):
    seen=[]
    cuda=types.SimpleNamespace(CUDAGraph=Graph,graph=recording,synchronize=lambda:None)
    proxy=types.SimpleNamespace(cuda=cuda,randn=torch.randn,zeros=torch.zeros,full=torch.full,
        full_like=torch.full_like,arange=torch.arange,bfloat16=torch.bfloat16,int32=torch.int32,
        float32=torch.float32,isfinite=torch.isfinite,count_nonzero=torch.count_nonzero)
    arms=[dict(qkv=torch.empty(1),ma_w=None,ma_scratch=None,nsm=132,ordinal=i) for i in range(4)]
    ns=dict(torch=proxy,_srp_budget=lambda:None,_fork_native_moe=lambda t,a:Native(t,a['ordinal'],mode,seen))
    exec(text,ns)
    with patch.object(torch.Tensor,'fill_',recorded_fill):checks=ns['_srp_occupied_empty'](arms)
    return checks,seen

class FixtureDiagnosis(unittest.TestCase):
    def test_exact_selected_native_predicate(self):
        for (t,on),c in SELECTED.items():
            flags=c['extra_cuda_cflags'];src=c['cuda_sources']
            self.assertIn('-DOKC_TINY=1',flags)
            self.assertNotIn('-DOKC_WGS=1',flags)
            self.assertFalse(any(f.startswith('-DOKC_SPLIT=') for f in flags))
            self.assertEqual('-DOKC_SHORT_FINAL_RELAXED=1' in flags,on)
            for literal in ('#define OKC_SPLIT 0','#define OKC_SKT 8','#define OKC_FT_T 64','#define OKC_WGS 0',
                'const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;',
                'if (nF > 0) __trap();','run_phase<false, BIG>(p, S, xbuf);'):
                self.assertIn(literal,src)
    def test_frozen_fixture_rejected_by_distinct_contract(self):
        with self.assertRaisesRegex(ContractViolation,r"1, 0, 'duplicate top-k IDs'"):
            exercise(FROZEN,'contract')
    def test_frozen_native_count_model_reaches_t8_off0_fat_predicate(self):
        with self.assertRaisesRegex(FatPredicate,r"8, 0, 'njF>0"):
            exercise(FROZEN,'native_predicate')
    def test_old_permissive_mock_misses_defect(self):
        checks,seen=exercise(FROZEN,'permissive')
        self.assertEqual(checks,16);self.assertEqual(max(v['max_expert_entries'] for v in seen),64)
        self.assertFalse(all(v['unique'] for v in seen))
    def test_proposed_legal_fixture_all_arms_transitions(self):
        checks,seen=exercise(PROPOSED,'contract')
        self.assertEqual(checks,16);self.assertEqual(len(seen),40)
        self.assertTrue(all(v['unique'] and v['max_expert_entries']<=v['t'] for v in seen))
        for t in (1,8):
            for arm in range(4):
                seq=[v for v in seen if v['t']==t and v['arm']==arm]
                self.assertEqual([v['occupied'] for v in seq],[True,True,False,True,False])
                self.assertEqual([v['max_expert_entries'] for v in seq],[t,t,0,t,0])
        # Replacing initialization alone is insufficient: fill_(0) reintroduces duplicates.
        with self.assertRaises(ContractViolation):exercise(FROZEN.replace(INIT,LEGAL_INIT),'contract')
        OBS.extend(seen)

if __name__=='__main__':
    start=time.monotonic();stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureDiagnosis))
    print(stream.getvalue(),end='')
    evidence=dict(source_sha256=PIN,GPU=False,scope='Exact frozen fixture under CPU route-contract/native-predicate facades; proposed repair in memory only, no device-cause localization.',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,observations=OBS,
        source_locations=dict(fixture_line=NODE.lineno,old_cpu_mock='tools/test_g_common_rmw_pair.py:native_run'),
        proposed_replacements=[dict(old=INIT,new=LEGAL_INIT),dict(old=TRANSITION,new=LEGAL_TRANSITION)],
        selected=[dict(t=t,on=on,name=c['name'],cuda_sha256=hashlib.sha256(c['cuda_sources'].encode()).hexdigest(),flags=c['extra_cuda_cflags']) for (t,on),c in SELECTED.items()])
    if len(sys.argv)>1:pathlib.Path(sys.argv[1]).write_text(json.dumps(evidence,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
