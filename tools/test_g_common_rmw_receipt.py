"""Synthetic owning-capsule negatives; no GPU/service calls or real new results."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'tools'))
# Prefer this checkout's new reader over root's (possibly not yet integrated) file.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from g_common_rmw_receipt import *
SOURCE=ROOT/'muse_decode_layer/cand/g1b4_common_short_rmw_release.py'

def fixture():
    r=json.loads((ROOT/'muse_decode_layer/profiles/scored.607eb52bdaf24160.json').read_text(encoding='utf8'))
    r['id']='synthetic123';r['output']={'log':'/tmp/synthetic123.log'}
    return r

def capsule():
    r=dict(arm='short-rmw',base_sha=BASE_SHA,**COUNTS,native_expert=True,rmw_exercised=True)
    r['identity']=dict(capability=[9,0],sm_count=132,cuda_driver_version=12080,cuda_runtime_version=12080,gpu_name='SYNTHETIC_NOT_GPU',total_memory=None,device_uuid='unreported',node_environment=None,node_physical_identity='unconfirmed',torch_version='mock',torch_build_cuda=None,nvcc_version=None)
    r['native_resources']={k:[96,8,0,1000,288,1024,1,132,1,132]*2 for k in KEYS}
    rows=[]
    for t,c in sorted(CELLS):
        b,l={1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[t]
        flat=int(c=='full' and b>1);coop=int(l==8 or(t==1 and c=='full'));kind=int(coop and(not flat or l==8))
        split=2 if b==1 else 1;key='tiny'if t==1 else'tiny2'if t==8 else'mid'if t==16 else'b4'if t==32 else'main'
        a=[1,l,b,flat,coop,0,0,split,0,1,2048 if c=='sliding'else 0,split,1 if flat else b,1,128,1000,96,0,0,1024,232448,2,132,1,kind,100 if flat else 3,0]
        rows.append([t,c,a,[t,key,True]])
    r['actual_attention']=rows;return r

def log(r):return 'G_COMMON_QUAL_DONE '+json.dumps(r)+'\n'

class Reader(unittest.TestCase):
    def test_valid_complete_and_comparison(self):
        out=analyze(fixture(),'synthetic123',log(capsule()),SOURCE,fixture(),control_job='synthetic123')
        self.assertTrue(out['qualification']['treatment_exercised']);self.assertFalse(out['paired']);self.assertIsNone(out['causal_RMW_effect'])
        self.assertEqual(len(out['cross_job_comparison']['rows']),12);self.assertEqual(len(out['cross_job_comparison']['weighted_loads']),6)
    def test_terminal_guards_before_capsule(self):
        for key,val in [('id','wrong'),('board','muse_bf16_proj'),('mode','check'),('status','running')]:
            r=fixture();r[key]=val
            with self.assertRaises(AssertionError):analyze(r,'synthetic123',log(capsule()),SOURCE)
        for key,val in [('passed',False),('returncode',1),('error','final owner guard'),('flags',['flag'])]:
            r=fixture();r['result'][key]=val
            with self.assertRaises(AssertionError):analyze(r,'synthetic123',log(capsule()),SOURCE)
        r=fixture();r['result']['records'][0]['prepare_s']=295
        with self.assertRaises(AssertionError):analyze(r,'synthetic123',log(capsule()),SOURCE)
        r=fixture();r['result']['records'].pop()
        with self.assertRaises(AssertionError):analyze(r,'synthetic123',log(capsule()),SOURCE)
        r=fixture();r['result']['speedup']*=1.01
        with self.assertRaises(AssertionError):analyze(r,'synthetic123',log(capsule()),SOURCE)
    def test_counts_pins_and_capsule_transport(self):
        check=source_contract(SOURCE)
        for key in COUNTS:
            r=capsule();r[key]-=1
            with self.assertRaises(AssertionError):qualification(log(r),check)
        for key,val in [('arm','common'),('base_sha','a'*64),('native_expert',False),('rmw_exercised',False)]:
            r=capsule();r[key]=val
            with self.assertRaises(AssertionError):qualification(log(r),check)
        for text in ('',log(capsule())+log(capsule()),log(capsule())[:-20]):
            with self.assertRaises((AssertionError,ValueError)):qualification(text,check)
    def test_actual_dispatch_resources_identity_negatives(self):
        changes=[lambda r:r['actual_attention'].pop(),lambda r:r['actual_attention'].__setitem__(0,r['actual_attention'][1]),lambda r:r['actual_attention'][0][2].__setitem__(24,0),lambda r:r['actual_attention'][0][2].__setitem__(10,47),lambda r:r['actual_attention'][0][3].__setitem__(2,False),lambda r:r['native_resources']['tiny'].__setitem__(6,0),lambda r:r['native_resources'].pop('main'),lambda r:r['identity'].__setitem__('sm_count',1),lambda r:r['identity'].__setitem__('node_physical_identity','invented'),lambda r:r['identity'].__setitem__('cuda_runtime_version',float('nan'))]
        check=source_contract(SOURCE)
        for change in changes:
            r=capsule();change(r)
            with self.assertRaises((AssertionError,ValueError)):qualification(log(r),check)
    def test_fallback_is_preserved_not_attributed(self):
        text='G_COMMON_QUAL_UNEXERCISED native_expert_Marlin_fallback rmw_exercised=0 actual_backend\n'
        out=analyze(fixture(),'synthetic123',text,SOURCE)
        self.assertEqual(out['qualification']['status'],'unexercised');self.assertFalse(out['qualification']['treatment_exercised']);self.assertIsNone(out['causal_RMW_effect'])
        for bad in (text+log(capsule()),text.replace('rmw_exercised=0','rmw_exercised=1')):
            with self.assertRaises(AssertionError):analyze(fixture(),'synthetic123',bad,SOURCE)
    def test_owning_log_and_source(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'synthetic123.log';p.write_bytes(log(capsule()).encode());r=fixture();r['output']['log']=str(p)
            self.assertEqual(owning_log(r,p)[0],log(capsule()))
            q=Path(d)/'other.log';q.write_bytes(p.read_bytes())
            with self.assertRaises(AssertionError):owning_log(r,q)
            with self.assertRaises(AssertionError):source_contract(q)

if __name__=='__main__':unittest.main()
