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
    def test_private_mode_is_explicit_for_success_failure_and_comparison(self):
        public=fixture();private=copy.deepcopy(public);private["mode"]="check"
        with self.assertRaises(AssertionError):analyze(private,"synthetic123",log(capsule()),SOURCE)
        out=analyze(private,"synthetic123",log(capsule()),SOURCE,public,"synthetic123",mode="check")
        self.assertEqual(out["terminal_fields"]["mode"],"check");self.assertFalse(out["paired"])
        with self.assertRaises(AssertionError):analyze(public,"synthetic123",log(capsule()),SOURCE,mode="check")
        with self.assertRaises(AssertionError):analyze(private,"synthetic123",log(capsule()),SOURCE,private,"synthetic123",mode="check")
        self.assertEqual(len(analyze(private,"synthetic123",log(capsule()),SOURCE,private,"synthetic123",mode="check",control_mode="check")["weighted_six"]),6)
        failed=read_receipt(ROOT/"muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.terminal.json");failed["mode"]="check"
        with self.assertRaises(AssertionError):observe(failed,failed["id"],"",SOURCE)
        self.assertEqual(observe(failed,failed["id"],"",SOURCE,mode="check")["qualification"]["status"],"setup_failed")
        for mode in ("unknown",None):
            with self.assertRaises(AssertionError):observe(failed,failed["id"],"",SOURCE,mode=mode)

    def test_valid_complete_and_comparison(self):
        out=analyze(fixture(),'synthetic123',log(capsule()),SOURCE,fixture(),control_job='synthetic123')
        self.assertTrue(out['qualification']['treatment_exercised']);self.assertFalse(out['paired']);self.assertIsNone(out['causal_RMW_effect'])
        self.assertEqual(len(out['cross_job_comparison']['rows']),12);self.assertEqual(len(out['cross_job_comparison']['weighted_loads']),6)
        self.assertEqual({r['load'] for r in out['cross_job_comparison']['rows'] if r['target']},{'decode 1','verify 1'})
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
    def test_budget_cache_stats(self):
        check=source_contract(SOURCE);check.requires_reference_cache=True
        for hit in (0,1,95,96):
            r=capsule();r['reference_cache']=dict(hit=hit,miss=144-hit,peak_entries=4)
            self.assertEqual(qualification(log(r),check)['status'],'complete_native')
        bad=[None,{},dict(hit=96,miss=48,peak_entries=3),dict(hit=97,miss=47,peak_entries=4),
             dict(hit=0,miss=145,peak_entries=4),dict(hit=96.,miss=48,peak_entries=4),
             dict(hit=True,miss=143,peak_entries=4),dict(hit=95,miss=48,peak_entries=4),
             dict(hit=96,miss=48,peak_entries=4,extra=0)]
        for value in bad:
            r=capsule();r['reference_cache']=value
            with self.assertRaises(AssertionError):qualification(log(r),check)
        r=capsule()
        with self.assertRaises(AssertionError):qualification(log(r),check)
        # Valid fallback remains unexercised, without fictional native cache stats.
        result=qualification('G_COMMON_QUAL_UNEXERCISED native_expert_Marlin_fallback rmw_exercised=0 backend\n',check)
        self.assertFalse(result['treatment_exercised'])

    def test_phase_coverage_and_new_source_registration(self):
        check=source_contract(SOURCE)
        check.requires_reference_cache=True;check.requires_prepare_phases=True
        valid=capsule();valid['reference_cache']=dict(hit=96,miss=48,peak_entries=4)
        valid['prepare_phases']=dict(PHASE_COUNTS,reference_back_miss=48)
        valid['prepare_phase_seconds']={phase:0.01 for phase in valid['prepare_phases']}
        valid['prepare_phase_elapsed_seconds']=sum(valid['prepare_phase_seconds'].values())
        self.assertEqual(qualification(log(valid),check)['status'],'complete_native')
        for key in valid['prepare_phases']:
            for replacement in (0,True,float(valid['prepare_phases'][key])):
                changed=copy.deepcopy(valid);changed['prepare_phases'][key]=replacement
                with self.assertRaisesRegex(AssertionError,'prepare phase'):qualification(log(changed),check)
        for replacement in (None,{},dict(valid['prepare_phases'],unexpected=1)):
            changed=copy.deepcopy(valid);changed['prepare_phases']=replacement
            with self.assertRaisesRegex(AssertionError,'prepare phase'):qualification(log(changed),check)
        self.assertNotIn('82234b757e64242fa23d704a909c6996c602e3f0dae327fb66a67260243544d8',ALLOWED_SOURCES)
        self.assertEqual(len(ALLOWED_SOURCES),3)
        for change in (
            lambda record:record.__setitem__('prepare_phase_seconds',{}),
            lambda record:record['prepare_phase_seconds'].__setitem__('cleanup_begin',-1),
            lambda record:record['prepare_phase_seconds'].__setitem__('cleanup_begin',True),
            lambda record:record['prepare_phase_seconds'].__setitem__('cleanup_begin',float('nan')),
            lambda record:record.__setitem__('prepare_phase_elapsed_seconds',295),
            lambda record:record.__setitem__('prepare_phase_elapsed_seconds',1.0),
            lambda record:record.__setitem__('prepare_phase_elapsed_seconds',True),
        ):
            changed=copy.deepcopy(valid);change(changed)
            with self.assertRaises((AssertionError,ValueError)):qualification(log(changed),check)

    def test_real_setup_failure_and_nonzero_cli_envelope(self):
        receipt_path=ROOT/'muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.terminal.json'
        raw=read_receipt(receipt_path)
        envelope=ROOT/'docs/steering/evidence/2026-11-01-rmw-wait-2258/persistent-wait-result.json'
        self.assertEqual(read_receipt(envelope),raw)
        archived=(ROOT/'muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.log').read_text()
        result=observe(raw,raw['id'],archived,SOURCE)
        self.assertEqual(result['qualification']['status'],'setup_failed')
        self.assertIsNone(result['official']['official_score']);self.assertFalse(result['release_decision'])
        self.assertIsNone(result['causal_RMW_effect'])
        for change in (
            lambda receipt:receipt.__setitem__('id','wrong'),
            lambda receipt:receipt.__setitem__('status','running'),
            lambda receipt:receipt.__setitem__('payload_ref','wrong'),
            lambda receipt:receipt['result'].__setitem__('speedup',1.1),
            lambda receipt:receipt['result'].__setitem__('returncode',0),
            lambda receipt:receipt['result']['records'].append(dict(event='benchmark',status='pass')),
        ):
            bad=copy.deepcopy(raw);change(bad)
            with self.assertRaises(AssertionError):observe(bad,raw['id'],archived,SOURCE)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for payload in ({'returncode':1},{'stdout':'not json'},{'stdout':'{}'}):
                path.write_text(json.dumps(payload))
                with self.assertRaises((AssertionError,ValueError)):read_receipt(path)

    def test_owning_log_and_source(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'synthetic123.log';p.write_bytes(log(capsule()).encode());r=fixture();r['output']['log']=str(p)
            self.assertEqual(owning_log(r,p)[0],log(capsule()))
            q=Path(d)/'other.log';q.write_bytes(p.read_bytes())
            with self.assertRaises(AssertionError):owning_log(r,q)
            with self.assertRaises(AssertionError):source_contract(q)

if __name__=='__main__':unittest.main()
