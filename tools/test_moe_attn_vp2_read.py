"""Synthetic receipt transport/decision tests; no claimed H200 measurements."""
import copy,json,unittest
from moe_attn_vp2_read import analyze,score_function,OWN,PUBLIC

def fixture():
 manifest={'seeds':[824731,824737,824743],'sch':0,'measurement_only':True,'owned_lf':OWN,'public_raw':PUBLIC}
 rows=[(b,l,w,s,[100.,100.,88. if (b,l,w)==(1,8,1) else 100.,88. if (b,l,w)==(1,8,1) else 100.]) for b in (1,16,32) for l in (1,8) for w in (0,1) for s in manifest['seeds']]
 fixed=[[b,l,w,4.] for b in (1,16,32) for l in (1,8) for w in (0,1)]
 plans=[[b,l,w,i,['VP2_grid_cluster_threads',32,16,256] if (b,l,w)==(1,8,1) and i>=2 else ['ONE_splits_copies_mode',33,1,15]] for b in (1,16,32) for l in (1,8) for w in (0,1) for i in range(4)]
 score=score_function(manifest)
 report=dict(schema='PRIVATE_VP2_BF16_V1',row_kind='raw_us',source=manifest,rows=rows,fixed_offsets=fixed,
  counts={'eager_extra':16,'normal':144,'fresh':336,'post':144},dispatch_counts={'ONE':284,'VP2':20},plans=plans,
  resources=[[8,15,111,0,90704,256,2]]*2+[[97,0,0,90000,256,2,14]]*2,
  raw_gate=score(rows),gate=score([(b,l,w,s,[t-4 for t in ts]) for b,l,w,s,ts in rows]))
 result=dict(passed=True,flags=[],speedup=2.,speedup_ci=[1.9,2.1],
  log_tail='VP2_PRIVATE_REPORT '+json.dumps(report)+'\nVP2_PRIVATE_DONE 36 private_only\n',records=[])
 for b,l,w,_ in fixed:
  result['records'].append(dict(event='benchmark',case=dict(batch=b,load='verify' if l==8 else 'decode',layer='sliding' if w else 'full'),median_raw_ns=100000.,median_ns=95000.))
 result['records'] += [dict(event='test',status='pass') for _ in range(16)]
 return dict(board='muse_attention_decode',status='done',id='CPU-SYNTHETIC-NOT-A-RUN',result=result)

class Reader(unittest.TestCase):
 def test_normalized_profile_is_explicitly_insufficient(self):
  with self.assertRaisesRegex(ValueError,'normalized profile JSON is insufficient'):
   analyze({'problem':'muse_attention_decode','job_id':'NORMALIZED-FIXTURE','benchmarks':[],'summary':{}})
 def test_contemporaneous_decision_recomputed(self):
  r=analyze(fixture());self.assertTrue(r['registered_decision']);self.assertTrue(r['registered_raw_sign_gate'])
  self.assertTrue(all(v[-1]==5. for v in r['contemporary_offsets']))
  self.assertNotEqual(r['contemporary_gate']['cross'],r['stale_gate']['cross'])
 def test_flags_coverage_and_nonpositive_rejected(self):
  r=fixture();r['result']['flags']=['bad']
  with self.assertRaises(AssertionError):analyze(r)
  r=fixture();text=r['result']['log_tail'];report=json.loads(text.splitlines()[0].partition(' ')[2]);report['dispatch_counts']['VP2']=0
  r['result']['log_tail']='VP2_PRIVATE_REPORT '+json.dumps(report)+'\nVP2_PRIVATE_DONE 36 private_only'
  with self.assertRaisesRegex(AssertionError,'coverage'):analyze(r)
  r=fixture()
  for v in r['result']['records']:
   if v['event']=='benchmark':v['median_ns']=-1000.
  with self.assertRaisesRegex(AssertionError,'BAD_TIMING'):analyze(r)

if __name__=='__main__':unittest.main()
