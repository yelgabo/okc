"""Synthetic score/qualification negatives; no official calls or GPU work."""
import ast,copy,json,unittest
from pathlib import Path
import proj_sparse_xor,proj_stage_xor
from proj_stage_xor_release_analyze import *

ROOT=Path(__file__).resolve().parents[1]

def fixture(fallback=None):
 meta=json.loads((ROOT/'docs/steering/evidence/2026-11-01-projection-sparse-xor/prototype-manifest.json').read_text(encoding='utf8'))['metadata']
 source='_XOR_CONFIG='+repr(meta)+'\n'
 for text,name in [(proj_sparse_xor.RUNTIME,'xor_expected_layout'),(proj_stage_xor.RUNTIME,'sxs_expected_layout')]:
  source+=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)+'\n'
 ns=layout_functions(source);lines=[];n=0;keys=0
 for (layer,m),name in TARGETS.items():
  if (layer,m)==fallback:
   lines.append(f'SXSRELEASE_ROUTE {layer} {m} unsupported_pack_original_C2 checks=0 capacity=0 arena_bytes=0 resources=[]');continue
  n+=1;keys+=m-(2 if m==8 else 9 if m==16 else 17)+1
  c=ns['sxs_expected_layout'](name,0,2);s=ns['sxs_expected_layout'](name,2,2)
  cv=[80,0,c['static'],c['dynamic'],c['threads'],c['threads'],2]
  sv=[60,0,s['static'],s['dynamic'],s['threads'],s['threads'],2]
  lines.append(f'SXSRELEASE_ROUTE {layer} {m} override checks=16 capacity=2 arena_bytes=16 resources='+repr([cv,sv,cv,sv]))
 lines.append(f'SXSRELEASE_DONE overrides={n} fallbacks={10-n} pick_keys={keys} checks={n*16} captures={n} fresh={n*3}')
 def receipt(job,treatment):
  records=[dict(event='setup',prepare_s=1)]+[dict(event='test',status='pass') for _ in range(9)]
  for m in MS:
   for l in LAYERS:
    k=l,m;value=9 if treatment and k in TARGETS and k!=fallback else 10
    case=dict(layer=l,m=m,weight=62,seed=123 if treatment else 456)
    records.append(dict(event='benchmark',case=case,status='pass',median_ns=value,median_raw_ns=value+2,baseline_ns=20,baseline_raw_ns=22))
    records.append(dict(event='profile',case=case,report='synthetic'))
  result=dict(records=records,passed=True,checks={'ok':True},flags=[],speedup_ci=[1,3])
  result['speedup']=score(benchmarks(result))
  return dict(id=job,board='muse_bf16_proj',mode='run',status='done',result=result)
 return receipt('synthetic-stage',True),receipt(CONTROL,False),'\n'.join(lines),source

class Analysis(unittest.TestCase):
 def test_all30_sixload_reconstruction_and_unpaired_scope(self):
  r,c,log,source=fixture();summary,rows,loads,resources=analyze_records(r,c,log,source,'synthetic-stage')
  self.assertEqual((len(rows),len(loads),len(resources)),(30,6,10))
  self.assertEqual(summary['coverage'],dict(overrides=10,fallbacks=0,pick_keys=110,math_checks=160,captures=10,fresh=30))
  expected=math.prod(50/(50-sum((l,m) in TARGETS for l in LAYERS)) for m in MS)**(1/6)
  self.assertAlmostEqual(summary['reported_score_ratio'],expected)
  self.assertAlmostEqual(summary['counterfactual_replace10_factor_vs_old'],expected)
  self.assertIn('not a paired result',summary['comparison_scope'])
  self.assertEqual(summary['target_regressions'],0)
  self.assertAlmostEqual(summary['raw_sixload_ratio_using_raw_baselines'],score(benchmarks(r['result']),'median_raw_ns','baseline_raw_ns'))
  self.assertNotAlmostEqual(summary['raw_sixload_ratio_using_raw_baselines'],score(benchmarks(r['result']),'median_raw_ns'))
  self.assertTrue(all(v['inferred_C2_pmax']==2 for v in resources.values()))
 def test_explicit_fallback_and_exact_totals(self):
  r,c,log,source=fixture(('qkv_gate',8));s,rows,_,resources=analyze_records(r,c,log,source,'synthetic-stage')
  self.assertEqual(s['coverage'],dict(overrides=9,fallbacks=1,pick_keys=103,math_checks=144,captures=9,fresh=27))
  self.assertFalse(next(x for x in rows if (x['layer'],x['m'])==('qkv_gate',8))['stage_override'])
  self.assertEqual(resources['qkv_gate:8']['vectors'],[])
  with self.assertRaisesRegex(AssertionError,'QUALIFICATION_TOTALS'):parse_qualification(log.replace('checks=144 captures=9','checks=160 captures=9'),source)
 def test_owning_terminal_and_score_negatives(self):
  for mutation,label in [('job','OWNING_RECEIPT'),('running','TERMINAL_STATUS'),('score','SCORE_RECONSTRUCTION'),('nan','FINITE_POSITIVE'),('duplicate','THIRTY_UNIQUE_CELLS')]:
   r,c,log,source=fixture()
   if mutation=='job':r['id']='foreign'
   elif mutation=='running':r['status']='running'
   elif mutation=='score':r['result']['speedup']*=1.01
   else:
    records=r['result']['records'];ids=[i for i,x in enumerate(records) if x['event']=='benchmark']
    if mutation=='nan':records[ids[0]]['median_raw_ns']=float('nan')
    else:records[ids[0]]=copy.deepcopy(records[ids[1]])
   with self.subTest(mutation=mutation),self.assertRaisesRegex(AssertionError,label):analyze_records(r,c,log,source,'synthetic-stage')
 def test_capacity_layout_admission_and_missing_rows_fail(self):
  _,_,log,source=fixture()
  for bad,label in [(log.replace('arena_bytes=16','arena_bytes=32',1),'CAPACITY_ARENA'),('\n'.join(log.splitlines()[1:]),'TEN_ROUTE_COVERAGE'),(log+'\n'+log.splitlines()[-1],'DONE_COVERAGE')]:
   with self.subTest(label=label),self.assertRaisesRegex(AssertionError,label):parse_qualification(bad,source)
  lines=log.splitlines();prefix,tail=lines[0].split('resources=');v=ast.literal_eval(tail);v[1][6]=1;lines[0]=prefix+'resources='+repr(v)
  with self.assertRaisesRegex(AssertionError,'C2_ADMISSION_REGRESSION'):parse_qualification('\n'.join(lines),source)
 def test_C2_pmax_zero_odd_and_full_nf_mismatch(self):
  _,_,log,source=fixture()
  for delta,label in [(-16,'CONTROL_PMAX_LAYOUT'),(8,'CONTROL_PMAX_LAYOUT'),(16,'CONTROL_FULL_NF_PMAX_MISMATCH')]:
   lines=log.splitlines();prefix,tail=lines[0].split('resources=');v=ast.literal_eval(tail);v[2][3]+=delta;lines[0]=prefix+'resources='+repr(v)
   with self.subTest(delta=delta),self.assertRaisesRegex(AssertionError,label):parse_qualification('\n'.join(lines),source)
 def test_duplicate_profiles_and_failed_benchmark(self):
  r,c,log,source=fixture();records=r['result']['records'];p=next(x for x in records if x['event']=='profile');records.append(copy.deepcopy(p))
  with self.assertRaisesRegex(AssertionError,'THIRTY_UNIQUE_PROFILES'):analyze_records(r,c,log,source,'synthetic-stage')
  r,c,log,source=fixture();next(x for x in r['result']['records'] if x['event']=='benchmark')['status']='fail'
  with self.assertRaisesRegex(AssertionError,'BENCHMARK_NOT_PASS'):analyze_records(r,c,log,source,'synthetic-stage')

if __name__=='__main__':unittest.main()
