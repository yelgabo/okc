"""Independent recomputation of archived terminal data; no GPU/official calls."""
import json,math,unittest
from pathlib import Path

D=Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-projection-stage-xor-terminal'

class Terminal(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=json.loads((D/'summary.json').read_text(encoding='utf8'));cls.rows=json.loads((D/'rows30.json').read_text(encoding='utf8'));cls.loads=json.loads((D/'loads6.json').read_text(encoding='utf8'));cls.resources=json.loads((D/'resources.json').read_text(encoding='utf8'))
 def test_official_sixload_and_raw_reconstruction(self):
  self.assertEqual((len(self.rows),len(self.loads)),(30,6))
  self.assertAlmostEqual(math.prod(r['candidate_baseline_sum_us']/r['candidate_sum_us'] for r in self.loads)**(1/6),self.s['official_score'],places=13)
  self.assertAlmostEqual(math.prod(r['archived_baseline_sum_us']/r['archived_C2_sum_us'] for r in self.loads)**(1/6),self.s['archived_C2_official_score'],places=13)
  self.assertAlmostEqual(math.prod(r['counterfactual_factor_vs_archived_load'] for r in self.loads)**(1/6),self.s['counterfactual_replace10_factor_vs_old'],places=13)
  for prefix,field in [('new','raw_sixload_ratio_using_raw_baselines'),('old','archived_raw_sixload_ratio_using_raw_baselines')]:
   ratio=math.prod(sum(r[prefix+'_baseline_raw_us'] for r in self.rows if r['m']==m)/sum(r[prefix+'_raw_us'] for r in self.rows if r['m']==m) for m in (1,8,16,32,128,256))**(1/6)
   self.assertAlmostEqual(ratio,self.s[field],places=13)
 def test_all_targets_lose_and_qualification_complete(self):
  target=[r for r in self.rows if r['target']]
  self.assertEqual(len(target),10)
  self.assertTrue(all(r['stage_override'] and r['delta_us']>0 and r['new_raw_us']>r['old_raw_us'] for r in target))
  self.assertEqual(self.s['coverage'],dict(overrides=10,fallbacks=0,pick_keys=110,math_checks=160,captures=10,fresh=30))
  self.assertLess(self.s['official_score'],self.s['archived_C2_official_score'])
 def test_forty_resources_capacities_and_input_pin(self):
  self.assertEqual(len(self.resources),10)
  for r in self.resources.values():
   self.assertEqual(len(r['vectors']),4)
   self.assertGreaterEqual(r['inferred_C2_pmax'],2);self.assertEqual(r['inferred_C2_pmax']%2,0)
   self.assertEqual(r['stage_record_bytes'],(2*r['stage_capacity']+15)&~15)
   for i,v in enumerate(r['vectors']):
    self.assertEqual(v[1:3],[0,0]);self.assertGreaterEqual(v[6],1)
    if i%2:self.assertGreaterEqual(v[6],r['vectors'][i-1][6])
  inputs=json.loads((D/'inputs.json').read_text(encoding='utf8'))
  self.assertEqual(inputs['stage_source']['bytes'],450781)
  self.assertEqual(inputs['stage_source']['sha256'],self.s['source_sha256'])
  self.assertEqual(inputs['stage_receipt']['sha256'],self.s['receipt_sha256'])
  self.assertEqual(self.s['job'],'dfcfdd01d6f04525')

if __name__=='__main__':unittest.main()
