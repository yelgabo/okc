import json,unittest,math
from pathlib import Path
from dl_sk_xor_applicability import factor,G_PIN
class Applicability(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=json.loads((Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-g-sk-xor-applicability/map.json').read_text())
 def test_actual_coverage_and_pins(self):
  d=self.data;self.assertEqual(d['G_source_sha256'],G_PIN);self.assertEqual(len(d['cells']),12);self.assertEqual(len(d['dense_policy']),12)
  for r in d['cells']:
   self.assertEqual(r['weight'],46 if r['layer']=='sliding' else 16)
   self.assertEqual(set(r['profile_components_us']),{'qkv','o','un','pool_router_fused'} if r['T']<=32 else set())
  for p in d['dense_policy']:
   self.assertEqual(p['N'],1024 if p['projection']=='un' and p['T']<=8 else 2304 if p['projection']=='qkv' else 8192)
 def test_score_neutrality_and_weighting(self):
  rows=self.data['cells'];self.assertEqual(factor(rows,lambda r:0),1.)
  delta=1.23;T=1;old=sum(r['weight']*r['median_us'] for r in rows if r['T']==T)
  expected=(old/(old-46*delta))**(1/6)
  self.assertAlmostEqual(factor(rows,lambda r:delta if r['T']==T and r['layer']=='sliding' else 0),expected)
  req=self.data['one_percent_six_load_requirements']['same_whole_cell_us_saved_at_T1_8_16_32']
  self.assertAlmostEqual(factor(rows,lambda r:req if r['T']<=32 else 0),1.01)
 def test_byte_scope(self):
  b=self.data['byte_plan'];self.assertTrue(b['not_a_built_layer_artifact']);self.assertLess(b['conservative_source_plan_sum'],1_000_000);self.assertEqual(b['unchanged_host_function_source_total'],sum(b['host_function_source_bytes'].values()))
if __name__=='__main__':unittest.main()
