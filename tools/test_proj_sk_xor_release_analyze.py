import json,math,unittest
from pathlib import Path
class Terminal(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-sk-xor-release-terminal';cls.s=json.loads((p/'summary.json').read_text());cls.r=json.loads((p/'rows30.json').read_text());cls.l=json.loads((p/'loads6.json').read_text());cls.resources=json.loads((p/'resources.json').read_text())
 def test_reconstruct_six_score(self):
  self.assertAlmostEqual(math.prod(r['candidate_baseline_sum_us']/r['candidate_sum_us'] for r in self.l)**(1/6),self.s['official_score'])
  self.assertAlmostEqual(math.prod(r['archived_baseline_sum_us']/r['archived_C2_sum_us'] for r in self.l)**(1/6),self.s['archived_C2_official_score'])
  self.assertAlmostEqual(math.prod(r['counterfactual_factor_vs_archived_load'] for r in self.l)**(1/6),self.s['counterfactual_replace10_factor_vs_old'])
 def test_full_resources_and_rows(self):
  self.assertEqual(len(self.r),30);self.assertEqual(sum(r['target'] for r in self.r),10);self.assertEqual(sum(r['delta_us']>0 for r in self.r if r['target']),10)
  self.assertEqual(len(self.resources),10)
  for r in self.resources.values():
   self.assertEqual(len(r['vectors']),4)
   for v in r['vectors']:self.assertEqual(v[1],0);self.assertGreater(v[6],0)
 def test_coverage(self):self.assertEqual(list(self.s['coverage'].values()),[10,0,110,160,10,30])
if __name__=='__main__':unittest.main()
