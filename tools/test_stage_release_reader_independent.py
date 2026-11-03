"""Independent negatives for the stage-local terminal reader; no official calls.
Set SK_STAGE_ANALYZER_TOOLS to the owner's tools directory before integration.
"""
import ast,copy,importlib,os,sys,unittest
from pathlib import Path
class ReaderContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=Path(os.environ.get('SK_STAGE_ANALYZER_TOOLS',Path(__file__).resolve().parent))
  if not (p/'proj_stage_xor_release_analyze.py').exists():raise unittest.SkipTest('Stage reader not integrated; set SK_STAGE_ANALYZER_TOOLS')
  sys.path.insert(0,str(p))
  try:
   cls.reader=importlib.import_module('proj_stage_xor_release_analyze');cls.fixtures=importlib.import_module('proj_test_stage_xor_release_analyze')
  finally:sys.path.pop(0)
 def changed_vectors(self,mutate):
  r,c,log,s=self.fixtures.fixture();lines=log.splitlines();h,v=lines[0].split('resources=');v=ast.literal_eval(v);mutate(v);lines[0]=h+'resources='+repr(v)
  return '\n'.join(lines),s
 def test_odd_and_different_control_table_rejected(self):
  for mutate in (lambda v:(v[0].__setitem__(3,v[0][3]+8),v[2].__setitem__(3,v[2][3]+8)),lambda v:v[2].__setitem__(3,v[2][3]+16)):
   log,s=self.changed_vectors(mutate)
   with self.assertRaises(AssertionError):self.reader.parse_qualification(log,s)
 def test_duplicate_profile_is_not_silent(self):
  r,c,log,s=self.fixtures.fixture();r['result']['records'].append(copy.deepcopy(next(x for x in r['result']['records'] if x['event']=='profile')))
  with self.assertRaises(AssertionError):self.reader.analyze_records(r,c,log,s,'synthetic-stage')
 def test_c2_direct_score_and_sensitivity(self):
  r,c,log,s=self.fixtures.fixture();summary,rows,loads,_=self.reader.analyze_records(r,c,log,s,'synthetic-stage')
  self.assertEqual(summary['control_job'],'9eba96894ee1402e');self.assertEqual(summary['reconstructed_score'],r['result']['speedup']);self.assertEqual(summary['reconstructed_C2_score'],c['result']['speedup'])
  self.assertIn('not a paired result',summary['comparison_scope'])
if __name__=='__main__':unittest.main()
