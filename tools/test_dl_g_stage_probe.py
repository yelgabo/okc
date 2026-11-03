import ast
from pathlib import Path
import random
import unittest
from dl_g_stage_probe import patch_device,DEVICE_SHA
from dl_g_stage_analyze import analyze_trace,expected_jobs,stage_order
class DeviceTests(unittest.TestCase):
 def test_source_pin_and_sync_census(self):
  root=Path(__file__).resolve().parents[2]/"okc"
  s=(root/"muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8")
  c=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC")
  new=patch_device(c)
  for word in ("wg_wait<0>()","wg_wait<1>()","wg_commit()","wg_fence()","mbar_arrive1(","CTA_SYNC();","WS_JOIN();","WGSMMA<NTT>::mma"):
   self.assertEqual(c.count(word),new.count(word),word)
  with self.assertRaises(ValueError):patch_device(c+" ")
  self.assertIn("gsp_mark<PB,0>",new);self.assertIn("gsp_mark<PB,5>",new)
 def test_partial_reordered_sequence_coverage(self):
  for pb in (0,1):
   for a in range(70):
    for b in range(a,101):
     order,sp=stage_order(pb,a,b)
     self.assertEqual(sorted(order),list(range(a,b)))
     self.assertEqual(len(order),len(set(order)))
     if pb:self.assertEqual(order,list(range(a,b)))
 def test_completeness_negative(self):
  # No jobs but fabricated nonzero selected stage must be rejected, never a partial pass.
  words=[0]*475;ids=[list(range(32,40)) for _ in range(128)]
  with self.assertRaises(ValueError):analyze_trace(words,ids,132,1)
if __name__=="__main__":unittest.main()
