import unittest
from moe_lowtoken_representation_audit import *
class Audit(unittest.TestCase):
 def test_exact_expanded_math(self):self.assertEqual(prove_expansion(),1904)
 def test_reload_oracle(self):
  for counts in ([0]*32,[1]+[0]*31,[9]+[0]*31,[256]+[0]*31,[8]*32):
   live=[min(8,c-o) for c in counts for o in range(0,c,8)]
   for grid in (1,7,132,136):
    got=reloads(counts,grid)
    for phase,u in enumerate(U):
     seen=set()
     for cta in range(grid):
      for v in range(len(live)*u*cta//grid,len(live)*u*(cta+1)//grid):seen.add((cta,v//u))
     self.assertEqual(got[phase]['job_cta_incidences'],len(seen))
     self.assertEqual(got[phase]['requested_activation_bytes'],sum(live[j]*8192 for _,j in seen))
 def test_same_ring_and_score_bounds(self):
  self.assertEqual(SHARED-SLOTS*UNIT+SLOTS*16*H*2,438144)
  self.assertGreater(438144,232448)
  self.assertAlmostEqual(1-1.01**(-6/2),.02940985207235558)
  with self.assertRaises(AssertionError):reloads([0]*31)
if __name__=='__main__':unittest.main()
