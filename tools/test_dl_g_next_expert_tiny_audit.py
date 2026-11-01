import random
import unittest
from dl_g_next_expert_tiny_audit import model,UNIT_BYTES
class TinyTests(unittest.TestCase):
 def test_unique_ownership(self):
  for seed in range(200):
   rng=random.Random(seed);counts=[0]*32
   for _ in range(8):
    for expert in rng.sample(range(256),8):
     if expert<32:counts[expert]+=1
   for grid in (1,32,132,136):
    stats=model(counts,grid)
    self.assertEqual(stats["duplicate_tiles"],0)
    self.assertEqual(stats["logical_weight_bytes"],sum(c>0 for c in counts)*768*UNIT_BYTES)
 def test_reuse_and_empty(self):
  self.assertEqual(model([0]*32)["logical_weight_bytes"],0)
  for c in range(1,9):self.assertEqual(model([c]+[0]*31)["logical_weight_bytes"],28311552)
if __name__=="__main__":unittest.main()
