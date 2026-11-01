"""Mapping and credit conservation only; not a memory-ordering/performance proof."""
import random
import unittest
from moe_weight_reuse_model import summary


class WeightReuseModel(unittest.TestCase):
    def test_maps_and_credits(self):
        rng=random.Random(2081)
        for m in (1,8,16,32):
            cases=[[0]*32,[m]+[0]*31,[m]*8+[0]*24]
            for _ in range(5):
                c=[0]*32
                for _ in range(m):
                    for e in rng.sample(range(256),8):
                        if e<32: c[e]+=1
                cases.append(c)
            for counts in cases:
                for clusters in (1,2,66):
                    s=summary(counts,clusters)
                    self.assertLessEqual(s['paired_saved_request_fraction'],.5)
                    if m<=8:
                        self.assertEqual(s['logical_duplicate_fraction'],0)
                        self.assertEqual(s['paired_saved_request_fraction'],0)

    def test_hot_synthetic_bounds_not_forecasts(self):
        s16=summary([16]*8+[0]*24)
        s32=summary([32]*8+[0]*24)
        self.assertEqual(s16['logical_duplicate_fraction'],.5)
        self.assertEqual(s32['logical_duplicate_fraction'],.75)
        self.assertEqual(s16['paired_saved_request_fraction'],.5)
        self.assertEqual(s32['paired_saved_request_fraction'],.5)
        self.assertEqual(s32['original_neighbor_same_step_pairs'],0)


if __name__=='__main__': unittest.main()
