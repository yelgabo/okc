"""CPU proof of grouped slot bijection, capacity, minimal tail separation."""
import math
import random
import unittest


class MaskedContract(unittest.TestCase):
    def test_routes_tails_bounds(self):
        rng=random.Random(3317)
        for m in (1,8,32,753,3012,16384):
            for b1,b2 in ((8,24),(64,128),(104,176),(128,160)):
                align=math.lcm(b1,b2)
                cap=(m+align-1)//align*align
                for mode in ('zero','one','local','mixed'):
                    counts=[0]*32
                    slots={}
                    for token in range(m):
                        ids=([32+i for i in range(8)] if mode=='zero' else
                             [0]+list(range(32,39)) if mode=='one' else
                             list(range(8)) if mode=='local' else rng.sample(range(256),8))
                        for k,e in enumerate(ids):
                            if e>=32: continue
                            rank=counts[e]; counts[e]+=1
                            slot=e*cap+rank
                            self.assertNotIn(slot,slots)
                            slots[slot]=8*token+k
                    self.assertTrue(all(c<=m for c in counts))
                    self.assertEqual(len(set(slots.values())),len(slots))
                    for e,c in enumerate(counts):
                        for bm in (b1,b2):
                            end=(c+bm-1)//bm*bm
                            self.assertLessEqual(end,cap)
                            self.assertLess(end-c,bm)
                            for row in range(c,end):
                                self.assertNotIn(e*cap+row,slots)
                            # No scheduled block or tail writes for zero count.
                            if c==0: self.assertEqual(end,0)
                    self.assertTrue(all(0<=slot<32*cap for slot in slots))


if __name__=='__main__': unittest.main()
