"""CPU contract oracle for fixed indexed capacity; does not execute Triton/Humming."""
import random
import unittest


class RouteCapacity(unittest.TestCase):
    def test_capacity_bijection_and_sentinel(self):
        rng=random.Random(20261031)
        for m in (1,3,32,753):
            r=m*8
            for ids in ([255]*r,[0]*r,[31]*r,
                        [i%256 for i in range(r)],
                        [rng.randrange(256) for _ in range(r)]):
                order=list(range(r)); rng.shuffle(order)
                counts=[0]*32; ranks={}
                for i in order:
                    if 0<=ids[i]<32:
                        e=ids[i]; ranks[i]=counts[e]; counts[e]+=1
                for bm in (8,16,24,32,64,104,128,160,176):
                    cap=((r+bm-1)//bm+32)*bm
                    padded=[(c+bm-1)//bm*bm for c in counts]
                    off=[0]
                    for c in padded: off.append(off[-1]+c)
                    self.assertLessEqual(off[-1],cap)
                    sorted_ids=[r]*cap; expert_ids=[0]*(cap//bm)
                    for e in range(32):
                        for b in range(off[e]//bm,off[e+1]//bm): expert_ids[b]=e
                    for i,rank in ranks.items(): sorted_ids[off[ids[i]]+rank]=i
                    found=[]
                    for j,i in enumerate(sorted_ids[:off[-1]]):
                        if i==r:
                            self.assertGreaterEqual(i//8,m)  # G1 padding masked
                            self.assertGreaterEqual(i,r)     # G2 padding masked
                        else:
                            self.assertEqual(expert_ids[j//bm],ids[i])
                            self.assertEqual(i//8,divmod(i,8)[0])
                            found.append(i)
                    self.assertEqual(sorted(found),sorted(ranks))


if __name__=='__main__': unittest.main()
