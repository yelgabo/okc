"""CPU contracts for owned mode3 cluster merge, including empty partitions."""
import unittest
from collections import Counter


class Contract(unittest.TestCase):
    def test_partition(self):
        for seq in (0,1,7,31,63,64,65,1024,2040,2047,2048,79999,80000,1000000):
            start=max(0,seq-2048+1);end=seq+8;n=end-start;tiles=(n+63)//64
            ranges=[(start+(tiles*c//32)*64,min(end,start+(tiles*(c+1)//32)*64)) for c in range(32)]
            self.assertEqual(ranges[0][0],start);self.assertEqual(ranges[-1][1],end)
            self.assertTrue(all(a<=b for a,b in ranges))
            self.assertTrue(all(ranges[c][1]==ranges[c+1][0] for c in range(31)))
            self.assertEqual(sum(b-a for a,b in ranges),n)
            self.assertEqual(sum((b-a+63)//64 for a,b in ranges),tiles)
        self.assertEqual(128*68*4,34816)
        self.assertLessEqual(34816,3*2*8192)

    def test_partial_and_merge_bijection(self):
        elements=Counter();sums=Counter()
        for tid in range(256):
            warp,lane=divmod(tid,32);hk=warp//4;wq=warp%4;g=lane//4;c=lane%4
            for hh in range(2):
                row=(2*wq+hh)*16+hk*8+g
                for pp in range(4):
                    for e in range(4):elements[row*68+c*4+pp*16+e]+=1
                if c==0:sums[row*68+64]+=1
        expected={r*68+d for r in range(128) for d in range(64)}
        self.assertEqual(set(elements),expected);self.assertEqual(set(elements.values()),{1})
        self.assertEqual(set(sums),{r*68+64 for r in range(128)})
        self.assertEqual(set(sums.values()),{1})
        vectors=Counter(e for tid in range(256) for e in range(tid,128*16,256))
        self.assertEqual(set(vectors),set(range(128*16)));self.assertEqual(set(vectors.values()),{1})


if __name__=='__main__':unittest.main()
