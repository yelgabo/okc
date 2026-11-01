"""Independent integer model of pinned VP2 DSM ownership; no donor code import."""
from collections import Counter
import unittest

class Mailbox(unittest.TestCase):
 def test_exact_destination_footprint(self):
  for source in range(16):
   owners=[Counter() for _ in range(16)]
   for w in range(4):
    for g in range(8):
     for t in range(4):
      for jj in range(8):
       owner=2*jj+t//2
       for e in range(2):
        row=2*(t%2)+e;offset=source*1120+row*272+(16*w+2*g)*4
        owners[owner].update(range(offset,offset+8))
     # WG0, t==0: four scalar metadata stores per lane group.
     a=4*w+g//4;b=a+2;row=g%4
     for owner in (a,b):
      for offset in (source*1120+1088+row*4,source*1120+1104+row*4):owners[owner].update(range(offset,offset+4))
   for coverage in owners:
    self.assertEqual(len(coverage),1056)
    self.assertEqual(set(coverage.values()),{1})
    self.assertGreaterEqual(min(coverage),source*1120)
    self.assertLess(max(coverage),(source+1)*1120)
  self.assertEqual(16*1056,16896)
 def test_output_and_new_cache_owners(self):
  outputs=Counter();cache=Counter()
  for hk in range(2):
   for rank in range(16):
    token=rank//2;head0=8*hk+4*(rank%2)
    outputs.update((token,head0+w) for w in range(4))
    if rank%2==0:cache.update((token,hk,lane) for lane in range(32))
  self.assertEqual(set(outputs),{(t,h) for t in range(8) for h in range(16)})
  self.assertEqual(set(outputs.values()),{1})
  self.assertEqual(len(cache),8*2*32)
  self.assertEqual(set(cache.values()),{1})
if __name__=='__main__':unittest.main()
