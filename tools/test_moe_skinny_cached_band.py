import ast
from pathlib import Path
import random
import unittest
from moe_skinny_cached_band import patch,ENTRIES
from moe_skinny_band import group_map,credit_at

ROOT=Path(__file__).resolve().parents[1]
class Cached(unittest.TestCase):
 def test_bounds_and_packed_schedule(self):
  rng=random.Random(993881)
  # All route-count totals and active counts; ceil bound includes arbitrary duplicate IDs.
  for routes in range(257):
   for active in range(1,min(32,routes)+1):
    jobs=(routes+7*active)//8
    for grid in (132,133,136):
     self.assertLessEqual((jobs*512+grid-1)//grid+(jobs*256+grid-1)//grid,ENTRIES)
  for _ in range(100):
   counts=[0]*32
   for r in range(rng.randrange(257)):counts[rng.randrange(32)]+=1
   jobs=[(e,k,min(8,c-8*k)) for e,c in enumerate(counts) for k in range((c+7)//8)]
   J=len(jobs);self.assertLessEqual(J,60)
   for cta in range(132):
    a0=J*512*cta//132;a1=J*512*(cta+1)//132
    b0=J*256*cta//132;b1=J*256*(cta+1)//132
    schedule=[]
    for pb,(lo,hi,U) in enumerate(((a0,a1,512),(b0,b1,256))):
     for vv in range(lo,hi):
      oldj=vv//U;e,k,_=jobs[oldj];base=oldj-k;nj=(counts[e]+7)//8
      pj,rb=group_map(vv-base*U,nj,U);v=(base+pj)*U+rb
      last,n=credit_at(a0-base*512,a1-base*512,nj,pj,512)
      credit=n if not pb and vv-base*U==last else 0
      word=v|(credit<<16);self.assertEqual(word&65535,v);self.assertEqual(word>>16,credit)
      self.assertLess(v,30720);self.assertLessEqual(credit,233)
      schedule.append(word)
    self.assertLessEqual(len(schedule),ENTRIES)

 def test_source_guards_and_setup_order(self):
  src=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf8')
  out=patch(src);tree=ast.parse(out)
  b=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK')
  self.assertIn('p.M<=32 && sms>=132',b)
  self.assertEqual(b.count('band_map(S,'),1) # setup only
  self.assertEqual(b.count('band_credit(S,'),1) # setup only
  self.assertEqual(b.count('S.schedule[q] & 65535u'),4)
  self.assertEqual(b.count('S.schedule[q] >> 16'),1)
  setup=b.index('for (int q = lane; q < n; q += 32)')
  issue=b.index('for (int q = 0; q < min(kNS, n); ++q) issue_unit')
  self.assertLess(setup,b.index('__syncwarp();',setup));self.assertLess(b.index('__syncwarp();',setup),issue)

if __name__=='__main__':unittest.main()
