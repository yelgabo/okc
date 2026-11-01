import ast
import hashlib
from pathlib import Path
import random
import unittest
from moe_skinny_band import patch,group_map,prefix,credit_at
from moe_multiply_decode import PIN

ROOT=Path(__file__).resolve().parents[1]
class Band(unittest.TestCase):
 def test_bijection_prefix_last(self):
  rng=random.Random(81751)
  for jobs in range(1,33):
   for units in (256,512):
    seq=[group_map(x,jobs,units) for x in range(jobs*units)]
    self.assertEqual(set(seq),{(j,u) for j in range(jobs) for u in range(units)})
    for x in (0,1,31,32,33,jobs*units-1,jobs*units,*[rng.randrange(jobs*units+1) for _ in range(12)]):
     for j in range(jobs):self.assertEqual(prefix(x,jobs,j,units),sum(a==j for a,b in seq[:x]))
    for grid in (1,7,132,136):
     total=[0]*jobs
     for c in range(grid):
      a=jobs*units*c//grid;b=jobs*units*(c+1)//grid
      for j in range(jobs):
       last,n=credit_at(a,b,jobs,j,units)
       expected=[a+i for i,(jj,u) in enumerate(seq[a:b]) if jj==j]
       self.assertEqual(n,len(expected));self.assertEqual(last,max(expected) if expected else None)
       total[j]+=n
     self.assertEqual(total,[units]*jobs)

 def test_source_scope(self):
  src=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf8')
  self.assertEqual(hashlib.sha256(src.encode()).hexdigest(),PIN)
  revised=patch(src);a=ast.parse(src);b=ast.parse(revised)
  for t in (a,b):
   for n in t.body:
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK':n.value=ast.Constant('ONLY_CHANGE')
  self.assertEqual(ast.dump(a),ast.dump(b))
  with self.assertRaises(AssertionError):patch(revised)
  blocks=[]
  for t in (ast.parse(src),ast.parse(revised)):
   blocks.append(next(n.value.value for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK'))
  # No synchronization primitive or precision operation removed/added except credit placement.
  for s in ('mbar_try(', 'named_sync(', 'named_arrive(', 'fence_proxy_async(', '__syncthreads(', 'bf16r(', 'mma_bf16('):
   self.assertEqual(blocks[0].count(s),blocks[1].count(s),s)

if __name__=='__main__':unittest.main()
