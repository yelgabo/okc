import ast
import collections
import hashlib
import os
from pathlib import Path
import unittest
from dl_g_next_expert_flush import patch_device,patch_source,DEVICE_SHA,OLD,NEW,SIG
BASE=Path(os.environ.get("OKC_G_STAGE_BASE",str(Path(__file__).resolve().parents[2]/"okc/.build/today/g1b4-stage-retire.py")))

class FlushTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.source=BASE.read_text(encoding="utf8")
  cls.cuda=next(ast.literal_eval(n.value) for n in ast.parse(cls.source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC")
 def test_exact_scope(self):
  output=patch_source(self.source);new=patch_device(self.cuda)
  self.assertEqual(hashlib.sha256(self.cuda.encode()).hexdigest(),DEVICE_SHA)
  self.assertNotIn(OLD,new);self.assertEqual(new.count(NEW),1)
  self.assertLess(new.index("stage(std::true_type{});"),new.index(NEW))
  for prefix in ("mbar_arrive1(","fence.proxy.async", "wg_wait<", "wgs_flush<PB>(p, S, pd)"):
   self.assertEqual(self.cuda.count(prefix),new.count(prefix),prefix)
  with self.assertRaises(ValueError):patch_device(self.cuda+" ")
  self.assertEqual(new[new.index("__device__ __forceinline__ void wgs_flush"):new.index(SIG)],self.cuda[self.cuda.index("__device__ __forceinline__ void wgs_flush"):self.cuda.index(SIG)])
 def test_pending_publication_model(self):
  cases=0
  # The same pending record is published once before the current epilogue
  # can overwrite it, including one-stage/partial/reordered units and no pending.
  for kind in range(4):
   for begin in range(16):
    for end in range(begin+1,17):
     for parity in range(2):
      for slot0 in range(3):
       events=[];pending=kind
       for stage in range(begin,end):
        slot=(slot0+stage-begin)%3
        events.extend([("math",stage),("drain",stage),("fence",stage),("release",slot)])
        if stage==begin and pending:
         events.append(("publish",pending));pending=0
       events.append(("current_epilogue",0))
       pubs=[i for i,e in enumerate(events) if e[0]=="publish"]
       self.assertEqual(len(pubs),bool(kind))
       if kind:
        self.assertGreater(pubs[0],events.index(("release",slot0)))
        self.assertLess(pubs[0],len(events)-1)
       self.assertEqual(sum(e[0]=="release" for e in events),end-begin)
       cases+=1
  self.assertEqual(cases,3264)
 def test_publication_dependency_preserved(self):
  # Publication relative order is invariant. Moving a flush past current math
  # adds no same-phase-publication dependency: A uses gathered inputs, B uses A.
  for n in range(1,100):
   old=[("publish",i) for i in range(n)]
   new=[]
   for i in range(n):new.extend([("stage_math_from_A_or_gather",i),("publish",i)])
   self.assertEqual([e for e in new if e[0]=="publish"],old)

if __name__=="__main__":unittest.main()
