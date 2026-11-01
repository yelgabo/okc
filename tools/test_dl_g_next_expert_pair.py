import ast
from pathlib import Path
import random
import unittest
from dl_g_next_expert_pair import patch_device,patch_source,OLD,NEW
from test_dl_g_next_expert_flush import BASE

class PairTests(unittest.TestCase):
 def test_exact_delta(self):
  s=BASE.read_text(encoding="utf8");c=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC")
  o=patch_device(c);patch_source(s)
  self.assertEqual(o.replace(NEW,OLD).replace("  const int first_seq = seq;\n",""),c)
  with self.assertRaises(ValueError):patch_device(c+" ")
 def test_every_partial_wrap_parity(self):
  cases=0
  for s0 in range(16):
   for s1 in range(s0+1,17):
    for first in range(18):
     pending=[];released=[]
     for seq in range(first,first+s1-s0):
      pending.append(seq)
      if ((seq-first)&1) or seq-first==s1-s0-1:
       got=([seq-1] if (seq-first)&1 else [])+[seq]
       self.assertEqual(got,pending);released.extend(got);pending=[]
     self.assertFalse(pending);self.assertEqual(released,list(range(first,first+s1-s0)))
     cases+=1
  self.assertEqual(cases,2448)
 def test_skewed_four_warpgroup_progress(self):
  # Finite abstract per-stage math; every reuse requires all4WG arrivals.
  # Includes mixed segments/phase boundary and arbitrarily skewed WG execution.
  for seed in range(600):
   r=random.Random(seed);segments=[r.randrange(1,17) for _ in range(r.randrange(1,12))]
   starts=[];ends=[];a=0
   for length in segments:
    starts.extend([a]*length);ends.extend([a+length]*length);a+=length
   total=a;producer=0;nexts=[0]*4;released=[set() for _ in range(4)];pending=[[] for _ in range(4)];slots={}
   for step in range(12*total+10):
    actions=[]
    if producer<total and (producer<3 or all(producer-3 in x for x in released)):actions.append(4)
    for wg,k in enumerate(nexts):
     if k<total and slots.get(k%3)==k:actions.append(wg)
    if not actions:break
    who=r.choice(actions)
    if who==4:
     if producer>=3:self.assertTrue(all(producer-3 in x for x in released))
     slots[producer%3]=producer;producer+=1
    else:
     k=nexts[who];pending[who].append(k)
     if ((k-starts[k])&1) or k==ends[k]-1:
      for old in pending[who]:released[who].add(old)
      pending[who]=[]
     nexts[who]+=1
   self.assertEqual(nexts,[total]*4,seed)
   for wg in range(4):self.assertEqual(released[wg],set(range(total)))

if __name__=="__main__":unittest.main()
