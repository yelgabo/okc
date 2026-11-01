import ast
import copy
from pathlib import Path
import types
import unittest
from dl_g_next_expert_overlap import _g_overlap_join,BODY,build,BASE

class Stream:
 def __init__(self,name,events):self.name=name;self.events=events;self.cuda_stream=id(self)
 def wait_stream(self,other):self.events.append((self.name,"wait",other.name))
class Tensor:
 device="cuda:0"
 def __init__(self,name,events):self.name=name;self.events=events
 def record_stream(self,stream):self.events.append((self.name,"record",stream.name))
 def is_contiguous(self):return True
class Context:
 def __init__(self,cuda,side):self.cuda=cuda;self.side=side
 def __enter__(self):self.cuda.events.append(("enter",self.side.name))
 def __exit__(self,*args):self.cuda.events.append(("exit",self.side.name))
class CUDA:
 def __init__(self,events,capture=False):self.events=events;self.capture=capture;self.parent=Stream("main",events)
 def current_stream(self,**kw):return self.parent
 def is_current_stream_capturing(self):return self.capture
 def stream(self,side):return Context(self,side)
 def Stream(self,device=None):return Stream("new",self.events)

class OverlapTests(unittest.TestCase):
 def test_real_join_logic_success_and_failure(self):
  for capture in (False,True):
   for fail in (None,"router","pool"):
    events=[];cuda=CUDA(events,capture);torch=types.SimpleNamespace(cuda=cuda);side=Stream("side",events)
    h,hr,logit,pool=[Tensor(n,events) for n in ("h","hr","logit","pool")]
    def callback(name,out):
     def f(a):
      events.append((name,"call"))
      if fail==name:raise RuntimeError("injected")
      return out
     return f
    if fail:
     with self.assertRaises(RuntimeError):_g_overlap_join(torch,h,hr,side,callback("pool",pool),callback("router",logit))
    else:self.assertEqual(_g_overlap_join(torch,h,hr,side,callback("pool",pool),callback("router",logit)),(pool,logit))
    self.assertEqual(events[0],("side","wait","main"))
    self.assertIn(("main","wait","side"),events)
    self.assertEqual(any(e[1]=="record" for e in events),not capture)
    if not fail:self.assertLess(events.index(("router","call")),events.index(("pool","call")))
 def test_owner_and_bootstrap_delegation(self):
  events=[];cuda=CUDA(events);old=object()
  ns={"torch":types.SimpleNamespace(cuda=cuda),"_pool_router_fp32":lambda*a:old,"prepare":lambda w:{},
      "_pdl":lambda T:False,"GLE_T":{},"bfp_SK2_PDL":0}
  exec(BODY,ns)
  owner=ns["_GOverlapStream"]("cuda:0");clone=copy.deepcopy(owner)
  self.assertIsNot(owner,clone);self.assertIsNot(owner.stream,clone.stream)
  tensor=Tensor("x",events)
  self.assertIs(ns["_pool_router_fp32"](tensor,tensor,1,{},"bfp"),old)
  cuda.capture=True
  with self.assertRaises(RuntimeError):copy.deepcopy(owner)
 def test_source_prefix_and_full_coverage(self):
  base=Path(__file__).resolve().parents[2]/"okc/muse_decode_layer/cand/g1b4_stage_release.py"
  source=base.read_text(encoding="utf8");output=build(source)
  self.assertTrue(output.startswith(source))
  self.assertIn("if checks!=32:",output)
  self.assertIn("G split-front shared writable storage",output)
  self.assertIn("close(route[2],rr[2]",output)
  with self.assertRaises(ValueError):build(source+"#bad")

if __name__=="__main__":unittest.main()
