import ast
import hashlib
import pathlib
import types
import unittest
from unittest.mock import Mock
from dl_owned_slice import build, PIN
BASE=pathlib.Path(__file__).resolve().parents[2]/"okc"/"muse_decode_layer/cand/layer_lt_boundary.py"

class SliceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.source=BASE.read_text(encoding="utf8");cls.output,cls.manifest=build(cls.source)
 def test_pin_and_literals(self):
  from dl_owned_attention_boundary import patch_owned_attention
  def vals(s):
   return {n.targets[0].id:n.value.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
  full=vals(patch_owned_attention(self.source)); small=vals(self.output)
  for k in self.manifest["device_literals"]: self.assertEqual(full[k],small[k],k)
  self.assertLess(self.manifest["bytes"],510000)
  with self.assertRaises(AssertionError):build(self.source+"#different")
 def test_registration(self):
  # Execute the actual emitted builder definitions, with inert extension API only.
  tree=ast.parse(self.output)
  wanted={"_ext","_one_load","_moe1_load","_moe1_source","_dlplt_build"}
  ns={"load_inline":Mock(return_value=object())}
  for n in tree.body:
   if isinstance(n,ast.Assign):
    try:exec(compile(ast.Module([n],[]),"slice","exec"),ns)
    except (NameError,AttributeError):pass
   elif isinstance(n,ast.FunctionDef) and n.name in wanted:exec(compile(ast.Module([n],[]),"slice","exec"),ns)
  ns["_ext"]();ns["_one_load"]();ns["_moe1_load"]();ns["_dlplt_build"]()
  calls=ns["load_inline"].call_args_list
  self.assertEqual(len(calls),4)
  self.assertEqual(calls[0].kwargs["functions"],["route_fwd","attn_boundary"])
  self.assertEqual(len(calls[0].kwargs["cuda_sources"]),2)
  host=calls[2].kwargs["cuda_sources"][1]
  self.assertIn("case 10: launch_main<51>",host)
  self.assertNotIn("case 9:",host)
 def test_runtime_scope_and_order(self):
  tree=ast.parse(self.output);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="kernel_fn")
  events=[]
  def log(name,result):
   def f(*a,**kw):events.append(name);return result
   return f
  class Z:
   def __getitem__(self,k):return self
   def t(self):return self
  z=Z();torch=types.SimpleNamespace(float32="fp32",bfloat16="bf16",mm=log("concat",z))
  ns={"torch":torch,"D_EXPERT":4096,"N_EXPERTS":256,
   "_slice_proj":lambda x,l,s:(events.append(l) or z),"_attn_one":log("one",z),
   "_abnd":log("abnd",(z,z,None)),"_route_fwd":log("route",(z,z,z)),"_moe1_run":log("moe",z)}
  exec(compile(ast.Module([n],[]),"slice","exec"),ns)
  cfg=dict(batch=32,q_len=8,tokens=256,page_size=1,window=2048,hidden=8192,num_q_heads=16,num_kv_heads=2,head_dim=64,n_experts=256,top_k=8,local_experts=32)
  data=dict(config=cfg,page_table=types.SimpleNamespace(shape=(32,80000)),attn_input=z,stream=z,local_start=z)
  state={"pool_router":z,"ffn_boundary":log("ffn_boundary",(z,z,None))}
  ns["kernel_fn"](data,state)
  self.assertEqual(events,["qkv_gate","one","o","abnd","concat","route","moe","unpooler","ffn_boundary"])
  cfg["tokens"]=128
  with self.assertRaises(AssertionError):ns["kernel_fn"](data,state)

if __name__=="__main__":unittest.main()
