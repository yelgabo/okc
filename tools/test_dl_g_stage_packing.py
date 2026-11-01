"""Actual TorchCPU byte comparison for standalone pack ABI vs exact layer native."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
NAMES={"_stored_pos","_kperm_src","_kperm_codes","_kperm_rows","_pack_codes_nat","_wgs_copy","_sc6_wgs","_pack_codes","_pack_scales","_fat_order","_unit_order","_split_nibbles","_scatter_code_bits"}
def load(path):
 s=path.read_text(encoding="utf8");ns={"torch":torch,"DQ7":True,"_REG_HJ":[(0,0),(1,0),(0,2),(1,2)]}
 for n in ast.parse(s).body:
  if isinstance(n,ast.FunctionDef) and n.name in NAMES:exec(compile(ast.Module([n],[]),"packing_only","exec"),ns)
 ns["_POS"]=ns["_stored_pos"]();return ns
if __name__=="__main__":
 layer,standalone,output=map(Path,sys.argv[1:]);a,b=load(layer),load(standalone);gen=torch.Generator().manual_seed(991707);report=[]
 for kind,R in [("W13",512),("W2",256)]:
  code=torch.randint(0,256,(R,2048),dtype=torch.uint8,generator=gen)
  sf=torch.randint(64,127,(R,256),dtype=torch.uint8,generator=gen)
  if kind=="W13":
   rb=torch.arange(R//16);g=torch.arange(8);rows=torch.stack((8*rb[:,None]+g[None,:],R//2+8*rb[:,None]+g[None,:]),1).reshape(-1)
  else:rows=torch.arange(R)
  for sc6 in (False,True):
   x=a["_wgs_copy"](code,sf,a["_kperm_rows"](rows) if kind=="W13" else rows,sc6);y=b["_wgs_copy"](code,sf,b["_kperm_rows"](rows) if kind=="W13" else rows,sc6)
   assert torch.equal(x,y),"WGS ABI byte mismatch"
   report.append((kind,"WGS",sc6,x.numel(),hashlib.sha256(x.numpy().tobytes()).hexdigest()))
  ca,cb=a["_pack_codes"](code,rows),b["_pack_codes"](code,rows);sa,sb=a["_pack_scales"](sf,rows),b["_pack_scales"](sf,rows)
  assert torch.equal(ca,cb) and torch.equal(sa,sb)
  assert torch.equal(a["_fat_order"](ca,sa),b["_fat_order"](cb,sb))
  assert torch.equal(a["_unit_order"](ca,sa),b["_unit_order"](cb,sb,False))
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({"torch":torch.__version__,"device":"CPU","all_bytes_equal":True,"records":report},indent=2)+"\n")
 print("PACK_ABI_ALL_BYTES_EQUAL",report)
