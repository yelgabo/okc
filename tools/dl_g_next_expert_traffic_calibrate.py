"""Offline source-fidelity check; no kernels or source copied into a submission."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from dl_g_next_expert_traffic import load_public_generator
reference,archived,output=map(Path,sys.argv[1:])
public,pin=load_public_generator(reference,torch)
s=archived.read_text(encoding="utf8");n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=="_gen_input")
ns=dict(torch=torch,HIDDEN=4096,N_EXPERTS_TOTAL=256,TOP_K=8,LOCAL_EXPERTS=32,LOCAL_START=0,SHARDING="ep8")
exec(compile(ast.Module([n],[]),"archived_generator_only","exec"),ns)
a=public(256,"ep8",4773,device="cuda");b=ns["_gen_input"](256,4773,"cuda")
result={"public_vs_archived_same_backend":{k:torch.equal(a[k],b[k]) for k in ("hidden_states","topk_ids","topk_weights")},
 "archived_generator_function_sha256":hashlib.sha256(ast.get_source_segment(s,n).encode()).hexdigest()}
x=json.loads(output.read_text(encoding="utf8"));x["local_census"]["source_fidelity"]=result
output.write_text(json.dumps(x,indent=2)+"\n",encoding="utf8");print(json.dumps(result))
