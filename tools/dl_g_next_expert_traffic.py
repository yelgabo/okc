"""Decode retained WGF1 / census public normal routing offline. Never a judge payload."""
import ast
import base64
import collections
import hashlib
import json
from pathlib import Path
import struct
import zlib

STAGE_BYTES=35840
PHASE_UNITS={"W13":32,"W2":16}

def model(counts,height=64):
 if len(counts)!=32 or min(counts)<0:raise ValueError("32 nonnegative local counts required")
 active=sum(c>0 for c in counts);jobs=sum((c+height-1)//height for c in counts)
 per={name:units*16*STAGE_BYTES for name,units in PHASE_UNITS.items()}
 return {"counts":counts,"local_routes":sum(counts),"active_experts":active,"max_count":max(counts),
  "jobs":jobs,"extra_jobs":jobs-active,
  "phase_payload":{name:{"mandatory_unique_bytes":active*b,"requested_bytes":jobs*b,"extra_bytes":(jobs-active)*b} for name,b in per.items()},
  "mandatory_unique_bytes":active*sum(per.values()),"requested_bytes":jobs*sum(per.values()),
  "extra_bytes":(jobs-active)*sum(per.values()),"extra_request_fraction":(jobs-active)/jobs if jobs else 0.}

def decode(log):
 lines=[x.split()[1] for x in log.splitlines() if x.startswith("WGSFRONT_TRACE ")]
 if len(lines)!=1:raise ValueError("exactly one capsule expected")
 raw=zlib.decompress(base64.b64decode(lines[0],validate=True));magic,grid,tokens=struct.unpack_from("<4sHH",raw)
 if magic!=b"WGF1" or not 1<=grid<=136 or len(raw)!=8+4*(32+3*grid+3*tokens):raise ValueError("invalid WGF1 extent")
 words=struct.unpack_from("<"+str((len(raw)-8)//4)+"I",raw,8);masks=list(words[-tokens:])
 if any(x.bit_count()>8 for x in masks):raise ValueError("more than8localexperts/token")
 counts=[sum(bool(mask&(1<<e)) for mask in masks) for e in range(32)]
 if any(bool(words[e])!=bool(counts[e]) for e in range(32)):raise ValueError("writer/input expert mismatch")
 return {"grid":grid,"tokens":tokens,"raw_sha256":hashlib.sha256(raw).hexdigest(),"masks":masks,
         "token_local_count_histogram":dict(sorted(collections.Counter(x.bit_count() for x in masks).items())),**model(counts)}

def load_public_generator(reference,torch):
 source=reference.read_text(encoding="utf8");tree=ast.parse(source)
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="generate_input")
 # Identical RNG/math/order. Add ONLY diagnostic selected-group metadata at return.
 n=ast.parse(ast.get_source_segment(source,n)).body[0]
 ret=next(x for x in n.body if isinstance(x,ast.Return))
 n.body.insert(n.body.index(ret),ast.parse("selected_group = int(pairs.argmax()) if rank == 'busiest' and count < N_EXPERTS else 0").body[0])
 ret.value=ast.Call(func=ast.Name(id="dict",ctx=ast.Load()),args=[ret.value],keywords=[ast.keyword(arg="selected_group",value=ast.Name(id="selected_group",ctx=ast.Load()))])
 ast.fix_missing_locations(n)
 ns={"torch":torch,"D_HIDDEN":4096,"D_EXPERT":4096,"N_EXPERTS":256,"ROUTED_PER_TOKEN":8,"LOCAL":{"ep8":(0,32)},"POPULARITY":.03}
 exec(compile(ast.Module([n],[]),"public_normal_input_diagnostic","exec"),ns)
 return ns["generate_input"],hashlib.sha256(source.encode()).hexdigest()

def census(reference,log):
 import torch
 generator,pin=load_public_generator(reference,torch);capsule=decode(log)
 def bank(m,seed):
  d=generator(m,"ep8",seed,dist="normal",rank="busiest",device="cuda")
  ids=d["topk_ids"].cpu().tolist();masks=[sum(1<<e for e in set(row) if 0<=e<32) for row in ids]
  counts=[sum(bool(mask&(1<<e)) for mask in masks) for e in range(32)]
  return {"m":m,"seed":seed,"selected_global_group":d["selected_group"],"masks":masks,**model(counts)}
 reproduced=bank(256,4517+256)
 fidelity={"h200_capsule_exact_masks":reproduced["masks"]==capsule["masks"],
  "h200_capsule_counts":capsule["counts"],"local_counts":reproduced["counts"],
  "local_mask_differences":sum(a!=b for a,b in zip(reproduced["masks"],capsule["masks"]))}
 calibrations=[]
 for m in (128,256):
  for seed in (901,902):
   b=bank(m,1009*seed+m);b.pop("masks");calibrations.append(b)
 records=[]
 for m in (128,256):
  for seed in range(65001,65033):
   b=bank(m,seed);b.pop("masks");records.append(b)
 return {"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(),
  "capability":torch.cuda.get_device_capability(),"reference_lf_sha256":pin,
  "distribution":"public standalone normal/popularity.03/top8/busiestep8; not actual layer logits",
  "prespecified_seeds":list(range(65001,65033)),"fidelity":fidelity,"calibrations":calibrations,"records":records}

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--log",type=Path,required=True);p.add_argument("--reference",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 log=a.log.read_text(encoding="utf8");out={"capsule":decode(log)}
 if a.reference:out["local_census"]=census(a.reference,log)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
 print(json.dumps({k:v for k,v in out["capsule"].items() if k!="masks"},indent=2))
 if "local_census" in out:print("LOCAL_FIDELITY",json.dumps(out["local_census"]["fidelity"]))
