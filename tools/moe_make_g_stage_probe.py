"""Build one private standalone attribution artifact; no dispatch."""
import argparse
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from dl_g_stage_probe import patch_device,DEVICE_SHA
import dl_g_stage_analyze as analyzer

def build(root,layer,upstream):
 sys.path.insert(0,str(root/"tools"))
 from moe_make_sc6_compare import make,RESOURCE
 s=make(upstream,(root/"muse_moe_w4a16/baseline.py").read_text(encoding="utf8"),"")
 device=next(ast.literal_eval(n.value) for n in ast.parse(layer).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="MOE_CUDA_SRC")
 if hashlib.sha256(device.encode()).hexdigest()!=DEVICE_SHA:raise ValueError("unreviewed layer device")
 node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="CUDA_SRC")
 lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=["CUDA_SRC="+repr(patch_device(device)+RESOURCE)+"\n"];s="".join(lines)
 helpers="import math,statistics\nORDINALS=(0,7,15,31)\nWORDS=475\n"+"\n\n".join(inspect.getsource(getattr(analyzer,n)) for n in ["expected_jobs","stage_order","analyze_trace","distortion","compact"])
 s+="\n"+helpers+"\n"+Path(__file__).with_name("moe_g_stage_body.py").read_text(encoding="utf8")
 ast.parse(s);assert len(s.encode())<1000000
 return s
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--layer",type=Path,required=True);p.add_argument("--upstream",type=Path,required=True);p.add_argument("output",type=Path);a=p.parse_args()
 s=build(a.root,a.layer.read_text(encoding="utf8"),a.upstream.read_text(encoding="utf8"));a.output.write_text(s,encoding="utf8",newline="\n");print(len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
