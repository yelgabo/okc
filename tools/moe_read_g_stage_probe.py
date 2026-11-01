"""Read a terminal stage-probe receipt. No speedup extrapolation from sparse samples."""
import argparse
import base64
import json
import struct
import zlib
from pathlib import Path
from dl_g_stage_analyze import analyze_trace

def decode_packet(capsule):
 raw=zlib.decompress(base64.b64decode(capsule,validate=True));tag,grid,m=struct.unpack_from("<4sHH",raw)
 if tag!=b"GSP1" or not 3<=grid<=136 or m!=256 or len(raw)!=8+1900*8+m*4:raise ValueError("unknown trace capsule")
 arrays=struct.unpack_from("<1900Q",raw,8);masks=struct.unpack_from(f"<{m}I",raw,8+1900*8)
 ids=[]
 for mask in masks:
  row=[e for e in range(32) if mask&(1<<e)]
  if len(row)>8:raise ValueError("invalid local route mask")
  ids.append(row+list(range(32,40-len(row))))
 return {"grid":grid,"tokens":m,"masks":masks,
  "endpoint0":analyze_trace(arrays[0:475],ids,grid,3),
  "endpoint1":analyze_trace(arrays[475:950],ids,grid,3),
  "ready":analyze_trace(arrays[950:1425],ids,grid,1),
  "retire":analyze_trace(arrays[1425:1900],ids,grid,2)}

def summarize(receipt):
 if receipt.get("board")!="muse_moe_w4a16" or receipt.get("status")!="done" or not receipt.get("result",{}).get("passed"):
  raise ValueError("requires terminal passing standalone MoE receipt")
 lines=receipt["result"]["log_tail"].splitlines()
 report=json.loads(next(l.partition(" ")[2] for l in lines if l.startswith("GSP_SUMMARY ")))
 if not any(l.startswith("GSP_DONE 6 ") for l in lines):raise ValueError("incomplete diagnostic")
 wanted={(m,s) for m in (128,256) for s in (76111,76117,76123)}
 if len(report["cases"])!=6 or {(r[0],r[1]) for r in report["cases"]}!=wanted:raise ValueError("missing case coverage")
 if report["tolerance"]!=[.008,.075] or report["tallies"]!={"normal":72,"fresh":36,"skew":36,"remote":36,"restored":36}:raise ValueError("math coverage mismatch")
 cap=decode_packet(next(l.partition(" ")[2] for l in lines if l.startswith("GSP_CAPSULE ")))
 valid={f"{r[0]}/{r[1]}":[bool(v[-1]) for v in r[4]] for r in report["cases"]}
 return {"job":receipt["id"],"report":report,"valid_ready_retire_rows":valid,"representative":cap,
         "interpretation":"only passed mode/case rows; observed original-order waits, not physical TMA latency, no unbiased sub-us or all-stage bound"}

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("receipt",type=Path);a=p.parse_args()
 print(json.dumps(summarize(json.loads(a.receipt.read_text(encoding="utf8"))),indent=2))
