"""Private four-cell source package, retaining the existing selected payload."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import dl_selected_pair
from dl_verify_pair_slices import build as slices, once


def build(root):
    original, old_manifest=dl_selected_pair.build(root)
    arms, arm_manifest=slices(root)
    source=original
    for name in ("G","owned"):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                  and getattr(n.targets[0],"id","")=="_PAIR_SOURCE_"+name)
        old=ast.get_source_segment(source,node)
        source=once(source,old,"_PAIR_SOURCE_"+name+" = "+dl_selected_pair.literal(arms[name]))
    pins=dict(old_manifest["arm_pins"])
    pins.update({k:hashlib.sha256(v.encode()).hexdigest() for k,v in arms.items()})
    source=once(source,"_PAIR_PINS = "+repr(old_manifest["arm_pins"]),"_PAIR_PINS = "+repr(pins))
    baseline_path=root/"muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.json"
    baseline=json.loads(baseline_path.read_text(encoding="utf8"))
    assert baseline["job_id"]=="607eb52bdaf24160"
    cells={(b["case"]["batch"],b["case"]["layer"]):b["median_ns"]/1000
           for b in baseline["benchmarks"] if b["case"]["load"].startswith("verify") and b["case"]["batch"] in (16,32)}
    assert set(cells)=={(b,l) for b in (16,32) for l in ("sliding","full")}
    costs={b:46*cells[b,"sliding"]+16*cells[b,"full"] for b in (16,32)}
    source=once(source,"_PAIR_PINS = "+repr(pins),"_PAIR_PINS = "+repr(pins)+
                "\n_VERIFY_G_CELLS = "+repr(cells)+"\n_VERIFY_G_COSTS = "+repr(costs))
    old_body=(root/"tools/dl_selected_pair_body.py").read_text(encoding="utf8")
    helpers="\n\n".join(ast.get_source_segment(old_body,n) for n in ast.parse(old_body).body
                         if isinstance(n,ast.FunctionDef) and n.name in ("_pair_restore","_pair_run","_pair_validate"))
    body=(root/"tools/dl_verify_pair_body.py").read_text(encoding="utf8")
    source=once(source,old_body,helpers+"\n\n"+body)
    source=once(source,"PRIVATE selected-cell comparison; never submit publicly.",
                "PRIVATE verify16/32 sliding/full comparison; never submit publicly.")
    ast.parse(source)
    assert len(source.encode())<1000000,"private verify payload exceeds size cap"
    return source,{"bytes":len(source.encode()),"sha256":hashlib.sha256(source.encode()).hexdigest(),
                   "scope":"verify16/32 sliding/full; other four loads unmeasured",
                   "seeds":[781051,781061],"arm_pins":pins,"slice_manifest":arm_manifest,
                   "leverage_baseline_job":baseline["job_id"],
                   "leverage_baseline_sha256":hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                   "one_microsecond_sensitivity":{str(b)+":"+l:(costs[b]/(costs[b]-(46 if l=="sliding" else 16)))**(1/6)
                       for b,l in cells},
                   "whole_six_load_score_claim":False,"H200_qualified":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();text,manifest=build(a.root)
    a.output.write_bytes(text.encode())
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print(json.dumps(manifest,indent=2))
