"""Owning outgoing seam reader: only arm3 is a treatment; no automatic promotion."""
import argparse
import ast
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

import g_common_rmw_pair_read as receipt_rules
from g_common_rmw_pair import function
from g_common_release_body import BODY
from g_outgoing_pair_body import SUPPORT

MANIFEST=None
scope={"bfp__sk2_src":lambda:None,"load_inline":lambda:None}
exec(SUPPORT,scope)
decision=scope["_os_score"]
edge_gate=scope["_os_edges"]
positive=receipt_rules.positive
close=receipt_rules.close
OFFICIAL=receipt_rules.OFFICIAL

def graph_evidence(report):
    resources=report["secondary_resources"]
    assert set(resources)=={"1","8"},"secondary coverage"
    for vectors in resources.values():
        assert len(vectors)==4 and vectors[0]==vectors[1] and vectors[2]==vectors[3],"secondary identity"
        for vector in vectors:
            assert len(vector)==8 and all(type(value)is int and value>=0 for value in vector),"secondary ABI"
            assert vector[0]>0 and vector[1]>0 and vector[5]>=416 and vector[6]>=1 and vector[7]==416,"secondary resource"
        raw,adapt=vectors[0],vectors[2]
        assert adapt[1]<=raw[1] and adapt[2]<=raw[2] and adapt[3:6]==raw[3:6] and adapt[6]>=raw[6],"secondary resource loss"
    observed=Counter()
    assert len(report["graph_edges"])==48,"graph coverage"
    for tokens,window,seed,arm,proof in report["graph_edges"]:
        assert tokens in (1,8) and window in (0,2048) and arm in range(4) and seed in (993811,993817,993821),"graph placement"
        assert proof["edge_data_nonnull"] is True,"missing edgeData"
        primary=proof["primary"];assert len(primary)==2 and len(set(primary))==2 and all(type(pointer)is int and pointer>0 for pointer in primary),"primary identity ABI"
        assert primary==report["primary_identities"]["tiny" if tokens==1 else "tiny2"],"wrong primary function"
        nodes=dict(proof["neighbor_kernels"]+[proof["producer"],proof["consumer"]])
        edges=proof["incoming"]+proof["seam"]+proof["outgoing"]
        rows=[[2,0,1,1]]+[[0,node,pointer] for node,pointer in nodes.items()]+[[1,*edge] for edge in edges]
        rebuilt=edge_gate(rows,primary,resources[str(tokens)][arm][0],arm==3)
        assert close(rebuilt,proof),"graph reconstruction"
        observed[tokens,window,seed,arm]+=1
    assert observed==Counter({(tokens,window,seed,arm):1 for tokens in (1,8) for window in (0,2048) for seed in (993811,993817,993821) for arm in range(4)}),"graph bank coverage"

def capsule(log):
    assert MANIFEST is not None,"manifest required"
    blobs=re.findall(r"^OUTGOING_REPORT ([A-Za-z0-9+/=]+)\r?$",log,re.M)
    assert len(blobs)==1 and len(blobs[0])<11000,"missing/duplicate/truncated capsule"
    payload=zlib.decompress(base64.b64decode(blobs[0],validate=True));assert len(payload)<1_000_000
    report=json.loads(payload,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
    assert report["schema"]==3 and report["arms"]==MANIFEST["arms"] and report["part"]=="all"
    for field,key in (("candidate_pin","control_sha256"),("g1b4_conformance","conformance"),("device_sha","device_sha256"),("harness_core_sha","core_sha256")):
        assert report[field]==MANIFEST[key],("source pin",field)
    assert report["seeds"]==MANIFEST["seeds"] and report["common_control"] is True and report["cleanup_restored"] is True
    assert report["math_checks"]==report["fresh_final_checks"]==144 and report["empty_native_checks"]==32 and report["occupied_to_empty_checks"]==16
    assert report["fresh_replays_per_bank"]==8 and report["judged_stage_on"] is True and report["judged_short_relaxed"] is False
    assert re.fullmatch("[a-f0-9]{64}",report["weight_signature"])
    assert len(report["owners_bytes"])==4 and all(type(value)is int and value>0 for value in report["owners_bytes"])
    assert positive(report["prepare_s"])<295
    cache=report["reference_cache"];assert set(cache)=={"hit","miss","peak_entries"} and all(type(value)is int for value in cache.values())
    assert cache["hit"]+cache["miss"]==1152 and 144<=cache["miss"]<=1152 and cache["peak_entries"]==4
    phases=report["prepare_phase_seconds"];assert set(phases)=={"normal_prepare","private","resources","empty","row","cleanup"}
    assert all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in phases.values()) and sum(phases.values())<=report["prepare_s"]
    assert set(report["resources"])==set(report["common_resources"])=={"tiny","tiny2","mid","b4","main"}
    for key,vectors in report["resources"].items():
        assert len(vectors)==4 and all(vector==vectors[0] for vector in vectors)
        assert len(vectors[0])==12 and all(type(value)is int and value>=0 for value in vectors[0]) and min(vectors[0][4],vectors[0][10])>=1
        modern=report["common_resources"][key];assert len(modern)==4 and all(vector==modern[0] for vector in modern)
        assert len(modern[0])==20 and all(type(value)is int and value>=0 for value in modern[0])
        for start,oldstart in ((0,0),(10,6)):
            vector=modern[0][start:start+10]
            assert 0<vector[4]<=vector[5] and vector[6]>=1 and 0<vector[7]<=136 and vector[8]==1 and vector[9]==vector[6]*vector[7]
            assert [vector[0],vector[1],vector[2],vector[3],vector[6],vector[4]]==vectors[0][oldstart:oldstart+6]
    flags=report["module_flags"];assert set(flags)=={str(arm)+":"+key for arm in (0,2) for key in report["resources"]}
    for name,values in flags.items():
        assert isinstance(values,list) and all(isinstance(value,str) for value in values)
        assert "-DOKC_SHORT_FINAL_RELAXED=1" not in values and "-DOKC_STAGE_RETIRE=0" not in values
        if name.split(":")[1] in ("main","b4"):assert ("-DOKC_B4FIN=1" in values)==name.endswith(":b4")
    audit={};exec(function(BODY,"_gq_attn_record"),audit);observed=Counter()
    assert len(report["actual_attention"])==432
    for tokens,arm,row in report["actual_attention"]:
        assert tokens in (1,8,16,32,128,256) and type(arm)is int and arm in range(4) and all(type(value)is int for value in row)
        batch,length={1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[tokens]
        assert row[10] in (0,2048);audit["_gq_attn_record"]({"config":dict(tokens=tokens,batch=batch,q_len=length,num_kv_heads=2,window=row[10])},row)
        observed[tokens,arm,row[10]]+=1
    assert observed==Counter({(tokens,arm,window):9 for tokens in (1,8,16,32,128,256) for arm in range(4) for window in (0,2048)})
    for row in report["rows"]:
        tokens=row["tokens"];key="tiny" if tokens<=4 else "tiny2" if tokens<=8 else "mid" if tokens<=16 else "b4" if tokens<=32 else "main"
        assert type(row["local_start"])is int and row["local_start"] in range(0,256,32)
        counts=row["reference_group_counts"];assert len(counts)==8 and all(type(count)is int and count>=0 for count in counts) and sum(counts)==tokens*8
        assert row["actual_routes"]==[[[tokens,key,False]] for _ in range(4)]
    graph_evidence(report)
    recomputed=decision(report["rows"]);assert close(recomputed,report["decision"]),"printed gate mismatch"
    return report,recomputed

read_scope=dict(vars(receipt_rules));read_scope.update(capsule=capsule)
exec(function(Path(receipt_rules.__file__).read_text(),"read"),read_scope)

def read(receipt,job,log,manifest):
    global MANIFEST
    MANIFEST=manifest;read_scope["MANIFEST"]=manifest
    return read_scope["read"](receipt,job,log)

def verify_source(path,manifest):
    raw=path.read_bytes();assert len(raw)==manifest["bytes"] and hashlib.sha256(raw).hexdigest()==manifest["source_sha256"],"wrong uploaded source"

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("receipt",type=Path);parser.add_argument("--job",required=True);parser.add_argument("--source",required=True,type=Path);parser.add_argument("--manifest",required=True,type=Path);parser.add_argument("--log",required=True,type=Path);parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());verify_source(args.source,manifest);receipt=json.loads(args.receipt.read_text())
    assert args.log.resolve()==Path(receipt["output"]["log"]).resolve() and args.job in args.log.name,"owning log"
    raw=args.log.read_bytes();output=read(receipt,args.job,raw.decode("utf8"),manifest);output["owning_log_sha256"]=hashlib.sha256(raw).hexdigest();args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n");print(json.dumps(dict(job=args.job,complete=True,private_raw_gate=output["private_raw_gate"])))
