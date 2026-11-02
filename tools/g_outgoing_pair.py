"""Rebase the frozen legal paired instrument on one outgoing seam; PRIVATE ONLY."""
import argparse
import ast
import json
from pathlib import Path

import g_common_release as exports
from g_common_rmw_pair import build as paired, function, sha
from g_integration_common import build as common
from g_short_rmw_check import once, replace_function
from g_short_outgoing_seam import SUPPORT as ADAPTER, PIN
from g_outgoing_pair_body import SUPPORT
from g_outgoing_seam_query import PRIMARY, SECONDARY, DECL
from moe_make_sc6_compare import RESOURCE

MARKER="\n# PRIVATE CHECK ONLY: outgoing-only seam four placements. NEVER SUBMIT.\n"

def build(root):
    root=Path(root)
    previous,prior_manifest=paired(root)
    _,suffix=previous.split("\n# PRIVATE CHECK ONLY: four-owner short-RMW diagnostic. NEVER SUBMIT.\n")
    base,_=common((root/"muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"),True)
    assert sha(base)==PIN
    device=exports.native_exports(exports.attention_exports(base))
    changes=[]
    for name in ("MOE_CUDA_SRC","_MA_ALT_SRCS['moe_helper0']"):
        node=exports.assignment(device,name)
        text=ast.literal_eval(node.value)+RESOURCE.replace("sc6_resources","wl_resources")+PRIMARY
        changes.append((node,name+" = "+exports.literal(text)))
    node=exports.assignment(device,"MOE_CPP_SRC")
    changes.append((node,"MOE_CPP_SRC = "+exports.literal(ast.literal_eval(node.value)+"\nstd::vector<int64_t> wl_resources();\nstd::vector<int64_t> seam_native_identity();\n")))
    node=exports.assignment(device,"bfp_CPP_SRC")
    changes.append((node,"bfp_CPP_SRC = "+exports.literal(ast.literal_eval(node.value)+DECL)))
    device=exports.replace_nodes(device,changes)
    device=once(device,' + ["qual_native_resources"],',' + ["wl_resources","qual_native_resources","seam_native_identity"],')
    device=once(device,'"gq", "sk2f", "sk2_set_pdl"],','"gq", "sk2f", "sk2_set_pdl","seam_unpool_resources","seam_graph_edges"],')
    adapter=ADAPTER.replace("@@ENABLED@@","False")
    adapter=replace_function(adapter,"prepare","")
    adapter=once(adapter,"def _os_install(state):","def _os_install(state,enabled):")
    adapter=once(adapter,"1073741826 if _OS_ENABLED else 0","1073741826 if enabled else 0")
    adapter=once(adapter,"if name != 'un' or m not in (1, 8):","if name != 'un' or m not in (1, 8) or '_os_unpool_bfp' not in st:")
    adapter=once(adapter,"return shared + bfp_MAP_SRC + source,", "return shared + bfp_MAP_SRC + source + _OS_QUERY_SOURCE,")
    adapter=once(adapter,"functions=['seam_sk2']","functions=['seam_sk2','seam_unpool_resources','seam_graph_edges']")
    adapter=once(adapter,"cpp_sources=cpp,","cpp_sources=cpp + _OS_QUERY_DECL,")
    device+="\n_OS_QUERY_SOURCE="+repr(SECONDARY)+"\n_OS_QUERY_DECL="+repr(DECL)+"\n"+adapter
    suffix=once(suffix,"arm['ma_ext']=dict(state['ma_ext']);arm['ma_ext'].pop('tiny',None);arm['ma_ext'].pop('tiny2',None)","arm['ma_ext']=dict(state['ma_ext'])")
    old=function(suffix,"_srm_resource_gate")
    suffix=once(suffix,old,"")
    suffix=replace_function(suffix,"_wl_summary","")
    suffix=once(suffix,"arms=[_wl_clone(state,readonly,i>=2) for i in range(4)]","arms=[_wl_clone(state,readonly,False) for _ in range(4)]\n    _OS_GRAPHS.clear();_OS_SECONDARY.clear()\n    for index,arm in enumerate(arms):\n        arm['_os_placement']=index\n        if index>=2:_os_install(arm,index==3)")
    suffix=once(suffix,"common_resources=_srp_native_resources(arms,budget)","common_resources=_srp_native_resources(arms,budget)\n    secondary_resources=_os_secondary_gate(arms,budget)")
    suffix=once(suffix,"for i in order:captures[i]=_wl_capture(data[i],arms[i])","for i in order:\n                    arms[i]['_os_bank']=seed;captures[i]=_wl_capture(data[i],arms[i])")
    suffix=once(suffix,"assert ('-DOKC_SHORT_FINAL_RELAXED=1' in flags)==(bool(state['_srm_on']) and key in ('tiny','tiny2')),'short RMW flag drift'","assert '-DOKC_SHORT_FINAL_RELAXED=1' not in flags,'outgoing RMW drift'\n        state['_os_primary']=list(ext.seam_native_identity())")
    suffix=once(suffix,"bool(state['_srm_on']),record","state['_os_placement'],record")
    suffix=once(suffix,"graph=torch.cuda.CUDAGraph()\n    with torch.cuda.graph(graph):out=_wl_run(data,state)","graph=torch.cuda.CUDAGraph(keep_graph=True)\n    with torch.cuda.graph(graph):out=_wl_run(data,state)\n    _os_graph(graph,data,state)\n    graph.instantiate()")
    suffix=once(suffix,"'schema':2,","'schema':3,'arms':list(_OS_ARMS),'graph_edges':list(_OS_GRAPHS),'secondary_resources':secondary_resources,'primary_identities':{key:list(ext.seam_native_identity()) for key,ext in arms[0]['ma_ext'].items()},")
    suffix=once(suffix,"'two_short_RMW_flags_all_WGS_stage_on'","'outgoing_only_four_placements'")
    suffix=once(suffix,"print('SRM_COMMON_REPORT '+blob","print('OUTGOING_REPORT '+blob")
    suffix=once(suffix,"global _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back","global _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back,load_inline")
    suffix=once(suffix,"        _ma_ext=_SRP_OBSERVE_BUILD","        load_inline=_os_load_inline;_OS_BUILDS.clear()\n        _ma_ext=_SRP_OBSERVE_BUILD")
    suffix=once(suffix,"        _WL_EXT_FLAGS.clear();_srp_phase('complete')","        load_inline=_OS_ORIGINAL_LOAD\n        _WL_EXT_FLAGS.clear();_srp_phase('complete')")
    suffix=once(suffix,"report.update(cleanup_restored=True,","report.update(build_census=list(_OS_BUILDS),cleanup_restored=True,")
    pin={"candidate_pin":PIN,"authorization":"c4739708830cdb68ca138f83206de6945c4b0c82","scope":"outgoing seam requires owning runtime qualification; historical RMW PASS is not seam proof"}
    for name,value in (("_WL_CONFORMANCE",pin),("_WL_DEVICE_SHA",sha(device)),("_WL_CORE_SHA","")):
        node=exports.assignment(suffix,name);suffix=once(suffix,ast.get_source_segment(suffix,node),name+"="+repr(value))
    suffix+="\n"+SUPPORT
    core=sha(suffix)
    suffix=once(suffix,"_WL_CORE_SHA=''","_WL_CORE_SHA="+repr(core))
    source=device+MARKER+suffix
    ast.parse(source);assert len(source.encode())<1_000_000
    return source,dict(source_sha256=sha(source),bytes=len(source.encode()),control_sha256=PIN,device_sha256=sha(device),core_sha256=core,conformance=pin,prior_instrument_sha256=prior_manifest["source_sha256"],arms=["rawOFF1","rawOFF2","adapterOFF","adapterON"],seeds=[993811,993817,993821],rows=36,rounds=24,flush_bytes=256<<20,total_guard_s=295,build_census={"primary_native":5,"added_secondary_adapter":1,"original_secondary_factory_reused":True,"attention_factory_reused":True,"other_original_preparation_factories_unchanged":True},host_compilation="NOT_RUN; owning check only",private_check_only=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("root",type=Path);parser.add_argument("out",type=Path);args=parser.parse_args()
    source,manifest=build(args.root);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(source.encode());args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest))
