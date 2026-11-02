"""Rebase frozen f042 paired machinery on repaired common OFF/ON; PRIVATE ONLY."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import g_short_rmw_check as prior
import g_common_release as qual
from g_common_release_budget import SUPPORT as MEMO_SUPPORT
from g_common_release_body import BODY as QUAL_BODY
from g_common_rmw_pair_body import SUPPORT
from g_integration_rmw_release import build as common, common_delta, COMMON


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def function(source,name):
    return ast.get_source_segment(source,next(n for n in ast.parse(source).body
        if isinstance(n,ast.FunctionDef) and n.name==name))


def build(root):
    root=Path(root)
    old,old_manifest=prior.build(root)
    assert sha(old)=='f0426f7160768de01997b271744542ae3442d0851a1c27e586ced322a8fc6864'
    marker='\n# PRIVATE CHECK ONLY: four-owner short-RMW diagnostic. NEVER SUBMIT.\n'
    device,suffix=old.split(marker)
    base=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')
    control,_=common(base,False);on,on_manifest=common(base,True)
    device,delta=common_delta(base,control,device)
    device=prior.once(device,' + ["wl_resources"],',',')
    device=qual.native_exports(qual.attention_exports(device))
    device=prior.once(device,' + ["qual_native_resources"],',' + ["wl_resources","qual_native_resources"],')
    suffix=prior.once(suffix,'_WL_PIN='+repr(prior.PIN),'_WL_PIN='+repr(COMMON))
    conformance={'candidate_pin':COMMON,'job_id':'c1931f0c9e834485',
        'scope':'common + ON single-arm qualification; OFF official tests required in owning paired receipt',
        'qualified_on_source_sha256':'978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a'}
    tree=ast.parse(suffix)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_WL_CONFORMANCE')
    suffix=prior.once(suffix,ast.get_source_segment(suffix,node),'_WL_CONFORMANCE='+repr(conformance))
    # Retain the old paired timing and numerical decision code.
    suffix=prior.once(suffix,'def _wl_prepare(weights):','def _wl_prepare(weights,state):\n    global _SRP_CACHE')
    suffix=prior.once(suffix,'    started=time.monotonic()','    started=_SRP_START')
    suffix=prior.once(suffix,'    state=_WL_PREPARE(weights);budget();weight_signature=_wl_weight_signature(weights)',
        '    budget();weight_signature=_wl_weight_signature(weights)')
    suffix=prior.once(suffix,'    resources=_srm_resource_gate(arms,budget)',
        "    _srp_phase('resources');resources=_srm_resource_gate(arms,budget)\n    common_resources=_srp_native_resources(arms,budget)")
    suffix=prior.once(suffix,'    empty_checks=_wl_empty(arms);budget()',
        "    _srp_phase('empty');empty_checks=_wl_empty(arms);budget()\n    occupied_empty_checks=_srp_occupied_empty(arms);budget()")
    suffix=prior.once(suffix,'                budget();base=_ab_inputs',
        "                _srp_phase('row');_SRP_CACHE={};budget();base=_ab_inputs")
    suffix=prior.once(suffix,'                budget()\n    expected=',
        '                _SRP_CACHE.clear();_SRP_CACHE=None;budget()\n    expected=')
    suffix=prior.once(suffix,'    graph.replay();torch.cuda.synchronize()\n    return graph,out',
        "    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])\n    graph.replay();torch.cuda.synchronize()\n    return graph,out")
    suffix=prior.once(suffix,"report={'schema':1", "report={'schema':2,'common_control':True,'actual_attention':list(_SRP_ACTUAL),'common_resources':common_resources,'occupied_to_empty_checks':occupied_empty_checks")
    start=suffix.index('    budget();blob=base64.b64encode')
    end=suffix.index('\ndef prepare(weights):',start)
    suffix=suffix[:start]+"    budget();_wl_check_owners([state,*arms],readonly);_check_guards()\n    return _wl_judged_state(state,arms[2]),report\n"+suffix[end:]
    suffix=prior.replace_function(suffix,'prepare','def prepare(weights):\n    return _srp_prepare(weights)')
    suffix=prior.once(suffix,'# Normal judge always receives the unchanged607 control modules.',
        '# Normal judge receives repaired-common OFF; private bindings are restored.')
    suffix=prior.once(suffix,"    if state.get('_wl_probe',False):state['_wl_attn_hits']+=1\n    return _WL_ATTN(data,state,qkv)",
        "    out=_WL_ATTN(data,state,qkv)\n    if state.get('_wl_probe',False):state['_wl_attn_hits']+=1\n    return out")
    run=function(suffix,'_wl_run')
    run=prior.once(run,"    try:out=_WL_KERNEL(data,state)\n    finally:state['_wl_probe']=False",
        "    audit=_att7();audit.qual_att7_begin()\n    try:out=_WL_KERNEL(data,state)\n    finally:\n        state['_wl_probe']=False\n        actual=list(audit.qual_att7_end())\n    record=_gq_attn_record(data,actual)\n    _SRP_ACTUAL.append([data['config']['tokens'],bool(state['_srm_on']),record])")
    suffix=prior.replace_function(suffix,'_wl_run',run)
    signature=function(MEMO_SUPPORT,'_gqb_signature')
    att_record=function(QUAL_BODY,'_gq_attn_record')
    support=signature+'\n'+att_record+'\n'+SUPPORT
    suffix+='\n_SRP_OBSERVE_BUILD=_ma_ext\n_SRP_OBSERVE_NATIVE=_fork_native_moe\n_SRP_OBSERVE_ATTN=_decode_attention7\n'
    suffix+='_ma_ext=_WL_MA_EXT\n_fork_native_moe=_WL_NATIVE\n_decode_attention7=_WL_ATTN\n'
    old_device=next(n for n in ast.parse(suffix).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_WL_DEVICE_SHA')
    suffix=prior.once(suffix,ast.get_source_segment(suffix,old_device),'_WL_DEVICE_SHA='+repr(sha(device)))
    core=sha(support+suffix)
    old_core=next(n for n in ast.parse(suffix).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_WL_CORE_SHA')
    suffix=prior.once(suffix,ast.get_source_segment(suffix,old_core),'_WL_CORE_SHA='+repr(core))
    at=suffix.index('def _wl_storage(')
    suffix=suffix[:at]+support+'\n'+suffix[at:]
    result=device+marker+suffix
    ast.parse(result);assert len(result.encode())<1_000_000
    pins={'_WL_PIN':COMMON,'_WL_CONFORMANCE':conformance,'_WL_DEVICE_SHA':sha(device),'_WL_CORE_SHA':core}
    manifest=dict(source_sha256=sha(result),bytes=len(result.encode()),capsule_pins=pins,
        historical_harness_sha256=sha(old),control_sha256=COMMON,
        fixed_on_raw_sha256=on_manifest['source_sha256'],common_deltas=delta,
        seeds=[993811,993817,993821],rows=36,arms=4,rounds=24,flush_bytes=256<<20,
        normal_judged='repaired-common OFF; all common progress/cooperative fixes retained',
        support_sha256=sha(support),private_check_only=True,total_guard_s=295)
    return result,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    s,m=build(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(s.encode())
    a.out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2))
