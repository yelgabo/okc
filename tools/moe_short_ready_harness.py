"""Exact selected-helper pair for the single short-ready representation check."""
import argparse
import ast
import json
from pathlib import Path

import expert_top100_moe_adapter as adapter
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings
from cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, ROOT, sha
from moe_short_ready import transform, HELPER_PIN
from moe_factor_y_harness import assignment, progress_runtime


def build(*,target_proposal=True):
    assert target_proposal is True
    prefix,_=adapter.build()
    assert sha(prefix)==ADAPTER_PIN
    helper=bindings(BASE.read_text())['_MA_ALT_SRCS']['moe_helper0']
    candidate,edits=transform(helper)
    experiment='moe_short_ready'
    prefix=assignment(prefix,'_EXPERT_PAIR_EDITS',[])
    prefix=assignment(prefix,'EXPERT_MOE_IDENTITIES',(experiment,BASE_PIN,HELPER_PIN,sha(candidate)))
    prefix=prefix.replace("if arm and key in ('b4', 'main'):","if arm and key in ('tiny', 'tiny2'):")
    prefix=prefix.replace('expert_top100_pregamma',experiment)
    prefix+='\n_MA_ALT_SRCS["moe_short_ready"] = '+repr(candidate)+'\n'
    reference=(ROOT/'muse_moe_w4a16/reference.py').read_text()
    nvfp4=(ROOT/'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference)==REFERENCE_PIN and sha(nvfp4)==NVFP4_PIN
    route_source='\n\n'.join(('D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0,32), "full": (0,256)}',
                              extract(nvfp4,'outlier_channels'),extract(reference,'generate_input')))
    calls,routes=capture(prefix)
    revised,revised_routes=capture(prefix.replace("arm = state.get('_expert_pair_arm', False)",'arm = True'))
    assert routes==revised_routes
    calls+=[c for c in revised if c not in calls]
    assert len(calls)==7
    descriptors=[]
    for call in calls:
        item=dict(call)
        for key in ('cpp_sources','cuda_sources'):item[key]=sha(item[key])
        descriptors.append(item)
    contract=(ROOT/'tools/expert_top100_moe_harness_contract.py').read_text()
    gate_source=(ROOT/'tools/moe_short_gate.py').read_text()
    node=next(n for n in ast.parse(gate_source).body if isinstance(n,ast.FunctionDef) and n.name=='short_decision')
    contract+='\n'+ast.get_source_segment(gate_source,node)+'\ndecision=short_decision\n'
    runtime=progress_runtime((ROOT/'tools/expert_top100_moe_harness_runtime.py').read_text())
    old="    report['factor_y_decision']=factor_y_decision(rows,decision(rows))"
    assert runtime.count(old)==1
    runtime=runtime.replace(old,'')
    node=next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name=='factor_y_decision')
    runtime=runtime.replace(ast.get_source_segment(runtime,node),'')
    old='            counters[\'observations\'] += 1'
    new="""            for state in states:
                require(int(torch.count_nonzero(state['ma_scratch'][0][136:136+state['nsm']]))==0,'completion flags reset')
                counters['flag_zero_checks']=counters.get('flag_zero_checks',0)+1
            counters['observations'] += 1"""
    assert runtime.count(old)==1
    runtime=runtime.replace(old,new)
    runtime=runtime.replace('MOE_FACTOR_Y','MOE_SHORT_READY').replace('_FY','_SR').replace('_fy_','_sr_')
    identity=sha(prefix+route_source+contract+runtime+json.dumps(descriptors,sort_keys=True))
    source=prefix+'\n'+route_source+'\n'+contract+'\n'
    source+=f'H_RUNTIME_ID={identity!r}\nH_HOLD=False\nH_BUILD_CALLS={descriptors!r}\n'+runtime
    ast.parse(source)
    manifest=dict(experiment='MOE-SHORT-READY-01',source_sha256=sha(source),bytes=len(source.encode()),runtime_id=identity,
                  control_native_sha256=NATIVE_PIN,control_helper_sha256=HELPER_PIN,candidate_native_sha256=sha(candidate),
                  official_arm='candidate2',hold=False,compiler_calls=descriptors,routes=routes,
                  adapter_control_sha256=ADAPTER_PIN,contract_sha256=sha(contract),runtime_sha256=sha(runtime),
                  reference_sha256=REFERENCE_PIN,nvfp4_sha256=NVFP4_PIN,public_submission=False,target_qualified=False)
    assert len(source.encode())<1_000_000
    return source,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source,manifest=build();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(source);args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
