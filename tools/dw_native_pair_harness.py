"""Narrow held adaptation of the existing whole-layer paired instrument."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import dw_native_pair_package as pair
import expert_top100_harness as old
import expert_top100_pair_package as old_pair

ROOT=pair.ROOT
once=old.once
sha=old.sha


def build():
    package,pm=pair.build();selection,calls=pair.selected()
    previous,prior=old.build();prefix,_=old_pair.build()
    core=previous[previous.index('\nimport math,time\n',len(prefix)):]
    core=once(core,'    src=MOE_CUDA_SRC',"    src=_DW_CUDA if '-DOKC_WGS=1' in flags else MOE_CUDA_SRC")
    core=once(core,'    for i,arm in enumerate(arms):\n        _expert_pair_bind(arm,i>=2)', '''    for i,arm in enumerate(arms):
        arm.pop('_expert_pair_arm',None)
        arm.pop('_expert_pair_arm_bound',None)
        _expert_pair_bind(arm,i>=2)''')
    core=once(core,"        assert not state.get('_expert_pair_arm',False),'EXPERT_OFFICIAL_ARM'", "        assert state.get('_expert_pair_arm',True) is True,'EXPERT_OFFICIAL_ARM'\n        _expert_pair_bind(state,True)")
    core=core.replace("official_arm='exact607'","official_arm='direct_weight'")
    core=once(core,"print('EXPERT_TOP100_REPORT '+blob,flush=True)","print('DW_NATIVE_PAIR_REPORT '+blob,flush=True)")
    pins=dict(control=prior['pins']['control'],candidate=pair.PIN,pair=pm['source_sha256'],core=sha(core))
    source=package+'\n_EH_OFFLINE_HOLD=True\n_EH_PINS='+repr(pins)+'\n_EH_SELECTED='+repr(selection)+'\n'+core
    ast.parse(source);assert len(source.encode())<1_000_000
    reader=Path(__file__).with_name('dw_native_pair_read.py')
    manifest=dict(source_sha256=sha(source),bytes=len(source.encode()),pins=pins,selected=selection,
        reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),official_arm='direct_weight',
        oracle=prior['oracle'],seeds=prior['seeds'],rows=36,placements=4,rounds=24,flush_bytes=256<<20,
        total_guard_s=295,target_fit='UNMEASURED',offline_hold=True,dispatch_ready=False,gpu_qualified=False,
        pending=['separate captain confirmation allocation if first check warrants','target additional two-control build and295s fit',
            'target graph/admission/owner/numerical/official13-12-12 and matched timing gates'])
    return source,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s,m=build();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(s.encode())
    a.out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
