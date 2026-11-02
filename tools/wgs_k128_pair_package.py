"""Held state-local K256/K128 packaging. No compiler, GPU, or judge actions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import wgs_k128_full_reuse as fixed
from g_expert_g224_prefix import function_source

ROOT=Path(__file__).resolve().parents[1]
FROZEN='440f41b118eeb3676867e15d109c2d5f7da0213873f84a17c9d68d1af868703f'
PAYLOAD=ROOT/'docs/steering/evidence/2026-11-02-worker-wgs-k128-full-reuse/generated/candidate.py'

SUPPORT=r'''
_K128_PAIR_BUILD_ON = _ma_ext
_K128_PAIR_BUILD_OFF = _K128_PARENT_EXT
_MA_ALT_SRCS['k128_pair_off'] = MOE_CUDA_SRC

def _ma_ext(defines=None):
    defs=tuple(MOE_DEFINES if defines is None else defines)
    if '@k128_pair_off' in defs:
        return _K128_PAIR_BUILD_OFF(tuple(d for d in defs if d!='@k128_pair_off'))
    return _K128_PAIR_BUILD_ON(defs)

def _k128_pair_signature(state):
    return tuple((str(t.device),int(t.untyped_storage().data_ptr()),
        int(t.untyped_storage().nbytes()),tuple(t.shape),str(t.dtype),int(t._version))
        for t in state['ma_w'])

def _k128_pair_assert(state):
    arm=state.get('_expert_pair_arm',True)
    if type(arm) is not bool:
        raise RuntimeError('K128_PAIR_ARM_TYPE')
    if '_expert_pair_arm_bound' in state:
        if state['_expert_pair_arm_bound'] is not arm:
            raise RuntimeError('K128_PAIR_REBOUND')
        if state['_k128_pair_layout'] != ('K128' if arm else 'K256'):
            raise RuntimeError('K128_PAIR_LAYOUT')
        if state['_k128_pair_weights'] != _k128_pair_signature(state):
            raise RuntimeError('K128_PAIR_WEIGHT_OWNER_OR_VERSION')
    elif not arm:
        raise RuntimeError('K128_PAIR_UNBOUND_OFF')
    return arm

def _k128_pair_mark(state,arm):
    state['_expert_pair_arm']=arm
    state['_expert_pair_arm_bound']=arm
    state['_k128_pair_layout']='K128' if arm else 'K256'
    state['_k128_pair_weights']=_k128_pair_signature(state)
    return state

def _k128_pair_repack(weights,state,arm):
    pack=_wgs_copy if arm else _k128_off_wgs_copy
    rowpack=_kperm_rows if arm else _k128_off_kperm_rows
    dev=weights['w13'].device
    r=torch.arange(4096,device=dev).view(-1,8)
    rowsa=torch.stack((r,r+4096),1).reshape(-1)
    rowsb=torch.arange(4096,device=dev)
    wa=torch.cat([pack(weights['w13'][e].view(torch.uint8),weights['w13_sf'][e],rowpack(rowsa),state['ma_sc6']) for e in range(32)])
    wb=torch.cat([pack(weights['w2'][e].view(torch.uint8),weights['w2_sf'][e],rowsb,state['ma_sc6']) for e in range(32)])
    return wa,wb

def _expert_pair_bind(state,arm,weights):
    # Called before capture. All per-arm weight storage is independently owned;
    # no process-global packer/native switch occurs during preparation or timing.
    if type(arm) is not bool or '_expert_pair_arm_bound' in state:
        raise RuntimeError('K128_PAIR_INVALID_BIND')
    if not state['_fork_native_eligible'] or not state['ma_sc6']:
        raise RuntimeError('K128_PAIR_NATIVE_SC6_REQUIRED')
    if len(state['ma_w'])!=14:
        raise RuntimeError('K128_PAIR_WEIGHT_ABI')
    state['ma_w']=[t.clone() for t in state['ma_w'][:12]]+list(_k128_pair_repack(weights,state,arm))
    factory=_ma_make_scratch if arm else _k128_off_ma_make_scratch
    mark=len(_GUARDS)
    state['ma_scratch']=[_guarded(t) for t in factory(weights['w13'].device)]
    state['_wl_guard_storage']+=_GUARDS[mark:]
    state['ma_ext']={}
    _k128_pair_mark(state,arm)
    for tokens in (1,8,16,32,128,256):_fork_native_moe(tokens,state)
    state['_k128_pair_modules']={key:id(ext) for key,ext in state['ma_ext'].items()}
    return state
'''


def build():
    candidate=PAYLOAD.read_bytes().decode()
    assert hashlib.sha256(candidate.encode()).hexdigest()==FROZEN
    base=(ROOT/fixed.BASE).read_text()
    assert fixed.sha(base)==fixed.BASE_SHA
    names=('_kperm_src','_kperm_codes','_kperm_rows','_wgs_copy','_ma_make_scratch')
    mapping={n:'_k128_off'+n for n in names}
    closure=[]
    for name in names:
        fn=function_source(base,name)
        fn=re.sub(r'\b('+'|'.join(map(re.escape,names))+r')\b',lambda m:mapping[m[0]],fn)
        closure.append(fn)
    selector=function_source(base,'_fork_native_moe')
    selector=fixed.once(selector,'    sfx = tuple(state["ma_sfx"])','    arm = _k128_pair_assert(state)\n    sfx = tuple(state["ma_sfx"])')
    selector=fixed.once(selector,'    ext = state["ma_ext"].get(key)','    if not arm and "-DOKC_WGS=1" in defines:\n        defines=tuple(defines)+("@k128_pair_off",)\n    ext = state["ma_ext"].get(key)')
    selector=fixed.once(selector,'    return ext', '''    if '_k128_pair_modules' in state and state['_k128_pair_modules'].get(key)!=id(ext):
        raise RuntimeError('K128_PAIR_MODULE_OWNER')
    return ext''')
    source=candidate+'\n'+'\n\n'.join(closure)+'\n'+SUPPORT+'\n'+selector+'\n'
    source+='''
_K128_PAIR_PRODUCTION_PREPARE=prepare
def prepare(weights):
    raise RuntimeError('K128_PAIR_PACKAGE_HOLD_NO_MEASUREMENT_RUNTIME')
'''
    ast.parse(source)
    assert len(source.encode())<1_000_000
    return source,dict(source_sha256=fixed.sha(source),bytes=len(source.encode()),frozen_candidate_sha256=FROZEN,
        control_sha256=fixed.BASE_SHA,native_control_sha256=fixed.NATIVE_SHA,native_candidate_sha256=fixed.sha(fixed.transform(fixed.literal(base,'MOE_CUDA_SRC'))),
        default_arm='candidate',packing='state-local K256/OFF or K128/ON dependency closures',
        status='HELD_PACKAGE_ONLY',dispatch_ready=False,gpu=False,compile=False,
        remaining=['adapt existing full-layer prepare to clear inherited binding before private clone',
            'retain candidate official state and validated cleanup', 'fresh full-generated-entry CPU failures/owner negatives',
            'strict owning reader/manifest binding', 'prospective 295s prepare fit and captain allocation'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s,m=build();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(s.encode())
    a.out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2))
