"""Held native-only direct-weight/control package; no compile/GPU/dispatch."""
import ast
import hashlib
from pathlib import Path
import tempfile
from g_expert_g224_prefix import function_source,once
from test_g_expert_g224_prefix import capture,BASE

ROOT=Path(__file__).resolve().parents[1]
PIN='a27a1baff54f5496e339044760d1a58927af8733fa6af6fadcc68c60dcf140bd'
PATH=ROOT/'docs/steering/evidence/2026-11-02-worker-direct-weight-registers/generated/candidate.py'

SUPPORT='''
_DWP_ON_BUILD=_ma_ext
_DWP_OFF_BUILD=_DW_PARENT_EXT
_MA_ALT_SRCS['dw_control']=MOE_CUDA_SRC
def _ma_ext(defines=None):
    defs=tuple(MOE_DEFINES if defines is None else defines)
    if '@dw_control' in defs:
        return _DWP_OFF_BUILD(tuple(v for v in defs if v!='@dw_control'))
    return _DWP_ON_BUILD(defs)

def _expert_pair_bind(state,arm):
    if type(arm) is not bool or '_expert_pair_arm_bound' in state:
        raise RuntimeError('DW_PAIR_INVALID_BIND')
    state['_expert_pair_arm']=arm
    state['_expert_pair_arm_bound']=arm
    for key in ('b4','main'):state['ma_ext'].pop(key,None)
    for tokens in (32,128):_fork_native_moe(tokens,state)
    return state
'''


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def build():
    candidate=PATH.read_bytes().decode();assert sha(candidate)==PIN
    selector=function_source(BASE,'_fork_native_moe')
    selector=once(selector,'    ext = state["ma_ext"].get(key)', '''    arm=state.get('_expert_pair_arm',True)
    if type(arm) is not bool or state.get('_expert_pair_arm_bound',arm) is not arm:
        raise RuntimeError('DW_PAIR_ARM_MUTATED')
    if not arm and "-DOKC_WGS=1" in defines:
        defines=tuple(defines)+('@dw_control',)
    ext = state["ma_ext"].get(key)''')
    source=candidate+'\n'+SUPPORT+'\n'+selector+'\n'
    ast.parse(source)
    return source,dict(source_sha256=sha(source),bytes=len(source.encode()),candidate=PIN,official_arm='direct_weight',dispatch_ready=False)


def selected():
    calls,e=capture(BASE,False);s,_=build();exec(s[len(BASE):],e)
    e['_MA_EXTS']={};e['_DW_EXTS']={};calls.clear();seen={};builder=e['_ma_ext']
    def observe(defines=None):
        mod=builder(defines);flags=tuple(defines);src=e['_DW_CUDA'] if '-DOKC_WGS=1' in flags else e['MOE_CUDA_SRC']
        for f in flags:
            if f.startswith('@'):src=e['_MA_ALT_SRCS'][f[1:]]
        seen[id(mod)]=dict(flags=list(flags),sha256=sha(src))
        return mod
    e['_ma_ext']=observe;records={}
    for arm in (False,True):
        st=dict(ma_ext={},ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=True)
        e['_expert_pair_bind'](st,arm)
        for t in (1,8,16,32,128,256):e['_fork_native_moe'](t,st)
        for k,mod in st['ma_ext'].items():records[str(int(arm))+':'+k]=seen[id(mod)]
    return records,calls
