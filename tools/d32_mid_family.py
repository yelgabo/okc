"""Fixed host-only17..32 MID-family integration from immutable ba9; no dispatch."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from g_expert_g224_prefix import BASE,BASE_SHA,NATIVE_SHA,sha,once,function_source

ROOT=Path(__file__).resolve().parents[1]
MID_NATIVE='b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'

ATTEST='''
def _d32_mid_attest(st):
    import hashlib
    defines=tuple(MOE_MID_DEFINES)+tuple(st['ma_sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defines:defines+=('-DOKC_DQ7=1',)
    if defines!=_D32_MID_EXPECTED_DEFINES:raise RuntimeError('D32_MID_FLAG_DRIFT')
    source=MOE_CUDA_SRC
    for flag in defines:
        if flag.startswith('@'):source=_MA_ALT_SRCS[flag[1:]]
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest!=_D32_MID_SOURCE:raise RuntimeError('D32_MID_SOURCE_DRIFT')
    ext=_fork_native_moe(32,st)
    if ext is not st['ma_ext']['mid'] or ext is not _fork_native_moe(16,st):
        raise RuntimeError('D32_MID_SELECTION_DRIFT')
    if ext._module is not _MA_EXTS.get(defines):raise RuntimeError('D32_MID_CACHE_IDENTITY')
    if _fork_native_moe(33,st) is not st['ma_ext']['main']:raise RuntimeError('D32_MID_M33_DRIFT')
    query=getattr(ext,'okc_attrs',None)
    attrs=list(query()) if callable(query) else None
    return dict(tokens=32,key='mid',source_sha256=digest,defines=list(defines),
        module=getattr(ext._module,'__name__',type(ext._module).__name__),resource_query=attrs)
'''


def generate(out):
    from g_native_patch_ladder import flags
    base=(ROOT/BASE).read_text();assert sha(base)==BASE_SHA
    candidate=once(base,'MOE_MID_MAX_M = 16','MOE_MID_MAX_M = 32')
    old_tasks=function_source(base,'_fork_build_tasks')
    new_tasks=once(old_tasks,'(1, 8, 16, 32, 128)','(1, 8, 32, 128)')
    candidate=once(candidate,old_tasks,new_tasks)
    old_prepare=function_source(base,'_ma_prepare')
    new_prepare=once(old_prepare,"{'tiny','tiny2','mid','b4','main'}","{'tiny','tiny2','mid','main'}")
    new_prepare=once(new_prepare,"    dev = weights['w13'].device","    mid_selection = _d32_mid_attest(st)\n    dev = weights['w13'].device")
    new_prepare=once(new_prepare,"    st['_fork_moe_status']='native repaired:","    print('D32_MID_SELECTION',mid_selection,flush=True)\n    st['_fork_moe_status']='native repaired:")
    candidate=once(candidate,old_prepare,new_prepare)
    defines=tuple(flags(base)['MOE_MID_DEFINES'])+('-DOKC_PRMTSC=1',)
    if flags(base)['DQ7'] and '-DOKC_DQ7=1' not in defines:defines+=('-DOKC_DQ7=1',)
    candidate+='\n_D32_MID_EXPECTED_DEFINES='+repr(defines)+'\n_D32_MID_SOURCE='+repr(MID_NATIVE)+'\n'+ATTEST
    ast.parse(candidate);assert len(candidate.encode())<1_000_000
    reverse=candidate[:candidate.index('\n_D32_MID_EXPECTED_DEFINES=')]
    reverse=once(reverse,new_prepare,old_prepare);reverse=once(reverse,new_tasks,old_tasks)
    reverse=once(reverse,'MOE_MID_MAX_M = 32','MOE_MID_MAX_M = 16')
    assert reverse==base
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('candidate.py',candidate),('host.diff',''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='ba9',tofile='d32-mid-fixed')))]:
        (out/name).write_text(value,encoding='utf8',newline='\n')
    m=dict(base_sha256=BASE_SHA,candidate_sha256=sha(candidate),candidate_bytes=len(candidate.encode()),
        mid_native_sha256=MID_NATIVE,main_native_sha256=NATIVE_SHA,mid_defines=list(defines),
        build_representatives=[1,8,32,128],prepared_keys=['tiny','tiny2','mid','main'],exact_reversal=True,
        shader_edits=False,packing_edits=False,scratch_edits=False,gpu=False,dispatch=False)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return candidate,m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.out)[1],indent=2))
