"""Build private stage-retirement component A/B; never dispatches or submits."""
import argparse,ast,hashlib,importlib.util,sys
from pathlib import Path

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


PURE_PREPARE = '''
_stage_validate_prepare = prepare
def prepare(weights):
    state = _stage_validate_prepare(weights)
    assert state['lo'] == 0 and state['cnt'] == 32
    assert not PDL and not state.get('ps')
    flags = tuple(_main_defines(state['sfx'])) + tuple(state['sfxm'])
    assert {'-DOKC_WGS=1','-DOKC_SC6W=1','-DOKC_PRMTSC=1'} <= set(flags)
    resources = _ext(flags).sc6_resources()
    assert len(resources) == 12 and min(resources[4], resources[10]) >= 1
    print('ST_PURE_ACTIVE', flags, resources, 'stage-retire candidate, component check only', flush=True)
    return state
'''

def build(root,upstream,body,candidate_only=False):
    sys.path.insert(0,str(root/'tools'))
    make=load(root/'tools/moe_make_sc6_compare.py','_stage_sc6_make').make
    stage=load(root/'tools/dl_g1_stage_retire.py','_stage_patch')
    baseline = '' if candidate_only else (root/'muse_moe_w4a16/baseline.py').read_text(encoding='utf8')
    source=make(upstream.read_text(encoding='utf8'),baseline,'' if candidate_only else body)
    tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA_SRC')
    old=ast.literal_eval(n.value);new=stage.patch_device(old,switch=not candidate_only)
    assert new.count('OKC_STAGE_RETIRE')==(0 if candidate_only else 2) and '-DOKC_STAGE_NULL' not in new
    lines=source.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['CUDA_SRC = '+repr(new)+'\n'];out=''.join(lines)
    if candidate_only:
        assert out.count('_SC_BASELINE = ') == 1
        out = out[:out.index('_SC_BASELINE = ')] + PURE_PREPARE
    ast.parse(out);assert len(out.encode())<1_000_000
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream',type=Path);p.add_argument('--candidate-only',action='store_true');a=p.parse_args()
    upstream=a.upstream or a.root/'.build/today/kel/kernels/muse_moe_w4a16.py'
    s=build(a.root,upstream,Path(__file__).with_name('dl_g1_stage_ab_body.py').read_text(encoding='utf8'),a.candidate_only);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(s.encode());print('STAGE_AB',len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
