"""Build one fixed owned SK10 treatment, with prepare-only qualification.

No comparative benchmark, candidate autotuning, runtime source I/O or dispatch.
"""
import argparse,ast,hashlib,importlib.util,json,re
from pathlib import Path

from proj_stage_xor import generate,PIN
from proj_stage_xor_release_body import BODY

AUGMENTED='b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e'
HELPER_BODY='92b16040b139ead412cc40c1d81e1f2ea36137ea7f22870036e5d6f06c1c1ba0'
HELPERS=('_sx_guard','_sx_storage','_sx_disjoint','_sx_reference','_sx_check','_sx_x','_sx_pack_key','_sx_layout','_sx_capture')


def audited_helpers(source):
    tree=ast.parse(source)
    value=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY')
    assert hashlib.sha256(value.encode()).hexdigest()==HELPER_BODY,'audited helper body drift'
    value=value.replace('@@PIN@@','{}').replace('@@MODEL@@','').replace('@@CODEC@@','')
    nodes=[n for n in ast.parse(value).body if isinstance(n,ast.FunctionDef) and n.name in HELPERS or
           isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_SX_TARGET' for t in n.targets)]
    assert len(nodes)==len(HELPERS)+1
    return '\n\n'.join(ast.get_source_segment(value,n) for n in nodes)+'\n'


def make(raw,augmented,helper_source):
    assert hashlib.sha256(raw).hexdigest()==PIN
    assert hashlib.sha256(augmented).hexdigest()==AUGMENTED
    c2=raw.decode('utf8').replace('\r\n','\n');aug=augmented.decode('utf8')
    assert aug.startswith(c2)
    helpers=audited_helpers(helper_source)
    pins=dict(control_raw=PIN,control_lf=hashlib.sha256(c2.encode()).hexdigest(),prototype=AUGMENTED,
              helper_body=HELPER_BODY,release_body=hashlib.sha256(BODY.encode()).hexdigest())
    suffix='\n# Fixed single-arm stage-local sparse repair; prepare gates only.\n_SXSR_PINS = '+repr(pins)+'\n'+helpers+'\n'+BODY
    # No new timing/codec machinery enters this production suffix.
    for n in ast.walk(ast.parse(suffix)):
        if isinstance(n,ast.Attribute):assert n.attr not in ('Event','elapsed_time','profiler')
        if isinstance(n,ast.Name):assert n.id not in ('_sx_bench','_sx_cold','_sx_score_model','_sx_encode')
    result=aug+suffix;tree=ast.parse(result)
    assert sum(isinstance(n,ast.FunctionDef) and n.name=='kernel_fn' for n in tree.body)==1
    # All treatment IDs must actually select STAGE+SX. Control IDs retain both
    # flags clear; full/NF differ only by the pre-existing FULL bit.
    meta=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SXS_CONFIG')
    cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SXS_CUDA')
    assert len(meta)==10
    for cfg in meta.values():
        assert len(cfg['ids'])==3 and len({i for pair in cfg['ids'] for i in pair})==6
        old=cfg['original_targs'].split(', ')
        for variant,pair in enumerate(cfg['ids']):
            for nearby,cfgid in enumerate(pair):
                args=old.copy();args[9]=str((int(old[9])&~32 if nearby else int(old[9]))|(4096 if variant else 0)|(8192 if variant==2 else 0))
                assert re.search(r'case '+str(cfgid)+r': launch_sk<'+re.escape(', '.join(args))+r'>',cuda),'stage dispatch flag/geometry drift'
    oldmath=next(n for n in ast.parse((Path(__file__).with_name('proj_sk_xor_release_body.py')).read_text(encoding='utf8')).body if isinstance(n,ast.Assign))
    oldbody=ast.literal_eval(oldmath.value)
    expected=next(n for n in ast.parse(oldbody.replace('_sxr_','_sxsr_').replace('SXRELEASE','SXSRELEASE')).body if isinstance(n,ast.FunctionDef) and n.name=='_sxsr_math')
    actual=next(n for n in ast.parse(BODY).body if isinstance(n,ast.FunctionDef) and n.name=='_sxsr_math')
    assert ast.dump(actual)==ast.dump(expected),'stage release weakened math gates'
    assert len(result.encode())<1000000
    return result,dict(pins=pins,bytes=len(result.encode()),sha256=hashlib.sha256(result.encode()).hexdigest(),
        fixed_targets=10,normal_prepare_calls=1,extra_native_modules=1,
        new_comparative_timings=False,H200_qualified=False,publication_authority='root only')


def build(root,helper_source=None):
    root=Path(root);p=root/'tools/proj_sk_xor_control.py'
    spec=importlib.util.spec_from_file_location('_release_restore_C2',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    raw=m.restore(root);aug,_,_=generate(raw)
    helper_source=(root/'tools/proj_make_bench_sk_xor_body.py').read_text(encoding='utf8') if helper_source is None else helper_source
    return make(raw,aug.encode(),helper_source)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--helper-body',type=Path);a=p.parse_args()
    result,manifest=build(a.root,a.helper_body.read_text(encoding='utf8') if a.helper_body else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(result.encode())
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,indent=2))
