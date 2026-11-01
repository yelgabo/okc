"""PRIVATE CHECK ONLY: exact607 stage-on control vs short final-RMW representation.

No dispatch. Normal judged entry remains607 control. Reuses audited whole-G
owner/FP32-reference/cold four-arm machinery; no old stage-off control.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from g_short_final_rmw import build as patch_rmw, PIN
from g_short_rmw_check_body import OVERRIDES


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b)


def load(path):
    spec=importlib.util.spec_from_file_location('_srm_'+path.stem,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def replace_function(source,name,new):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[new+'\n']
    return ''.join(lines)


def build(root):
    original=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
    receipt_path=root/'muse_decode_layer/profiles/scored.607eb52bdaf24160.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    result=receipt['result']
    tests=[r for r in result['records'] if r.get('event')=='test']
    assert receipt['id']=='607eb52bdaf24160' and receipt['board']=='muse_decode_layer' and receipt['mode']=='run'
    assert receipt['status']=='done' and result['passed'] and result['checks']['ok'] and not result['flags']
    assert len(tests)==13 and all(r['status']=='pass' for r in tests),'607 conformance receipt required'
    source,_=patch_rmw(original,False)
    # Per-owner compile key, not a per-input/time-selected dispatch.
    source=once(source,'    ext = state["ma_ext"].get(key)',
        '    if state.get("_srm_on",False) and key in ("tiny","tiny2"):\n'
        '        defines=tuple(defines)+("-DOKC_SHORT_FINAL_RELAXED=1",)\n'
        '    ext = state["ma_ext"].get(key)')
    resource=load(root/'tools/moe_make_sc6_compare.py').RESOURCE.replace('sc6_resources','wl_resources')
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            value=ast.literal_eval(n.value)+resource
            edits.append((n.lineno-1,n.end_lineno,ast.unparse(n.targets[0])+' = r"""'+value+'"""\n'))
    assert len(edits)==2
    for a,b,s in reversed(edits):lines[a:b]=[s]
    source=''.join(lines)
    source=once(source,'std::vector<int64_t> okc_attrs();','std::vector<int64_t> okc_attrs();\nstd::vector<int64_t> wl_resources();')
    source=once(source,'functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),',
        'functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []) + ["wl_resources"],')
    gate=load(root/'tools/dl_g1_layer_gate.py')
    pub=ast.parse((root/'tools/dl_make_bench_pub.py').read_text(encoding='utf-8'))
    reference=next(ast.literal_eval(n.value) for n in pub.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY')
    funcs={n.name:ast.get_source_segment(reference,n) for n in ast.parse(reference).body if isinstance(n,ast.FunctionDef)}
    helpers='\n\n'.join(funcs[name] for name in gate.NAMES)
    helpers=once(helpers,'    if not torch.isfinite(o).all():',
        '    import math\n    assert math.isfinite(float(tol)) and bool(torch.isfinite(r).all()),"SRM_REFERENCE_NONFINITE"\n    if not torch.isfinite(o).all():')
    helpers=once(helpers,'    def bound(e):\n',
        '    def bound(e):\n        import math\n        assert math.isfinite(e),"SRM_ADAPTIVE_NONFINITE"\n')
    owners=(root/'tools/dl_g1_layer_owners.py').read_text(encoding='utf-8')
    owners=once(owners,"arm['ma_ext'].pop('main',None);arm['ma_ext'].pop('b4',None)",
                       "arm['ma_ext'].pop('tiny',None);arm['ma_ext'].pop('tiny2',None)")
    owners=owners.replace("arm['_wl_stage_on']=bool(on)","arm['_srm_on']=bool(on)")
    body=(root/'tools/dl_g1_layer_body.py').read_text(encoding='utf-8')
    overrides={n.name:ast.get_source_segment(OVERRIDES,n) for n in ast.parse(OVERRIDES).body if isinstance(n,ast.FunctionDef)}
    for name in ('_fork_native_moe','_wl_judged_state','_wl_empty'):
        body=replace_function(body,name,overrides[name])
    body=body.replace('_WL_SEEDS=(391031,391032,391033)','_WL_SEEDS=(993811,993817,993821)')
    body=body.replace('_WL_AFFECTED=(32,128,256)','_WL_AFFECTED=(1,8)')
    body=body.replace('math.log(1.002)','math.log(1.01)')
    # Full all-row raw factors are the decision; affected-only factors remain diagnostic.
    body=body.replace('min(raw_gains)>max(0.,2*null)','min(raw_gains)>max(math.log(1.01),2*null)')
    body=once(body,'min(gains)>max(math.log(1.01),2*null) and min(raw_gains)', 'min(raw_gains)')
    body=body.replace("assert time.monotonic()-started<300,'WL_INCOMPLETE_BEFORE_300S'",
                      "assert time.monotonic()-started<295,'SRM_INCOMPLETE_BEFORE_295S'")
    a=body.index('    resources={}\n',body.index('def _wl_prepare'))
    b=body.index('    empty_checks=',a)
    body=body[:a]+'    resources=_srm_resource_gate(arms,budget)\n'+body[b:]
    body=body.replace("'two_flags_per_main_B4_dedup'","'two_short_RMW_flags_all_WGS_stage_on'")
    body=body.replace("'judged_stage_on':True", "'judged_stage_on':True,'judged_short_relaxed':False")
    body=body.replace("print('WL_STAGE_REPORT '+blob", "print('SRM_REPORT '+blob")
    # Inner and final budget checks; no row can silently complete past deadline.
    body=body.replace('                for _ in range(8):\n','                for _ in range(8):\n                    budget()\n')
    body=body.replace('                for rep in range(24):\n','                for rep in range(24):\n                    budget()\n')
    body=body.replace('                    for i in order:\n                        data[i]', '                    for i in order:\n                        budget()\n                        data[i]')
    body=body.replace('samples[i].append(a.elapsed_time(z)*1000)',
        "elapsed=a.elapsed_time(z)*1000\n                        assert math.isfinite(elapsed) and elapsed>0,'SRM_BAD_TIMING'\n                        samples[i].append(elapsed)")
    body=body.replace('import statistics,random,json,zlib,base64','import statistics,random,json,zlib,base64,math')
    body=once(body,"            cost=[46*pair['sliding'][i]+16*pair['full'][i] for i in range(4)]",
        "            assert all(len(v)==4 and all(math.isfinite(x) and x>0 for x in v) for v in pair.values()),'SRM_BAD_SUMMARY_TIMING'\n            cost=[46*pair['sliding'][i]+16*pair['full'][i] for i in range(4)]")
    # Identical natural routing decisions are required across all four owners.
    for anchor in ('                out_intervals=', '                for i in range(4):_wl_copy(data[i],base);'):
        body=once(body,anchor,"                assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]),'SRM_ROUTING_ARM_MISMATCH'\n"+anchor)
    core=owners+body+OVERRIDES
    config=dict(_WL_PART='all',_WL_LOADS=(1,8,16,32,128,256),_WL_PIN=PIN,
                _WL_CONFORMANCE={'candidate_pin':PIN,'job_id':'607eb52bdaf24160',
                    'receipt_sha256':sha(receipt_path.read_text(encoding='utf-8')),
                    'scope':'previous public incumbent, current official tests required'},
                _WL_DEVICE_SHA=sha(source),_WL_CORE_SHA=sha(core))
    # Body's captured _WL_NATIVE must remain the per-owner-policy source function.
    output=source+'\n# PRIVATE CHECK ONLY: four-owner short-RMW diagnostic. NEVER SUBMIT.\n'+gate.PRELUDE+'\n'+helpers+'\n'+owners+'\n'
    output+='\n'.join(name+'='+repr(value) for name,value in config.items())+'\n'
    output+=overrides['_srm_resource_gate']+'\n'+body
    output=once(output,'# okc: problem=muse_decode_layer',
                '# okc: problem=muse_decode_layer\n# PRIVATE CHECK ONLY: FOUR-ARM DIAGNOSTIC. NEVER SUBMIT.')
    ast.parse(output)
    assert len(output.encode())<1_000_000
    manifest=dict(source_sha256=sha(output),bytes=len(output.encode()),control_sha256=PIN,
        normal_judged='607 control; short flag0; WGS stage retirement retained',
        affected=(1,8),seeds=(993811,993817,993821),rows=36,arms=4,
        owners_sha256=sha(owners),body_sha256=sha(body),resource_sha256=sha(resource),
        generation_only=True,gpu_execution=False,private_check_only=True)
    return output,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();source,manifest=build(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))
