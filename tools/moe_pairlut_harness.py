"""Compare an exact pair lookup and its lossless WGS packing with the incumbent."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import expert_top100_moe_adapter as adapter
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings
from cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, ROOT, sha
from moe_pairlut import transform,table


def assignment(source, name, value):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
    old = ast.get_source_segment(source, node)
    assert source.count(old) == 1
    return source.replace(old, name + ' = ' + repr(value))


def build(*, target_proposal=True):
    assert target_proposal is True
    prefix, ignored = adapter.build()
    assert sha(prefix) == ADAPTER_PIN
    native = bindings(BASE.read_text())['MOE_CUDA_SRC']
    assert sha(native) == NATIVE_PIN
    candidate, edits = transform(native)
    experiment = 'moe_pairlut'
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', edits)
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', (experiment, BASE_PIN, NATIVE_PIN, sha(candidate)))
    prefix = prefix.replace('expert_top100_pregamma', experiment)
    prefix = prepare_codec(prefix)
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join((
        'D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
        extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    calls, routes = capture(prefix)
    alternatives = dict(bindings(prefix)['_MA_ALT_SRCS'])
    alternatives[experiment] = candidate
    candidate_prefix = prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True')
    candidate_calls, candidate_routes = capture(candidate_prefix + '\n_MA_ALT_SRCS = ' + repr(alternatives))
    assert candidate_routes == routes
    calls += [call for call in candidate_calls if call not in calls]
    assert len(calls) == 7
    descriptors = []
    for call in calls:
        item = dict(call)
        for key in ('cpp_sources', 'cuda_sources'):
            item[key] = sha(item[key])
        descriptors.append(item)
    contract = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    runtime = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    runtime = progress_runtime(runtime)
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = False\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    assert len(source.encode()) < 1_000_000
    ast.parse(source)
    manifest = dict(experiment=experiment, source_sha256=sha(source), bytes=len(source.encode()),
                    runtime_id=identity, control_native_sha256=NATIVE_PIN, candidate_native_sha256=sha(candidate),
                    official_arm='candidate2', hold=False, compiler_calls=descriptors, routes=routes,
                    adapter_control_sha256=ADAPTER_PIN, contract_sha256=sha(contract), runtime_sha256=sha(runtime),
                    reference_sha256=REFERENCE_PIN, nvfp4_sha256=NVFP4_PIN,
                    public_submission=False, target_qualified=False,
                    pair_table_bytes=65536,pair_table_sha256=__import__('hashlib').sha256(__import__('struct').pack('<16384i',*table())).hexdigest(),
                    packing_sha256={n.name:sha(ast.get_source_segment(prefix,n)) for n in ast.parse(prefix).body
                                    if isinstance(n,ast.FunctionDef) and n.name in ('_pack_codes_nat_pairlut','_wgs_copy_pairlut','_fork_native_ma_prepare')})
    return source, manifest


def prepare_codec(prefix):
    nodes=ast.parse(prefix).body
    pack=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_pack_codes_nat')
    packing=ast.get_source_segment(prefix,pack).replace('_pack_codes_nat','_pack_codes_nat_pairlut')
    old="""            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]"""
    assert packing.count(old)==1
    packing=packing.replace(old,'            word |= code << (Rg*8+up*4)')
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_wgs_copy')
    copy=ast.get_source_segment(prefix,node).replace('_wgs_copy','_wgs_copy_pairlut').replace('_pack_codes_nat(','_pack_codes_nat_pairlut(')
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_ma_prepare')
    oldprep=ast.get_source_segment(prefix,node)
    assert oldprep.count('_wgs_copy(')==2
    prep=oldprep.replace('_wgs_copy(',"(_wgs_copy_pairlut if st.get('_expert_pair_arm',False) and st['ma_sc6'] else _wgs_copy)(")
    anchor='    st["ma_scratch"] = '
    assert prep.count(anchor)==1
    prep=prep.replace(anchor,"    if st.get('_expert_pair_arm',False) and st['ma_sc6']:\n        st['ma_w'][6]=torch.cat((st['ma_w'][6],torch.tensor(_PAIR_WORDS,dtype=torch.int32,device=dev)))\n"+anchor)
    prefix=prefix.replace(oldprep,prep)
    return prefix+'\n'+packing+'\n'+copy+'\n_PAIR_WORDS = '+repr(tuple(table()))+'\n'


def progress_runtime(runtime):
    def once(old,new):
        nonlocal runtime
        assert runtime.count(old)==1,old
        runtime=runtime.replace(old,new)
    once('def prepare(weights):','def _pl_prepare_inner(weights):')
    once("    require(descriptor in H_BUILD_CALLS, 'exact compiler source/flags/name/exports')",
         "    require(descriptor in H_BUILD_CALLS, 'exact compiler source/flags/name/exports')\n    _PL_BUILDS.append(H_BUILD_CALLS.index(descriptor))")
    once('        owners = _h_ownership(states)',
         '        owners = _h_ownership(states)\n'
         '        require(sorted(set(_PL_BUILDS))==list(range(7)),"all registered builds observed")\n'
         '        require([s["ma_w"][6].numel() for s in states]==[256,256,16640,16640],"bound generic LUT owners")\n'
         '        require(all(torch.equal(s["ma_w"][6][:256],states[0]["ma_w"][6]) for s in states),"original LUT prefix")')
    once('owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters)',
         'owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters,\n'
         '                      loaded_builds=sorted(set(_PL_BUILDS)),table_words=[s["ma_w"][6].numel() for s in states])')
    once('    report = None\n',
         '    report = None\n    _PL_LIVE.update(rows=rows,counters=counters,began=began,stage="preparation")\n')
    once('                    row = _h_case(',
         '                    _PL_LIVE["stage"]=[seed,tokens,dist]\n                    row = _h_case(')
    once('    finally:\n        load_inline = original_loader\n        _h_reset(states)',
         '    except BaseException as error:\n        _PL_LIVE["primary_error"]=error\n        raise\n'
         '    finally:\n        load_inline = original_loader\n        _PL_LIVE["stage"]="cleanup"\n'
         '        try:_h_reset(states)\n'
         '        except BaseException as cleanup_error:\n'
         '            _PL_LIVE["cleanup_error"]=str(cleanup_error)[:300]\n'
         '            if "primary_error" in _PL_LIVE:raise _PL_LIVE["primary_error"] from cleanup_error\n'
         '            raise')
    once('    _expert_moe_budget(deadline)\n    report.update(',
         '    _PL_LIVE["stage"]="post-cleanup-budget"\n    _expert_moe_budget(deadline)\n    report.update(')
    once("    encoded = json.dumps(report, separators=(',', ':'), allow_nan=False)",
         '    _PL_LIVE["stage"]="serialization"\n    encoded = _pl_encode(report)')
    once("    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')",
         "    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')\n    _PL_LIVE[\"stage\"]=\"final-budget\"")
    once("    report['component_pass'] = decision(rows)['component_pass']",
         "    report['component_pass'] = decision(rows)['component_pass']\n    report['pairlut_decision']=pairlut_decision(rows,decision(rows))")
    extra=(ROOT/'tools/moe_pairlut_contract.py').read_text()
    return extra+'\n'+PARTIAL+'\n'+runtime+'\n'+ENTRY


PARTIAL = '''
_PL_LIVE={}
_PL_BUILDS=[]
def _pl_encode(report):
    return json.dumps(report,separators=(',',':'),allow_nan=False)

def _pl_partial(error):
    import base64,zlib
    report={k:v for k,v in _PL_LIVE.items() if k!='primary_error'}
    report.update(runtime_id=H_RUNTIME_ID,status='FAILED_PARTIAL_NOT_QUALIFIED',
                  elapsed_s=time.monotonic()-report['began'],error=type(error).__name__+':'+str(error)[:300])
    blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    require(len(blob)<12000,'bounded partial evidence')
    print('MOE_PAIRLUT_PARTIAL '+blob,flush=True)
'''
ENTRY = '''
def prepare(weights):
    _PL_LIVE.clear()
    _PL_LIVE.update(rows=[],counters={},began=time.monotonic(),stage='entry',loaded_builds=_PL_BUILDS)
    try:return _pl_prepare_inner(weights)
    except BaseException as error:
        try:_pl_partial(error)
        except BaseException as reporting_error:
            print('MOE_PAIRLUT_REPORTING_FAILURE '+type(reporting_error).__name__+':'+str(reporting_error)[:300],flush=True)
        raise
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
