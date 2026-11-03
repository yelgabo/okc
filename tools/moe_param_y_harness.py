"""Compare prepared gamma plus exact factorized storage with the incumbent."""
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
from moe_param_y import transform
from moe_pairlut_harness import progress_runtime as retained_runtime


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
    experiment = 'moe_param_y'
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', edits)
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', (experiment, BASE_PIN, NATIVE_PIN, sha(candidate)))
    prefix = prefix.replace('expert_top100_pregamma', experiment)
    prefix = prepare_gamma(prefix)
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
                    public_submission=False,target_qualified=False,prepared_gamma=dict(raw_index=7,derived_index=14,stored_weights=15,native_arguments=14,dtype='bfloat16',candidate_tokens=[32,33,128,256,257]))
    return source, manifest


def progress_runtime(runtime):
    source=retained_runtime(runtime)
    source=source.replace('[256,256,16640,16640]','[256,256,256,256]')
    source=source.replace('bound generic LUT owners','original lookup owners')
    source=source.replace('_PL','_PG').replace('_pl_','_pg_')
    source=source.replace('MOE_PAIRLUT','MOE_PARAM_Y')
    source=source.replace('        owners = _h_ownership(states)', '        owners = _h_ownership(states)\n        gamma_checks = _param_y_check(states)')
    source=source.replace('table_words=[s["ma_w"][6].numel() for s in states])', 'table_words=[s["ma_w"][6].numel() for s in states],gamma_binding_checks=gamma_checks,gamma_words=[s["ma_w"][14].numel() for s in states])')
    return source.replace('pairlut_decision','param_y_decision')



def prepare_gamma(prefix):
    node=next(n for n in ast.parse(prefix).body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_ma_prepare')
    old=ast.get_source_segment(prefix,node)
    anchor='    st["ma_scratch"] = '
    assert old.count(anchor)==1
    new=old.replace(anchor,'    assert len(st["ma_w"])==14\n    st["ma_w"].append((st["ma_w"][7].float()+1.0).to(torch.bfloat16).contiguous())\n'+anchor)
    prefix=prefix.replace(old,new)
    argument="        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],"
    assert prefix.count(argument)==1
    prefix=prefix.replace(argument,"        _param_y_weights(state,tokens,arm), state['ma_scratch'], data['hidden_states'], data['topk_ids'],")
    return prefix+'\n'+PARAMETERS


PARAMETERS='''
def _param_y_weights(state,tokens,arm):
    result=state['ma_w'][:14]
    if arm and tokens>16:result[7]=state['ma_w'][14]
    return result

def _param_y_check(states):
    count=0
    for state in states:
        require(len(state['ma_w'])==15,'prepared gamma inventory')
        raw,gamma=state['ma_w'][7],state['ma_w'][14]
        require(gamma.dtype==torch.bfloat16 and gamma.shape==raw.shape and gamma.device==raw.device
                and gamma.is_contiguous() and gamma.data_ptr()%16==0,'prepared gamma metadata')
        for tokens in TOKENS:
            arm=state['_expert_pair_arm']
            args=_param_y_weights(state,tokens,arm)
            expected=list(state['ma_w'][:14])
            if arm and tokens>16:expected[7]=gamma
            require(len(args)==14 and all(a.data_ptr()==b.data_ptr() for a,b in zip(args,expected)),'selected gamma argument binding')
            count+=1
    require(count==36,'all gamma size/owner bindings')
    return count
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
