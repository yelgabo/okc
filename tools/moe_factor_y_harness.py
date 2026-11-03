"""Compare exact factorized expert-output storage with the incumbent FP32 buffer."""
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
from moe_factor_y import transform
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
    experiment = 'moe_factor_y'
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', edits)
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES', (experiment, BASE_PIN, NATIVE_PIN, sha(candidate)))
    prefix = prefix.replace('expert_top100_pregamma', experiment)
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
                    public_submission=False, target_qualified=False)
    return source, manifest


def progress_runtime(runtime):
    source=retained_runtime(runtime)
    source=source.replace('[256,256,16640,16640]','[256,256,256,256]')
    source=source.replace('bound generic LUT owners','original lookup owners')
    source=source.replace('_PL','_FY').replace('_pl_','_fy_')
    source=source.replace('MOE_PAIRLUT','MOE_FACTOR_Y')
    return source.replace('pairlut_decision','factor_y_decision')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
