"""Generate a held exact-source measurement payload. Never dispatch here."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import expert_top100_moe_adapter as adapter
from expert_top100_pregamma import capture, transform
from prod_size_compile_g import bindings

ROOT = adapter.ROOT
ADAPTER_PIN = '19aeaf356be6329dea1031966a4b7efbab4f039748f122acc78300e670cb5b57'
REFERENCE_PIN = 'a7e0b4fa86aa477a4e7bb5745b2b258954d72636b6da153bd239a717927cba44'
NVFP4_PIN = 'cfdf37c7bc14b10f7353aba7d4632f4767401b92b4b073941a8030d9a961aaaf'


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def extract(source, name):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, node)


def build():
    prefix, adapter_manifest = adapter.build()
    adapter.require(sha(prefix) == ADAPTER_PIN, 'harness adapter prefix')
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text(encoding='utf8')
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text(encoding='utf8')
    adapter.require(sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN, 'routing extraction pins')
    route_source = '\n\n'.join((
        'D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
        extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    calls, routes = capture(prefix)
    values = bindings(prefix)
    alternatives = dict(values['_MA_ALT_SRCS'])
    alternatives['expert_top100_pregamma'] = transform(values['MOE_CUDA_SRC'])[0]
    candidate_source = prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True')
    candidate_calls, unused_routes = capture(candidate_source + '\n_MA_ALT_SRCS = ' + repr(alternatives))
    calls += [call for call in candidate_calls if call not in calls]
    adapter.require(len(calls) == 7, 'seven exact native builds')
    descriptors = []
    for call in calls:
        descriptor = dict(call)
        for key in ('cpp_sources', 'cuda_sources'):
            descriptor[key] = sha(descriptor[key])
        descriptors.append(descriptor)
    contract = Path(__file__).with_name('expert_top100_moe_harness_contract.py').read_text(encoding='utf8')
    runtime = Path(__file__).with_name('expert_top100_moe_harness_runtime.py').read_text(encoding='utf8')
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = True\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    adapter.require(len(source.encode()) < 1_000_000, 'harness size')
    ast.parse(source)
    manifest = dict(allocation_commit='ca5cf8f00402b1804eb9176227a6072170226a14',
                    source_sha256=sha(source), bytes=len(source.encode()), runtime_id=identity,
                    adapter_sha256=ADAPTER_PIN, native_control_sha256=adapter.CONTROL_NATIVE,
                    native_candidate_sha256=adapter.CANDIDATE_NATIVE, reference_sha256=REFERENCE_PIN,
                    nvfp4_sha256=NVFP4_PIN, route_source_sha256=sha(route_source),
                    compiler_calls=descriptors, official_arm='candidate2', adapter_official_arm='control',
                    hold=True, dispatch_ready=False, gpu_qualified=False,
                    pending=['parent source/host review and captain allocation; explicit held-source revision',
                             'target build/admission and four-placement memory/295s fit',
                             'independent official FP32 candidate correctness and all observed target replays',
                             'owning private receipt, six-load every-bank/cross/null/regression gate',
                             'separate whole-layer qualification; no component-to-layer inference'])
    return source, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps(manifest, indent=2))
