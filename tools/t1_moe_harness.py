"""Generate a HELD exact-native T1-only standalone component instrument."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import symtable
import builtins

import g_t1_three_launch as t1
import expert_top100_moe_adapter as extraction
from expert_top100_pregamma import capture
from expert_top100_moe_harness import extract, REFERENCE_PIN, NVFP4_PIN
from prod_size_compile_g import bindings

ROOT = t1.ROOT
NATIVE_PIN = 'dbdd3381cc9b823afb9e7d8678bba536cd5d6e3731896153aee9908652237447'
CONTROL_PIN = extraction.CONTROL_NATIVE
ALLOCATION = 'aa4e5cfec1f1b081ac50e3c62c53fdea511d282c'


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError('T1_MOE_SOURCE: ' + message)


def replace_once(text, old, new):
    require(text.count(old) == 1, 'boundary ' + old[:60])
    return text.replace(old, new)


def adapter_source():
    base = t1.BASE.read_text(encoding='utf8')
    require(sha(base) == t1.BASE_SHA, 'ba9 control')
    values = bindings(base)
    values.update(extraction.flags(base))
    require(sha(values['MOE_CUDA_SRC']) == CONTROL_PIN, 'native control')
    native = (t1.EVIDENCE / 'generated/selected.cu').read_text(encoding='utf8')
    require(t1.native_source(base).replace('\r\n', '\n') == native, 'frozen native regeneration')
    require(sha(native) == NATIVE_PIN, 'frozen T1 native')
    frozen = (t1.EVIDENCE / 'generated/candidate.py').read_text(encoding='utf8')
    frozen_values = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(frozen).body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ('_T1_THREE_CUDA', '_T1_THREE_CPP', '_T1_THREE_FLAGS')}
    require(frozen_values == dict(_T1_THREE_CUDA=native, _T1_THREE_CPP=t1.CPP,
                                 _T1_THREE_FLAGS=t1.FLAGS), 'frozen native/binding/ordered flags')
    t1.native_contract(native)
    tree = ast.parse(base)
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assignments = {node.targets[0].id: node for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    parts = ['# okc: problem=muse_moe_w4a16',
             '"""HELD PRIVATE T1 COMPONENT PROPOSAL; no dispatch authorization."""',
             '\n'.join(base.splitlines()[1:17]),
             'import time\nimport torch\nfrom torch.utils.cpp_extension import load_inline',
             f'EXPERT_MOE_IDENTITIES = {(t1.BASE_SHA, CONTROL_PIN, NATIVE_PIN)!r}']
    for name in extraction.CONSTANTS:
        if name != '_EXPERT_PAIR_EDITS':
            parts.append(f'{name} = {values[name]!r}')
    parts.extend(ast.get_source_segment(base, assignments[name]) for name in extraction.EXPRESSIONS)
    parts.extend(('MOE_CUDA_SRC = ' + extraction.source_literal(values['MOE_CUDA_SRC']),
                  '_MA_EXTS = {}',
                  "_MA_ALT_SRCS = {'moe_helper0': " + extraction.source_literal(values['_MA_ALT_SRCS']['moe_helper0']) + '}',
                  "_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']"))
    closure = {}
    for name in (*extraction.FUNCTIONS[:-2], '_RouterExt'):
        text = ast.get_source_segment(base, definitions[name])
        original = text
        if name == '_fork_native_ma_prepare':
            text = replace_once(text, '[_guarded(t) for t in _ma_make_scratch(dev)]',
                                '[_expert_moe_guarded(t, st) for t in _ma_make_scratch(dev)]')
        parts.append(text)
        closure[name] = dict(original_sha256=sha(original), emitted_sha256=sha(text))
    notice_start = base.index('#                                  Apache License')
    notice_end = base.index('# Added by yelgabo, November 1, 2026:', notice_start)
    notices = base[notice_start:notice_end]
    require('MIT License:' in notices and 'END OF TERMS AND CONDITIONS' in notices,
            'complete retained Apache/NOTICE/MIT block')
    parts.extend(('_POS = _stored_pos()', notices))
    body = (ROOT / 'tools/expert_top100_moe_body.py').read_text(encoding='utf8')
    body = replace_once(body, '_expert_pair_bind(state, arm)',
                        "state['_t1_arm'] = arm\n        state['_t1_arm_bound'] = arm")
    body = replace_once(body, '        return state', '        _t1_install(state)\n        _expert_moe_budget(deadline)\n        return state')
    body = body.replace("state.get('_expert_pair_arm')", "state.get('_t1_arm')")
    body = body.replace("state.get('_expert_pair_arm_bound')", "state.get('_t1_arm_bound')")
    run_start = body.index('def _expert_moe_run(data, state, arm):')
    data_start = body.index('    if not isinstance(data, dict)', run_start)
    data_end = body.index("    output = torch.empty_like", data_start)
    validation = 'def _expert_moe_data(data, state):\n' + body[data_start:data_end]
    validation = replace_once(validation, "    tokens = config.get('bs')",
        "    if config.get('sharding') != 'ep8':\n        raise RuntimeError('EXPERT_MOE_CONFIG_UNSUPPORTED_SHARD')\n    tokens = config.get('bs')")
    body = body[:run_start] + validation + '\n\ndef _expert_moe_run(data, state, arm):\n    return _t1_run(data, state, arm)\n'
    body = replace_once(body, "for full, words in state['guards']:",
                        "for full, words in state['guards'] + state.get('t1_guards', []):")
    parts.extend((body, f'_T1_THREE_CUDA = {native!r}\n_T1_THREE_CPP = {t1.CPP!r}\n_T1_THREE_FLAGS = {t1.FLAGS!r}',
                  (ROOT / 'tools/t1_moe_body.py').read_text(encoding='utf8')))
    source = '\n\n'.join(parts).rstrip() + '\n'
    require('_expert_pair_bind' not in source and '_expert_pair_source' not in source
            and 'expert_top100_pregamma' not in source, 'no pregamma path')
    unresolved = extraction.global_references(source) - {
        symbol.get_name() for symbol in symtable.symtable(source, '<t1>', 'exec').get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()} - set(dir(builtins))
    require(not unresolved, 'closure ' + repr(unresolved))
    return source, closure


def build(*, target_proposal=False):
    require(type(target_proposal) is bool, 'boolean mode')
    prefix, closure = adapter_source()
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text(encoding='utf8')
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text(encoding='utf8')
    require(sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN, 'routing pins')
    route = '\n\n'.join(('D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
                         extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    calls, routes = capture(t1.BASE.read_text(encoding='utf8'))
    descriptors = []
    for call in calls:
        descriptor = dict(call)
        for name in ('cpp_sources', 'cuda_sources'):
            descriptor[name] = sha(descriptor[name])
        descriptors.append(descriptor)
    descriptors.append(dict(name='t1_three_launch_v1', cpp_sources=sha(t1.CPP), cuda_sources=NATIVE_PIN,
                            functions=['t1_three_run'], extra_cflags=['-O3', '-std=c++20'],
                            extra_cuda_cflags=t1.FLAGS, verbose=False))
    require(len(descriptors) == 6, 'five control plus frozen T1')
    contract = Path(__file__).with_name('t1_moe_contract.py').read_text(encoding='utf8')
    runtime = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text(encoding='utf8')
    runtime = runtime.replace("state['_expert_pair_arm']", "state['_t1_arm']")
    runtime = replace_once(runtime, "        owned, byte_count = set(), 0",
        "        require(len(state['t1_scratch']) == len(state['t1_guards']) == (3 if index >= 2 else 0), 'T1 guard inventory')\n        owned, byte_count = set(), 0")
    runtime = replace_once(runtime, 'owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters)',
        'owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters, affected_tokens=[1])')
    runtime = replace_once(runtime, "*(full for full, words in state['guards'])",
                           "*(full for full, words in state['guards'] + state['t1_guards'])")
    runtime = replace_once(runtime, "zip(state['ma_scratch'], state['guards'])",
                           "zip(state['ma_scratch'] + state['t1_scratch'], state['guards'] + state['t1_guards'])")
    runtime = replace_once(runtime, "            del initial", "            for tensor in state.get('t1_scratch', []):\n                tensor.zero_()\n            del initial")
    identity = sha(prefix + route + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = {not target_proposal!r}\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    ast.parse(source)
    require(len(source.encode()) < 1_000_000, 'plain size')
    manifest = dict(allocation_commit=ALLOCATION, source_sha256=sha(source), bytes=len(source.encode()),
        runtime_id=identity, base_sha256=t1.BASE_SHA, native_control_sha256=CONTROL_PIN,
        native_candidate_sha256=NATIVE_PIN, t1_cuda_flags=t1.FLAGS, t1_host_flags=['-O3', '-std=c++20'],
        compiler_calls=descriptors, closure=closure, official_arm='candidate2', affected_tokens=[1],
        original_control_tokens=[4, 8, 16, 32, 33, 128, 256, 257], hold=not target_proposal,
        dispatch_ready=False, gpu_qualified=False, reference_sha256=REFERENCE_PIN, nvfp4_sha256=NVFP4_PIN,
        contract_sha256=sha(contract), reader_canonical_lf_sha256=sha(Path(__file__).with_name('t1_moe_read.py').read_text(encoding='utf8')),
        pending=['captain review and separate target allocation', 'target C++20 host build/load and four-state295s fit',
                 'official FP32 and actual CUDA replay qualification', 'frozen equal-six/null/T1-regression gate',
                 'separate layer qualification; no public/default promotion'])
    return source, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps({key: manifest[key] for key in ('source_sha256', 'bytes', 'runtime_id', 'hold', 'dispatch_ready')}))
