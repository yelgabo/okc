"""Generate an exact-source private component adapter, not a qualified benchmark."""
import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable

import expert_top100_pair_package as pair
from expert_top100_pregamma import ROOT, PIN
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings

CANDIDATE_PIN = 'dff221ae6ccc2de35d7a4c123aea3e3eb1345d3685f6eb97e52cbbfef6a0c8c4'
CONTROL_NATIVE = 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'
CANDIDATE_NATIVE = '3cc12ae09f1954013f5cc9be858120d29b12bf14fff267391b94c5000d2256ea'
FUNCTIONS = (
    '_stored_pos', '_pack_codes', '_pack_scales', '_sc6_ok', '_sc6_wgs',
    '_fat_order', '_unit_order', '_pack_codes_nat', '_kperm_src', '_kperm_codes',
    '_kperm_rows', '_wgs_copy', '_wgs_on', '_ma_scales_prmt_exact',
    '_main_defines', '_ma_ext', '_df_inter', '_ma_make_scratch',
    '_fork_native_ma_prepare', '_fork_native_moe', '_expert_pair_source',
    '_expert_pair_bind',
)
CONSTANTS = (
    'MOE_CPP_SRC', 'MOE_DEFINES', 'MOE_TINY_DEFINES', 'MOE_TINY_MAX_M',
    'MOE_TINY2_DEFINES', 'MOE_TINY2_MAX_M', 'MOE_MID_DEFINES', 'MOE_MID_MAX_M',
    'MOE_HAS_PDL', 'DQ7', '_REG_HJ', 'SC6_WGS', '_MA_PRMT_SCALES',
    '_MA_SFX_IF_PRMT', 'MAIN_FALLBACK_DEFINES', '_MA_MAIN_NS', '_MA_PDL',
    'SCRATCH_SENTINEL', '_EXPERT_PAIR_EDITS',
)
EXPRESSIONS = ('_MA_MMAX', '_MA_EMAX', 'WGS_FIXF', 'WGS_XROWS', 'SCRATCH_GUARD')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_literal(text):
    require("'''" not in text and not text.endswith('\\'), 'native literal delimiter')
    return "r'''" + text + "'''"


def require(condition, message):
    if not condition:
        raise RuntimeError('EXPERT_MOE_SOURCE_DRIFT: ' + message)


def global_references(source):
    result = set()

    def visit(table):
        result.update(symbol.get_name() for symbol in table.get_symbols()
                      if symbol.is_global() and symbol.is_referenced())
        for child in table.get_children():
            visit(child)

    visit(symtable.symtable(source, '<component>', 'exec'))
    return result


def build():
    base = pair.BASE.read_text(encoding='utf8')
    candidate = (ROOT / 'tools/expert_top100_candidate.py').read_text(encoding='utf8')
    require(sha(base) == PIN, 'complete control')
    require(sha(candidate) == CANDIDATE_PIN, 'complete candidate')
    packed, pair_manifest = pair.build()
    require(pair_manifest['native_control_sha256'] == CONTROL_NATIVE, 'control native')
    require(pair_manifest['native_candidate_sha256'] == CANDIDATE_NATIVE, 'candidate native')
    require(sha(bindings(candidate)['MOE_CUDA_SRC']) == CANDIDATE_NATIVE, 'delivered candidate native')
    values = bindings(packed)
    values.update(flags(packed))
    require(sha(values['MOE_CUDA_SRC']) == CONTROL_NATIVE, 'packaged control native')
    tree = ast.parse(packed)
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assignments = {node.targets[0].id: node for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    parts = ['# okc: problem=muse_moe_w4a16',
             '"""PRIVATE SOURCE-ONLY ADAPTER. dispatch_ready=False; never submit publicly."""',
             '\n'.join(base.splitlines()[1:17]),
             'import time\nimport torch\nfrom torch.utils.cpp_extension import load_inline',
             'EXPERT_MOE_DISPATCH_READY = False',
             f'EXPERT_MOE_IDENTITIES = {(PIN, CANDIDATE_PIN, CONTROL_NATIVE, CANDIDATE_NATIVE)!r}']
    parts.extend(f'{name} = {values[name]!r}' for name in CONSTANTS)
    parts.extend(ast.get_source_segment(packed, assignments[name]) for name in EXPRESSIONS)
    parts.append('MOE_CUDA_SRC = ' + source_literal(values['MOE_CUDA_SRC']))
    parts.append("_MA_EXTS = {}\n_MA_ALT_SRCS = {'moe_helper0': " + source_literal(values['_MA_ALT_SRCS']['moe_helper0']) + '}')
    require(values['_MA_ALT_SRCS']['moe_helper0df'] == values['_MA_ALT_SRCS']['moe_helper0'], 'helper alias')
    parts.append("_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']")
    closure = {}
    for name in (*FUNCTIONS, '_RouterExt'):
        text = ast.get_source_segment(packed, definitions[name])
        closure[name] = dict(source_sha256=sha(text), globals=sorted(global_references(text)),
                             packaged_line=definitions[name].lineno)
        if name == '_fork_native_ma_prepare':
            old = '[_guarded(t) for t in _ma_make_scratch(dev)]'
            require(text.count(old) == 1, 'scratch owner boundary')
            text = text.replace(old, '[_expert_moe_guarded(t, st) for t in _ma_make_scratch(dev)]')
        closure[name]['emitted_sha256'] = sha(text)
        parts.append(text)
    parts.append('_POS = _stored_pos()')
    parts.append("_MA_ALT_SRCS['expert_top100_pregamma'] = _expert_pair_source()")
    body = Path(__file__).with_name('expert_top100_moe_body.py').read_text(encoding='utf8')
    parts.append(body)
    source = '\n\n'.join(parts).rstrip() + '\n'
    table = symtable.symtable(source, '<component>', 'exec')
    defined = {symbol.get_name() for symbol in table.get_symbols()
               if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
    unresolved = global_references(source) - defined - set(dir(builtins))
    require(not unresolved, 'unresolved closure ' + repr(sorted(unresolved)))
    require(len(source.encode()) < 1_000_000, 'payload size')
    manifest = dict(
        evidence_base='a58f058548249a3c5c1a5b5885b516fb15869603',
        base_sha256=PIN, candidate_sha256=CANDIDATE_PIN,
        native_control_sha256=CONTROL_NATIVE, native_candidate_sha256=CANDIDATE_NATIVE,
        helper_sha256=sha(values['_MA_ALT_SRCS']['moe_helper0']),
        source_sha256=sha(source), bytes=len(source.encode()), closure=closure,
        adapter_body_sha256=sha(body), official_arm='control',
        supported_local_range=[0, 32], supported_tokens=[1, 4, 8, 16, 32, 33, 128, 256, 257],
        mutable_ownership='per preparation: weights, scratch, guard tails, extension map; outputs per call',
        shared='compiled modules and input-independent native module metadata; PDL stays zero',
        input_cache=False,
        dependencies={path: sha((ROOT / path).read_text(encoding='utf8')) for path in (
            'tools/expert_top100_pair_package.py', 'tools/expert_top100_pregamma.py',
            'tools/prod_size_compile_g.py', 'tools/g_native_patch_ladder.py',
            'muse_moe_w4a16/reference.py', 'muse_moe_w4a16/TASK.md')},
        preparation_guard_seconds=295, dispatch_ready=False, gpu_qualified=False,
        pending=['target build and memory/preparation fit', 'real packing/numerical qualification',
                 'fresh observed candidate graph replays and empty/occupied transitions',
                 'four-placement ownership validation on GPU',
                 'heldout paired cold timings, all crosses and duplicate nulls',
                 'strict owning job/source/result reader', 'separate whole-layer qualification'])
    return source, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    payload, receipt = build()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_bytes(payload.encode())
    arguments.out.with_suffix('.manifest.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({key: value for key, value in receipt.items() if key != 'closure'}, indent=2))
