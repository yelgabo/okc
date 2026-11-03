"""Prepare the MoE factor-y candidate arm as a standalone release candidate."""
import argparse
import ast
import json
from pathlib import Path

from cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, ROOT, sha
from expert_top100_moe_adapter import build as adapter
from expert_top100_pregamma import capture
from moe_factor_y import transform
from moe_factor_y_harness import build as paired
from prod_size_compile_g import bindings

PAIRED_PIN = '78506b743ace84e1fc0b088fe7fe3b23009f3de0bf5136e20413504026e07492'
PAIRED_JOB = '7479372115c64702'
PAIRED_FILE = ROOT / 'muse_moe_w4a16/cand/factor_y_paired.py'
PAIRED_MANIFEST = ROOT / 'docs/steering/evidence/2026-11-02-moe-factor-y/paired.manifest.json'
MARKER = '\nD_HIDDEN = D_EXPERT = 4096\n'


def candidate_calls(prefix, candidate):
    """The compiler calls the paired adapter makes on its candidate arm, captured the way the harness did."""
    arm_true = prefix.replace("arm = state.get('_expert_pair_arm', False)", 'arm = True')
    assert arm_true != prefix
    alternatives = dict(bindings(prefix)['_MA_ALT_SRCS'])
    alternatives['moe_factor_y'] = candidate
    return capture(arm_true + '\n_MA_ALT_SRCS = ' + repr(alternatives))


def strip(call):
    return {key: value for key, value in call.items() if key != 'name'}


def descriptor(call):
    item = dict(call)
    for key in ('cuda_sources', 'cpp_sources'):
        item[key] = sha(item[key])
    return item


def build():
    source, manifest = paired()
    assert sha(source) == PAIRED_PIN == manifest['source_sha256']
    assert PAIRED_FILE.read_bytes() == source.encode()
    assert adapter()[1]['base_sha256'] == BASE_PIN
    control = bindings(BASE.read_text())['MOE_CUDA_SRC']
    assert sha(control) == NATIVE_PIN == manifest['control_native_sha256']
    candidate, _ = transform(control)
    assert sha(candidate) == manifest['candidate_native_sha256']
    helper = manifest['compiler_calls'][0]['cuda_sources']
    prefix = source[:source.index(MARKER)]
    old_calls, old_routes = candidate_calls(prefix, candidate)
    assert len(old_calls) == 5

    tree = ast.parse(prefix)
    lines = prefix.splitlines(keepends=True)
    removed = {'_expert_pair_source', 'expert_moe_prepare_candidate', 'expert_moe_prepare_four', 'expert_moe_candidate_fn'}
    for node in reversed(tree.body):
        discard = isinstance(node, ast.FunctionDef) and node.name in removed
        discard |= (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('_EXPERT_PAIR_EDITS', 'EXPERT_MOE_DISPATCH_READY'))
        discard |= (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                    and ast.get_source_segment(prefix, node.targets[0]) == "_MA_ALT_SRCS['moe_factor_y']")
        if discard:
            lines[node.lineno - 1:node.end_lineno] = []
    release = ''.join(lines)

    def replace(old, new, count=1):
        nonlocal release
        assert release.count(old) == count, (old[:80], release.count(old))
        release = release.replace(old, new)

    replace(control, candidate)
    identity = next(node for node in ast.parse(release).body if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'EXPERT_MOE_IDENTITIES')
    replace(ast.get_source_segment(release, identity),
            'EXPERT_MOE_IDENTITIES = ' + repr(('moe-factor-y-release', BASE_PIN, sha(candidate), helper)))
    replace('PRIVATE SOURCE-ONLY ADAPTER. dispatch_ready=False; never submit publicly.',
            'Standalone factor-y release candidate. Public submission requires approval.')
    replace("    if arm and key in ('b4', 'main'):\n        defines = tuple(defines) + ('@moe_factor_y',)\n", '')
    replace("if type(arm) is not bool or '_expert_pair_arm_bound' in state:",
            "if arm is not True or '_expert_pair_arm_bound' in state:")
    replace('_expert_pair_bind', '_moe_bind_factor_y', 2)
    replace('def prepare(weights):\n    return _expert_moe_prepare(weights, False)\n',
            'def prepare(weights):\n    return _expert_moe_prepare(weights, True)\n')
    replace('def kernel_fn(data, state):\n    return _expert_moe_run(data, state, False)\n',
            'def kernel_fn(data, state):\n    return _expert_moe_run(data, state, True)\n')
    replace("        if not state['ma_sfx']:\n            raise RuntimeError('EXPERT_MOE_NATIVE_SCALE_CONTRACT_NO_FALLBACK')\n", '')
    replace("        if set(state['ma_ext']) != {'tiny', 'tiny2', 'mid', 'b4', 'main'}:",
            "        expected = {'tiny', 'tiny2', 'mid', 'main'} | ({'b4'} if state['ma_sfx'] else set())\n"
            "        if set(state['ma_ext']) != expected:")

    for token in ('moe_factor_y', '_EXPERT_PAIR_EDITS', '_expert_pair_source', 'EXPERT_MOE_DISPATCH_READY',
                  'expert_moe_prepare_candidate', 'expert_moe_prepare_four', 'expert_moe_candidate_fn',
                  'dispatch_ready', 'H_RUNTIME_ID', 'H_BUILD_CALLS', 'D_HIDDEN'):
        assert token not in release, token
    assert control not in release
    env = bindings(release)
    assert sha(env['MOE_CUDA_SRC']) == sha(candidate)
    assert {sha(value) for value in env['_MA_ALT_SRCS'].values()} == {helper}
    calls, routes = capture(release)
    assert len(calls) == 5 and routes == old_routes
    assert [strip(call) for call in calls] == [strip(call) for call in old_calls]
    evidence = json.loads(PAIRED_MANIFEST.read_text())
    assert evidence['source_sha256'] == PAIRED_PIN and evidence['routes'] == routes
    expected = [strip(call) for call in evidence['compiler_calls'] if call['cuda_sources'] != NATIVE_PIN]
    assert [strip(descriptor(call)) for call in calls] == expected
    ast.parse(release)
    return release, dict(
        source_sha256=sha(release), bytes=len(release.encode()), native_sha256=sha(candidate),
        helper_sha256=helper, control_native_sha256_removed=NATIVE_PIN, paired_source_sha256=PAIRED_PIN,
        paired_job=PAIRED_JOB, paired_judged_arm=evidence['official_arm'],
        adapter_control_sha256=manifest['adapter_control_sha256'],
        compiler_calls=[descriptor(call) for call in calls],
        candidate_arm_call_names=[call['name'] for call in old_calls], routes=routes,
        judged_arm='candidate, the only arm', public_submission_approved=False,
        qualification='pending exact-file private check',
        scope='Factorized BF16 down-result storage with exact FP32 gamma reconstruction in B4/MAIN native ecd1df11; '
              'original tiny/tiny2/MID helper b8ed8543; incumbent scale fallbacks restored; original three scratch '
              'buffers; no control arm, pair edits, dispatch gate or private harness')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
