"""Fail-closed source/CPU retirement gate; does not emit a VMM candidate."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
BASE_SHA256 = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-pr-owned-buffer'
STOP = 'STOP_MISSING_REAL_CALLER_GRAPH_RETIREMENT_CONTRACT'


class MissingCallerRetirement(RuntimeError):
    pass


def pinned_source():
    payload = BASE.read_bytes()
    if len(payload) != 869076 or hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError('exact ba9 source identity changed')
    if b'\r' in payload or not payload.endswith(b'\n'):
        raise ValueError('base is not physical LF with final newline')
    return payload.decode('utf-8')


def definitions(source, name):
    return [node for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]


def segment(source, name, index=0):
    return ast.get_source_segment(source, definitions(source, name)[index])


def graft_prepare(source):
    candidates = [node for node in definitions(source, 'prepare')
                  if any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                         and child.func.id == '_graft_lt_plan' for child in ast.walk(node))]
    if len(candidates) != 1:
        raise ValueError('expected one actual pre-plan prepare integration entry')
    return ast.get_source_segment(source, candidates[0]), candidates[0].lineno


def source_evidence():
    source = pinned_source()
    prepare_text, prepare_line = graft_prepare(source)
    nodes = ast.parse(source).body
    picks = next(ast.literal_eval(node.value) for node in nodes if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == '_GRAFT_LT_PICKS'
                         for target in node.targets))
    pr_picks = {str(tokens): list(pick[:3]) for (name, tokens), pick in picks.items() if name == 'pr'}
    assert pr_picks == {'256': ['tn', 2, True], '128': ['tn', 1, True]}
    assert prepare_text.index('_vj_prepare(weights)') < prepare_text.index('_graft_lt_plan(st)')
    plan = segment(source, '_graft_lt_plan')
    run = segment(source, '_graft_lt_run')
    final = segment(source, 'prepare', -1)
    assert 'weight=weight' in plan and 'src[name]' in plan
    assert '_GP(plan["weight"].data_ptr())' in run
    assert final.index('_release_normal_prepare(weights)') < final.index('_release_expert_self_test(')
    assert final.rstrip().endswith('return state')
    return dict(decision=STOP, candidate_emitted=False, base_sha256=BASE_SHA256,
                base_bytes=869076, pr_bytes=4608 * 8192 * 2, pr_picks=pr_picks,
                pre_plan_prepare_line=prepare_line,
                final_prepare_line=definitions(source, 'prepare')[-1].lineno,
                plan_line=definitions(source, '_graft_lt_plan')[0].lineno,
                run_line=definitions(source, '_graft_lt_run')[0].lineno,
                added_host_bindings=[], added_driver_linkage=[], changed_native_sources=0,
                compiler_calls=0, driver_calls=0, gpu_calls=0, service_calls=0,
                missing=['real caller graph/state ownership and destruction order',
                         'no future replay/submission before context completion',
                         'safe no-throw cleanup policy after completion/context errors'],
                not_implemented=['VMM allocation/capability/property stages',
                                 'independent compressed PR bit-copy',
                                 'rollback/fallback for allocation/copy/plan failures'],
                limitations=['CPU pointer-lifetime example is not an observed judge failure',
                             'No claim of target support, compression, eligibility or speed'])


def build():
    pinned_source()
    raise MissingCallerRetirement(STOP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EVIDENCE)
    args = parser.parse_args()
    result = source_evidence()
    from test_g_pr_compressed_owner import contract_evidence
    result['CPU_contracts'] = contract_evidence()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'decision.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
