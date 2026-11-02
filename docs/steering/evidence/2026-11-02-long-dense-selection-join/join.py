import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
PRIOR = ROOT / 'docs/steering/evidence/2026-11-01-long-dense-archive-join/analysis.json'
DONOR = ROOT / 'docs/steering/evidence/2026-11-01-ruwwww-complete-qualification/analysis.json'
prior = json.loads(PRIOR.read_text())
donor = json.loads(DONOR.read_text())
assert donor['source']['sha256'] == '3cee8ba759569172a9f8a38c59e6589dd8045db0d414e6b33294604aad11ba0e'
assert 'c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d' in str(donor['remote_pin'])
FUNCTIONS = {'dense', '_rq_gemm', '_graft_lt_plan', '_graft_lt_run', '_graft_layouts', '_fork_lt_no_split', '_policy_variant', '_graft_after_attention', '_graft_boundary', 'dense_ng'}
CONSTANTS = {'_GRAFT_LT_PICKS', '_GRAFT_QKV256', '_GRAFT_LT_T'}


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def keys(value, path=''):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = path + '/' + key
            if any(term in key.lower() for term in ('compose_selection', 'graft_lt_status', 'mm_pick')):
                found.append({'path': location, 'value': str(item)[:500]})
            found.extend(keys(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(keys(item, path+'/'+str(index)))
    return found


class Tensor:
    def __init__(self, shape):
        self.shape = shape

    def view(self, *shape):
        return Tensor(shape)


result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checkout': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'scope': 'Source/archive and CPU dispatcher-only model; no donor import, target APIs, network, compiler, GPU, challenge CLI or production edits', 'prior_evidence': identity(PRIOR.read_bytes()), 'donor_evidence': identity(DONOR.read_bytes()), 'jobs': {}, 'cpu_tests': []}
for job in ('607eb52bdaf24160', 'c5e4f89672f743bb'):
    old = prior['jobs'][job]
    path = ROOT / old['source']['path']
    raw = path.read_bytes()
    assert identity(raw)['sha256'] == old['source']['sha256']
    source = raw.decode()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    constants[target.id] = {'line': node.lineno, 'value': repr(ast.literal_eval(node.value)), 'text': ast.get_source_segment(source, node)}
    extracted = {name: {'line': functions[name].lineno, 'text': ast.get_source_segment(source, functions[name]), 'ast_sha256': hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()} for name in FUNCTIONS}
    extracted['_GraftLt'] = {'line': classes['_GraftLt'].lineno, 'text': ast.get_source_segment(source, classes['_GraftLt'])}
    for name, saved in old['functions'].items():
        assert ast.dump(functions[name], include_attributes=False) == ast.dump(ast.parse(saved['text']).body[0], include_attributes=False)
    log_raw = (ROOT / old['log']['path']).read_bytes()
    assert identity(log_raw)['sha256'] == old['log']['sha256']
    accepted_lines = [{'line': index+1, 'text': line} for index, line in enumerate(log_raw.decode(errors='replace').splitlines()) if 'Lt plans' in line]
    assert len(accepted_lines) == 1
    accepted = ast.literal_eval(accepted_lines[0]['text'].split('Lt plans ', 1)[1])
    assert set(accepted) == {(name, tokens) for name in ('qkv', 'o', 'pr', 'un') for tokens in (128, 256)}
    receipt_raw = (ROOT / old['receipt']['path']).read_bytes()
    assert identity(receipt_raw)['sha256'] == old['receipt']['sha256']
    receipt = json.loads(receipt_raw)
    records = (receipt.get('result') or {}).get('records', [])
    profiles = receipt.get('profiles', []) or [row for row in records if row.get('event') == 'profile']
    long_profiles = [row for row in profiles if row['case']['load'] in ('verify 16', 'verify 32')]
    assert len(long_profiles) == 4
    profile_rows = []
    for row in long_profiles:
        assert '_graft_row_boundary' in row['report'] and '_graft_route' in row['report']
        profile_rows.append({'case': row['case'], 'rows': [line for line in row['report'].splitlines() if any(term in line for term in ('nvjet_', 'cublas', '_graft_row_boundary', '_graft_route'))]})
    picks = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '_GRAFT_LT_PICKS' for target in node.targets)))
    dimensions = {'qkv': (2304, 8192), 'o': (8192, 1024), 'pr': (4608, 8192), 'un': (8192, 4096)}
    plans = {}
    plan_records = []
    for name, tokens in accepted:
        n, k = dimensions[name]
        mode, batch, fp32, words = picks.get((name, tokens), ('tn', 1, False, None))
        plans[name, tokens] = {'n': n, 'fp32': fp32, 'name': name, 'tokens': tokens}
        plan_records.append({'name': name, 'M_N_K': [tokens, n, k], 'output': 'FP32' if fp32 else 'BF16', 'mode': mode, 'N_partitions_not_K_split': batch, 'algorithm_words': words, 'explicit_qkv256_configuration': constants['_GRAFT_QKV256'] if (name, tokens) == ('qkv', 256) else None, 'accepted': True, 'rejection_reason': None})
    calls = []
    def lt_run(value, plan):
        calls.append(('Lt', plan['name'], plan['tokens'], plan['fp32']))
        return Tensor((value.shape[0], plan['n']))
    def fallback(value, weight, tokens, state, name):
        calls.append(('fallback', name, tokens))
        return Tensor((1, tokens, dimensions[name][0])), 1
    namespace = {'_graft_lt_run': lt_run, '_vj_dense': fallback}
    isolated = ast.Module(body=[functions[name] for name in ('dense', '_rq_gemm', '_policy_variant')], type_ignores=[])
    exec(compile(isolated, '<isolated-final-dispatch-only>', 'exec'), namespace)
    state = {'graft_lt': plans, 'mm_pick': {}, 'compose_graft_enabled': {}, 'compose_boundary_warps': {}}
    for name, tokens in accepted:
        for context in ('sliding', 'full'):
            calls.clear()
            output, splits = namespace['dense'](Tensor((tokens, dimensions[name][1])), None, tokens, state, name)
            assert calls == [('Lt', name, tokens, plans[name, tokens]['fp32'])] and splits == 1
            assert output.shape == (1, tokens, dimensions[name][0])
            result['cpu_tests'].append({'job': job, 'name': name, 'T': tokens, 'context': context, 'call': calls[:], 'result': 'PASS'})
    qkv_entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'kernel_fn' and '_graft_lt_run' in ast.get_source_segment(source, node))
    extracted['qkv_direct_entry'] = {'line': qkv_entry.lineno, 'text': ast.get_source_segment(source, qkv_entry)}
    extracted['pool_router_definitions'] = [{'line': node.lineno, 'text': ast.get_source_segment(source, node)} for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_pool_router_fp32']
    namespace.update(_supported7=lambda config: True, _att_fits7=lambda data, config: True, _decode_attention7=lambda data, state, qkv: Tensor((data['config']['tokens'], 1024)), _after_attention=lambda data, state, value, gate, tokens: 'after_attention', _vj_kernel_fn=lambda data, state: 'qkv_fallback')
    exec(compile(ast.Module(body=[qkv_entry], type_ignores=[]), '<isolated-source-QKV-entry>', 'exec'), namespace)
    state['att_ok'] = True
    for tokens in (128, 256):
        for context in ('sliding', 'full'):
            calls.clear()
            data = {'config': {'tokens': tokens, 'q_len': 8, 'layer': context}, 'attn_input': Tensor((tokens, 8192))}
            assert namespace['kernel_fn'](data, state) == 'after_attention'
            assert calls == [('Lt', 'qkv', tokens, False)]
            result['cpu_tests'].append({'job': job, 'name': 'actual_source_direct_qkv_entry', 'T': tokens, 'context': context, 'call': calls[:], 'result': 'PASS'})
    for enabled, warps in ((True, 16), (False, 16), (False, 8), (False, 4)):
        assert namespace['_policy_variant'](state, 128, (enabled, warps, ()))['graft_lt'] == plans
    for name, tokens in accepted:
        calls.clear()
        missing = dict(state, graft_lt={key: value for key, value in plans.items() if key != (name, tokens)})
        namespace['dense'](Tensor((tokens, dimensions[name][1])), None, tokens, missing, name)
        assert calls == [('fallback', name, tokens)]
    result['jobs'][job] = {'source': dict(old['source']), 'source_identity_revalidated': identity(raw), 'accepted_log': accepted_lines, 'receipt': dict(old['receipt']), 'selection_keys_found': keys(receipt), 'plans': plan_records, 'profile_rows': profile_rows, 'final_functions': extracted, 'constants': constants, 'negative_tests': 'PASS eight missing-plan fallbacks and four no-drop composition modes', 'state_limit': 'Accepted keys logged; exact runtime status/workspace words and composition timing object not exported'}
result['donor'] = {'pin': donor['remote_pin'], 'source': donor['source'], 'job': donor['receipt']['job_id'], 'source_receipt_association': 'Pinned source/receipt co-publication, no authenticated upload hash', 'dense': donor['selected_functions']['dense']['remote'], 'prepare': donor['selected_functions']['prepare']['remote'], 'after_attention': donor['selected_functions']['_after_attention']['remote'], 'long_profiles': [{'case': row['case'], 'rows': [line for line in row['report'].splitlines() if any(term in line for term in ('nvjet_', 'cublas', '_boundary1', '_boundary2', '_route'))]} for row in donor['receipt']['profiles'] if row['case']['load'] in ('verify 16', 'verify 32')]}
result['comparison'] = {name: len({job['final_functions'][name]['ast_sha256'] for job in result['jobs'].values()}) == 1 for name in FUNCTIONS}
assert all(result['comparison'][name] for name in FUNCTIONS - {'_graft_after_attention'})
result['decision'] = 'SCOPED_STOP: no newly identifiable retained-slower or equivalent-contract alternative selected plan. Existing join1942ff1 already established intended Lt policy; revalidated against unchanged owning files. No new implementation/test allocation justified.'
result['score_scope'] = {'affected_weight_if_both_long_loads_and_contexts': '2*(46+16)/(6*(46+16)) = 1/3', 'whole_3pct_requires_entire_affected_load_reduction': 1-1/(1.03**3), 'one_long_load_both_contexts_3pct_reduction': 1-1/(1.03**6), 'qualification': 'Uniform whole-load arithmetic scenarios only; not operator exposure or predicted savings'}
(OUT / 'join.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'decision': result['decision'], 'cpu_dispatch_cases': len(result['cpu_tests']), 'jobs': {job: {'accepted': len(record['plans']), 'profiles': len(record['profile_rows']), 'selection_keys': record['selection_keys_found']} for job,record in result['jobs'].items()}, 'score_scope': result['score_scope']}, indent=2))
