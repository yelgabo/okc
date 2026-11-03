"""Offline receipt/route audit only: no imports of submission, CUDA or judge calls.

Run from repository root: python tools/top100_profile_audit.py
"""
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from dl_compare_receipts import validate, compare, LOADS, shape_name
from hotspots import load, case_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tools/top100_profile_evidence'
FRESH = 'docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json'
PUBLIC = 'muse_decode_layer/profiles/scored.607eb52bdaf24160.json'
BASE = 'muse_decode_layer/baseline_profile/status.7d42771c3384449d.json'
BASE_PROFILE = 'muse_decode_layer/baseline_profile/profiles/baseline.7d42771c3384449d.json'
SOURCE = 'muse_decode_layer/cand/g1b4_stage_release.py'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    OUT.mkdir(exist_ok=True)
    receipts = [read(p) for p in (FRESH, PUBLIC, BASE)]
    checked = [validate(r, job, mode) for r, job, mode in zip(receipts,
               ('cc474883117947c0', '607eb52bdaf24160', '7d42771c3384449d'), ('check', 'run', 'check'))]
    fresh, public, baseline = checked
    assert fresh['content_hash'] == public['content_hash']
    source = (ROOT / SOURCE).read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert digest == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    assert len(source.encode()) == 869076
    assert '56 verified_replays' in receipts[0]['result']['log_tail']
    # Independently reconcile normalized historical profiles against owning status.
    normalized = load(ROOT / BASE_PROFILE)
    original = load(ROOT / BASE)
    assert {case_key(x['case']): x['report'] for x in normalized['profiles']} == {
        case_key(x['case']): x['report'] for x in original['profiles']}
    assert len(normalized['benchmarks']) == len(original['benchmarks']) == 12
    for a, b in zip(normalized['benchmarks'], original['benchmarks']):
        assert all(b[k] == v for k, v in a.items())
        assert set(b) - set(a) <= {'flags'} and not b.get('flags')

    tree = ast.parse(source)
    env = {'_num_sms7': lambda _: 132}
    # Execute only literal policy definitions and pure routing functions, never module code.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith(('MOE_', 'OKC_')) or name == 'MAIN_FALLBACK_DEFINES':
                try:
                    env[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id.startswith('MOE_'):
            env[node.target.id] += ast.literal_eval(node.value)
    for name in ('_plan7', '_main_defines', '_fork_native_moe'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE, 'exec'), env)
    env['_ma_ext'] = lambda flags: flags
    env['_RouterExt'] = lambda flags: flags
    routes = {}
    for tokens in (1, 8, 16, 32, 128, 256):
        state = {'ma_sfx': ('-DOKC_PRMTSC=1',), 'ma_sc6': True, 'ma_ext': {}}
        flags = env['_fork_native_moe'](tokens, state)
        routes[tokens] = {'key': next(iter(state['ma_ext'])), 'flags': flags,
                          'builder_extra_cuda_flags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a',
                            '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_DQ7=1']}
    # Later MID marker rewrite is a nonliteral assignment, excluded above; retain it accurately.
    routes[16]['flags'] = tuple('@moe_helper0df' if x == '@moe_helper0' else x for x in routes[16]['flags'])
    plans = {f'{b}x{q}': env['_plan7'](b, 1, q, 0, 86008, 1, 'cuda')
             for b in (16, 32) for q in (1, 8)}
    assert all(p == dict(n_splits=264, flat=1, cs=0, ncl=1, nc=1) for p in plans.values())
    cells = []
    for record in original['benchmarks']:
        assert shape_name(record['case']) in fresh['shapes']
    for record in load(ROOT / FRESH)['profiles']:
        name = shape_name(record['case'])
        rows = []
        for line in record['report'].splitlines()[1:]:
            match = re.match(r'(.+?)\s+([\d.]+)\s*µs\s+([\d.]+)%\s+(\d+)\s*$', line)
            if match:
                rows.append({'symbol': match[1].strip(), 'activity_us': float(match[2]), 'calls': int(match[4])})
        expert = [x for x in rows if 'okc::moe_fused<' in x['symbol']]
        attention = [x for x in rows if 'attn_kernel<' in x['symbol']]
        assert len(expert) == len(attention) == 1
        b = fresh['shapes'][name]
        tokens = b['case']['batch'] * b['case'].get('q_len', 1)
        assert ('<true>' in expert[0]['symbol']) == (tokens >= 128)
        cells.append(dict(shape=name, tokens=tokens, case=b['case'], median_us=b['median_ns']/1000,
                          baseline_us=b['baseline_ns']/1000, modeled_floor_us=b['speed_of_light_ns']/1000,
                          expert_activity_us=expert[0]['activity_us'], attention_activity_us=attention[0]['activity_us'],
                          expert_route=routes[tokens], activities=rows))
    indexed = {x['shape']:x for x in cells}
    loads = []
    for name in LOADS:
        latency = fresh['loads'][name]['median_ns']
        weighted = lambda key: (46*indexed[name+'s'][key]+16*indexed[name+'f'][key])/62
        loads.append(dict(load=name, endpoint_us=latency, reference_us=fresh['loads'][name]['baseline_ns'],
                          expert_activity_us=weighted('expert_activity_us'),
                          full_attention_activity_us=indexed[name+'f']['attention_activity_us'],
                          whole_load_one_us_log_score_percent=100/(6*latency),
                          sliding_cell_one_us_log_score_percent=100*46/(62*6*latency),
                          full_cell_one_us_log_score_percent=100*16/(62*6*latency)))
    # Conditional endpoint arithmetic, NOT conversion of activities to removable time.
    def scenario(names, full_only=False):
        return math.exp(sum(math.log(x['endpoint_us']/(x['endpoint_us']-
            .1*(16/62*x['full_attention_activity_us'] if full_only else x['expert_activity_us'])))
            for x in loads if x['load'] in names)/6)
    threshold = next(x for x in read('tools/top100_profile_evidence/layer.json')['data']['rows'] if x['rank']==100)['speedup']
    ev0, ev1 = receipts[0]['result']['evaluator'], receipts[2]['result']['evaluator']
    report = dict(evidence_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_sha256_lf=digest, source_bytes_lf=len(source.encode()),
                  inputs={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (FRESH,PUBLIC,BASE,BASE_PROFILE,SOURCE)},
                  validation='3 owning receipts each 13 passing tests / 12 distinct benchmarks / 12 matching profiles; scores reconstructed',
                  old_to_fresh=compare(public,fresh), baseline_created_at=receipts[2]['created_at'],
                  evaluator_file_changes={k:dict(historical=v,fresh=ev0['files'].get(k)) for k,v in ev1['files'].items() if ev0['files'].get(k)!=v},
                  evaluator_images=dict(historical=ev1['image'],fresh=ev0['image']),
                  full_attention_plans=plans, cells=cells, loads=loads,
                  scenarios_NOT_exposed_savings=dict(long_experts_10pct=scenario(('d16','d32','v16','v32')),
                      short_experts_10pct=scenario(('d1','v1')),
                      four_full_attention_10pct=scenario(('d16','d32','v16','v32'),True)),
                  layer_cutoff_rounded=threshold, layer_gap_factor=threshold/public['score'],
                  uniform_endpoint_reduction_fraction=1-public['score']/threshold,
                  limitations=['Activity sums are not endpoint budgets; scenarios are not additive.',
                    'Pure source dispatch audit plus recorded symbols is not compiled instruction validation.',
                    'Historical SGLang profile and fresh baseline endpoints are distinct evidence.'])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cmd=[sys.executable,'tools/hotspots.py',FRESH,'--baseline',BASE_PROFILE]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env={**__import__('os').environ,'PYTHONIOENCODING':'utf-8'},check=True)
    (OUT/'hotspots.md').write_text(result.stdout,encoding='utf-8')
    print(json.dumps({'loads':loads,'scenarios':report['scenarios_NOT_exposed_savings'],'gap':report['layer_gap_factor']},indent=2))


if __name__ == '__main__':
    main()
