"""Reproduce T1 layer sizing from immutable owning receipts; no device or POST."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

BASE = '5f68780152ff6f33bc5e8c967e85d41b23e391e7'
OUT = Path(__file__).parent


def blob(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path])


def load(path):
    return json.loads(blob(path))


def endpoints(path, job):
    receipt = load(path)
    assert receipt['id'] == job and receipt['board'] == 'muse_decode_layer'
    records = receipt['result']['records']
    assert receipt['result']['passed']
    cells = [r for r in records if r['event'] == 'benchmark' and r['case']['load'] == 'decode 1']
    profiles = [r for r in records if r['event'] == 'profile' and r['case']['load'] == 'decode 1']
    assert len(cells) == len(profiles) == 2
    assert {r['case']['layer']: r['case']['weight'] for r in cells} == {'sliding': 46, 'full': 16}
    d = sum(r['median_ns'] / 1000 * r['case']['weight'] for r in cells) / 62
    activity = {}
    for r in profiles:
        lines = [line for line in r['report'].splitlines() if 'okc::moe_fused<' in line]
        assert len(lines) == 1
        activity[r['case']['layer']] = float(re.search(r'\s([0-9.]+)\s+µs', lines[0])[1])
    e = (46 * activity['sliding'] + 16 * activity['full']) / 62
    gain = lambda saving: (d / (d - saving)) ** (1 / 6) - 1
    required = lambda factor: d * (1 - factor ** -6)
    return {
        'path': path, 'job': job, 'receipt_git_blob_sha256': hashlib.sha256(blob(path)).hexdigest(),
        'weighted_decode1_us': d, 'expert_activity_us': activity, 'weighted_expert_activity_us': e,
        'activity_fraction_of_endpoint': e / d,
        'net_endpoint_scenarios': [{'saving_us': x, 'score_gain': gain(x)} for x in (1, 2, 3, 5, 10)],
        'conditional_activity_scenarios': [{'activity_reduction': x, 'net_saving_us_if_fully_exposed': e*x,
                                           'score_gain_if_fully_exposed': gain(e*x)} for x in (.1, .2, .3, .5)],
        'score_thresholds': [{'score_gain': x, 'required_net_us': required(1+x),
                              'fraction_of_profiled_activity': required(1+x)/e} for x in (.01, .02, .03)],
        'conditional_layer_gain_at_one_percent_component_gate': gain(e * (1 - 1.01**-6)),
    }


fresh = endpoints('docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json', 'cc474883117947c0')
public = endpoints('muse_decode_layer/profiles/scored.607eb52bdaf24160.json', '607eb52bdaf24160')
reader = json.loads((OUT / 'independent-reader.json').read_text())
assert reader == load('docs/steering/evidence/2026-11-02-worker-t1-moe-check/frozen-reader-result.json')
t1rows = [r for r in reader['report']['rows'] if r[1] == 1]
factors = [x for r in t1rows for x in r[5]]
assert len(t1rows) == 6 and len(factors) == 24 and not reader['decision']['component_pass']
assert len(reader['decision']['affected_regressions']) == 6
factor = math.exp(sum(math.log(x) for x in factors) / len(factors))
snapshot = json.loads((OUT / 'snapshot.json').read_text())
public_score = load(public['path'])['result']['speedup']
layer100 = snapshot['layer_cutoff100']['speedup']
target = layer100 / public_score
output = {
    'evidence_commit': BASE, 'fresh': fresh, 'public': public,
    'model': 'Fixed baselines, other five loads unchanged; score factor=(D/(D-net_weighted_endpoint_saving))^(1/6). Activity conversion is hypothetical, not exposed savings or a bound.',
    'matched_t1_result': {'job': reader['job'], 'factors': len(factors), 'control_candidate_factor_range': [min(factors), max(factors)],
                          'descriptive_geometric_control_candidate_factor': factor,
                          'descriptive_candidate_latency_increase': 1/factor - 1,
                          'component_pass': False, 'full_layer_gain': 'unmeasured'},
    'current_layer100_scenario': {'public_score': public_score, 'sampled_layer100': layer100,
                                  'required_score_factor': target,
                                  'required_decode1_net_saving_us_using_fresh_endpoint': fresh['weighted_decode1_us'] * (1-target**-6),
                                  'fraction_of_fresh_profiled_activity': fresh['weighted_decode1_us'] * (1-target**-6) / fresh['weighted_expert_activity_us'],
                                  'overall_rank_prediction': 'not inferred'},
    'review_blocker': 'none: review c66f5ce6 complete; target 28248eb5 completed; captain 5f687801 parks exact source with no follow-on allocation',
}
(OUT / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'fresh_D_us': fresh['weighted_decode1_us'], 'layer_1pct_required_us': fresh['score_thresholds'][0]['required_net_us'],
                  'component_1pct_conditional_layer_gain': fresh['conditional_layer_gain_at_one_percent_component_gate'],
                  'actual_t1_latency_increase': 1/factor-1, 'layer100_required_net_us': output['current_layer100_scenario']['required_decode1_net_saving_us_using_fresh_endpoint']}))
