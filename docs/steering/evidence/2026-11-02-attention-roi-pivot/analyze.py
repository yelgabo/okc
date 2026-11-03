"""Offline source/owning-receipt sizing. No compilation, device access or POST."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

BASE = '8a05389d1052ac5a01090965e1e5e17b03775299'
OUT = Path(__file__).parent
LOADS = ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')
LONG = ('decode 16', 'decode 32', 'verify 16', 'verify 32')


def blob(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path])


def load(path):
    return json.loads(blob(path))


fresh = load('docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json')
assert fresh['id'] == 'cc474883117947c0' and fresh['result']['passed']
records = fresh['result']['records']
endpoints, attention, expert = {}, {}, {}
for name in LOADS:
    benches = [r for r in records if r['event'] == 'benchmark' and r['case']['load'] == name]
    assert {r['case']['layer']: r['case']['weight'] for r in benches} == {'sliding': 46, 'full': 16}
    endpoints[name] = sum(r['median_ns']/1000*r['case']['weight'] for r in benches)/62
    if name not in LONG:
        continue
    profiles = [r for r in records if r['event'] == 'profile' and r['case']['load'] == name]
    expert[name] = 0
    for r in profiles:
        lines = r['report'].splitlines()
        moe = [s for s in lines if 'okc::moe_fused<' in s]
        assert len(moe) == 1
        expert[name] += float(re.search(r'\s([0-9.]+)\s+µs', moe[0])[1])*r['case']['weight']/62
        if r['case']['layer'] == 'full':
            att = [s for s in lines if 'attn_kernel<' in s]
            assert len(att) == 1
            attention[name] = float(re.search(r'\s([0-9.]+)\s+µs', att[0])[1])


def factor(savings):
    assert all(0 <= savings.get(n, 0) < endpoints[n] for n in LOADS)
    return math.exp(sum(math.log(endpoints[n]/(endpoints[n]-savings.get(n, 0))) for n in LOADS)/6)


def inverse(fn, target, high):
    low = 0
    assert fn(high) >= target
    for _ in range(100):
        mid = (low+high)/2
        if fn(mid) < target:
            low = mid
        else:
            high = mid
    return (low+high)/2


snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
public = load('muse_decode_layer/profiles/scored.607eb52bdaf24160.json')
target = snapshot['layer_cutoff100']['speedup']/public['result']['speedup']
equal_full = lambda x: factor({n: 16/62*x for n in LONG})
attention_fraction = lambda x: factor({n: 16/62*attention[n]*x for n in LONG})
expert_fraction = lambda x: factor({n: expert[n]*x for n in LONG})
manifest = load('docs/steering/evidence/2026-11-02-attention-continuity/manifest.json')
candidate = blob('docs/steering/evidence/2026-11-02-attention-continuity/candidate.py')
assert hashlib.sha256(candidate).hexdigest() == manifest['files']['candidate.py']['sha256']
source = blob('muse_decode_layer/cand/g1b4_stage_release.py')
assert hashlib.sha256(source).hexdigest() == 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
output = {
    'evidence_commit': BASE, 'fresh_job': fresh['id'], 'public_job': public['id'],
    'weighted_layer_endpoints_us': endpoints, 'long_full_attention_activity_us': attention,
    'long_weighted_expert_activity_us': expert, 'current_layer_score_ratio_needed': target,
    'net_equal_full_cell_scenarios': [{'saving_us': x, 'layer_score_gain': equal_full(x)-1} for x in (1, 3, 5, 7, 10)],
    'net_equal_full_cell_required': [{'score_gain': gain, 'saving_us_each': inverse(equal_full, 1+gain, 100)} for gain in (.01, .03, target-1)],
    'conditional_whole_attention_activity_scenarios': [{'activity_reduction': x, 'layer_score_gain_if_fully_exposed': attention_fraction(x)-1} for x in (.1, .2, .3, .5, .9)],
    'conditional_long_expert_activity_scenarios': [{'activity_reduction': x, 'layer_score_gain_if_fully_exposed': expert_fraction(x)-1} for x in (.1, .2, .9)],
    'conditional_attention_fraction_needed_for_cutoff': inverse(attention_fraction, target, 1),
    'conditional_expert_fraction_needed_for_cutoff': inverse(expert_fraction, target, 1),
    'conditional_expert_3pct_fraction': inverse(expert_fraction, 1.03, 1),
    'boundary_coverage': [{'batch': b, 'grid': 264, 'cross_request_ctas_at_most': b-1,
                           'fraction_at_most': (b-1)/264} for b in (16, 32)],
    'coverage_warning': 'This is not a timing bound. Boundary CTAs can be critical; CONT init/tail/resource effects can apply elsewhere. Actual scored CUDA sequence lengths/nseg were not archived.',
    'source_candidate': manifest['files']['candidate.py'],
    'decision': 'DEPRIORITIZE_CONTINUITY; HOLD_TARGET; freeze delivered source; fund architecture-first multi-load dominant-computation decision',
    'measured_attention_gain': None,
    'model_warning': 'Activity reductions are conditional sizing, not exposed savings, predictions or recoverable ceilings. Fixed baselines and unchanged other work are assumed.',
}
(OUT/'analysis.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'target_factor': target, '7us_full_cell_layer_gain': equal_full(7)-1,
                  '1pct_us_full_cell': inverse(equal_full, 1.01, 100),
                  'top100_us_full_cell': inverse(equal_full, target, 100),
                  'expert20pct_layer_gain_conditional': expert_fraction(.2)-1,
                  'expert10x_layer_gain_conditional': expert_fraction(.9)-1}))
