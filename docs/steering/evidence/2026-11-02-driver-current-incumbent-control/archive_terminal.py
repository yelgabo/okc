"""Archive own unchanged-control receipt without rounding 64-bit seeds."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
JOB = '94ef698e91964dfa'
raw = (HERE / 'status-wait.raw.json').read_bytes()
try:
    receipt_text, receipt_encoding = raw.decode('utf-8'), 'utf-8'
except UnicodeDecodeError:
    receipt_text, receipt_encoding = raw.decode('cp1252'), 'cp1252'
d = json.loads(receipt_text)
assert (d['id'], d['board'], d['team'], d['mode'], d['status']) == (
    JOB, 'muse_decode_layer', 'yelgabo', 'check', 'done')
r = d['result']
assert r['passed'] and r['returncode'] == 0 and r['status'] == 'ok'
records = r['records']
counts = dict(collections.Counter(x['event'] for x in records))
assert counts == dict(setup=1, test=13, benchmark=12, profile=12)
assert all(x['status'] == 'pass' for x in records if x['event'] in ('test', 'setup'))
assert all('total ' in x['report'] for x in records if x['event'] == 'profile')
log = (HERE / 'profiles' / f'control.{JOB}.log').read_text(encoding='utf-8')
startup = [x for x in log.splitlines() if 'G_RELEASE_STARTUP_OK native' in x]
assert len(startup) == 1 and '56 verified_replays' in startup[0]
base = (ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes()
payload = (HERE / 'control.py').read_bytes()
assert payload[:len(base)] == base and len(payload) == 869198
assert hashlib.sha256(payload).hexdigest() == '04314b16743988c5b172a945e0f10437d78a70bb847ddd865f4a5c1101d6b305'
bench = [x for x in records if x['event'] == 'benchmark']
rows = []
for load in dict.fromkeys(x['case']['load'] for x in bench):
    xs = [x for x in bench if x['case']['load'] == load]
    assert sorted(x['case']['weight'] for x in xs) == [16, 46]
    values = {key: sum(x['case']['weight'] * x[key] / 1000 for x in xs)
              for key in ('median_ns', 'median_raw_ns', 'baseline_ns', 'baseline_raw_ns')}
    rows.append(dict(load=load, weighted_us=values['median_ns'],
        endpoint_us=values['median_ns']/62, baseline_weighted_us=values['baseline_ns'],
        baseline_endpoint_us=values['baseline_ns']/62,
        ratio=values['baseline_ns']/values['median_ns'],
        raw_weighted_us=values['median_raw_ns'], raw_baseline_weighted_us=values['baseline_raw_ns']))
score = math.prod(x['ratio'] for x in rows)**(1/6)
assert abs(score-r['speedup']) < 1e-12
comparisons = []
for path in ('muse_decode_layer/cand/profiles/g1b4_stage_release.cc474883117947c0.json',
             'docs/steering/evidence/2026-11-02-driver-direct-weight-first-check/status-wait.raw.json'):
    old = json.loads((ROOT / path).read_text())
    oldr = old.get('result', old)
    oldbench = oldr.get('benchmarks', [x for x in oldr.get('records', []) if x['event'] == 'benchmark'])
    oldev = oldr.get('evaluator', {})
    comparisons.append(dict(path=path, evaluator=oldev,
        all_benchmark_seeds_different=all(x['case']['seed'] != y['case']['seed'] for x,y in zip(bench,oldbench))))
summary = dict(job=JOB, score=r['speedup'], CI=r['speedup_ci'], reconstructed_score=score,
    counts=counts, prepare_s=records[0]['prepare_s'], rows=rows,
    d32_cells=[x for x in bench if x['case']['load'] == 'decode 32'], startup=startup,
    machine=r['machine'], cluster=d['cluster'], evaluator=r['evaluator'], comparisons=comparisons,
    raw_receipt_encoding=receipt_encoding,
    limitation='Unchanged incumbent calibration; no optimization gain or matched causal runner effect.')
(HERE / 'terminal-canonical.utf8.json').write_text(json.dumps(d, indent=2, ensure_ascii=True)+'\n', encoding='utf-8', newline='\n')
(HERE / 'terminal-analysis.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8', newline='\n')
manifest = {str(p.relative_to(ROOT)).replace('\\','/'): dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.name != 'archive-manifest.json'}
(HERE / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps({k:summary[k] for k in ('job','score','CI','counts','prepare_s','rows')},indent=2))
