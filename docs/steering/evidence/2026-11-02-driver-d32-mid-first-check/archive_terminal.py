"""Reconstruct own MID result; preserve receipt bytes and integer seeds."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
JOB = 'a6fe2769779e4299'


def load(path):
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode('utf-8'), 'utf-8'
    except UnicodeDecodeError:
        text, encoding = raw.decode('cp1252'), 'cp1252'
    return json.loads(text), encoding


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n', encoding='utf-8', newline='\n')


d, raw_encoding = load(HERE / 'status-wait.raw.json')
assert (d['id'], d['team'], d['board'], d['mode'], d['status']) == (
    JOB, 'yelgabo', 'muse_decode_layer', 'check', 'done')
r = d['result']
assert r['passed'] and r['status'] == 'ok' and r['returncode'] == 0 and not r['flags']
records = r['records']
counts = dict(Counter(x['event'] for x in records))
assert counts == dict(setup=1, test=13, benchmark=12, profile=12)
assert all(x['status'] == 'pass' for x in records if x['event'] in ('setup','test'))
assert all('total ' in x['report'] for x in records if x['event'] == 'profile')
source = ROOT / 'docs/steering/evidence/2026-11-02-worker-d32-mid-family/generated/candidate.py'
b = source.read_bytes()
assert len(b) == 870649 and b'\r' not in b
assert hashlib.sha256(b).hexdigest() == 'e607fb894b7ca9fb8c0245d986e1c1894024295efd5b1509ac5b8314329ca089'
association, _ = load(HERE / 'upload-association.json')
assert association['server_content_identifier'] == d['content_hash']
assert association['physical_source_sha256'] == hashlib.sha256(b).hexdigest()
log_path = ROOT / 'docs/steering/evidence/2026-11-02-worker-d32-mid-family/generated/profiles' / f'candidate.{JOB}.log'
log = log_path.read_text(encoding='utf-8')
lines = log.splitlines()
markers = [x for x in lines if x.startswith('D32_MID_SELECTION ')]
assert len(markers) == 1
selection = ast.literal_eval(markers[0].split(' ', 1)[1])
assert selection['tokens'] == 32 and selection['key'] == 'mid'
assert selection['source_sha256'] == 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'
manifest, _ = load(source.parent / 'manifest.json')
assert selection['defines'] == manifest['mid_defines']
assert selection['resource_query'] is None
startup = [x for x in lines if x.startswith('G_RELEASE_STARTUP_OK native ')]
assert len(startup) == 1 and '56 verified_replays' in startup[0]
control, control_encoding = load(ROOT / 'docs/steering/evidence/2026-11-02-driver-current-incumbent-control/status-wait.raw.json')
assert control['id'] == '94ef698e91964dfa' and control['result']['passed']
bench = [x for x in records if x['event'] == 'benchmark']
control_bench = [x for x in control['result']['records'] if x['event'] == 'benchmark']
rows = []
for name in dict.fromkeys(x['case']['load'] for x in bench):
    xs = [x for x in bench if x['case']['load'] == name]
    cs = [x for x in control_bench if x['case']['load'] == name]
    assert sorted(x['case']['weight'] for x in xs) == [16,46]
    def weighted(items, key):
        return sum(x['case']['weight'] * x[key]/1000 for x in items)
    current, old = weighted(xs,'median_ns'), weighted(cs,'median_ns')
    rows.append(dict(load=name, C_us=current, endpoint_C_over_62_us=current/62,
        baseline_B_us=weighted(xs,'baseline_ns'), ratio=weighted(xs,'baseline_ns')/current,
        raw_C_us=weighted(xs,'median_raw_ns'), raw_B_us=weighted(xs,'baseline_raw_ns'),
        control_C_us=old, descriptive_latency_change_pct=(current/old-1)*100))
reconstructed = math.prod(x['ratio'] for x in rows)**(1/6)
assert abs(reconstructed-r['speedup']) < 1e-12
assert r['evaluator'] == control['result']['evaluator']
assert all(x['case']['seed'] != y['case']['seed'] for x,y in zip(bench,control_bench))
write(HERE / 'terminal-canonical.utf8.json', d)
summary = dict(job=JOB, status='done/ok', score=r['speedup'], CI=r['speedup_ci'],
    reconstructed_score=reconstructed, counts=counts, prepare_s=records[0]['prepare_s'],
    source_sha256=hashlib.sha256(b).hexdigest(), source_bytes=len(b),
    server_content_identifier=d['content_hash'], selection=selection, startup=startup,
    rows=rows, d32_cells=[x for x in bench if x['case']['load'] == 'decode 32'],
    machine=r['machine'], cluster=d['cluster'], evaluator=r['evaluator'],
    raw_receipt_encoding=raw_encoding, control_raw_encoding=control_encoding,
    canonical_derivative='terminal-canonical.utf8.json: ASCII-escaped UTF-8 JSON; original raw bytes preserved',
    all_twelve_seeds_differ_from_control=True, matched_causal_effect=False,
    decision='PARK exact e607 fixed MID17..32; no useful D32 signal, no automatic second check',
    subsequent_D32_owner='Decode32 Worker under direct human D32-02/9dea190b; root retires pre-existing job only')
write(HERE / 'terminal-analysis.json', summary)
files = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'archive-manifest.json']
files += list(log_path.parent.glob(f'candidate.{JOB}.*'))
write(HERE / 'archive-manifest.json', {str(p.relative_to(ROOT)).replace('\\','/'):
    dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)})
print(json.dumps({k:summary[k] for k in ('job','score','CI','counts','prepare_s','raw_receipt_encoding','rows')},indent=2))
