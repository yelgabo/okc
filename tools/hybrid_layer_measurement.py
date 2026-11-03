"""Strict owning-receipt comparison for a one-arm layer measurement.

This reports cross-job observations, never paired attribution or promotion.
Keep failed owning receipts even when this complete-result reader rejects them.
"""
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

LOADS = ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')
CELLS = {(load, layer) for load in LOADS for layer in ('sliding', 'full')}


def positive(value):
    assert isinstance(value, (float, int)) and math.isfinite(value) and value > 0
    return value


def score(rows, median='median_ns', baseline='baseline_ns'):
    return math.exp(sum(math.log(
        sum((46 if layer == 'sliding' else 16) * rows[load, layer][baseline]
            for layer in ('sliding', 'full')) /
        sum((46 if layer == 'sliding' else 16) * rows[load, layer][median]
            for layer in ('sliding', 'full'))
    ) for load in LOADS) / 6)


def read(receipt, job, mode='run'):
    assert receipt['id'] == job and receipt['board'] == 'muse_decode_layer'
    assert mode in ('run', 'check') and receipt['mode'] == mode, 'unexpected measurement mode'
    assert receipt['status'] == 'done', 'receipt is not terminal'
    assert re.fullmatch('[0-9a-f]{64}', receipt['content_hash'])
    assert receipt['payload_ref'] == 'muse_decode_layer/'+receipt['content_hash']+'.py'
    result = receipt['result']
    assert result['status'] == 'ok'
    assert result['passed'] is True and result['returncode'] == 0
    assert result['checks']['ok'] is True and not result['checks'].get('flagged')
    assert not result['checks'].get('rejected') and not result['flags'] and not result['error']
    records = result['records']
    tests = [r for r in records if r.get('event') == 'test']
    setup = [r for r in records if r.get('event') == 'setup']
    benchmarks = [r for r in records if r.get('event') == 'benchmark']
    profiles = [r for r in records if r.get('event') == 'profile']
    assert len(tests) == 13 and all(r['status'] == 'pass' for r in tests)
    assert len(setup) == 1 and setup[0]['status'] == 'pass' and setup[0]['error'] is None
    assert positive(setup[0]['prepare_s']) < 300
    assert len(benchmarks) == len(profiles) == 12
    rows = {(r['case']['load'], r['case']['layer']): r for r in benchmarks}
    profile_rows = {(r['case']['load'], r['case']['layer']): r for r in profiles}
    assert rows.keys() == profile_rows.keys() == CELLS
    for key, row in rows.items():
        assert row['status'] == 'pass' and not row['flags']
        assert row['case']['weight'] == (46 if key[1] == 'sliding' else 16)
        for field in ('median_ns', 'median_raw_ns', 'baseline_ns', 'baseline_raw_ns'):
            positive(row[field])
        assert profile_rows[key]['case'] == row['case']
        assert profile_rows[key]['report'].strip()
    official = positive(result['speedup'])
    assert math.isclose(score(rows), official, rel_tol=1e-10, abs_tol=1e-12)
    ci = result['speedup_ci']
    assert len(ci) == 2 and positive(ci[0]) <= official <= positive(ci[1])
    return rows, dict(job=job, official_score=official, official_ci=ci,
                     reconstructed_score=score(rows), prepare_seconds=setup[0]['prepare_s'],
                     tests=13, benchmarks=12, profiles=12, machine=result.get('machine'),
                     evaluator=result.get('evaluator'), server_content_hash=receipt['content_hash'],
                     payload_ref=receipt['payload_ref'])


def qualification(log):
    done = re.findall(r'^HYBRID_QUAL_DONE (.*)$', log, re.M)
    assert len(done) == 1, 'missing or ambiguous startup completion'
    fields = dict(re.findall(r'(\w+)=([^\s]+)', done[0]))
    for name, count in (('selected', 4), ('fallback', 0), ('checks', 48), ('captures', 8), ('fresh', 24),
                        ('empty_checks', 40), ('empty_captures', 8), ('restored', 32)):
        assert fields.get(name) == str(count), ('incomplete startup', name, fields)
    rows = re.findall(r'^HYBRID_QUAL_RESOURCE (\w+) (\[[^\n]+\])$', log, re.M)
    assert len(rows) == 3
    resources = {name: ast.literal_eval(vector) for name, vector in rows}
    assert resources.keys() == {'moe51', 'one8_mode3', 'one8_mode51'}
    for vector in resources.values():
        assert len(vector) == 10 and all(type(x) is int and x >= 0 for x in vector)
        assert vector[1] == 0 and 0 < vector[4] <= vector[5]
        assert vector[3] <= 227*1024
        assert vector[6] >= 1 and vector[7] > 0 and vector[8] == 1
        assert vector[9] == vector[6] * vector[7]
    assert resources['moe51'][4] == 576
    return dict(startup_fields=fields, resources=resources,
                extra_startup_lines=[s for s in log.splitlines() if s.startswith('HYBRID_QUAL_')])


def owning_log(receipt, path):
    # Run in the environment that owns the CLI output path (WSL for this run).
    # An arbitrary supplied log cannot certify another job's qualification.
    expected = Path(receipt['output']['log'])
    assert path.resolve() == expected.resolve(), 'log is not the owning CLI output'
    assert receipt['id'] in expected.name
    raw = path.read_bytes()
    return raw.decode('utf8'), hashlib.sha256(raw).hexdigest()


def compare(new, old, job, control_job, log=None, mode='run', control_mode='run'):
    rows, summary = read(new, job, mode)
    control, control_summary = read(old, control_job, control_mode)
    output = []
    for load in LOADS:
        for layer in ('sliding', 'full'):
            n, b = rows[load, layer], control[load, layer]
            output.append(dict(load=load, layer=layer, target=load in ('verify 16', 'verify 32'),
                new_seed=n['case']['seed'], old_seed=b['case']['seed'],
                new_adjusted_us=n['median_ns']/1000, old_adjusted_us=b['median_ns']/1000,
                adjusted_change_percent=100*(n['median_ns']/b['median_ns']-1),
                new_raw_us=n['median_raw_ns']/1000, old_raw_us=b['median_raw_ns']/1000,
                raw_change_percent=100*(n['median_raw_ns']/b['median_raw_ns']-1),
                new_baseline_us=n['baseline_ns']/1000, old_baseline_us=b['baseline_ns']/1000,
                new_overhead_us=(n['median_raw_ns']-n['median_ns'])/1000,
                old_overhead_us=(b['median_raw_ns']-b['median_ns'])/1000))
    loads = []
    for load in LOADS:
        cost = lambda data: sum((46 if l == 'sliding' else 16)*data[load,l]['median_ns']
                               for l in ('sliding', 'full'))/62000
        n, b = cost(rows), cost(control)
        loads.append(dict(load=load, new_weighted_us=n, old_weighted_us=b,
                          latency_change_percent=100*(n/b-1)))
    summary.update(control=control_summary,
        score_ratio_to_archived_control=summary['official_score']/control_summary['official_score'],
        rows=output, weighted_loads=loads, paired=False, release_decision=False,
        scope='Separate jobs may differ in inputs, routing, tuning, baseline, overhead and machine. '
              'Full official scores and twelve raw/adjusted medians are observations, not paired causal evidence.')
    if log is not None:
        assert summary['prepare_seconds'] < 295
        summary['qualification'] = qualification(log)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path); p.add_argument('control', type=Path)
    p.add_argument('--job', required=True); p.add_argument('--control-job', default='607eb52bdaf24160')
    p.add_argument('--log', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text())
    log, log_sha256 = owning_log(receipt, a.log)
    result = compare(receipt, json.loads(a.control.read_text()), a.job, a.control_job, log)
    result['owning_log_sha256'] = log_sha256
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({k:result[k] for k in ('job','official_score','official_ci','prepare_seconds','score_ratio_to_archived_control')}))
