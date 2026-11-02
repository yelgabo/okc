"""Compare two complete owning layer receipts without claiming a matched experiment."""
import argparse
import json
import math
from pathlib import Path

LOADS = ('d1', 'd16', 'd32', 'v1', 'v16', 'v32')


def shape_name(case):
    prefix = 'd' if case.get('q_len', 1) == 1 else 'v'
    if case.get('q_len', 1) not in (1, 8):
        raise ValueError('unexpected query length')
    if case.get('layer') not in ('sliding', 'full'):
        raise ValueError('unexpected layer kind')
    return f"{prefix}{case['batch']}{'s' if case['layer'] == 'sliding' else 'f'}"


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('latency/score must be numeric')
    if not math.isfinite(value) or value <= 0:
        raise ValueError('latency/score must be finite and positive')
    return value


def geometric_mean(values):
    return math.exp(sum(math.log(positive(value)) for value in values) / len(values))


def validate(receipt, expected_job, expected_mode, expected_team='yelgabo'):
    if (receipt.get('id') != expected_job or receipt.get('board') != 'muse_decode_layer'
            or receipt.get('mode') != expected_mode or receipt.get('team') != expected_team):
        raise ValueError('owning job/board/mode/team mismatch')
    result = receipt.get('result') or {}
    content_hash = receipt.get('content_hash', '')
    if (len(content_hash) != 64 or any(character not in '0123456789abcdef' for character in content_hash)
            or receipt.get('payload_ref') != f'muse_decode_layer/{content_hash}.py'):
        raise ValueError('server content/payload association mismatch')
    if (receipt.get('status') != 'done' or result.get('passed') is not True
            or result.get('returncode') != 0):
        raise ValueError('receipt is not a successful terminal job')
    checks = result.get('checks') or {}
    if checks.get('ok') is not True or checks.get('rejected') or checks.get('flagged'):
        raise ValueError('official checks not clean')
    records = result.get('records', [])
    tests = [record for record in records if record.get('event') == 'test']
    if len(tests) != 13 or any(record.get('status') != 'pass' for record in tests):
        raise ValueError('expected thirteen passing official tests')
    expected_shapes = {load + suffix for load in LOADS for suffix in 'sf'}
    by_event = {}
    for event in ('benchmark', 'profile'):
        selected = [record for record in records if record.get('event') == event]
        indexed = {}
        for record in selected:
            name = shape_name(record['case'])
            if name in indexed:
                raise ValueError('duplicate scored shape')
            indexed[name] = record
        if len(selected) != 12 or indexed.keys() != expected_shapes:
            raise ValueError('expected twelve distinct scored shapes')
        by_event[event] = indexed
    shapes = by_event['benchmark']
    for name, record in shapes.items():
        if record.get('status') != 'pass':
            raise ValueError('benchmark not passing')
        weight = 46 if name.endswith('s') else 16
        if record['case'].get('weight') != weight:
            raise ValueError('unexpected scored weight')
        positive(record['median_ns'])
        positive(record['baseline_ns'])
        if 'median_raw_ns' in record:
            positive(record['median_raw_ns'])
        profile = by_event['profile'][name]
        if profile['case'] != record['case']:
            raise ValueError('profile/benchmark case mismatch')
        if not isinstance(profile.get('report'), str) or not profile['report']:
            raise ValueError('empty profile report')
    loads = {}
    for load in LOADS:
        cells = [shapes[load + suffix] for suffix in 'sf']
        loads[load] = {field: sum(record['case']['weight'] * record[field]
                                 for record in cells) / 62 / 1000
                       for field in ('median_ns', 'baseline_ns')}
        loads[load]['raw_us'] = (sum(record['case']['weight'] * record['median_raw_ns']
                                    for record in cells) / 62 / 1000
                                if all('median_raw_ns' in record for record in cells) else None)
    recomputed = geometric_mean([values['baseline_ns'] / values['median_ns']
                                 for values in loads.values()])
    if not math.isclose(recomputed, positive(result['speedup']), rel_tol=1e-9):
        raise ValueError('reported score does not match twelve weighted cells')
    return dict(shapes=shapes, loads=loads, score=recomputed,
                job=expected_job, mode=expected_mode, team=expected_team,
                content_hash=receipt.get('content_hash'), payload_ref=receipt.get('payload_ref'),
                machine=result.get('machine'), gpu=result.get('gpu'),
                speedup_ci=result.get('speedup_ci'))


def compare(control, candidate):
    rows = []
    for load in LOADS:
        before, after = control['loads'][load], candidate['loads'][load]
        rows.append(dict(load=load, control_adjusted_us=before['median_ns'],
                         candidate_adjusted_us=after['median_ns'],
                         control_raw_us=before['raw_us'], candidate_raw_us=after['raw_us'],
                         latency_factor=before['median_ns'] / after['median_ns'],
                         latency_change_percent=100 * (after['median_ns'] / before['median_ns'] - 1),
                         baseline_factor=after['baseline_ns'] / before['baseline_ns']))
    latency_factor = geometric_mean([row['latency_factor'] for row in rows])
    baseline_factor = geometric_mean([row['baseline_factor'] for row in rows])
    score_factor = candidate['score'] / control['score']
    if not math.isclose(score_factor, latency_factor * baseline_factor, rel_tol=1e-12):
        raise ValueError('score-factor decomposition failed')
    cells = []
    for name in sorted(control['shapes']):
        before, after = control['shapes'][name], candidate['shapes'][name]
        cells.append(dict(shape=name, control_seed=before['case'].get('seed'),
                          candidate_seed=after['case'].get('seed'),
                          same_seed=before['case'].get('seed') == after['case'].get('seed'),
                          latency_factor=before['median_ns'] / after['median_ns']))
    return dict(control={key:value for key,value in control.items() if key not in ('shapes', 'loads')},
                candidate={key:value for key,value in candidate.items() if key not in ('shapes', 'loads')},
                loads=rows, cells=cells, adjusted_latency_factor=latency_factor,
                baseline_factor=baseline_factor, official_score_factor=score_factor,
                descriptive_screen=latency_factor >= 1.01 and all(
                    row['latency_change_percent'] <= 1.0 for row in rows),
                performance_qualified=False, promotion_authorized=False,
                limitations=['Independent jobs, not a matched causal experiment',
                             'Screen requires separate selected-route/profile verification',
                             'Reported score CIs are not a confidence interval on treatment effect',
                             'Content hash is server identity, not physical upload SHA256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--control-job', required=True)
    parser.add_argument('--candidate-job', required=True)
    parser.add_argument('--control-mode', choices=('check', 'run'), default='check')
    parser.add_argument('--candidate-mode', choices=('check', 'run'), default='check')
    args = parser.parse_args()
    control = validate(json.loads(args.control.read_text(encoding='utf-8')), args.control_job, args.control_mode)
    candidate = validate(json.loads(args.candidate.read_text(encoding='utf-8')), args.candidate_job, args.candidate_mode)
    print(json.dumps(compare(control, candidate), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
