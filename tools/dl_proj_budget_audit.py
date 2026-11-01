"""Offline arithmetic on retained layer profiles; never estimates exposed GPU time."""
import argparse
import json
import math
from pathlib import Path
import re


FILES = {
    'g0': 'muse_decode_layer.3a5f879df0d44825.json',
    'g1b2': 'g1b2.profile.736a5cbc977f4ad4.json',
    'g1a': 'g1a.profile.606f47f251f34873.json',
    'owned': 'combo_pure.47bec92021ef4bbe.profile.json',
}


def dense_rows(report):
    rows = []
    for line in report.splitlines():
        match = re.match(r'(.+?)\s+([0-9.]+) µs\s+[0-9.]+%\s+(\d+)$', line)
        if match and any(s in match[1] for s in ('sk2_kernel', 'sk_kernel', 'wd_kernel',
                                               'nvjet_', 'void rq<', 'cgemv', '::router_kernel')):
            rows.append((match[1], float(match[2]), int(match[3])))
    return rows


def load_times(data, field='median_ns'):
    result = {}
    for b in data['benchmarks']:
        c = b['case']
        result.setdefault(c['load'], []).append((c['weight'], b[field] / 1000))
    assert len(result) == 6 and all(sorted(w for w, _ in rows) == [16, 46] for rows in result.values())
    return {key: sum(w * t for w, t in rows) / 62 for key, rows in result.items()}


def counterfactual(data, savings):
    original = load_times(data)
    changed = {'benchmarks': []}
    for b in data['benchmarks']:
        c = b['case']
        t = b['median_ns'] / 1000 - savings.get((c['load'], c['layer']), 0)
        assert t > 0
        changed['benchmarks'].append(dict(b, median_ns=t * 1000))
    updated = load_times(changed)
    return math.exp(sum(math.log(original[k] / updated[k]) for k in original) / 6)


def main(root):
    datasets = {k: json.loads((root / 'muse_decode_layer/profiles' / v).read_text(encoding='utf-8')) for k, v in FILES.items()}
    base = datasets['g1b2']
    cell_savings = {}
    print('load layer weight G1b2_us G0_us dense_G1b2 dense_G0 dense_G1a dense_owned')
    for i, b in enumerate(base['benchmarks']):
        c = b['case']
        assert all((d['benchmarks'][i]['case']['load'], d['benchmarks'][i]['case']['layer']) ==
                   (c['load'], c['layer']) for d in datasets.values())
        dense = {k: sum(t for _, t, _ in dense_rows(d['profiles'][i]['report'])) for k, d in datasets.items()}
        # Only larger verify loads have changed large-M projection dispatch.
        cell_savings[c['load'], c['layer']] = max(0, dense['g1b2'] - dense['g0']) if i >= 8 else 0
        print(c['load'], c['layer'], c['weight'], *(round(x, 6) for x in
              (b['median_ns'] / 1000, datasets['g0']['benchmarks'][i]['median_ns'] / 1000,
               dense['g1b2'], dense['g0'], dense['g1a'], dense['owned'])))
    for key, data in datasets.items():
        baseline, ours = load_times(data, 'baseline_ns'), load_times(data)
        score = math.exp(sum(math.log(baseline[k] / ours[k]) for k in ours) / 6)
        assert abs(score - data['summary']['speedup']) < 1e-9
        print('LOADS', key, ours, 'SCORE', score)
    print('LARGEM_POSITIVE_DENSE_GAP_FACTOR', counterfactual(base, cell_savings))
    # Best supported group comparison: replace qkv+O together. Retained NNT rows
    # combine two calls, so no per-op attribution is inferred here.
    group = {}
    for i in (8, 9):
        current = [t for name, t, calls in dense_rows(base['profiles'][i]['report'])
                   if name == 'nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT' and calls == 2]
        owned = [t for name, t, calls in dense_rows(datasets['owned']['profiles'][i]['report'])
                 if 'wd_kernel<1, 128, 3, 1, false' in name or
                 name == 'nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT']
        assert len(current) == 1 and len(owned) == 2
        c = base['benchmarks'][i]['case']
        group[c['load'], c['layer']] = current[0] - sum(owned)
    print('OWNED_QKV_O_GROUP_FACTOR', counterfactual(base, group))
    for i in range(8, 12):
        print('LARGE_CELL_ROWS', i, {k: dense_rows(d['profiles'][i]['report']) for k, d in datasets.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
