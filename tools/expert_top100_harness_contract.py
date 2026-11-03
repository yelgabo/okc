"""Frozen private experiment contract. Pure CPU; no dispatch or promotion."""
import math

BANKS = (1102031, 1102037, 1102049)
LOADS = (1, 8, 16, 32, 128, 256)
AFFECTED = (32, 128, 256)
CROSSES = ((0, 2), (0, 3), (1, 2), (1, 3))
CELLS = {(t, c, s) for t in LOADS for c in ('sliding', 'full') for s in BANKS}


def positive(value):
    assert type(value) in (int, float) and math.isfinite(value) and value > 0, 'invalid timing'
    return value


def decision(rows):
    assert len(rows) == 36 and {(r['tokens'], r['context'], r['seed']) for r in rows} == CELLS, 'row coverage'
    summaries = []
    for seed in BANKS:
        gains = [0.] * 4
        null = 0.
        regressions, unchanged = [], []
        weighted = []
        for t in LOADS:
            pair = {r['context']: r['times_us'] for r in rows if r['tokens'] == t and r['seed'] == seed}
            for times in pair.values():
                assert len(times) == 4, 'placement coverage'
                for value in times:
                    positive(value)
            cost = [(46 * pair['sliding'][i] + 16 * pair['full'][i]) / 62 for i in range(4)]
            weighted.append({'tokens': t, 'us': cost})
            for k, (c, v) in enumerate(CROSSES):
                gains[k] += math.log(cost[c] / cost[v]) / 6
                for context, times in pair.items():
                    ratio = times[v] / times[c]
                    row = [t, context, c, v, ratio]
                    if t in AFFECTED and ratio >= 1.01:
                        regressions.append(row)
                    if t not in AFFECTED:
                        unchanged.append(row)
            null += max(abs(math.log(cost[0] / cost[1])), abs(math.log(cost[2] / cost[3]))) / 6
        summaries.append(dict(seed=seed, cross_log_gains=gains, duplicate_log_null=null,
            weighted=weighted, affected_regressions=regressions, unchanged_drift=unchanged,
            gate=min(gains) >= math.log(1.01) and min(gains) > 2 * null and not regressions))
    return dict(gate=all(s['gate'] for s in summaries), banks=summaries)


def route_key(tokens):
    assert tokens in LOADS, 'unknown load'
    return {1: 'tiny', 8: 'tiny2', 16: 'mid', 32: 'b4', 128: 'main', 256: 'main'}[tokens]


def validate_rows(rows):
    result = decision(rows)
    for row in rows:
        t = row['tokens']
        assert len(row['component_us']) == 4, 'component placements'
        for value in row['component_us']:
            positive(value)
        assert row['actual_routes'] == [[[t, route_key(t), i >= 2]] for i in range(4)], 'actual route mismatch'
        assert type(row['local_start']) is int and row['local_start'] in range(0, 256, 32), 'local start'
        assert len(row['reference_group_counts']) == 8 and sum(row['reference_group_counts']) == t * 8, 'routing counts'
        assert all(type(n) is int and n >= 0 for n in row['reference_group_counts']), 'routing counts'
    return result
