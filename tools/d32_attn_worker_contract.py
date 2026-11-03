"""Prospective D32 matched timing decision, independent of whole-score gates."""
import math

LOADS = (1, 8, 16, 32, 128, 256)
CONTEXTS = ('sliding', 'full')
BANKS = (1102031, 1102037, 1102049)


def decision(rows):
    expected = {(t, c, s) for t in LOADS for c in CONTEXTS for s in BANKS}
    indexed = {}
    for row in rows:
        key = row['tokens'], row['context'], row['seed']
        values = row['times_us']
        if key in indexed or len(values) != 4 or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('invalid paired timing record')
        indexed[key] = values
    if set(indexed) != expected:
        raise ValueError('missing or unexpected load, context, or seed')
    decisions = []
    for seed in BANKS:
        stable = True
        unchanged = True
        for t in LOADS:
            for context in CONTEXTS:
                a, b, c, d = indexed[t, context, seed]
                duplicate = max(abs(a - b), abs(c - d))
                stable &= duplicate <= max(.5, .01 * min(a, b, c, d))
                if (t, context) != (32, 'full'):
                    unchanged &= max(c, d) - min(a, b) <= max(.5, 2 * duplicate)
        sliding = indexed[32, 'sliding', seed]
        full = indexed[32, 'full', seed]
        costs = [46 * sliding[i] + 16 * full[i] for i in range(4)]
        duplicate = max(abs(costs[0] - costs[1]), abs(costs[2] - costs[3]))
        saving = min(costs[:2]) - max(costs[2:])
        full_duplicate = max(abs(full[0] - full[1]), abs(full[2] - full[3]))
        full_saving = min(full[:2]) - max(full[2:])
        passed = stable and unchanged and saving > 2 * duplicate and full_saving > 2 * full_duplicate
        decisions.append(dict(seed=seed, weighted_us=costs, weighted_saving_us=saving,
                              weighted_duplicate_us=duplicate, full_saving_us=full_saving,
                              full_duplicate_us=full_duplicate, stable=stable,
                              other_cells_preserved=unchanged, passed=passed))
    return {'passed': all(row['passed'] for row in decisions), 'banks': decisions}
