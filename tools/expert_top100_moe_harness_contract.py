"""Frozen standalone gates, embedded verbatim in the payload."""
import math
import statistics

BANKS = (1102031, 1102037, 1102049)
LOADS = (1, 8, 16, 32, 128, 256)
TOKENS = (1, 4, 8, 16, 32, 33, 128, 256, 257)
DISTRIBUTIONS = ('normal', 'skewed')
CROSSES = ((0, 2), (0, 3), (1, 2), (1, 3))
GLOBAL_TOL = .015
ROW_TOL = .075
REPEATS = 8
CAPSULE_LIMIT = 14000


def require(condition, message):
    if not condition:
        raise RuntimeError('EXPERT_MOE_HARNESS: ' + message)


def positive(value):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0, 'positive finite timing')
    return value


def paired_samples(samples):
    require(len(samples) == 4 and all(len(sample) == REPEATS for sample in samples), 'paired sample count')
    for sample in samples:
        for value in sample:
            positive(value)

    def paired_logs(left, right):
        return [(math.log(samples[left][index] / samples[right][index])
                 + math.log(samples[left][index + 1] / samples[right][index + 1])) / 2
                for index in range(0, REPEATS, 2)]
    ratios = [math.exp(statistics.median(paired_logs(left, right))) for left, right in CROSSES]
    nulls = [statistics.mean(abs(value) for value in paired_logs(left, right)) for left, right in ((0, 1), (2, 3))]
    return [statistics.median(sample) for sample in samples], ratios, nulls


def decision(rows):
    expected = {(seed, tokens, dist) for seed in BANKS for tokens in LOADS for dist in DISTRIBUTIONS}
    require(isinstance(rows, list) and len(rows) == len(expected), '36 timed rows')
    require({tuple(row[:3]) for row in rows} == expected, 'bank/load/distribution coverage')
    summaries, drift, regressions = [], [], []
    for row in rows:
        require(len(row) == 7 and len(row[3]) == 4 and len(row[4]) == 3
                and len(row[5]) == 4 and len(row[6]) == 2, 'timing row layout')
        for value in row[3]:
            positive(value)
        for value in row[5]:
            positive(value)
        require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in row[6]), 'duplicate log null')
        total, occupied, maximum = row[4]
        require(all(type(value) is int for value in row[4]) and 1 <= occupied <= 32
                and 1 <= maximum <= row[1] and maximum <= total <= row[1] * 8, 'route census capacity')
        ratios = row[5]
        if row[1] >= 32 and any(ratio <= 1 / 1.01 for ratio in ratios):
            regressions.append(row[:3])
        if row[1] < 32:
            drift.append([*row[:3], ratios])
    for seed in BANKS:
        for dist in DISTRIBUTIONS:
            selected = [row for row in rows if row[0] == seed and row[2] == dist]
            nulls = [sum(row[6][index] for row in selected) / 6 for index in range(2)]
            logs = [sum(math.log(row[5][index]) for row in selected) / 6 for index in range(4)]
            passed = all(gain >= math.log(1.01) and gain > 2 * max(nulls) for gain in logs)
            summaries.append(dict(bank=seed, dist=dist, gains=[math.exp(gain) for gain in logs],
                                  duplicate_log_nulls=nulls, passed=passed))
    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),
                groups=summaries, affected_regressions=regressions, unchanged_drift=drift)
