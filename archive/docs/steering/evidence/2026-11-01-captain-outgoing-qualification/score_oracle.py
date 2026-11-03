"""Independent CPU arithmetic only, not a receipt or runtime qualifier."""
import math

ARMS = ('raw_off_1', 'raw_off_2', 'adapter_off', 'adapter_on')
SEEDS = (993811, 993817, 993821)
TOKENS = (1, 8, 16, 32, 128, 256)
CELLS = {(seed, tokens, context) for seed in SEEDS for tokens in TOKENS for context in ('sliding', 'full')}


def evaluate(rows, arms=ARMS):
    assert tuple(arms) == ARMS, 'arm mapping'
    assert len(rows) == len(CELLS), 'row count'
    indexed = {}
    for row in rows:
        assert type(row['seed']) is int and type(row['tokens']) is int, 'cell types'
        key = (row['seed'], row['tokens'], row['context'])
        assert key in CELLS and key not in indexed, 'cell identity/duplicate'
        times = row['times_us']
        assert len(times) == len(ARMS), 'arm count'
        assert all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in times), 'invalid time'
        indexed[key] = times
    assert set(indexed) == CELLS, 'coverage'
    banks = []
    for seed in SEEDS:
        net, short, adaptation = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        isolated, null = 0.0, 0.0
        regression = [True, True]
        for tokens in TOKENS:
            sliding = indexed[seed, tokens, 'sliding']
            full = indexed[seed, tokens, 'full']
            costs = [46 * sliding[arm] + 16 * full[arm] for arm in range(4)]
            null += abs(math.log(costs[0] / costs[1])) / 6
            isolated += math.log(costs[2] / costs[3]) / 6
            for control in (0, 1):
                gain = math.log(costs[control] / costs[3]) / 6
                net[control] += gain
                adaptation[control] += math.log(costs[control] / costs[2]) / 6
                if tokens in (1, 8):
                    short[control] += gain
                regression[control] &= all(times[3] - times[control] <= max(0.5, 0.01 * times[control]) for times in (sliding, full))
        threshold = max(math.log(1.01), 2 * null)
        banks.append(dict(seed=seed, net_log_gains=net, short_log_contributions=short, adapter_off_log_gains=adaptation, isolated_pdl_log_gain=isolated, duplicate_log_null=null, threshold_log_gain=threshold, context_regression_gates=regression, arithmetic_gate=min(net) > threshold and all(regression)))
    return dict(arms=list(ARMS), banks=banks, arithmetic_gate=all(bank['arithmetic_gate'] for bank in banks), scope='Arithmetic only; owning receipt/source/runtime/correctness evidence required separately')
