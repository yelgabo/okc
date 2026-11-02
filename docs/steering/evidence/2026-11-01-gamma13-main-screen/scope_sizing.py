import hashlib
import json
import math
from pathlib import Path
import re

root = Path(__file__).resolve().parents[4]
source = root / 'muse_decode_layer/profiles/g-common-rmw-v2.c1931f0c9e834485.json'
data = json.loads(source.read_text())
groups = {}
for benchmark, profile in zip(data['benchmarks'], data['profiles']):
    assert benchmark['case'] == profile['case']
    line = next(line for line in profile['report'].splitlines() if 'moe_fused' in line)
    duration = float(re.search(r'\s(\d+\.\d+)\s', line)[1])
    case = benchmark['case']
    groups.setdefault(case['load'], []).append(dict(weight=case['weight'],
        adjusted_us=benchmark['median_ns'] / 1000, profiled_expert_us=duration))

selected = {'decode 32', 'verify 16', 'verify 32'}

def factor(fraction):
    return math.prod(
        sum(row['weight'] * row['adjusted_us'] for row in rows)
        / sum(row['weight'] * (row['adjusted_us'] - fraction * row['profiled_expert_us']) for row in rows)
        if load in selected else 1 for load, rows in groups.items()) ** (1 / 6)

low, high = 0, 0.5
for iteration in range(60):
    middle = (low + high) / 2
    if factor(middle) < 2.2562 / 2.1261:
        low = middle
    else:
        high = middle
print(json.dumps(dict(source=str(source.relative_to(root)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    groups=groups, selected_loads=sorted(selected), scenarios={str(fraction): factor(fraction) for fraction in (0.1, 0.15, 0.2, 0.25)},
    modeled_expert_reduction_for_cutoff=high,
    disclaimer='Profile duration subtraction from official adjusted medians, full exposure/no replacement assumption; NOT a bound, forecast, exposed epilogue saving or candidate score.'), indent=2))
