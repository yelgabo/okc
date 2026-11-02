"""Reconcile exact owning G0/607 receipts offline; no GPU or challenge calls."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted(records, field):
    grouped = {}
    for record in records:
        case = record['case']
        grouped.setdefault(case['load'], []).append((case['weight'], record[field] / 1000))
    assert len(grouped) == 6
    assert all(sorted(weight for weight, _ in cells) == [16, 46] for cells in grouped.values())
    return {load: sum(weight * value for weight, value in cells) / 62 for load, cells in grouped.items()}


def score(baseline, latency):
    return math.exp(sum(math.log(baseline[load] / latency[load]) for load in latency) / 6)


def kernels(report):
    rows = []
    for line in report.splitlines():
        match = re.fullmatch(r'(.+?)\s+([0-9.]+)\s+(?:µ|Âµ)s\s+[0-9.]+%\s+(\d+)', line)
        if match:
            rows.append({'name': match[1].strip(), 'us': float(match[2]), 'calls': int(match[3])})
    assert rows
    return rows


def reconcile(root):
    sources = [('.build/today/kel/kernels/muse_decode_layer.py', '1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a'),
               ('muse_decode_layer/cand/g1b4_stage_release.py', 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab')]
    source_pins = []
    for relative, expected in sources:
        path = root / relative
        assert digest(path) == expected
        normalized = path.read_bytes().replace(b'\r\n', b'\n')
        source_pins.append({'path': relative, 'sha256': expected, 'bytes': path.stat().st_size,
                            'lf_sha256': hashlib.sha256(normalized).hexdigest(), 'lf_bytes': len(normalized)})
    specifications = [('G0', 'kel.3a5f879df0d44825.json', '3a5f879df0d44825', 'check'),
                      ('G607', 'scored.607eb52bdaf24160.json', '607eb52bdaf24160', 'run')]
    jobs = []
    for label, filename, job_id, mode in specifications:
        relative = 'muse_decode_layer/profiles/' + filename
        path = root / relative
        receipt = json.loads(path.read_text(encoding='utf-8'))
        assert receipt['id'] == job_id and receipt['mode'] == mode
        assert receipt['board'] == 'muse_decode_layer' and receipt['status'] == 'done'
        result = receipt['result']
        assert result['passed'] and result['returncode'] == 0 and result['gpu'] == 'H200'
        records = result['records']
        benchmarks = [record for record in records if record['event'] == 'benchmark']
        profiles = [record for record in records if record['event'] == 'profile']
        tests = [record for record in records if record['event'] == 'test']
        assert len(benchmarks) == len(profiles) == 12 and len(tests) == 13
        latency, baseline = weighted(benchmarks, 'median_ns'), weighted(benchmarks, 'baseline_ns')
        reconstructed = score(baseline, latency)
        assert abs(reconstructed - result['speedup']) < 1e-12
        jobs.append({'label': label, 'receipt': relative, 'sha256': digest(path), 'id': job_id,
                     'content_hash': receipt['content_hash'], 'mode': mode,
                     'node': result['machine']['node'], 'evaluator': result['evaluator']['code_sha256'],
                     'score': reconstructed, 'ci': result['speedup_ci'], 'latency_us': latency,
                     'baseline_us': baseline, 'benchmarks': benchmarks, 'profiles': profiles})
    original, fork = jobs
    assert original['evaluator'] == fork['evaluator']
    cells = []
    for before, after, old_profile, new_profile in zip(original['benchmarks'], fork['benchmarks'], original['profiles'], fork['profiles']):
        old_case, new_case = before['case'], after['case']
        assert {key: value for key, value in old_case.items() if key != 'seed'} == {key: value for key, value in new_case.items() if key != 'seed'}
        assert old_profile['case'] == old_case and new_profile['case'] == new_case
        old_kernels, new_kernels = kernels(old_profile['report']), kernels(new_profile['report'])
        expert_before = [row['us'] for row in old_kernels if 'okc::moe_fused' in row['name']]
        expert_after = [row['us'] for row in new_kernels if 'okc::moe_fused' in row['name']]
        assert len(expert_before) == len(expert_after) == 1
        cells.append({'load': new_case['load'], 'layer': new_case['layer'],
                      'G0_seed': str(old_case['seed']), 'G607_seed': str(new_case['seed']),
                      'seeds_equal': old_case['seed'] == new_case['seed'],
                      'G0_adjusted_us': before['median_ns'] / 1000, 'G607_adjusted_us': after['median_ns'] / 1000,
                      'G0_raw_us': before['median_raw_ns'] / 1000, 'G607_raw_us': after['median_raw_ns'] / 1000,
                      'G0_expert_profile_us': expert_before[0], 'G607_expert_profile_us': expert_after[0],
                      'G0_profile_rows': old_kernels, 'G607_profile_rows': new_kernels})
    loads = [{'load': load, 'G0_us': latency, 'G607_us': fork['latency_us'][load],
              'G607_latency_increase_pct': 100 * (fork['latency_us'][load] / latency - 1)}
             for load, latency in original['latency_us'].items()]
    hybrid = dict(fork['latency_us'])
    for load in ('decode 1', 'verify 1'):
        hybrid[load] = original['latency_us'][load]
    for job in jobs:
        del job['benchmarks'], job['profiles']
    return {'source_pins': source_pins, 'jobs': jobs, 'cells': cells, 'loads': loads,
            'official_score_ratio_G0_over_G607': original['score'] / fork['score'],
            'conditional_all_original_times_factor': score(fork['baseline_us'], original['latency_us']) / fork['score'],
            'conditional_short_original_times_factor': score(fork['baseline_us'], hybrid) / fork['score'],
            'same_evaluator': True, 'same_gpu_node': original['node'] == fork['node'],
            'matched_inputs': all(cell['seeds_equal'] for cell in cells),
            'scope': 'Historical unmatched whole-load observations. Rounded profiler activities are not exposed critical-path times or causal repair costs.'}


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2])
    result = reconcile(root)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('official_score_ratio_G0_over_G607', 'conditional_short_original_times_factor', 'same_evaluator', 'same_gpu_node', 'matched_inputs')}))
