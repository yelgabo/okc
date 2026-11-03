"""Input-only local CUDA census. No model weights, kernel, benchmark or selection."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

SEEDS = tuple(range(65001, 65033))
UNIT_BYTES = 36864
UNITS = (512, 256)


def summarize(counts):
    assert len(counts) == 32 and all(isinstance(c, int) and c >= 0 for c in counts)
    active = sum(c > 0 for c in counts)
    jobs = sum((c + 7) // 8 for c in counts)
    per = UNIT_BYTES * sum(UNITS)
    return dict(counts=counts, local_routes=sum(counts), active_experts=active,
                max_count=max(counts), jobs=jobs, extra_jobs=jobs-active,
                unique_payload_bytes=active*per, requested_bytes=jobs*per,
                extra_bytes=(jobs-active)*per,
                extra_request_fraction=(jobs-active)/jobs if jobs else 0.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--loader', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    spec = importlib.util.spec_from_file_location('_local_input_ast_loader', args.loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generator, reference_pin = mod.load_public_generator(args.reference, torch)
    receipt = json.loads(args.receipt.read_text())
    assert receipt['id'] == 'abe7651dbaa44458' and receipt['board'] == 'muse_moe_w4a16'
    rows = [r for r in receipt['result']['records'] if r.get('event') == 'benchmark']
    medians = {r['case']['bs']: r['median_ns']/1000 for r in rows}
    records = []
    for m in (16, 32):
        for seed in SEEDS:
            d = generator(m, 'ep8', seed, dist='normal', rank='busiest', device='cuda')
            ids = d['topk_ids'].cpu().tolist()
            assert all(len(row) == len(set(row)) == 8 for row in ids)
            counts = [sum(e in row for row in ids) for e in range(32)]
            records.append(dict(m=m, seed=seed,
                                selected_global_group=d['selected_group'], **summarize(counts)))
    torch.cuda.synchronize()
    aggregate = {}
    for m in (16, 32):
        rs = [r for r in records if r['m'] == m]
        fs = [r['extra_request_fraction'] for r in rs]
        aggregate[str(m)] = dict(banks=len(rs), banks_with_duplicates=sum(f > 0 for f in fs),
                                mean_fraction=statistics.mean(fs), median_fraction=statistics.median(fs),
                                max_fraction=max(fs), extra_jobs=sum(r['extra_jobs'] for r in rs),
                                max_count=max(r['max_count'] for r in rs))
    scenarios = []
    for seed in SEEDS:
        pair = [r for r in records if r['seed'] == seed]
        assert len(pair) == 2
        factor = math.prod(1/(1-r['extra_request_fraction']) for r in pair)**(1/6)
        scenarios.append(dict(seed=seed, factor=factor,
                              saved_us={str(r['m']): medians[r['m']]*r['extra_request_fraction'] for r in pair}))
    fs = [r['factor'] for r in scenarios]
    report = dict(schema='OWNED8_LOCAL_INPUT_ONLY_V1', prespecified_seeds=list(SEEDS),
                  dimensions=[16,32], distribution='public normal/popularity.03/top8/busiestep8',
                  scope='Independent local GPU distribution; not H200 replay, traffic, timing or bound.',
                  torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                  reference_lf_sha256=reference_pin,
                  loader_sha256=hashlib.sha256(args.loader.read_bytes()).hexdigest(),
                  prior_cross_h200_fidelity='FAILED: earlier M256/4773 differed249/256 masks; not rerun here.',
                  payload_model=dict(job_height=8, unit_bytes=UNIT_BYTES, phase_units=list(UNITS),
                                     per_expert_payload_bytes=UNIT_BYTES*sum(UNITS)),
                  analytical_m1_m8_duplicates=0,
                  receipt='abe7651dbaa44458', owned_medians_us=medians,
                  aggregate=aggregate, records=records, proportional_scenarios=scenarios,
                  scenario_summary=dict(geomean=math.prod(fs)**(1/len(fs)),
                                        median=statistics.median(fs), minimum=min(fs),maximum=max(fs),
                                        above_one_percent=sum(f>1.01 for f in fs)))
    assert len(records) == 64 and {(r['m'],r['seed']) for r in records} == {(m,s) for m in (16,32) for s in SEEDS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(aggregate=aggregate, scenario=report['scenario_summary'],
                          reference_lf_sha256=reference_pin, device=report['device']), indent=2))


if __name__ == '__main__':
    main()
