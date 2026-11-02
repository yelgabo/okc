"""Strict offline owning-receipt reader. Parsing never authorizes dispatch/release."""
import argparse
import json
from pathlib import Path
import re

from expert_top100_moe_harness import build, sha
from expert_top100_moe_harness_contract import CAPSULE_LIMIT, GLOBAL_TOL, ROW_TOL, decision, positive, require

OFFICIAL = {(1, 'decode 1'), (16, 'decode 16'), (32, 'decode 32'),
            (8, 'verify 1'), (128, 'verify 16'), (256, 'verify 32')}
TESTS = {(4, 'normal'), (33, 'normal'), (257, 'normal'), (128, 'skewed'), (1, 'normal')}
TEST_ORDER = ((4, 'normal'), (33, 'normal'), (257, 'normal'), (128, 'skewed'), (1, 'normal'))
BENCH_ORDER = ((1, 'decode 1'), (16, 'decode 16'), (32, 'decode 32'),
               (8, 'verify 1'), (128, 'verify 16'), (256, 'verify 32'))


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise RuntimeError('EXPERT_MOE_HARNESS: nonfinite JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def capsule(log, manifest):
    marker = 'EXPERT_MOE_CAPSULE '
    require(log.count(marker) == 1, 'exactly one capsule')
    lines = [line[len(marker):] for line in log.splitlines() if line.startswith(marker)]
    require(len(lines) == 1 and len(lines[0].encode()) < CAPSULE_LIMIT, 'bounded standalone capsule')
    report = strict_json(lines[0])
    require(report['runtime_id'] == manifest['runtime_id'] and report['official_arm'] == 'candidate2', 'capsule identity/arm')
    require(report['placement_arms'] == [False, False, True, True] and report['cleanup_restored'] is True, 'placements/cleanup')
    require(report['shared_tensor_storage'] == [], 'no unproven tensor sharing')
    require(len(report['owners_bytes']) == 4 and all(type(value) is int and value > 0 for value in report['owners_bytes']), 'owners')
    require(positive(report['prepare_s']) < 295, 'total preparation budget')
    counters = report['counters']
    require(counters['observations'] == 378 and counters['restorations'] == 54, '54 cases/seven observations')
    require(counters['captures'] == 216 and counters['timed_replays'] == 1152, 'capture/timing coverage')
    require(len(counters['worst']) == 2 and all(type(value) in (int, float) and 0 <= value <= limit
            for value, limit in zip(counters['worst'], (GLOBAL_TOL, ROW_TOL))), 'paired tolerances')
    verdict = decision(report['rows'])
    require(report['component_pass'] is verdict['component_pass'], 'printed component gate')
    return report, verdict


def read(receipt, job, log, manifest, payload, upload, log_attestation):
    expected_source, expected_manifest = build()
    require(manifest == expected_manifest and payload == expected_source.encode(), 'registered source/manifest')
    require(receipt['id'] == job and re.fullmatch('[a-f0-9]{16}', job) is not None, 'own private job')
    require(receipt['team'] == 'yelgabo' and receipt['board'] == 'muse_moe_w4a16'
            and receipt['mode'] == 'check' and receipt['status'] == 'done', 'team/problem/private/terminal')
    content = receipt['content_hash']
    require(re.fullmatch('[a-f0-9]{64}', content) is not None, 'server content identifier')
    require(receipt['payload_ref'] == 'muse_moe_w4a16/' + content + '.py', 'server payload association')
    require(upload == dict(job=job, team='yelgabo', mode='check', board='muse_moe_w4a16',
                          content_hash=content, payload_ref=receipt['payload_ref'],
                          source_sha256=manifest['source_sha256'], source_bytes=len(payload)), 'upload association')
    owning_log = receipt['output']['log']
    require(job in owning_log.replace('\\', '/').split('/')[-1], 'owning log path')
    require(log_attestation == dict(job=job, owning_log=owning_log, sha256=sha(log),
                                   source_sha256=sha(payload)), 'owning log attestation')
    result = receipt['result']
    require(result['status'] == 'ok' and result['passed'] is True and result['returncode'] == 0
            and not result['flags'] and result['error'] is None, 'official success')
    require(result['checks']['ok'] is True and not result['checks']['flagged']
            and not result['checks']['rejected'], 'static checks')
    records = result['records']
    require(len(records) == 18, 'exact official record count')
    require(all(type(record['index']) is int for record in records if record['event'] != 'setup'), 'integer indexes')
    setup = [record for record in records if record['event'] == 'setup']
    require(len(setup) == 1 and setup[0]['status'] == 'pass' and setup[0]['error'] is None
            and positive(setup[0]['prepare_s']) < 295, 'setup')
    tests = [record for record in records if record['event'] == 'test']
    require(len(tests) == 5 and {record['index'] for record in tests} == set(range(5)), 'five test indexes')
    require({(record['case']['bs'], record['case'].get('dist', 'normal')) for record in tests} == TESTS, 'official tests')
    for record in tests:
        require(record['status'] == 'pass' and record['case']['sharding'] == 'ep8'
                and type(record['case']['seed']) is int
                and (record['case']['bs'], record['case'].get('dist', 'normal')) == TEST_ORDER[record['index']], 'test result')
    for event in ('benchmark', 'profile'):
        selected = [record for record in records if record['event'] == event]
        require(len(selected) == 6 and {record['index'] for record in selected} == set(range(6)), 'six ' + event)
        require({(record['case']['bs'], record['case']['load']) for record in selected} == OFFICIAL, 'official cases')
        for record in selected:
            case = record['case']
            require(case['sharding'] == 'ep8' and case['weight'] == 60
                    and case.get('dist', 'normal') == 'normal' and type(case['seed']) is int
                    and (case['bs'], case['load']) == BENCH_ORDER[record['index']], 'standalone case schema')
            if event == 'benchmark':
                require(record['status'] == 'pass' and not record['flags'], 'benchmark pass')
                for key in ('median_ns', 'median_raw_ns', 'baseline_ns', 'baseline_raw_ns'):
                    positive(record[key])
            else:
                require(isinstance(record['report'], str) and bool(record['report'].strip()), 'profile text')
    benchmarks = {record['index']: record['case'] for record in records if record['event'] == 'benchmark'}
    require(all(record['case'] == benchmarks[record['index']] for record in records if record['event'] == 'profile'), 'profile/bench identity')
    report, verdict = capsule(log, manifest)
    return dict(job=job, source_sha256=sha(payload), official_arm='candidate2', report=report,
                decision=verdict, dispatch_ready=False, release_eligible=False,
                target_qualified=not manifest['hold'] and verdict['component_pass'],
                hold=manifest['hold'], scope='Standalone component only; no layer-speed inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('receipt', 'manifest', 'source', 'upload', 'log', 'log-attestation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    load = lambda path: strict_json(path.read_text(encoding='utf8'))
    result = read(load(args.receipt), args.job, args.log.read_bytes().decode('utf8'), load(args.manifest),
                  args.source.read_bytes(), load(args.upload), load(args.log_attestation))
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
