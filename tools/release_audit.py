"""Inventory passing own receipts below the current public elapsed score."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

PROBLEMS = ('muse_moe_w4a16', 'muse_bf16_proj', 'muse_attention_decode',
            'muse_decode_layer', 'muse_prefill_layer')


def public_rows(folder):
    result = {}
    for name in PROBLEMS:
        data = json.loads((folder / (name+'.leaderboard.json')).read_text())
        rows = [row for row in data['rows'] if row.get('team') == 'yelgabo']
        if len(rows) != 1:
            raise ValueError('Expected one own public row: '+name)
        row = rows[0]
        result[name] = dict(job=row['job'], score_us=row['value'], rank=row.get('rank'),
                            speedup=row['speedup'], snapshot=str(folder / (name+'.leaderboard.json')))
    return result


def own_jobs_index(data):
    """Read an explicitly supplied authenticated own-jobs listing, never a file-name hint."""
    if isinstance(data, dict):
        data = data.get('data', data)
        data = data.get('jobs') if isinstance(data, dict) else data
    if not isinstance(data, list):
        raise ValueError('--own-jobs must contain a jobs list')
    jobs = {}
    for row in data:
        if (not isinstance(row, dict) or not isinstance(row.get('job_id'), str)
                or not isinstance(row.get('problem'), str)):
            raise ValueError('Invalid own-jobs identity')
        if row.get('team', 'yelgabo') != 'yelgabo':
            raise ValueError('Foreign team in own-jobs index: '+row['job_id'])
        previous = jobs.get(row['job_id'])
        if previous is not None and previous != row:
            raise ValueError('Conflicting own-jobs entry: '+row['job_id'])
        jobs[row['job_id']] = row
    return jobs


def explicitly_disqualified(data):
    return isinstance(data, dict) and (data.get('disqualified') is True or any(
        isinstance(data.get(field), dict) and data[field].get('disqualified') is True
        for field in ('result', 'summary')))


def disqualification(data, own_jobs=None):
    """A fresh own listing can revoke an archived passing receipt without rewriting it."""
    if not isinstance(data, dict):
        return None
    is_cli = cli_export(data)
    identity = data.get('job_id') if is_cli else data.get('id')
    problem = data.get('problem') if is_cli else data.get('board')
    job = (own_jobs or {}).get(identity)
    if (job is not None and (explicitly_disqualified(job) or explicitly_disqualified(data))
            and problem != job['problem']):
        raise ValueError('Conflicting own receipt: '+identity+' problem')
    if job is not None and explicitly_disqualified(job):
        return dict(job=identity, problem=problem, basis='explicit own-jobs index',
                    reason=job.get('disqualified_reason'))
    if explicitly_disqualified(data) and (data.get('team') == 'yelgabo' or job is not None):
        reason = data.get('disqualified_reason')
        for field in ('result', 'summary'):
            if reason is None and isinstance(data.get(field), dict):
                reason = data[field].get('disqualified_reason')
        return dict(job=identity, problem=problem, basis='receipt', reason=reason)
    return None


def clean_own_job(row):
    return (row.get('problem') in PROBLEMS and row.get('status') == 'done'
            and row.get('result_status') == 'ok' and row.get('passed') is True
            and row.get('flags') == [] and not row.get('error') and not explicitly_disqualified(row))


def cli_export(data):
    return isinstance(data, dict) and 'job_id' in data and 'summary' in data


def cli_receipt_row(data, jobs):
    job = jobs.get(data.get('job_id'))
    if job is None or not clean_own_job(job):
        return None
    identity = data['job_id']
    def conflict(field):
        raise ValueError('Conflicting own CLI export: '+identity+' '+field)
    if data.get('problem') != job['problem']:
        conflict('problem')
    summary = data.get('summary')
    if not isinstance(summary, dict):
        conflict('summary')
    for field, expected in (('status', job['result_status']), ('passed', job['passed']), ('error', None)):
        if field not in summary or summary[field] != expected:
            conflict(field)
    if summary['passed'] is not True or summary.get('flags'):
        conflict('passed/flags')
    score = summary.get('score_us')
    if (type(score) not in (int, float) or not math.isfinite(score) or score <= 0
            or type(job.get('score_us')) not in (int, float) or score != job['score_us']):
        conflict('score_us')
    tests = summary.get('tests')
    if type(tests) is not int or tests <= 0 or type(summary.get('tests_passed')) is not int or summary['tests_passed'] != tests:
        conflict('test counts')
    benchmarks, profiles = data.get('benchmarks'), data.get('profiles')
    if (not isinstance(benchmarks, list) or not benchmarks or not isinstance(profiles, list)
            or len(benchmarks) != len(profiles)):
        conflict('benchmark/profile counts')
    for event, records in (('benchmark', benchmarks), ('profile', profiles)):
        if (any(not isinstance(row, dict) or row.get('event') != event
                or type(row.get('index')) is not int for row in records)
                or sorted(row['index'] for row in records) != list(range(len(records)))):
            conflict(event+' records')
        reported = summary.get(event+'s', len(records))
        if type(reported) is not int or reported != len(records):
            conflict(event+' count')
    benchmark_cases = {row['index']: row.get('case') for row in benchmarks}
    if (any(row.get('status') != 'pass' or not isinstance(row.get('case'), dict) for row in benchmarks)
            or any(row.get('case') != benchmark_cases[row['index']]
                   or not isinstance(row.get('report'), str) or not row['report'].strip() for row in profiles)):
        conflict('benchmark/profile coverage')
    return dict(job=identity, problem=job['problem'], mode=job.get('mode'), score_us=score,
                speedup=summary.get('speedup'), ci=summary.get('speedup_ci'),
                finished_at=None, content_hash=None, payload_ref=None, markers=[],
                receipt_format='cli-export', ownership_basis='explicit own-jobs index',
                record_counts=dict(test=tests, benchmark=len(benchmarks), profile=len(profiles)),
                record_count_basis=dict(test='export summary; individual test records unavailable',
                                        benchmark='export records', profile='export records'))


def receipt_row(data, own_jobs=None):
    if disqualification(data, own_jobs) is not None or explicitly_disqualified(data):
        return None
    if cli_export(data):
        return cli_receipt_row(data, own_jobs or {})
    if not isinstance(data, dict) or data.get('team') != 'yelgabo':
        return None
    if data.get('board') not in PROBLEMS or data.get('status') != 'done':
        return None
    result = data.get('result', {})
    if not isinstance(result, dict):
        return None
    score = result.get('score_us')
    if (result.get('passed') is not True or result.get('returncode') != 0 or result.get('status') != 'ok'
            or result.get('flags') or result.get('error')
            or type(score) not in (int,float) or not math.isfinite(score) or score <= 0):
        return None
    checks = result.get('checks', {})
    if (checks and checks.get('ok') is not True) or checks.get('flagged') or checks.get('rejected'):
        return None
    records = result.get('records', [])
    counts = {event: sum(row.get('event') == event for row in records)
              for event in ('test','benchmark','profile')}
    log = result.get('log_tail', '')
    return dict(job=data['id'], problem=data['board'], mode=data.get('mode'), score_us=score,
                speedup=result.get('speedup'), ci=result.get('speedup_ci'),
                finished_at=datetime.fromtimestamp(data['finished_at'],timezone.utc).isoformat(),
                content_hash=data.get('content_hash'), payload_ref=data.get('payload_ref'),
                receipt_format='raw', record_counts=counts, markers=sorted({marker for marker in (
                    'EXPERT_MOE_CAPSULE','D32_ATTENTION_PAIR_REPORT','component_pass',
                    'stage-retire candidate','SC6_COMPARE','WGSFRONT','D32_SPLIT_COPIES') if marker in log}))


def merge_receipt(receipts, row, path):
    previous = receipts.get(row['job'])
    if previous is None:
        row.update(receipt_paths=[path], receipt_formats=[row['receipt_format']])
        receipts[row['job']] = row
        return
    for key in ('score_us', 'problem'):
        if previous[key] != row[key]:
            raise ValueError('Conflicting own receipt: '+row['job']+' '+key)
    if row['receipt_format'] == 'raw' and 'raw' in previous['receipt_formats']:
        if previous['content_hash'] != row['content_hash']:
            raise ValueError('Conflicting own receipt: '+row['job']+' content_hash')
    paths = previous['receipt_paths'] + [path]
    formats = sorted(set(previous['receipt_formats'] + [row['receipt_format']]))
    if row['receipt_format'] == 'raw' and previous['receipt_format'] != 'raw':
        receipts[row['job']] = row
    receipts[row['job']].update(receipt_paths=paths, receipt_formats=formats)


def inventory(root, snapshots, tracked_only=False, own_jobs_path=None):
    public = public_rows(snapshots)
    own_jobs = (own_jobs_index(json.loads(own_jobs_path.read_text(encoding='utf-8-sig')))
                if own_jobs_path is not None else None)
    cli_coverage = dict(own_jobs_index=str(own_jobs_path) if own_jobs_path is not None else None,
                        indexed_jobs=len(own_jobs or {}), exports_seen=0, bound_clean_exports=0,
                        excluded_no_index=0, excluded_unlisted=0, excluded_unclean_job=0)
    bound_cli_jobs = set()
    command = (['git','ls-files','--','*.json'] if tracked_only else
               ['rg','--files','--hidden','--no-ignore','-g','!.git','-g','*.json'])
    files = subprocess.check_output(command,cwd=root).decode().splitlines()
    receipts, failures, disqualified = {}, [], []
    for name in files:
        path = root / name
        try:
            raw = path.read_bytes()
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeError:
                text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else raw.decode('cp1252')
            data = json.loads(text)
        except (UnicodeError,ValueError,OSError) as error:
            failures.append(dict(path=name,error=type(error).__name__))
            continue
        is_cli = cli_export(data)
        if is_cli:
            cli_coverage['exports_seen'] += 1
            if own_jobs is None:
                cli_coverage['excluded_no_index'] += 1
            elif data.get('job_id') not in own_jobs:
                cli_coverage['excluded_unlisted'] += 1
            elif not clean_own_job(own_jobs[data['job_id']]):
                cli_coverage['excluded_unclean_job'] += 1
        excluded = disqualification(data, own_jobs)
        if excluded is not None:
            disqualified.append(dict(excluded, path=name))
        row = receipt_row(data, own_jobs)
        if row is None:
            continue
        if is_cli:
            cli_coverage['bound_clean_exports'] += 1
            bound_cli_jobs.add(row['job'])
        merge_receipt(receipts, row, name)
    leads=[]
    for row in receipts.values():
        baseline=public[row['problem']]
        if row['score_us'] < baseline['score_us']:
            row.update(public_job=baseline['job'],public_score_us=baseline['score_us'],
                       elapsed_reduction_pct=100*(1-row['score_us']/baseline['score_us']),
                       numerical_lead_only=True)
            leads.append(row)
    leads.sort(key=lambda row:(row['problem'],row['score_us']))
    cli_coverage['distinct_bound_jobs'] = len(bound_cli_jobs)
    cli_scope = ('CLI exports excluded: no --own-jobs index supplied.' if own_jobs is None else
                 'CLI exports included only for clean matching jobs in the explicitly supplied '
                 +str(len(own_jobs))+'-job index; unlisted exports excluded. This is not full historical CLI coverage.')
    dq_scope = (' Archived receipts have no fresh post-adjudication check without --own-jobs.' if own_jobs is None else
                ' Explicit disqualifications in the supplied own-jobs index override archived passing receipts.')
    return dict(observed_at=datetime.now(timezone.utc).isoformat(),public=public,
                json_files_scanned=len(files),distinct_clean_passing_own_receipts=len(receipts),
                unreadable_json=failures,leads=leads,cli_export_coverage=cli_coverage,
                disqualification_coverage=dict(
                    indexed_disqualified_jobs=sum(explicitly_disqualified(row) for row in (own_jobs or {}).values()),
                    excluded_files=len(disqualified), distinct_excluded_jobs=len({row['job'] for row in disqualified}),
                    exclusions=disqualified),
                scope=('Tracked JSON only. ' if tracked_only else 'Saved JSON, including archives and ignored files. ')
                      +cli_scope+dq_scope+' Lower elapsed is a numerical lead, not release eligibility or matched gain.')


def ledger_gaps(audit, ledger, root=None):
    entries = {row['job']:row for row in ledger['runs']}
    gaps = []
    for lead in audit['leads']:
        entry = entries.get(lead['job'])
        if entry is None:
            gaps.append(dict(job=lead['job'],reason='missing promotion disposition'))
            continue
        for field in ('owner','judged_arm','source_identity','release_state','reason','evidence','disclosure_report'):
            if not isinstance(entry.get(field),str) or not entry[field].strip():
                gaps.append(dict(job=lead['job'],reason='missing '+field))
        if root is not None:
            for field in ('evidence','disclosure_report'):
                if not (root / entry.get(field,'')).is_file():
                    gaps.append(dict(job=lead['job'],reason='missing file '+field))
        if entry.get('problem') != lead['problem'] or entry.get('score_us') != lead['score_us']:
            gaps.append(dict(job=lead['job'],reason='receipt identity/score changed'))
    return gaps


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--snapshots',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ledger',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--tracked-only',action='store_true')
    parser.add_argument('--own-jobs',type=Path,help='Explicit authenticated own okc jobs JSON; also accepts data.jobs in a saved monitor snapshot')
    args=parser.parse_args()
    data=inventory(args.root,args.snapshots,args.tracked_only,args.own_jobs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    if args.check:
        if args.ledger is None:parser.error('--check requires --ledger')
        gaps=ledger_gaps(data,json.loads(args.ledger.read_text()),args.root)
        print(json.dumps(dict(promotion_gaps=gaps,leads=len(data['leads']),
                              scope=data['scope'],cli_export_coverage=data['cli_export_coverage'],
                              disqualification_coverage=data['disqualification_coverage']),indent=2))
        raise SystemExit(bool(gaps))
    print(json.dumps(dict(json_files=data['json_files_scanned'],receipts=data['distinct_clean_passing_own_receipts'],
                         scope=data['scope'],cli_export_coverage=data['cli_export_coverage'],
                         disqualification_coverage=data['disqualification_coverage'],
                         better_runs=len(data['leads']),best={p:next((x for x in data['leads'] if x['problem']==p),None)
                                                          for p in PROBLEMS}),indent=2))
