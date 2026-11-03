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


def receipt_row(data):
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
                record_counts=counts, markers=sorted({marker for marker in (
                    'EXPERT_MOE_CAPSULE','D32_ATTENTION_PAIR_REPORT','component_pass',
                    'stage-retire candidate','SC6_COMPARE','WGSFRONT','D32_SPLIT_COPIES') if marker in log}))


def inventory(root, snapshots, tracked_only=False):
    public = public_rows(snapshots)
    command = (['git','ls-files','--','*.json'] if tracked_only else
               ['rg','--files','--hidden','--no-ignore','-g','!.git','-g','*.json'])
    files = subprocess.check_output(command,cwd=root).decode().splitlines()
    receipts, failures = {}, []
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
        row = receipt_row(data)
        if row is None:
            continue
        previous = receipts.get(row['job'])
        if previous is None:
            row['receipt_paths'] = [name]
            receipts[row['job']] = row
        else:
            for key in ('score_us','problem','content_hash'):
                if previous[key] != row[key]:
                    raise ValueError('Conflicting own receipt: '+row['job']+' '+key)
            previous['receipt_paths'].append(name)
    leads=[]
    for row in receipts.values():
        baseline=public[row['problem']]
        if row['score_us'] < baseline['score_us']:
            row.update(public_job=baseline['job'],public_score_us=baseline['score_us'],
                       elapsed_reduction_pct=100*(1-row['score_us']/baseline['score_us']),
                       numerical_lead_only=True)
            leads.append(row)
    leads.sort(key=lambda row:(row['problem'],row['score_us']))
    return dict(observed_at=datetime.now(timezone.utc).isoformat(),public=public,
                json_files_scanned=len(files),distinct_clean_passing_own_receipts=len(receipts),
                unreadable_json=failures,leads=leads,
                scope='Saved own receipts, including archives and ignored files. Lower elapsed is a numerical lead, not release eligibility or matched gain.')


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
    args=parser.parse_args()
    data=inventory(args.root,args.snapshots,args.tracked_only)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    if args.check:
        if args.ledger is None:parser.error('--check requires --ledger')
        gaps=ledger_gaps(data,json.loads(args.ledger.read_text()),args.root)
        print(json.dumps(dict(promotion_gaps=gaps,leads=len(data['leads'])),indent=2))
        raise SystemExit(bool(gaps))
    print(json.dumps(dict(json_files=data['json_files_scanned'],receipts=data['distinct_clean_passing_own_receipts'],
                         better_runs=len(data['leads']),best={p:next((x for x in data['leads'] if x['problem']==p),None)
                                                          for p in PROBLEMS}),indent=2))
