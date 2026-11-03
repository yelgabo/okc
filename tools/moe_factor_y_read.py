"""Validate this experiment's owning receipt with the existing strict MoE reader."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import expert_top100_moe_harness_read as reader
from moe_factor_y_harness import build
from expert_top100_moe_harness_contract import decision
from moe_pairlut_contract import pairlut_decision as factor_y_decision


def read_partial(folder,job):
    require=reader.require
    receipt=reader.strict_json((folder/'status-wait.raw.json').read_text())
    manifest=reader.strict_json((folder/'paired.manifest.json').read_text())
    accepted=reader.strict_json((folder/'accepted.json').read_text())
    payload=(folder/'paired.py').read_bytes()
    require(receipt['id']==accepted['job_id']==job and receipt['team']=='yelgabo'
            and receipt['mode']=='check' and receipt['board']=='muse_moe_w4a16'
            and receipt['status']=='done','own terminal partial receipt')
    result=receipt['result']
    require(result['passed'] is False and result['returncode']!=0
            and result['score_us'] is None,'failed partial remains failed')
    require(reader.sha(payload)==accepted['source_sha256']==manifest['source_sha256']
            and manifest==build()[1],'partial source binding')
    log=(folder/'profiles'/Path(receipt['output']['log']).name).read_text()
    prefix='MOE_FACTOR_Y_PARTIAL '
    lines=[line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    require(len(lines)==1 and len(lines[0])<12000,'bounded unique partial')
    require(prefix+lines[0] in result['log_tail'],'partial owning log association')
    decoder=zlib.decompressobj()
    raw=decoder.decompress(base64.b64decode(lines[0],validate=True),100000)
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'bounded decoded partial')
    data=reader.strict_json(raw.decode())
    require(data['runtime_id']==manifest['runtime_id']
            and data['status']=='FAILED_PARTIAL_NOT_QUALIFIED','partial runtime/status')
    rows=data['rows'];counters=data['counters']
    expected={(seed,tokens,dist) for seed in (1102031,1102037,1102049)
              for tokens in (1,8,16,32,128,256) for dist in ('normal','skewed')}
    require(isinstance(rows,list) and len(rows)<=36
            and len({tuple(row[:3]) for row in rows})==len(rows)
            and {tuple(row[:3]) for row in rows}<=expected,'partial row coverage')
    for key,limit in (('observations',378),('restorations',54),('captures',216),('timed_replays',1152)):
        require(type(counters.get(key,0)) is int and 0<=counters.get(key,0)<=limit,'partial '+key)
    require(all(type(value) is int and 0<=value<7 for value in data['loaded_builds']),'partial loaded builds')
    complete=len(rows)==36
    if complete:decision(rows)
    summary=dict(job=job,qualified=False,terminal=data['status'],source_sha256=manifest['source_sha256'],
                 retained_rows=len(rows),complete_timing_coverage=complete,counters=counters,
                 stage=data['stage'],error=data['error'],elapsed_s=data['elapsed_s'],log_sha256=reader.sha(log),
                 scope='Failed partial evidence only. Complete timing rows cannot qualify a failed setup.')
    (folder/'partial.json').write_text(json.dumps(data,indent=2)+'\n')
    (folder/'partial-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def read(folder, job):
    receipt = reader.strict_json((folder / 'status-wait.raw.json').read_text())
    manifest = reader.strict_json((folder / 'paired.manifest.json').read_text())
    payload = (folder / 'paired.py').read_bytes()
    accepted = reader.strict_json((folder / 'accepted.json').read_text())
    if accepted['job_id'] != job or accepted['source_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('Accepted source/job mismatch')
    if receipt.get('id') != job or receipt.get('board') != 'muse_moe_w4a16' or receipt.get('mode') != 'check':
        raise ValueError('Owning private MoE receipt mismatch')
    owning_log = receipt['output']['log']
    log_path = folder / 'profiles' / Path(owning_log).name
    log = log_path.read_bytes().decode('utf8')
    upload = dict(job=job, team='yelgabo', mode='check', board='muse_moe_w4a16',
                  content_hash=receipt['content_hash'], payload_ref=receipt['payload_ref'],
                  source_sha256=manifest['source_sha256'], source_bytes=len(payload))
    attestation = dict(job=job, owning_log=owning_log, sha256=reader.sha(log),
                       source_sha256=reader.sha(payload))
    previous = reader.build
    try:
        reader.build = lambda **kwargs: build(**kwargs)
        result = reader.read(receipt, job, log, manifest, payload, upload, attestation)
    finally:
        reader.build = previous
    if result['report']['loaded_builds']!=list(range(7)) or result['report']['table_words']!=[256,256,256,256]:
        raise ValueError('Observed source/lookup binding mismatch')
    verdict=factor_y_decision(result['report']['rows'],decision(result['report']['rows']))
    if result['report']['factor_y_decision']!=verdict:raise ValueError('Factorized-output gate mismatch')
    result['factor_y_decision']=verdict
    result['target_qualified']=bool(result['target_qualified'] and verdict['passed'])
    result['official_speedup'] = receipt['result']['speedup']
    result['official_speedup_ci'] = receipt['result']['speedup_ci']
    (folder / 'upload-association.json').write_text(json.dumps(upload, indent=2) + '\n')
    (folder / 'log-attestation.json').write_text(json.dumps(attestation, indent=2) + '\n')
    (folder / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.partial:summary=read_partial(args.folder,args.job)
    else:
        result=read(args.folder,args.job)
        summary=dict(job=args.job,target_qualified=result['target_qualified'],
                     official_speedup=result['official_speedup'],decision=result['decision'])
    print(json.dumps(summary,indent=2))
