"""Validate an owning split-finalizer receipt without weakening its result."""
import argparse
import base64
import json
import math
import re
from pathlib import Path
import zlib

import expert_top100_moe_harness_read as original
from gopt_ws_split import build, contract_source, sha


def capsule(log, manifest):
    marker = 'GOPT_WS_SPLIT_REPORT '
    lines = [line[len(marker):] for line in log.splitlines() if line.startswith(marker)]
    original.require(len(lines)==1 and len(lines[0])<12000, 'bounded unique WS split capsule')
    data = original.strict_json(zlib.decompress(base64.b64decode(lines[0],validate=True)).decode())
    original.require(data['runtime_id']==manifest['runtime_id'] and data['official_arm']=='candidate2', 'split identity/arm')
    original.require(data['placement_arms']==[False,False,True,True] and data['cleanup_restored'] is True, 'owners/cleanup')
    original.require(data['shared_tensor_storage']==[] and len(data['owners_bytes'])==4
                     and all(type(x)is int and x>0 for x in data['owners_bytes']), 'independent storage')
    counts = data['counters']
    original.require(tuple(counts[x] for x in ('observations','restorations','captures','timed_replays','finalizer_reference_cases'))
                     == (378,54,216,1152,6), 'complete split and finalizer reference coverage')
    original.require(original.positive(data['prepare_s'])<295, 'prepare budget')
    original.require(len(counts['worst'])==2 and 0<=counts['worst'][0]<=.015 and 0<=counts['worst'][1]<=.075, 'math tolerances')
    env = {}
    exec(contract_source(),env)
    verdict = env['decision'](data['rows'])
    original.require(data['component_pass'] is verdict['component_pass'], 'paired gate recomputes')
    return data, verdict


def read(folder, job):
    receipt = original.strict_json((folder/'status-wait.raw.json').read_text())
    manifest = original.strict_json((folder/'paired.manifest.json').read_text())
    payload = (folder/'paired.py').read_bytes()
    accepted = original.strict_json((folder/'accepted.json').read_text())
    original.require(accepted['job_id']==job and accepted['source_sha256']==sha(payload), 'accepted job/source')
    owning = receipt['output']['log']
    local_log = folder/'profiles'/Path(owning).name
    log = local_log.read_text() if local_log.is_file() else Path(owning).read_text()
    upload = dict(job=job,team='yelgabo',mode='check',board='muse_moe_w4a16',content_hash=receipt['content_hash'],
                  payload_ref=receipt['payload_ref'],source_sha256=manifest['source_sha256'],source_bytes=len(payload))
    attestation = dict(job=job,owning_log=owning,sha256=sha(log),source_sha256=sha(payload))
    previous = original.build, original.capsule
    try:
        original.build = lambda target_proposal: build()[:2]
        original.capsule = capsule
        result = original.read(receipt,job,log,manifest,payload,upload,attestation)
    finally:
        original.build, original.capsule = previous
    result.update(official_score_us=receipt['result']['score_us'],official_speedup=receipt['result']['speedup'],
                  official_ci=receipt['result']['speedup_ci'])
    for name, value in [('analysis',result),('upload-association',upload),('log-attestation',attestation)]:
        (folder/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    return result


def partial(folder, job):
    receipt = original.strict_json((folder/'status-wait.raw.json').read_text())
    manifest = original.strict_json((folder/'paired.manifest.json').read_text())
    accepted = original.strict_json((folder/'accepted.json').read_text())
    payload = (folder/'paired.py').read_bytes()
    original.require(receipt['id']==accepted['job_id']==job and receipt['team']=='yelgabo'
                     and receipt['mode']=='check' and receipt['board']=='muse_moe_w4a16'
                     and receipt['status']=='done', 'own terminal failure')
    original.require(sha(payload)==accepted['source_sha256']==manifest['source_sha256']==build()[1]['source_sha256'], 'partial source identity')
    content = receipt['content_hash']
    original.require(re.fullmatch('[a-f0-9]{64}', content) is not None and receipt['payload_ref']=='muse_moe_w4a16/'+content+'.py', 'server payload association')
    original.require(job in Path(receipt['output']['log']).name, 'owning partial log')
    result = receipt['result']
    original.require(result['passed'] is False and result['returncode']!=0 and result['score_us'] is None, 'retain failed result')
    log = result['log_tail']
    lines = [line[len('GOPT_WS_SPLIT_PARTIAL '):] for line in log.splitlines() if line.startswith('GOPT_WS_SPLIT_PARTIAL ')]
    original.require(len(lines)==1 and len(lines[0])<12000, 'unique bounded partial')
    data = original.strict_json(zlib.decompress(base64.b64decode(lines[0],validate=True)).decode())
    original.require(data['runtime_id']==manifest['runtime_id'] and data['status']=='FAILED_PARTIAL_NOT_QUALIFIED', 'failed capsule identity')
    original.require(len(data['rows'])<=36 and len({tuple(row[:3]) for row in data['rows']})==len(data['rows']), 'retained partial coverage')
    env = {}
    exec(contract_source(),env)
    expected = {(seed,tokens,dist) for seed in env['BANKS'] for tokens in env['LOADS'] for dist in env['DISTRIBUTIONS']}
    for row in data['rows']:
        original.require(len(row)==7 and tuple(row[:3]) in expected and len(row[3])==4
                         and len(row[4])==3 and len(row[5])==4 and len(row[6])==2, 'partial row schema')
        for value in row[3]+row[5]: original.positive(value)
        original.require(all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in row[6]), 'partial nulls')
        total,occupied,maximum=row[4]
        original.require(all(type(x)is int for x in row[4]) and 1<=occupied<=32
                         and 1<=maximum<=row[1] and maximum<=total<=row[1]*8, 'partial census')
    report = dict(job=job,qualified=False,status='FAILED_PARTIAL_NOT_QUALIFIED',report=data,
                  receipt_returncode=result['returncode'],receipt_error=result['error'],
                  flags=result['flags'],source_sha256=manifest['source_sha256'])
    (folder/'partial-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--partial',action='store_true')
    a=p.parse_args()
    if a.partial:
        r=partial(a.folder,a.job)
        print(json.dumps({key:r[key] for key in ('job','status','receipt_returncode','receipt_error')},indent=2))
    else:
        r=read(a.folder,a.job)
        print(json.dumps({key:r[key] for key in ('job','target_qualified','official_score_us','decision')},indent=2))
