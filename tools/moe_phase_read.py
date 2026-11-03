"""Validate the source-bound sparse observer receipt and its official control arm."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

import expert_top100_moe_harness_read as original
from expert_top100_moe_harness_contract import decision,positive,require
from moe_phase_harness import build
from moe_phase_analysis import observer_decision


def capsule(log,manifest):
    prefix='MOE_PHASE_REPORT '
    lines=[x[len(prefix):] for x in log.splitlines() if x.startswith(prefix)]
    require(len(lines)==1 and len(lines[0])<12000,'bounded unique phase capsule')
    data=original.strict_json(zlib.decompress(base64.b64decode(lines[0],validate=True)).decode())
    require(data['runtime_id']==manifest['runtime_id'] and data['official_arm']=='control0','observer identity/official control')
    require(data['placement_arms']==[False,False,True,True] and data['cleanup_restored'] is True,'owners/cleanup')
    require(data['shared_tensor_storage']==[] and len(data['owners_bytes'])==4
            and all(type(x)is int and x>0 for x in data['owners_bytes']),'independent storage')
    counts=data['counters'];require((counts['observations'],counts['restorations'],counts['captures'],counts['timed_replays'])==(378,54,216,4608),'full observer coverage')
    require(counts['trace_replays']==180,'repeated cold traces')
    require(positive(data['prepare_s'])<295,'prepare budget')
    require(0<=counts['worst'][0]<=.015 and 0<=counts['worst'][1]<=.075,'math coverage')
    verdict=decision(data['rows']);require(data['component_pass']==verdict['component_pass'],'paired summary recomputes')
    selected={(r[0],r[1],r[2]) for r in data['rows'] if r[1]>=32}
    require(len(data['phases'])==18 and {tuple(x[:3]) for x in data['phases']}==selected,'phase coverage')
    for seed,tokens,dist,owners in data['phases']:
        require(len(owners)==2,'two observed owners')
        for owner in owners:
            require(0<owner['grid']<=136 and 0<owner['jobs']<=88,'observed geometry')
            for key in ('entry_to_A_begin_us','A_to_B_end_us','grid_wait_us','finalizer_us','first_B_ready_us'):
                values=owner[key];require(len(values)==4 and all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in values)
                                        and values==sorted(values),'boundary summary')
    require(data['observer_valid'] is observer_decision(data['rows'],data['sampling'])['valid'],'perturbation gate recomputes')
    require(len(data['repeated'])==18 and {tuple(x[:3]) for x in data['repeated']}==selected,'repeated trace coverage')
    for row in data['repeated']:
        require(len(row[3])==2,'repeated independent owners')
        for owner in row[3]:
            require(len(owner)==3,'repeated metrics')
            for values in owner:
                require(len(values)==3 and values==sorted(values) and all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in values),'repeated ranges')
    raw=data['raw_trace'];require(0<raw['grid']<=136 and 0<len(raw['critical_ctas'])<=15,'critical raw trace')
    require({0,raw['grid']//2,raw['grid']-1}<={r[0] for r in raw['critical_ctas']},'fixed raw CTA coverage')
    require(all(len(r)==25 for r in raw['critical_ctas']),'raw row layout')
    first=data['phases'][0][3][0]
    require(raw['grid']==first['grid'] and raw['base_ns']>0,'raw trace binding')
    recovered=[]
    for row in raw['critical_ctas']:
        require(0<=row[0]<raw['grid'],'raw CTA index')
        values=[raw['base_ns']+x if x>=0 else 0 for x in row[1:19]]+row[19:]
        require(values[0]<=values[1]<=values[2] and values[11]<=values[12]<=values[13], 'raw boundary ordering')
        recovered.append(values)
    span=round((max(r[13] for r in recovered)-min(r[0] for r in recovered))/1000,2)
    spread=round((max(r[11] for r in recovered)-min(r[11] for r in recovered))/1000,2)
    require(span==first['boundary_span_us'] and spread==first['consumer_finish_spread_us'],'raw critical extrema')
    return data,verdict


def read(folder,job):
    receipt=original.strict_json((folder/'status-wait.raw.json').read_text())
    manifest=original.strict_json((folder/'paired.manifest.json').read_text())
    payload=(folder/'paired.py').read_bytes();accepted=json.loads((folder/'accepted.json').read_text())
    require(accepted['job_id']==job and accepted['source_sha256']==original.sha(payload),'accepted binding')
    owning=receipt['output']['log'];log=(folder/'profiles'/Path(owning).name).read_text()
    upload=dict(job=job,team='yelgabo',mode='check',board='muse_moe_w4a16',content_hash=receipt['content_hash'],
                payload_ref=receipt['payload_ref'],source_sha256=manifest['source_sha256'],source_bytes=len(payload))
    attestation=dict(job=job,owning_log=owning,sha256=original.sha(log),source_sha256=original.sha(payload))
    old_build,old_capsule=original.build,original.capsule
    try:
        original.build=build;original.capsule=capsule
        result=original.read(receipt,job,log,manifest,payload,upload,attestation)
    finally:original.build,original.capsule=old_build,old_capsule
    result['official_arm']='control0'
    result['target_qualified']=False
    result['observer_valid']=result['report']['observer_valid']
    result['observer_decision']=observer_decision(result['report']['rows'],result['report']['sampling'])
    result['scope']='Uninstrumented control official score, sparse diagnostic validity; no treatment gain or pure-release qualification'
    result['official_score_us']=receipt['result']['score_us']
    result['official_speedup']=receipt['result']['speedup']
    result['official_ci']=receipt['result']['speedup_ci']
    for name,data in [('analysis.json',result),('upload-association.json',upload),('log-attestation.json',attestation)]:
        (folder/name).write_text(json.dumps(data,indent=2)+'\n')
    return result


def read_partial(folder,job):
    receipt=original.strict_json((folder/'status-wait.raw.json').read_text())
    manifest=original.strict_json((folder/'paired.manifest.json').read_text())
    accepted=original.strict_json((folder/'accepted.json').read_text())
    require(receipt['id']==accepted['job_id']==job and receipt['team']=='yelgabo'
            and receipt['mode']=='check' and receipt['board']=='muse_moe_w4a16'
            and receipt['status']=='done','own terminal partial receipt')
    result=receipt['result']
    require(result['passed'] is False and result['returncode']!=0 and result['score_us'] is None
            and not result['flags'] and result['checks']['ok'],'failed partial remains failed')
    payload=(folder/'paired.py').read_bytes()
    require(original.sha(payload)==accepted['source_sha256']==manifest['source_sha256']==build()[1]['source_sha256'],'partial source binding')
    log=(folder/'profiles'/Path(receipt['output']['log']).name).read_text()
    require(log==result['log_tail'],'complete owning partial log')
    lines=[x[len('MOE_PHASE_PARTIAL '):] for x in log.splitlines() if x.startswith('MOE_PHASE_PARTIAL ')]
    require(len(lines)==1 and len(lines[0])<12000,'bounded unique partial')
    data=original.strict_json(zlib.decompress(base64.b64decode(lines[0],validate=True)).decode())
    require(data['runtime_id']==manifest['runtime_id'] and data['status']=='FAILED_PARTIAL_NOT_QUALIFIED','partial runtime/status')
    decision(data['rows'])
    gate=observer_decision(data['rows'],data['sampling'])
    counts=data['counters']
    require(tuple(counts[k] for k in ('observations','restorations','captures','timed_replays','trace_replays'))
            ==(378,54,216,4608,180),'complete retained private coverage')
    expected={tuple(r[:3]) for r in data['rows'] if r[1] in (32,128,256)}
    require(len(data['phases'])==len(data['repeated'])==18
            and {tuple(r[:3]) for r in data['phases']}==expected
            and {tuple(r[:3]) for r in data['repeated']}==expected,'retained diagnostic coverage')
    require(data['raw_trace'] is None and data['raw_trace_omitted_for_failure_bound'] is True,'raw omission disclosed')
    return dict(job=job,terminal='FAILED_PARTIAL_NOT_QUALIFIED',qualified=False,
                source_sha256=manifest['source_sha256'],observer_valid=gate['valid'],
                valid_rows=sum(r['valid'] for r in gate['rows']),total_rows=18,counters=counts,
                log_sha256=original.sha(log),scope='Failed owning receipt; complete retained private rows only, no official score or phase qualification')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.partial:data=read_partial(args.folder,args.job)
    else:
        result=read(args.folder,args.job)
        data=dict(job=args.job,official_arm='control0',official_score_us=result['official_score_us'],observer_valid=result['observer_valid'])
    print(json.dumps(data,indent=2))
