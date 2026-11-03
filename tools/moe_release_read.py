"""Read the exact standalone control's official private qualification result."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(value,message):
    if not value:raise ValueError('MOE_RELEASE: '+message)


def positive(value):
    require(type(value) in (int,float) and math.isfinite(value) and value>0,'finite positive result')
    return value


def validate(receipt,accepted,manifest,payload):
    job=accepted['job_id']
    require(receipt['id']==job and receipt['board']=='muse_moe_w4a16' and receipt['team']=='yelgabo'
            and receipt['mode']=='check' and receipt['status']=='done','own terminal problem/job')
    require(hashlib.sha256(payload).hexdigest()==accepted['source_sha256']==manifest['source_sha256'],'exact release source')
    require(receipt['payload_ref']=='muse_moe_w4a16/'+receipt['content_hash']+'.py','server payload association')
    result=receipt['result']
    passed=(result['passed'] is True and result['returncode']==0 and result['checks']['ok'] is True
            and not result['flags'] and not result['checks']['flagged'] and not result['checks']['rejected'])
    summary=dict(job=job,source_sha256=manifest['source_sha256'],passed=passed,public_submission_approved=False,
                 material_release_lead=False,scope='Exact-file official release qualification; public comparison is cross-job, not a matched mechanism gain')
    if not passed:
        summary.update(returncode=result['returncode'],error=result.get('error'))
        return summary
    rows=result['records']
    tests=[x for x in rows if x.get('event')=='test']
    benchmarks=[x for x in rows if x.get('event')=='benchmark']
    profiles=[x for x in rows if x.get('event')=='profile']
    require(len(tests)==5 and all(x['status']=='pass' for x in tests),'five official correctness tests')
    require(len(benchmarks)==len(profiles)==6,'six benchmark/profile records')
    require({x['case']['bs'] for x in benchmarks}=={1,8,16,32,128,256},'all scored loads')
    require({x['case']['bs'] for x in profiles}=={1,8,16,32,128,256},'all profiled loads')
    require(all(x['status']=='pass' and not x['flags'] for x in benchmarks),'benchmark correctness/flags')
    score=positive(result['score_us']);speedup=positive(result['speedup']);ci=result['speedup_ci']
    require(len(ci)==2 and positive(ci[0])<=speedup<=positive(ci[1]),'official uncertainty')
    baseline=6069.607719056422
    summary.update(score_us=score,speedup=speedup,speedup_ci=ci,official_tests=5,official_benchmarks=6,official_profiles=6,
                   public_baseline=dict(job='abe7651dbaa44458',score_us=baseline,speedup=2.2157),
                   lower_elapsed_percent=100*(1-score/baseline),material_release_lead=score<=.99*baseline and ci[0]>2.2157,
                   target_us=5559.8,target_met=score<=5559.8,
                   loads=[dict(tokens=x['case']['bs'],load=x['case']['load'],adjusted_us=positive(x['median_ns'])/1000,
                               sglang_us=positive(x['baseline_ns'])/1000,speedup=positive(x['speedup'])) for x in benchmarks])
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();folder=args.folder;source=args.source
    data=validate(json.loads((folder/'status-wait.raw.json').read_text()),json.loads((folder/'accepted.json').read_text()),
                  json.loads(source.with_suffix('.manifest.json').read_text()),source.read_bytes())
    (folder/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
