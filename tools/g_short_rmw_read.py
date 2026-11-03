"""CPU-only owning reader for PRIVATE f042 short-RMW check. Never a public gain claim."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

MANIFEST=json.loads(Path(__file__).with_name('g_short_rmw_reader_manifest.json').read_text())
SEEDS=(993811,993817,993821)
LOADS=(1,8,16,32,128,256)
CELLS={(t,c,s) for t in LOADS for c in ('sliding','full') for s in SEEDS}
OFFICIAL={(l,c) for l in ('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32') for c in ('sliding','full')}

def positive(x):
    assert type(x) in (int,float) and math.isfinite(x) and x>0,'nonpositive/nonfinite'
    return x

def decision(rows):
    assert len(rows)==36 and {(r['tokens'],r['context'],r['seed']) for r in rows}==CELLS,'row coverage'
    summaries=[]
    for seed in SEEDS:
        gains=[0.]*4;raw=[0.]*4;null=0.;regression=True
        for t in LOADS:
            pair={r['context']:r['times_us'] for r in rows if r['tokens']==t and r['seed']==seed}
            assert set(pair)=={'sliding','full'}
            for v in pair.values():
                assert len(v)==4
                for x in v:positive(x)
            cost=[46*pair['sliding'][i]+16*pair['full'][i] for i in range(4)]
            for c,v in ((0,2),(0,3),(1,2),(1,3)):
                regression &= all(pair[k][v]-pair[k][c]<=max(.5,.01*pair[k][c]) for k in pair)
            for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):raw[k]+=math.log(cost[c]/cost[v])/6
            null+=max(abs(math.log(cost[0]/cost[1])),abs(math.log(cost[2]/cost[3])))/6
            if t in (1,8):
                for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):gains[k]+=math.log(cost[c]/cost[v])/6
        passed=min(raw)>max(math.log(1.01),2*null) and regression
        summaries.append(dict(seed=seed,sixload_log_contribution=gains,raw_allrow_log_contribution=raw,duplicate_log_null_contribution=null,regression_gate=regression,partial_or_complete_gate=passed))
    return dict(complete_sixload=True,gate=all(s['partial_or_complete_gate'] for s in summaries),seeds=summaries)

def close(a,b):
    if type(a) is float or type(b) is float:
        return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def capsule(log):
    blobs=re.findall(r'^SRM_REPORT ([A-Za-z0-9+/=]+)\r?$',log,re.M)
    assert len(blobs)==1 and len(blobs[0])<11000,'missing/duplicate/truncated capsule'
    payload=zlib.decompress(base64.b64decode(blobs[0],validate=True))
    assert len(payload)<1_000_000
    report=json.loads(payload,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    pins=MANIFEST['capsule_pins']
    for field,key in (('candidate_pin','_WL_PIN'),('g1b4_conformance','_WL_CONFORMANCE'),('device_sha','_WL_DEVICE_SHA'),('harness_core_sha','_WL_CORE_SHA')):
        assert report[field]==pins[key],('source pin',field)
    assert report['schema']==1 and report['part']=='all' and report['seeds']==list(SEEDS)
    assert report['math_checks']==report['fresh_final_checks']==144 and report['empty_native_checks']==32
    assert report['fresh_replays_per_bank']==8 and report['judged_stage_on'] is True and report['judged_short_relaxed'] is False
    positive(report['prepare_s']);assert report['prepare_s']<295
    assert re.fullmatch('[a-f0-9]{64}',report['weight_signature'])
    assert len(report['owners_bytes'])==4 and all(type(x)is int and x>0 for x in report['owners_bytes'])
    assert set(report['resources'])=={'tiny','tiny2','mid','b4','main'}
    for key,vectors in report['resources'].items():
        assert len(vectors)==4 and all(v==vectors[0] for v in vectors),'resource parity'
        for v in vectors:
            assert len(v)==12 and all(type(x)is int and x>=0 for x in v)
            assert v[4]>=1 and v[10]>=1 and v[0]>0 and v[6]>0 and v[5]>0 and v[11]>0
    flags=report['module_flags']
    assert set(flags)=={str(i)+':'+key for i in (0,2) for key in ('tiny','tiny2','mid','b4','main')}
    for name,fs in flags.items():
        i,key=name.split(':');assert isinstance(fs,list) and all(isinstance(x,str) for x in fs)
        assert ('-DOKC_SHORT_FINAL_RELAXED=1' in fs)==(i=='2' and key in ('tiny','tiny2'))
        assert '-DOKC_STAGE_RETIRE=0' not in fs
        if key in ('main','b4'):assert ('-DOKC_B4FIN=1' in fs)==(key=='b4')
    for row in report['rows']:
        t=row['tokens'];expected='tiny' if t<=4 else 'tiny2' if t<=8 else 'mid' if t<=16 else 'b4' if t<=32 else 'main'
        assert type(row['local_start'])is int and row['local_start'] in range(0,256,32)
        counts=row['reference_group_counts'];assert len(counts)==8 and all(type(n)is int and n>=0 for n in counts) and sum(counts)==t*8
        assert row['actual_routes']==[[[t,expected,i>=2]] for i in range(4)],'backend hits'
    recomputed=decision(report['rows']);assert close(recomputed,report['decision']),'printed gate mismatch'
    return report,recomputed

def read(receipt,job,log):
    assert receipt['id']==job and receipt['board']=='muse_decode_layer' and receipt['mode']=='check','own private job required'
    assert receipt['status']=='done','not terminal'
    assert re.fullmatch('[a-f0-9]{64}',receipt['content_hash'])
    assert receipt['payload_ref']=='muse_decode_layer/'+receipt['content_hash']+'.py'
    r=receipt['result'];assert r['status']=='ok' and r['passed'] is True and r['returncode']==0
    assert r['checks']['ok'] is True and not r['checks'].get('flagged') and not r['checks'].get('rejected')
    assert not r['flags'] and not r['error'],'final guard/error failure'
    records=r['records'];tests=[x for x in records if x.get('event')=='test'];setup=[x for x in records if x.get('event')=='setup']
    assert len(tests)==13 and all(x['status']=='pass' for x in tests)
    assert len(setup)==1 and setup[0]['status']=='pass' and setup[0]['error'] is None and positive(setup[0]['prepare_s'])<295
    for event in ('benchmark','profile'):
        rs=[x for x in records if x.get('event')==event]
        assert len(rs)==12 and {(x['case']['load'],x['case']['layer']) for x in rs}==OFFICIAL
        for x in rs:
            assert x['case']['weight']==(46 if x['case']['layer']=='sliding' else 16)
            if event=='benchmark':
                assert x['status']=='pass' and not x['flags']
                for k in ('median_ns','median_raw_ns','baseline_ns','baseline_raw_ns'):positive(x[k])
            else:assert x['report'].strip()
    report,recomputed=capsule(log)
    return dict(job=job,source_sha256=MANIFEST['source_sha256'],source_bytes=MANIFEST['bytes'],complete=True,private_raw_gate=recomputed['gate'],decision=recomputed,report=report,scope='Raw cold GPU-event whole-graph gate; normal official arm is607 control. Not an official candidate paired score or automatic release.')

def verify_source(path):
    raw=path.read_bytes()
    assert len(raw)==MANIFEST['bytes'] and hashlib.sha256(raw).hexdigest()==MANIFEST['source_sha256'],'wrong uploaded source artifact'
    return MANIFEST['source_sha256']

def owning_log(receipt,path,attestation=None):
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();expected=receipt['output']['log']
    assert receipt['id'] in Path(expected).name
    if attestation is None:assert path.resolve()==Path(expected).resolve(),'wrong owning log'
    else:
        assert attestation==dict(job=receipt['id'],owning_log=expected,sha256=digest,source_sha256=MANIFEST['source_sha256']),'archive attestation mismatch'
    return raw.decode('utf8'),digest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--job',required=True);p.add_argument('--source',required=True,type=Path);p.add_argument('--log',required=True,type=Path);p.add_argument('--archive-attestation',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();verify_source(a.source);r=json.loads(a.receipt.read_text());att=json.loads(a.archive_attestation.read_text()) if a.archive_attestation else None
    log,digest=owning_log(r,a.log,att);out=read(r,a.job,log);out['owning_log_sha256']=digest
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(job=a.job,complete=True,private_raw_gate=out['private_raw_gate'])))
