"""Validate complete private VP2 receipt and recompute all paired score sensitivities."""
import ast,json,math,statistics
from pathlib import Path

def score_function(manifest):
    source=Path(__file__).with_name('moe_attn_vp2_body.py').read_text(encoding='utf8')
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_vp_score')
    ns={'math':math,'statistics':statistics,'_VP_MANIFEST':manifest}
    exec(compile(ast.Module([fn],[]),'<frozen-vp2-score>','exec'),ns)
    return ns['_vp_score']

def close(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or isinstance(a,str):return a==b
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12

def analyze(receipt):
    if 'result' not in receipt or 'board' not in receipt:
        raise ValueError('Requires raw terminal status JSON with result.log_tail and official test/benchmark records; normalized profile JSON is insufficient')
    assert receipt['board']=='muse_attention_decode' and receipt['status']=='done'
    result=receipt['result'];assert result['passed'] and not result.get('flags'), 'requires passing official comparator'
    lines=result['log_tail'].splitlines()
    assert any(l.startswith('VP2_PRIVATE_DONE 36 ') for l in lines),'incomplete private preparation'
    r=json.loads(next(l.partition(' ')[2] for l in lines if l.startswith('VP2_PRIVATE_REPORT ')))
    assert r['schema']=='PRIVATE_VP2_BF16_V1' and r['row_kind']=='raw_us'
    assert r['counts']=={'eager_extra':16,'normal':144,'fresh':336,'post':144}
    assert r['dispatch_counts']=={'ONE':284,'VP2':20},'all-fallback/incomplete coverage'
    assert r['source']['sch']==0 and r['source']['measurement_only']
    assert len(r['plans'])==48 and sum(p[4][0]=='VP2_grid_cluster_threads' for p in r['plans'])==2
    assert all(p[:3]==[1,8,1] and p[3] in (2,3) and p[4][1:]==[32,16,256] for p in r['plans'] if p[4][0]=='VP2_grid_cluster_threads')
    res=r['resources'];assert len(res)==4 and res[0]==res[1] and res[2]==res[3]
    assert all(len(res[i])==7 and res[i][1]==0 and res[i][-1]>=2 for i in (2,3))
    score=score_function(r['source']);raw=score(r['rows']);assert close(raw,r['raw_gate'])
    fixed={tuple(v[:3]):v[3] for v in r['fixed_offsets']};assert len(fixed)==12
    adjusted=[(b,l,w,s,[t-fixed[(b,l,w)] for t in ts]) for b,l,w,s,ts in r['rows']]
    stale=score(adjusted);assert close(stale,r['gate'])
    benchmarks=[v for v in result['records'] if v.get('event')=='benchmark']
    tests=[v for v in result['records'] if v.get('event')=='test']
    assert len(benchmarks)==12 and len(tests)==16 and all(v['status']=='pass' for v in tests)
    offsets={}
    for v in benchmarks:
        c=v['case'];key=(c['batch'],8 if c['load'].startswith('verify') else 1,int(c['layer']=='sliding'))
        offsets[key]=(v['median_raw_ns']-v['median_ns'])/1000
    assert len(offsets)==12
    current=score([(b,l,w,s,[t-offsets[(b,l,w)] for t in ts]) for b,l,w,s,ts in r['rows']])
    raw_sign=all(min(v[1])>1 for v in raw['per_seed'])
    decision=current['keep'] and raw_sign and not raw['regression']
    return {'job':receipt['id'],'official_private_comparator_score':result['speedup'],
            'score_ci':result['speedup_ci'],'raw_gate':raw,'stale_gate':stale,'contemporary_gate':current,
            'contemporary_offsets':[[*k,v] for k,v in offsets.items()], 'report':r,
            'registered_raw_sign_gate':raw_sign,'registered_decision':decision,
            'scope':'Private donor measurement only. Paired sensitivities are not official candidate scores; root reviews contemporaneous gate before any owned rewrite. No public/adoption authorization.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
    print(json.dumps(analyze(json.loads(a.receipt.read_text(encoding='utf8'))),indent=2))
