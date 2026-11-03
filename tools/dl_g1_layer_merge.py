"""Merge COMPLETE terminal decode+verify reports. Never treats partial checks as sixload PASS."""
import argparse,base64,json,math,re,zlib
from pathlib import Path

def read(path):
    s=path.read_text(encoding='utf8')
    if path.suffix=='.json':
        obj=json.loads(s);r=obj.get('result',obj)
        assert obj.get('board')=='muse_decode_layer' and obj.get('mode')=='check' and obj.get('id'),'private layer job required'
        assert obj.get('status')=='done' and r.get('passed') and r.get('checks',{}).get('ok') and r.get('flags')==[],'clean terminal official receipt required'
        tests=[x for x in r.get('records',[]) if x.get('event')=='test']
        assert len(tests)==13 and all(x.get('status')=='pass' for x in tests),'13officialcandidate tests required'
        for event in ('benchmark','profile'):
            cells=[x for x in r['records'] if x.get('event')==event]
            assert len(cells)==12 and {x['index'] for x in cells}==set(range(12)),'incomplete official cells'
        s=r['log_tail']
    matches=re.findall(r'WL_STAGE_REPORT ([A-Za-z0-9+/=]+)',s)
    assert len(matches)==1,'exactly one complete WL report'
    return json.loads(zlib.decompress(base64.b64decode(matches[0])))

def decide(reports):
    assert len(reports) in (1,2)
    for r in reports:
        assert r['judged_stage_on'] and r['empty_native_checks']==12
        assert r['math_checks']==r['fresh_final_checks']==4*len({x['tokens'] for x in r['rows']})*2*3
    first=reports[0]
    assert all((r['candidate_pin'],r.get('g1b4_conformance'),r['device_sha'],r['harness_core_sha'],r['weight_signature'],r['seeds'],r['resources'])==(first['candidate_pin'],first.get('g1b4_conformance'),first['device_sha'],first['harness_core_sha'],first['weight_signature'],first['seeds'],first['resources']) for r in reports),'mismatched source/weight/seed/resource gates'
    rows=[x for r in reports for x in r['rows']]
    keys={(x['tokens'],x['context'],x['seed']) for x in rows}
    assert len(rows)==len(keys)==36,'need BOTH contexts/all6loads/3seeds exactlyonce'
    decisions=[];out=[]
    for seed in first['seeds']:
        logs=[0.]*4;raw_logs=[0.]*4;null=0.;regress=True
        for t in (1,8,16,32,128,256):
            pair={r['context']:r['times_us'] for r in rows if r['tokens']==t and r['seed']==seed};assert set(pair)=={'sliding','full'}
            cost=[46*pair['sliding'][i]+16*pair['full'][i] for i in range(4)]
            for c,v in ((0,2),(0,3),(1,2),(1,3)):regress &= all(pair[k][v]-pair[k][c]<=max(.5,.01*pair[k][c]) for k in pair)
            for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):raw_logs[k]+=math.log(cost[c]/cost[v])/6
            null+=max(abs(math.log(cost[0]/cost[1])),abs(math.log(cost[2]/cost[3])))/6
            if t in (32,128,256):
                for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):logs[k]+=math.log(cost[c]/cost[v])/6
        passed=min(logs)>max(math.log(1.002),2*null) and min(raw_logs)>max(0.,2*null) and regress
        decisions.append(passed);out.append({'seed':seed,'fourcross_sixload':[math.exp(v) for v in logs],'raw_allrow_fourcross':[math.exp(v) for v in raw_logs],'null_log':null,'regression':regress,'pass':passed})
    return {'decision':all(decisions),'seeds':out,'all36rows':True,'purequalification_required':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',nargs='+',type=Path);a=p.parse_args()
    assert all(p.suffix=='.json' for p in a.receipts),'promotion gate requires terminal receipts, not bare logs'
    print(json.dumps(decide([read(p) for p in a.receipts]),indent=2))
