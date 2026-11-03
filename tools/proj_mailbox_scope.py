"""No-GPU active-reducer/whole-score sensitivity from the pinned C2 audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    evidence=a.root/'docs/steering/evidence/2026-11-01-projection-waits/audit.json'
    audit=json.loads(evidence.read_text(encoding='utf8'))
    oldpath=a.root/'muse_bf16_proj/profiles/qkv_push.4edaa20c98904404.json'
    old=json.loads(oldpath.read_text(encoding='utf8'))
    assert old['id']=='4edaa20c98904404' and old['board']=='muse_bf16_proj' and old['result']['passed']
    sums={int(m):v for m,v in audit['load_sums_us'].items()}
    rows=[]
    for row in audit['rows']:
        r=dict(row);profile=r.pop('c2_profile');r.pop('sglang_baseline_profile')
        r['reducer_family']=next((name for name in ('sk','rt','rd','wd') if name+'_kernel<' in profile),'other')
        r['reducer_time_measured_us']=None
        rows.append(r)
    def scenario(names,cut):
        return math.prod(sums[m]/(sums[m]-sum(cut for r in rows if r['m']==m and r['reducer_family'] in names)) for m in sums)**(1/6)
    result=dict(c2_job=audit['c2_job'],c2_score=audit['reconstructed_score'],
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        old_push_job=old['id'],old_push_receipt_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        load_sums_us=sums,rows=rows,
        scenarios={','.join(names):dict(cells=sum(r['reducer_family'] in names for r in rows),
                   one_us_per_operator=scenario(names,1.),half_us_per_operator=scenario(names,.5))
                   for names in [('sk',),('rt',),('rd',),('sk','rt','rd'),('wd',)]},
        scope='Hypothetical uniform whole-operator cuts, NOT measured reducer cost, speedup forecast or attainable bound. No GPU execution.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
