"""Source/SASS census and exact30-cell C2 score sensitivity; no GPU execution."""
import argparse
import json
import math
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--compiled',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.root/'.build/today/profiles/projection_best.9eba96894ee1402e.json').read_text(encoding='utf8'))
    baseline=json.loads((a.root/'muse_bf16_proj/baseline_profile/profiles/baseline.bde8b4c49c4a46bd.json').read_text(encoding='utf8'))
    sas=(a.compiled/'device.sass').read_text();ptxas=(a.compiled/'ptxas.log').read_text()
    entries=[]
    for c in re.split(r'(?=\.section\s+\.text\.)',sas):
        if not c.startswith('.section'):continue
        head=c.splitlines()[0]
        if not any(x in head for x in ('wd_kernel','sk_kernel','rt_kernel','rd_kernel')):continue
        name=head.split('.text.',1)[1].split(',',1)[0]
        pm=re.search("Compiling entry function '"+re.escape(name)+r"'.*?(?=ptxas info\s+: Compiling entry|\Z)",ptxas,re.S)
        assert pm,name
        props=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',pm[0])
        regs=re.search(r'Used (\d+) registers, used (\d+) barriers',pm[0])
        counts={k:len(re.findall(pattern,c)) for k,pattern in [('hgmma',r'\bHGMMA\.'),('hmma',r'\bHMMA\.'),('wait1',r'WARPGROUP\.DEPBAR[^;]*0x1'),('wait0',r'WARPGROUP\.DEPBAR[^;]*0x0')]}
        if 'wd_kernel' in head:
            assert counts['wait1']==counts['wait0']==1 and counts['hgmma'] in (4,8)
        else:assert counts['hgmma']==counts['wait1']==counts['wait0']==0
        entries.append(dict(name=name,**counts,stack=int(props[1]),spill_stores=int(props[2]),spill_loads=int(props[3]),registers=int(regs[1]),barriers=int(regs[2]),
            wait_excerpt=[l for l in c.splitlines() if 'WARPGROUP.DEPBAR' in l],
            loop_excerpt=[line for i,l in enumerate(c.splitlines()) if 'WARPGROUP.DEPBAR' in l for line in c.splitlines()[max(0,i-2):i+7]]))
    assert len(entries)==22 and sum(e['hgmma']>0 for e in entries)==3
    bs=receipt['benchmarks'];assert len(bs)==30 and {b['case']['weight'] for b in bs}=={62}
    def key(c):return c['layer'],c['m']
    prof={key(r['case']):r['report'] for r in receipt['profiles']}
    baseprof={key(r['case']):r['report'] for r in baseline['profiles']}
    loads={m:sum(b['median_ns']/1000 for b in bs if b['case']['m']==m) for m in (1,8,16,32,128,256)}
    score=math.prod(sum(b['baseline_ns'] for b in bs if b['case']['m']==m)/sum(b['median_ns'] for b in bs if b['case']['m']==m) for m in loads)**(1/6)
    assert abs(score-receipt['summary']['speedup'])<1e-12
    rows=[]
    for b in bs:
        c=b['case'];layer,m=key(c);t=b['median_ns']/1000;floor=b['speed_of_light_ns']/1000
        first=prof[(layer,m)].splitlines()[1]
        family='wd' if 'wd_kernel' in first else 'lt' if m>=128 else 'short'
        rows.append(dict(layer=layer,m=m,weight=c['weight'],family=family,c2_us=t,paired_sglang_us=b['baseline_ns']/1000,floor_us=floor,
             one_us_score_factor=(loads[m]/(loads[m]-1))**(1/6),floor_score_factor=(loads[m]/(loads[m]-t+floor))**(1/6),
             c2_profile=prof[(layer,m)],sglang_baseline_profile=baseprof[(layer,m)]))
    wd=[r for r in rows if r['family']=='wd'];assert len(wd)==3
    def factor(scale=None,floors=False):
        return math.prod(loads[m]/(loads[m]-sum(r['c2_us']-(r['floor_us'] if floors else scale*r['c2_us']) for r in wd if r['m']==m)) for m in loads)**(1/6)
    result=dict(c2_job=receipt['job_id'],baseline_profile_job=baseline['job_id'],reconstructed_score=score,load_sums_us=loads,rows=rows,
       wd_scenarios=dict(all3_operator_times_cut10percent=factor(.9),all3_at_task_floor=factor(floors=True),all3_removed_impossible=factor(0)),
       entries=entries,serialization_warnings=ptxas.count('C751'),scope='Static local sm90a audit plus cross-receipt profile map; scenarios not measured speedups or bounds on changing a wait.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('entries','rows')},indent=2))
    print('ENTRIES',len(entries),'WD',[(e['hgmma'],e['wait1'],e['wait0'],e['registers'],e['spill_stores'],e['spill_loads']) for e in entries if e['hgmma']])


if __name__=='__main__':main()
