"""Final-day T1/T8 acceptance against independent duplicate controls."""
import math

from expert_top100_moe_harness_contract import BANKS, LOADS, DISTRIBUTIONS, CROSSES, positive, require


def short_decision(rows):
    expected={(seed,t,dist) for seed in BANKS for t in LOADS for dist in DISTRIBUTIONS}
    require(len(rows)==36 and len({tuple(r[:3]) for r in rows})==36
            and {tuple(r[:3]) for r in rows}==expected,'complete short-load screen')
    regressions=[]
    drift=[]
    savings={1:[],8:[]}
    groups=[]
    for row in rows:
        require(len(row)==7 and len(row[3])==4 and len(row[4])==3
                and len(row[5])==4 and len(row[6])==2,'timing record')
        total,occupied,maximum=row[4]
        require(all(type(n) is int for n in row[4]) and 1<=occupied<=32
                and 1<=maximum<=row[1] and maximum<=total<=row[1]*8
                and total<=occupied*maximum,'route census capacity')
        for value in row[3]+row[5]:positive(value)
        require(all(math.isfinite(n) and n>=0 for n in row[6]),'duplicate nulls')
        if row[1] in savings:
            saving=[row[3][a]-row[3][b] for a,b in CROSSES]
            resolved=all(s>=2. for s in saving) and all(math.log(r)>2*max(row[6]) for r in row[5])
            savings[row[1]].append(dict(bank=row[0],dist=row[2],saving_us=saving,resolved=resolved))
            if any(r<=1/1.01 for r in row[5]):regressions.append(row[:3])
        else:
            limit=max(math.log(1.01),2*max(row[6]))
            if any(abs(math.log(r))>limit for r in row[5]):drift.append(row[:3])
    for seed in BANKS:
        for dist in DISTRIBUTIONS:
            selected=[r for r in rows if r[0]==seed and r[2]==dist]
            null=max(sum(r[6][i] for r in selected)/6 for i in (0,1))
            logs=[sum(math.log(r[5][i]) for r in selected)/6 for i in range(4)]
            groups.append(dict(bank=seed,dist=dist,factors=[math.exp(x) for x in logs],
                               duplicate_log_null=null,resolved=all(x>2*null for x in logs)))
    qualifying=[t for t,items in savings.items() if all(x['resolved'] for x in items)]
    result=dict(passed=bool(qualifying) and not regressions and not drift and all(g['resolved'] for g in groups),
                qualifying_tokens=qualifying,savings=savings,groups=groups,
                affected_regressions=regressions,unchanged_drift=drift)
    result['component_pass']=result['passed']
    return result
