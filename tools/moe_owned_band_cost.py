"""CPU index/request model for fixed band32; not cache traffic or timing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def cost(counts, units, grid=132):
    sizes = [min(8,c-k*8) for c in counts for k in range((c+7)//8)]
    groups=[];start=0
    for c in counts:
        nj=(c+7)//8
        if nj:groups.append((start,nj));start+=nj
    old=[(j,rb) for j in range(len(sizes)) for rb in range(units)]
    new=[(j,rb) for j0,nj in groups for band in range(0,units,32)
         for j in range(j0,j0+nj) for rb in range(band,band+32)]
    assert Counter(old)==Counter(new) and len(set(new))==len(new)
    def runs(seq):
        total=live=credits=0
        for cta in range(grid):
            local=seq[len(seq)*cta//grid:len(seq)*(cta+1)//grid]
            jobs=[j for j,rb in local]
            credits+=len(set(jobs))
            for i,j in enumerate(jobs):
                if i==0 or j!=jobs[i-1]:total+=1;live+=sizes[j]*4096*2
        return dict(job_runs=total, activation_requested_bytes=live, unique_cta_jobs=credits)
    a,b=runs(old),runs(new)
    return dict(units=units,old=a,band32=b,
                extra_activation_requested_bytes=b['activation_requested_bytes']-a['activation_requested_bytes'],
                band_weight_unique_bytes=32*36864,
                max_expert_jobs=max((n for _,n in groups),default=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('census',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=json.loads(a.census.read_text());rows=[]
    for r in data['records']:
        rows.append(dict(m=r['m'],seed=r['seed'],phaseA=cost(r['counts'],512),phaseB=cost(r['counts'],256)))
    aggregate={}
    for m in (16,32):
        rs=[r for r in rows if r['m']==m];out={}
        for ph in ('phaseA','phaseB'):
            out[ph]={key:statistics.mean(r[ph][key] for r in rs) for key in ('extra_activation_requested_bytes',)}
            for arm in ('old','band32'):
                out[ph][arm]={k:statistics.mean(r[ph][arm][k] for r in rs)
                              for k in ('job_runs','activation_requested_bytes','unique_cta_jobs')}
            out[ph]['max_extra_activation_requested_bytes']=max(r[ph]['extra_activation_requested_bytes'] for r in rs)
        aggregate[str(m)]=out
    result=dict(schema='OWNED_BAND32_CPU_REQUEST_MODEL',grid=132,band=32,
                source='owned8-row/fullK16weightrows; xr reload only at job/phase change',
                limits='Logical loads/ownership only. No L1/L2/DRAM bytes, latency or speedup.',
                aggregate=aggregate,records=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
