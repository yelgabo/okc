"""Conditional source/instruction budget, not a measured or forecast speedup."""
import argparse,json,math
from pathlib import Path

def run(root,grid=132):
    d=json.loads((root/'muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.json').read_text(encoding='utf8'))
    assert d['job_id']=='607eb52bdaf24160'
    loads={}
    for b in d['benchmarks']:
        c=b['case'];loads.setdefault(c['load'],[]).append((c['weight'],b['median_ns']/1000))
    loads={k:sum(w*t for w,t in v)/62 for k,v in loads.items()}
    short=[loads[k]for k in ('decode 1','verify 1')]
    factor=lambda us:math.exp(sum(math.log(t/(t-us))for t in short)/6)
    lo,hi=0.,min(short)*.5
    for _ in range(80):
        mid=(lo+hi)/2
        if factor(mid)<1.01:lo=mid
        else:hi=mid
    rows=[]
    for t,j in ((1,8),(8,32)):
        a=math.ceil(512*j/grid);b=math.ceil(256*j/grid)
        intended=a*(1 if t==1 else 2)+b*2
        rows.append(dict(tokens=t,maximum_jobs=j,assumed_grid=grid,max_assigned_A_units=a,max_assigned_B_units=b,
                         intended_saved_warp_conversion_instructions_before_unpack_cost=intended,
                         ns_exposed_saving_per_intended_instruction_needed_for_3us=3000/intended))
    return dict(job='607eb52bdaf24160',weighted_short_us=short,conditional_factor_if_3us_saved_in_both_short_loads=factor(3),
                equal_short_us_cut_for_1percent_factor=hi,unit_scenarios=rows,
                assumptions=['same US cut in both contexts of both short loads; other4loads unchanged',
                             'grid132 is an explicit scenario; exact runtime grid/jobcounts/stalls not retained here',
                             'alllocaljobs upperlimits8(T1),32(T8); actualroutingcanhavefewer and balancedunitassignmentassumed',
                             'counting intended packed-conversion savings ignores extra unpack/PRMT and gate dependency; compiled prototype fails this premise'],
                actual_exposed_time_us=None,latency_forecast=None,GPU=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();d=run(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8');print(json.dumps(d,indent=2))
