"""Validate sparse WGS timestamps and price their complete-call perturbation."""
import math
import statistics

PH_FIELDS = 24
PH_LOADS = (32,128,256)


def _ph_require(value, message):
    if not value:
        raise ValueError('MOE_PHASE: '+message)


def _ph_stats(values):
    values=sorted(values)
    return [values[0],statistics.median(values),values[math.ceil(.95*len(values))-1],values[-1]]


def phase_trace(flat, grid):
    _ph_require(type(grid) is int and 0 < grid <= 136 and len(flat) >= grid*PH_FIELDS,'trace extent')
    rows=[flat[i*PH_FIELDS:(i+1)*PH_FIELDS] for i in range(grid)]
    _ph_require(all(type(x) is int and x >= 0 for row in rows for x in row),'trace values')
    jobs=rows[0][19]
    _ph_require(0 < jobs <= 88 and all(r[19]==jobs for r in rows),'job metadata')
    for phase,columns in ((0,(20,21)),(1,(22,23))):
        ranges=[(r[columns[0]],r[columns[1]]) for r in rows]
        _ph_require(ranges[0][0]==0 and ranges[-1][1]==jobs*(512 if phase==0 else 256),'range bounds')
        _ph_require(all(a<=b for a,b in ranges) and all(ranges[i][1]==ranges[i+1][0] for i in range(grid-1)),'range coverage')
    for r in rows:
        _ph_require(r[0]>0 and r[0]<=r[1]<=r[2],'entry/routing/gather order')
        _ph_require(all(r[2]<=r[3+i]<=r[7+i]<=r[11] for i in range(4)),'warpgroup/CTA order')
        _ph_require(r[11]<=r[12]<=r[13],'grid/finalizer order')
        _ph_require(all(x==0 or x>=r[0] for x in r[14:18]),'producer epoch')
        if r[23]>r[22]:
            _ph_require(r[14]>0 and r[15]>=r[14] and r[17]>0,'first B producer stamps')
    _ph_require(min(r[12] for r in rows)>=max(r[11] for r in rows),'global join order')
    start=min(r[0] for r in rows)
    result=dict(grid=grid,jobs=jobs,unique_smids=len({r[18] for r in rows}),
                boundary_span_us=(max(r[13] for r in rows)-start)/1000,
                consumer_finish_spread_us=(max(r[11] for r in rows)-min(r[11] for r in rows))/1000,
                A_finish_spread_us=(max(max(r[3:7]) for r in rows)-min(max(r[3:7]) for r in rows))/1000,
                last_compute_to_last_exit_us=(max(r[13] for r in rows)-max(r[11] for r in rows))/1000,
                entry_to_A_begin_us=_ph_stats([(r[2]-r[0])/1000 for r in rows]),
                A_to_B_end_us=_ph_stats([(r[7+i]-r[3+i])/1000 for r in rows for i in range(4)]),
                grid_wait_us=_ph_stats([(r[12]-r[11])/1000 for r in rows]),
                finalizer_us=_ph_stats([(r[13]-r[12])/1000 for r in rows]),
                first_B_ready_us=_ph_stats([(r[15]-r[14])/1000 for r in rows if r[14]]),
                A_stages=_ph_stats([r[21]-r[20] for r in rows]),
                B_stages=_ph_stats([r[23]-r[22] for r in rows]))
    for key,value in result.items():
        if isinstance(value,float):result[key]=round(value,2)
        elif isinstance(value,list):result[key]=[round(x,2) if isinstance(x,float) else x for x in value]
    selected={0,grid//2,grid-1}
    for key in (lambda r:r[0],lambda r:r[11],lambda r:r[13],
                lambda r:max(r[3:7]),lambda r:r[15]-r[14],lambda r:r[12]-r[11]):
        selected.add(min(range(grid),key=lambda i:key(rows[i])))
        selected.add(max(range(grid),key=lambda i:key(rows[i])))
    packed=[]
    for index in sorted(selected):
        row=rows[index]
        packed.append([index,*[x-start if x else -1 for x in row[:18]],*row[18:]])
    return result,dict(grid=grid,base_ns=start,critical_ctas=packed,
                       scope='Raw fixed/critical CTA rows; summaries validated against every CTA in setup')


def sampling_summary(samples):
    _ph_require(len(samples)==4 and all(len(x)==32 for x in samples),'sampling coverage')
    deltas=[sum(samples[o][k] for o in (2,3) for k in (i,i+1))/4
            -sum(samples[c][k] for c in (0,1) for k in (i,i+1))/4 for i in range(0,32,2)]
    mean=statistics.mean(deltas)
    half=2.131*statistics.stdev(deltas)/4
    iqrs=[statistics.quantiles(x,n=4,method='inclusive')[2]
          -statistics.quantiles(x,n=4,method='inclusive')[0] for x in samples]
    return [round(x,4) for x in (statistics.median(deltas),mean-half,mean+half,max(iqrs[:2]),max(iqrs[2:]))]


def observer_decision(rows, sampling):
    selected=[r for r in rows if r[1] in PH_LOADS]
    _ph_require(len(selected)==18,'observer row coverage')
    _ph_require(len(sampling)==18 and {tuple(x[:3]) for x in sampling}=={tuple(x[:3]) for x in selected},'sampling rows')
    evidence={tuple(x[:3]):x[3] for x in sampling}
    outcomes=[]
    for row in selected:
        c0,c1,o0,o1=row[3]
        control=(c0+c1)/2;observer=(o0+o1)/2
        limit=max(.5,.01*control)
        stable=max(abs(c0-c1),abs(o0-o1))<=limit
        distortion=observer-control
        uncertainty=evidence[tuple(row[:3])]
        _ph_require(len(uncertainty)==5 and all(math.isfinite(x) for x in uncertainty)
                    and uncertainty[1]<=uncertainty[2] and min(uncertainty[3:])>=0,'sampling uncertainty')
        bounded=uncertainty[1]>=-limit and uncertainty[2]<=limit
        outcomes.append(dict(seed=row[0],tokens=row[1],dist=row[2],control_us=control,
                             observer_us=observer,distortion_us=distortion,limit_us=limit,
                             duplicates_stable=stable,sampling=uncertainty,
                             valid=stable and abs(distortion)<=limit and bounded))
    return dict(valid=all(r['valid'] for r in outcomes),rows=outcomes,
                scope='Sparse boundary attribution only; no candidate gain or hardware-ceiling claim')
