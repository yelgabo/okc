"""G-layer-only dense budget from exact retained receipts. No kernel changes/jobs."""
import argparse,json,math,re,hashlib
from pathlib import Path
from dl_proj_budget_audit import counterfactual,dense_rows,load_times

def low_op(name,m):
    parts=name.split('<',1)[1].split(',')
    nt,nwr,wsk,upw,ns=map(int,parts[2:7])
    if nt==3:return 'qkv'
    if nt==4 and wsk==8:return 'pr'
    if nt==1:return 'un'
    if m<=8:return 'o'
    if m==16:return 'un' if ns==5 else 'o'
    if m==32:return 'un' if nt==4 and nwr==2 else 'o'
    raise ValueError((name,m))

def audit(root):
    paths=[root/'muse_decode_layer/profiles'/f for f in ('g1-stage-release.607eb52bdaf24160.json','muse_decode_layer.3a5f879df0d44825.json')]
    current,old=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    assert [d['job_id'] for d in (current,old)]==['607eb52bdaf24160','3a5f879df0d44825']
    source=root/'muse_decode_layer/cand/g1b4_stage_release.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
    assert len(current['benchmarks'])==len(old['benchmarks'])==12
    allgap={};lowgap={};qouzero={};large={};group={};rows=[];lowqou={};lgindividual={}
    for i,(b,p,q) in enumerate(zip(current['benchmarks'],current['profiles'],old['profiles'])):
        c=b['case'];assert (c['load'],c['layer'])==(old['benchmarks'][i]['case']['load'],old['benchmarks'][i]['case']['layer'])
        a,z=dense_rows(p['report']),dense_rows(q['report']);key=(c['load'],c['layer'])
        allgap[key]=max(0,sum(t for _,t,_ in a)-sum(t for _,t,_ in z))
        row=dict(load=c['load'],layer=c['layer'],weight=c['weight'],whole_us=b['median_ns']/1000,dense_us=sum(t for _,t,_ in a),g0_dense_us=sum(t for _,t,_ in z))
        if i<8:
            m=c['batch']*c.get('q_len',1)
            aa={low_op(n,m):t for n,t,_ in a};zz={low_op(n,m):t for n,t,_ in z}
            assert set(aa)==set(zz)=={'qkv','o','pr','un'},(i,aa,zz)
            row.update(ops=aa,g0_ops=zz)
            lowgap[key]=sum(max(0,aa[k]-zz[k]) for k in ('qkv','o','un'))
            qouzero[key]=sum(aa[k] for k in ('qkv','o','un'))
            for op in ('qkv','o','un'):
                lowqou.setdefault(op,{})[key]=max(0,aa[op]-zz[op])
        else:large[key]=allgap[key]
        rows.append(row)
    # Existing actual owned whole-layer profile comparison: grouped QKV+O only.
    for i,other in ((8,14.8+7.8),(9,14.5+7.6)):
        t=[t for n,t,calls in dense_rows(current['profiles'][i]['report']) if n=='nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT' and calls==2]
        assert len(t)==1
        c=current['benchmarks'][i]['case'];group[c['load'],c['layer']]=max(0,t[0]-other)
    loads,base=load_times(current),load_times(current,'baseline_ns')
    score=math.exp(sum(math.log(base[k]/loads[k]) for k in loads)/6)
    assert abs(score-current['summary']['speedup'])<1e-9
    return dict(job_id=current['job_id'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),score=score,weighted_load_us=loads,rows=rows,factors=dict(all_positive_dense_group_gaps=counterfactual(current,allgap),large_positive_dense_group_gaps=counterfactual(current,large),low_positive_qkv_o_un_gaps=counterfactual(current,lowgap),joint_low_plus_large=counterfactual(current,large|lowgap),owned_128_qkv_o_group=counterfactual(current,group),zero_all_low_qkv_o_un_rows=counterfactual(current,qouzero),ten_percent_all_low_qkv_o_un_rows=counterfactual(current,{k:.1*v for k,v in qouzero.items()}),individual_low_positive={op:counterfactual(current,s) for op,s in lowqou.items()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
if __name__=='__main__':main()
