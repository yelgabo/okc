"""CPU-only owning reader for frozen979 common+RMW single-arm measurement.

No GPU/service access. A public point score is not a paired causal RMW effect.
"""
import argparse,ast,hashlib,json,math,re
from pathlib import Path
from hybrid_layer_measurement import read as official_read, owning_log, compare

SOURCE_SHA='97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7'
SOURCE_BYTES=914572
BASE_SHA='2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd'
COUNTS=dict(cells=12,checks=72,captures=12,fresh_checked=36,empty_checks=60,empty_captures=12,restored=48)
LOADS=(1,8,16,32,128,256)
CELLS={(t,c) for t in LOADS for c in ('sliding','full')}
KEYS={'tiny','tiny2','mid','b4','main'}

def source_contract(path):
    raw=path.read_bytes();assert len(raw)==SOURCE_BYTES and hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'wrong frozen source'
    tree=ast.parse(raw.decode('utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gq_attn_record')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen CPU metadata predicate>','exec'),ns)
    return ns['_gq_attn_record']

def finite_tree(x):
    if isinstance(x,float):assert math.isfinite(x),'nonfinite capsule'
    elif isinstance(x,dict):
        for v in x.values():finite_tree(v)
    elif isinstance(x,list):
        for v in x:finite_tree(v)

def qualification(log,check_attention):
    done=re.findall(r'^G_COMMON_QUAL_DONE (.*)\r?$',log,re.M)
    fallback=re.findall(r'^G_COMMON_QUAL_UNEXERCISED (.*)\r?$',log,re.M)
    assert len(done)+len(fallback)==1,'missing/duplicate/ambiguous startup capsule'
    if fallback:
        line=fallback[0].strip()
        assert re.fullmatch(r'(?:native_expert_Marlin_fallback|unsupported_weight_interface) rmw_exercised=0 .+',line),'invalid unexercised capsule'
        return dict(status='unexercised',treatment_exercised=False,raw_fallback=line,complete_native_startup=False,causal_RMW_effect=None)
    assert len(done[0])<11000,'startup capsule oversized'
    r=json.loads(done[0],parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)));finite_tree(r)
    assert r['arm']=='short-rmw' and r['base_sha']==BASE_SHA,'wrong fixed arm/base'
    for k,v in COUNTS.items():assert type(r[k])is int and r[k]==v,('coverage',k)
    assert r['native_expert'] is True and r['rmw_exercised'] is True,'treatment not exercised'
    identity=r['identity'];assert isinstance(identity,dict)
    assert identity['capability']==[9,0] and type(identity['sm_count'])is int and identity['sm_count']>0
    for k in ('cuda_driver_version','cuda_runtime_version'):assert type(identity[k])is int and identity[k]>0
    for k in ('gpu_name','total_memory','device_uuid','node_environment','node_physical_identity','torch_version','torch_build_cuda','nvcc_version'):assert k in identity,('missing identity field',k)
    assert identity['node_physical_identity']=='unconfirmed','cannot invent physical node identity'
    resources=r['native_resources'];assert set(resources)==KEYS
    for key,v in resources.items():
        assert len(v)==20 and all(type(x)is int and x>=0 for x in v),'expert resource shape/domain'
        for off in (0,10):
            a=v[off:off+10]
            assert a[4]>0 and a[4]<=a[5] and a[6]>=1
            assert a[7]==identity['sm_count'] and 0<a[7]<=136 and a[8]==1 and a[9]==a[6]*a[7],'expert admission'
    rows=r['actual_attention'];assert len(rows)==12 and all(isinstance(x,list)and len(x)==4 for x in rows)
    assert {(x[0],x[1])for x in rows}==CELLS,'attention cell coverage'
    for tokens,context,row,expert in rows:
        assert type(tokens)is int and len(row)==27 and all(type(x)is int for x in row),'attention row ABI'
        batch,length={1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[tokens]
        cfg=dict(batch=batch,q_len=length,window=2048 if context=='sliding' else 0,num_kv_heads=2)
        assert all(row[i]>0 for i in (11,12,13,14,16,19,21,22)) and all(row[i]>=0 for i in (15,17,18,20,26)),'attention resource domain'
        assert row[25] in {0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33,100},'unknown function tag'
        check_attention({'config':cfg},row)
        assert row[22]==identity['sm_count'],'inconsistent actual attention GPU'
        key='tiny' if tokens<=4 else 'tiny2' if tokens<=8 else 'mid' if tokens<=16 else 'b4' if tokens<=32 else 'main'
        assert expert==[tokens,key,True],'actual expert/backend coverage'
    return dict(status='complete_native',treatment_exercised=True,complete_native_startup=True,capsule=r,causal_RMW_effect=None)

def analyze(receipt,job,log,source,control=None,control_job='607eb52bdaf24160'):
    # Terminal success first: DONE is printed before the final prepare deadline.
    official_rows,summary=official_read(receipt,job)
    assert summary['prepare_seconds']<295,'qualification prepare deadline'
    q=qualification(log,source_contract(source))
    result=dict(official=summary,qualification=q,source_sha256=SOURCE_SHA,source_bytes=SOURCE_BYTES,paired=False,release_decision=False,causal_RMW_effect=None,scope='Single-arm official observation with runtime startup gates. Flags are source-asserted, not separately printed per-module flag vectors. Native exercise is necessary but not sufficient for causal attribution.')
    result['official_rows']=[official_rows[k] for k in sorted(official_rows)]
    result['weighted_six']=[dict(load=l,**{metric:sum((46 if c=='sliding' else 16)*official_rows[l,c][field] for c in ('sliding','full'))/62000 for metric,field in (('adjusted_us','median_ns'),('raw_us','median_raw_ns'),('baseline_adjusted_us','baseline_ns'),('baseline_raw_us','baseline_raw_ns'))}) for l in ('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32')]
    rr=receipt['result']
    result['terminal_fields']=dict(status=receipt['status'],mode=receipt['mode'],result_status=rr['status'],passed=rr['passed'],returncode=rr['returncode'],checks=rr['checks'],flags=rr['flags'],error=rr['error'],gpu=rr.get('gpu'),setup=[x for x in rr['records'] if x.get('event')=='setup'])
    if control is not None:result['cross_job_comparison']=compare(receipt,control,job,control_job,log=None)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--job',required=True);p.add_argument('--source',required=True,type=Path);p.add_argument('--log',required=True,type=Path);p.add_argument('--control',type=Path);p.add_argument('--control-job',default='607eb52bdaf24160');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    receipt=json.loads(a.receipt.read_text(encoding='utf8'));log,digest=owning_log(receipt,a.log)
    result=analyze(receipt,a.job,log,a.source,json.loads(a.control.read_text(encoding='utf8'))if a.control else None,a.control_job)
    result['owning_log_sha256']=digest
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(job=a.job,score=result['official']['official_score'],qualification=result['qualification']['status'],paired=False)))
