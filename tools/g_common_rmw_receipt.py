"""CPU-only owning reader for frozen979 common+RMW single-arm measurement.

No GPU/service access. A public point score is not a paired causal RMW effect.
"""
import argparse,ast,hashlib,json,math,re
from pathlib import Path
from hybrid_layer_measurement import read as official_read, owning_log, compare

SOURCE_SHA='97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7'
SOURCE_BYTES=914572
# Reviewed immutable source epochs only. Draft822 is deliberately excluded.
ALLOWED_SOURCES={SOURCE_SHA:dict(bytes=SOURCE_BYTES,reference_cache=False)}
ALLOWED_SOURCES['978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a']=dict(bytes=918978,reference_cache=True,prepare_phases=True)
ALLOWED_SOURCES['32196b8bc1c954e9b7f2c6667e6b3e0aeed14da0fed6fe53bc153c9d563e5047']=dict(bytes=918490,reference_cache=True,prepare_phases=True,arm='common',base_sha='4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280')
PHASE_COUNTS=dict(normal_prepare_begin=1,normal_prepare_end=1,clone_begin=1,
    clone_end_resources_begin=1,resources_end=1,cell_begin=12,base_front_begin=12,
    base_front_end_fresh_front_begin=12,fresh_front_end_input_clone_begin=12,
    input_allocate_begin=24,front_bf16_begin=24,front_fp8_begin=24,front_end=24,
    empty_begin=12,empty_end=12,validate_begin=72,validate_end=72,cell_end=12,
    cleanup_begin=1,cleanup_end=1)
BASE_SHA='2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd'
COUNTS=dict(cells=12,checks=72,captures=12,fresh_checked=36,empty_checks=60,empty_captures=12,restored=48)
LOADS=(1,8,16,32,128,256)
CELLS={(t,c) for t in LOADS for c in ('sliding','full')}
KEYS={'tiny','tiny2','mid','b4','main'}

def source_contract(path):
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest in ALLOWED_SOURCES,'wrong frozen source'
    contract=ALLOWED_SOURCES[digest]
    assert len(raw)==contract['bytes'],'wrong frozen source length'
    tree=ast.parse(raw.decode('utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gq_attn_record')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen CPU metadata predicate>','exec'),ns)
    check=ns['_gq_attn_record']
    check.source_sha256=digest;check.source_bytes=len(raw)
    check.requires_reference_cache=contract['reference_cache']
    check.requires_prepare_phases=contract.get('prepare_phases',False)
    check.arm=contract.get('arm','short-rmw');check.base_sha=contract.get('base_sha',BASE_SHA)
    return check

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
    arm=getattr(check_attention,'arm','short-rmw')
    assert r['arm']==arm and r['base_sha']==getattr(check_attention,'base_sha',BASE_SHA),'wrong fixed arm/base'
    for k,v in COUNTS.items():assert type(r[k])is int and r[k]==v,('coverage',k)
    assert r['native_expert'] is True and r['rmw_exercised'] is (arm=='short-rmw'),'wrong treatment exercise'
    if getattr(check_attention,'requires_reference_cache',False):
        cache=r.get('reference_cache')
        assert isinstance(cache,dict) and set(cache)=={'hit','miss','peak_entries'},'reference cache schema'
        assert all(type(v)is int for v in cache.values()),'reference cache integer stats'
        assert 0<=cache['hit']<=96 and 48<=cache['miss']<=144 and cache['hit']+cache['miss']==144,'reference cache coverage'
        assert cache['peak_entries']==4,'reference cache owner coverage'
    if getattr(check_attention,'requires_prepare_phases',False):
        expected=dict(PHASE_COUNTS,reference_back_miss=r['reference_cache']['miss'])
        phases=r.get('prepare_phases')
        assert isinstance(phases,dict) and set(phases)==set(expected),'prepare phase schema'
        assert all(type(value)is int for value in phases.values()) and phases==expected,'prepare phase coverage'
        seconds=r.get('prepare_phase_seconds');elapsed=r.get('prepare_phase_elapsed_seconds')
        assert isinstance(seconds,dict) and set(seconds)==set(expected),'prepare timing schema'
        assert all(type(value)in(int,float) and 0<=value<295 for value in seconds.values()),'prepare timing values'
        assert type(elapsed)in(int,float) and 0<elapsed<295,'prepare timing elapsed'
        assert math.isclose(sum(seconds.values()),elapsed,rel_tol=1e-8,abs_tol=1e-6),'prepare timing coverage'
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
    return dict(status='complete_native',treatment_exercised=r['rmw_exercised'],complete_native_startup=True,capsule=r,causal_RMW_effect=None)

def analyze(receipt,job,log,source,control=None,control_job='607eb52bdaf24160',mode='run',control_mode='run'):
    # Terminal success first: DONE is printed before the final prepare deadline.
    official_rows,summary=official_read(receipt,job,mode)
    assert summary['prepare_seconds']<295,'qualification prepare deadline'
    contract=source_contract(source)
    q=qualification(log,contract)
    if contract.requires_prepare_phases and q['status']=='complete_native':
        assert q['capsule']['prepare_phase_elapsed_seconds']<=summary['prepare_seconds']+1e-3,'phase elapsed exceeds owning setup'
    result=dict(official=summary,qualification=q,source_sha256=contract.source_sha256,source_bytes=contract.source_bytes,paired=False,release_decision=False,causal_RMW_effect=None,scope='Single-arm official observation with runtime startup gates. Flags are source-asserted, not separately printed per-module flag vectors. Native exercise is necessary but not sufficient for causal attribution.')
    result['official_rows']=[official_rows[k] for k in sorted(official_rows)]
    result['weighted_six']=[dict(load=l,**{metric:sum((46 if c=='sliding' else 16)*official_rows[l,c][field] for c in ('sliding','full'))/62000 for metric,field in (('adjusted_us','median_ns'),('raw_us','median_raw_ns'),('baseline_adjusted_us','baseline_ns'),('baseline_raw_us','baseline_raw_ns'))}) for l in ('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32')]
    rr=receipt['result']
    result['terminal_fields']=dict(status=receipt['status'],mode=receipt['mode'],result_status=rr['status'],passed=rr['passed'],returncode=rr['returncode'],checks=rr['checks'],flags=rr['flags'],error=rr['error'],gpu=rr.get('gpu'),setup=[x for x in rr['records'] if x.get('event')=='setup'])
    if control is not None:
        comparison=compare(receipt,control,job,control_job,log=None,mode=mode,control_mode=control_mode)
        # Generic helper retains the older owned-verify target labels.
        for row in comparison['rows']:row['target']=contract.arm=='short-rmw' and row['load'] in ('decode 1','verify 1')
        comparison['target_scope']='Tiny/Tiny2 short-RMW only: decode1/verify1 both contexts' if contract.arm=='short-rmw' else 'COMMON OFF observation; historical comparison does not isolate an individual repair'
        result['cross_job_comparison']=comparison
    return result

def read_receipt(path):
    envelope=json.loads(path.read_text(encoding='utf8'))
    if 'id' in envelope:return envelope
    assert isinstance(envelope.get('stdout'),str),'missing owning CLI stdout'
    receipt=json.loads(envelope['stdout'])
    assert isinstance(receipt,dict) and 'id' in receipt,'invalid owning CLI stdout'
    return receipt

def observe(receipt,job,log,source,control=None,control_job='607eb52bdaf24160',mode='run',control_mode='run'):
    if receipt.get('result',{}).get('passed') is not False:
        return analyze(receipt,job,log,source,control,control_job,mode,control_mode)
    contract=source_contract(source)
    assert receipt['id']==job and receipt['board']=='muse_decode_layer','wrong failure owner'
    assert mode in ('run','check') and receipt['mode']==mode and receipt['status']=='done','failure not terminal expected mode'
    assert re.fullmatch('[0-9a-f]{64}',receipt['content_hash']),'invalid failure content identity'
    assert receipt['payload_ref']=='muse_decode_layer/'+receipt['content_hash']+'.py','wrong failure payload'
    result=receipt['result'];records=result['records']
    assert result['status']=='ok' and type(result['returncode'])is int and result['returncode']!=0,'invalid judge failure'
    assert len(records)==1 and records[0].get('event')=='setup' and records[0].get('status')=='fail','unsupported failure scope'
    assert isinstance(records[0].get('error'),str) and records[0]['error'],'missing setup failure'
    assert result.get('speedup')is None and result.get('speedup_ci')is None,'failed setup cannot carry score'
    context=[line for line in log.splitlines() if line.startswith(('GQUAL_FAILURE_CONTEXT ','GQUAL_BUDGET_CONTEXT '))]
    return dict(official=dict(job=job,official_score=None,official_ci=None,tests=0,benchmarks=0,profiles=0,
            server_content_hash=receipt['content_hash'],payload_ref=receipt['payload_ref']),
        qualification=dict(status='setup_failed',complete_native_startup=False,treatment_exercised=None),
        source_sha256=contract.source_sha256,source_bytes=contract.source_bytes,
        terminal_fields=dict(status=receipt['status'],mode=receipt['mode'],result_status=result['status'],passed=False,
            returncode=result['returncode'],setup=records,checks=result.get('checks'),flags=result.get('flags'),
            error=result.get('error'),machine=result.get('machine'),evaluator=result.get('evaluator')),
        failure_context=context[-2:],paired=False,release_decision=False,causal_RMW_effect=None,
        scope='Owning setup failure only; no completed layer qualification or performance result. Source association remains separate from server content identity.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--job',required=True);p.add_argument('--source',required=True,type=Path);p.add_argument('--log',required=True,type=Path);p.add_argument('--control',type=Path);p.add_argument('--control-job',default='607eb52bdaf24160');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=('run','check'),default='run');p.add_argument('--control-mode',choices=('run','check'),default='run');a=p.parse_args()
    receipt=read_receipt(a.receipt);log,digest=owning_log(receipt,a.log)
    result=observe(receipt,a.job,log,a.source,read_receipt(a.control)if a.control else None,a.control_job,a.mode,a.control_mode)
    result['owning_log_sha256']=digest
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(job=a.job,score=result['official']['official_score'],qualification=result['qualification']['status'],paired=False)))
