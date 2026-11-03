"""One fixed model arm, prepare-only qualification. No timing or comparison."""
BODY=r'''
_GQ_NORMAL_PREPARE=prepare
_GQ_KERNEL=kernel_fn
_GQ_NATIVE_BUILD=_ma_ext
_GQ_ATTENTION=_decode_attention7
_GQ_MOE=_moe
_GQ_CURRENT=None
_GQ_MODULE_FLAGS={}
_GQ_REF_BACK=_ab_ref_back

def _gq_budget(start):
    assert time.monotonic()-start<295,'GQUAL_PREPARE_BUDGET'

def _gq_ref_back(front,weights,ids):
    result=_GQ_REF_BACK(front,weights,ids)
    for i,value in enumerate(result):_qual_finite(value,'g_reference_'+str(i))
    return result

_ab_ref_back=_gq_ref_back

def _gq_build(defines=None):
    flags=tuple(MOE_DEFINES if defines is None else defines)
    ext=_GQ_NATIVE_BUILD(defines)
    _GQ_MODULE_FLAGS[id(ext)]=flags
    return ext

def _gq_attention(data,state,qkv):
    out=_GQ_ATTENTION(data,state,qkv)
    if _GQ_CURRENT is not None:
        assert state is _GQ_CURRENT['state'],'GQUAL_ATTENTION_OWNER'
        _GQ_CURRENT['attention_success']+=1  # AFTER successful native return
    return out

def _gq_moe(M=None,state=None):
    ext=_GQ_MOE(M,state)
    if _GQ_CURRENT is not None:
        assert state is _GQ_CURRENT['state'],'GQUAL_EXPERT_OWNER'
        native=bool(state['_fork_native_eligible'])
        key='tiny' if M<=4 else 'tiny2' if M<=8 else 'mid' if M<=16 else 'b4' if M<=32 else 'main'
        if native:
            assert ext is state['ma_ext'][key],'GQUAL_EXPERT_CACHE'
            flags=_GQ_MODULE_FLAGS[id(ext._module)]
            expected=_GQ_ARM=='short-rmw' and key in ('tiny','tiny2')
            assert ('-DOKC_SHORT_FINAL_RELAXED=1' in flags)==expected,'GQUAL_EXPERT_POLICY'
            assert '-DOKC_FORK_GAMMA_EFFECTIVE=1' not in flags,'GQUAL_CLOSED_GAMMA'
            assert '-DOKC_STAGE_RETIRE=0' not in flags,'GQUAL_STAGE_DRIFT'
        _GQ_CURRENT['experts'].append((M,key,native))
    return ext

def _gq_attn_record(data,rows):
    assert len(rows)==27,'GQUAL_ATTENTION_ACTUAL_COVERAGE'
    r=[int(v) for v in rows];cfg=data['config']
    assert r[0]==1 and r[1]==cfg['q_len'] and r[2]==cfg['batch'],'GQUAL_ATTENTION_ACTUAL_CONFIG'
    assert all(r[i] in (0,1) for i in (3,4,5,23)) and r[10]==cfg['window'],'GQUAL_ATTENTION_FLAGS_WINDOW'
    assert r[11]==r[7] and r[12]==(1 if r[3] else cfg['batch']) and r[13]==cfg['num_kv_heads']//2,'GQUAL_ATTENTION_GRID'
    assert r[14]>0 and r[14]<=r[19] and 0<=r[15]<=r[20] and r[21]>0 and r[22]>0 and r[23]==1,'GQUAL_ATTENTION_RESOURCE'
    blocks=r[11]*r[12]*r[13]
    expected_wait=bool(r[4] and ((not r[3] and r[8]<=1) or (r[3] and r[1]==8)))
    assert (r[24]==1)==expected_wait,'GQUAL_ATTENTION_WAITING_LAUNCH'
    assert (r[24]==2)==(r[8]>1),'GQUAL_ATTENTION_CLUSTER_KIND'
    if r[24]==1:
        assert r[4] and blocks<=r[21]*r[22],'GQUAL_ATTENTION_COOPERATIVE'
    elif r[24]==2:
        assert r[8]>1 and r[26]>0,'GQUAL_ATTENTION_CLUSTER'
    else:assert r[24]==0,'GQUAL_ATTENTION_LAUNCH_KIND'
    assert r[25]>=0,'GQUAL_ATTENTION_KERNEL'
    return r

def _gq_identity():
    import os
    props=torch.cuda.get_device_properties(torch.cuda.current_device())
    versions=[int(v) for v in _att7().qual_att7_versions()]
    assert len(versions)==5 and min(versions[:2])>0,'GQUAL_RUNTIME_IDENTITY'
    return dict(gpu_name=getattr(props,'name',None),sm_count=props.multi_processor_count,
        capability=(props.major,props.minor),total_memory=getattr(props,'total_memory',None),
        device_uuid=str(getattr(props,'uuid','unreported')),
        node_environment=os.environ.get('HOSTNAME'),node_physical_identity='unconfirmed',
        torch_version=str(torch.__version__),torch_build_cuda=getattr(torch.version,'cuda',None),
        cuda_driver_version=versions[0],cuda_runtime_version=versions[1],
        nvcc_version=versions[2:] if versions[2] else None)

def _gq_run(data,state):
    global _GQ_CURRENT
    assert _GQ_CURRENT is None,'GQUAL_NESTED_MODEL'
    audit=_att7();audit.qual_att7_begin()
    current={'state':state,'attention_success':0,'experts':[]}
    _GQ_CURRENT=current
    try:
        out=_GQ_KERNEL(data,state)
    finally:
        _GQ_CURRENT=None
        rows=list(audit.qual_att7_end())  # disabled even when the model/fallback fails
    assert current['attention_success']==1,'GQUAL_CAUGHT_ATTENTION_FALLBACK'
    assert len(current['experts'])==1 and current['experts'][0][0]==data['config']['tokens'],'GQUAL_EXPERT_COVERAGE'
    record=_gq_attn_record(data,rows)
    return out,record,current['experts'][0]

def _gq_inputs(tokens,layer,seed,weights):
    batch,length={1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[tokens]
    data=_ab_inputs(batch,length,layer,seed)
    refs=(_ab_ref_front(data,weights),_ab_ref_front(data,weights,FP8))
    counts=torch.bincount((refs[0]['biased'].topk(8,dim=-1).indices//32).flatten(),minlength=8)
    local=32*int(counts.argmax());assert int(counts.max())>0,'GQUAL_NATURAL_ROUTING'
    data['local_start'].fill_(local)
    for front in refs:front['local_start']=local
    return data,refs

def _gq_resources(state):
    rows={}
    if not state['_fork_native_eligible']:
        assert state['_fork_fallback_kind']=='Marlin','GQUAL_UNREVIEWED_EXPERT_FALLBACK'
        return rows
    for key,tokens in (('tiny',1),('tiny2',8),('mid',16),('b4',32),('main',128)):
        ext=_fork_native_moe(tokens,state)
        torch.cuda.synchronize()  # execution errors cannot be an optional getter miss
        values=[int(v) for v in ext.qual_native_resources()]
        assert len(values)==20,'GQUAL_EXPERT_RESOURCE_ABI'
        for i in (0,10):
            r=values[i:i+10]
            assert all(v>=0 for v in r) and r[4]>0 and r[4]<=r[5] and r[6]>=1,'GQUAL_EXPERT_RESOURCE'
            assert r[7]==state['nsm'] and 0<r[7]<=136 and r[8]==1 and r[9]==r[6]*r[7],'GQUAL_EXPERT_ADMISSION'
        flags=_GQ_MODULE_FLAGS[id(ext._module)]
        assert ('-DOKC_SHORT_FINAL_RELAXED=1' in flags)==(_GQ_ARM=='short-rmw' and key in ('tiny','tiny2')),'GQUAL_RESOURCE_POLICY'
        rows[key]=values
    return rows

def _gq_empty(state,tokens,start):
    _gq_budget(start);dev=state['qkv'].device
    x=torch.randn((tokens,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.full((tokens,8),-1,device=dev,dtype=torch.int32)
    scores=torch.full((tokens,8),.125,device=dev,dtype=torch.float32)
    ext=_GQ_MOE(tokens,state);out=torch.empty_like(x)
    def call():
        out.fill_(17)  # INSIDE capture, so every replay must overwrite poison
        ext.okc_run(state['ma_w'],state['ma_scratch'],x,ids,scores,out,0,32,state['nsm'])
    call();torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'GQUAL_EMPTY_EAGER'
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    graph._gq_inputs=(x,ids,scores,out)
    graph.replay();torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'GQUAL_EMPTY_CAPTURE'
    return graph,out

def _gq_private(weights,production,start):
    import copy
    _gq_budget(start)
    readonly=_wl_readonly(production,weights)
    mutable=sum(n for _,_,n in _wl_intervals(production,readonly))
    free,total=torch.cuda.mem_get_info()
    assert mutable<=2<<30 and free>mutable+(8<<30),'GQUAL_PRIVATE_MEMORY'
    private=_wl_clone(production,readonly,False)
    _wl_check_owners([production,private],readonly)
    assert '_g_preexp_effective' in private and private['_g_preexp_effective'].dtype==torch.bfloat16,'GQUAL_PREEXPERT_FIELD'
    assert not _wl_overlap([_wl_storage(private['_g_preexp_effective'])],[_wl_storage(production['_g_preexp_effective'])]),'GQUAL_PREEXPERT_OWNER'
    resources=_gq_resources(private);_gq_budget(start)
    checks=captures=fresh_checks=empty_checks=empty_captures=restored=0
    covered=set();records=[]
    for tokens in (1,8,16,32,128,256):
        for layer in ('sliding','full'):
            _gq_budget(start);seed=949001+tokens*1009+(layer=='full')*100
            base,refs=_gq_inputs(tokens,layer,seed,weights)
            fresh,fresh_refs=_gq_inputs(tokens,layer,seed+101,weights)
            data=copy.deepcopy(base)
            assert not _wl_overlap([_wl_storage(v) for v in data.values() if isinstance(v,torch.Tensor)],
                [_wl_storage(v) for v in base.values() if isinstance(v,torch.Tensor)]),'GQUAL_INPUT_ALIAS'
            empty_graph,empty_out=_gq_empty(private,tokens,start);empty_checks+=2;empty_captures+=1
            warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for i in range(2):
                    _gq_budget(start);_qual_restore(data,base);_qual_poison(data)
                    output,record,expert=_gq_run(data,private);torch.cuda.synchronize()
                    _qual_validate(base,data,output,refs,weights);checks+=1
                    if i==0:restored+=1
            torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
            _qual_restore(data,base);_qual_poison(data)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output,record,expert=_gq_run(data,private)
            graph._gq_inputs=data
            graph.replay();torch.cuda.synchronize()
            _qual_validate(base,data,output,refs,weights);checks+=1;captures+=1
            for source,want in ((fresh,fresh_refs),(base,refs),(fresh,fresh_refs)):
                _gq_budget(start);empty_graph.replay();torch.cuda.synchronize()
                assert bool(torch.isfinite(empty_out).all()) and int(torch.count_nonzero(empty_out))==0,'GQUAL_EMPTY_REPLAY'
                empty_checks+=1
                _qual_poison(data,output);_qual_restore(data,source);_qual_poison(data)
                graph.replay();torch.cuda.synchronize()
                _qual_validate(source,data,output,want,weights)
                checks+=1;fresh_checks+=1;restored+=1
            covered.add((tokens,layer));records.append((tokens,layer,record,expert))
            _wl_check_owners([production,private],readonly)
            for raw,words in private['_wl_guard_storage']:
                assert bool((raw.view(torch.int32)[words:]==SCRATCH_SENTINEL).all()),'GQUAL_PRIVATE_GUARD'
            _check_guards();_gq_budget(start)
            del graph,empty_graph,output,empty_out,data,base,fresh,refs,fresh_refs,source,want,warm
    assert len(covered)==12 and (checks,captures,fresh_checks)==(72,12,36),'GQUAL_MODEL_COVERAGE'
    assert (empty_checks,empty_captures,restored)==(60,12,48),'GQUAL_EMPTY_COVERAGE'
    _wl_check_owners([production,private],readonly);_check_guards();_gq_budget(start)
    report={'arm':_GQ_ARM,'base_sha':_GQ_BASE_SHA,'identity':_gq_identity(),'cells':12,'checks':checks,'captures':captures,
        'fresh_checked':fresh_checks,'empty_checks':empty_checks,'empty_captures':empty_captures,
        'restored':restored,'native_expert':bool(private['_fork_native_eligible']),
        'rmw_exercised':bool(private['_fork_native_eligible'] and _GQ_ARM=='short-rmw'),
        'native_resources':resources,'actual_attention':records}
    private.clear()
    return report

def prepare(weights):
    global _ma_ext,_decode_attention7,_moe
    start=time.monotonic();prior_tf32=torch.backends.cuda.matmul.allow_tf32
    normal_build=_ma_ext;normal_attention=_decode_attention7;normal_moe=_moe
    guard_mark=None;guarded_ids=None;audit=None
    try:
        _ma_ext=_gq_build
        production=_GQ_NORMAL_PREPARE(weights);_gq_budget(start)
        # The generic clone does not own Marlin's cyclic adapter/bank. Preserve
        # the normal G56-checked fallback unchanged rather than fake coverage.
        if not production['_fork_native_eligible']:
            assert production['_fork_fallback_kind']=='Marlin','GQUAL_UNREVIEWED_FALLBACK'
            unexercised='native_expert_Marlin_fallback'
            report=None
        elif tuple(production['qkv'].shape)!=(QKV_ROWS,HIDDEN):
            unexercised='unsupported_weight_interface'
            report=None
        else:
            assert production['att_ok'],'GQUAL_ATTENTION_NOT_NATIVE'
            guard_mark=len(_GUARDS);guarded_ids=set(_GUARDED_IDS)
            audit=_att7();audit.qual_att7_end()  # force the prepare witness initially off
            _decode_attention7=_gq_attention;_moe=_gq_moe
            _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
            report=_gq_private(weights,production,start)
        _gq_budget(start)
    finally:
        _ma_ext=normal_build;_decode_attention7=normal_attention;_moe=normal_moe
        if audit is not None:audit.qual_att7_end()  # forcibly disabled before return/failure
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=prior_tf32
        if guard_mark is not None:
            del _GUARDS[guard_mark:]
            _GUARDED_IDS.clear();_GUARDED_IDS.update(guarded_ids)
    _gq_budget(start)
    if report is None:
        print('G_COMMON_QUAL_UNEXERCISED '+unexercised,'rmw_exercised=0',
              str(production.get('_fork_moe_status','backend_status_unreported')),flush=True)
        _gq_budget(start);return production
    import json
    encoded=json.dumps(report,separators=(',',':'))
    assert len(encoded)<11000,'GQUAL_COMPLETE_REPORT_SIZE'
    print('G_COMMON_QUAL_DONE '+encoded,flush=True)
    _gq_budget(start)
    return production
'''
