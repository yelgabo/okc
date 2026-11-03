"""Narrow runtime additions to the frozen four-placement RMW machinery."""
SUPPORT = r'''
_SRP_START=None
_SRP_PHASE='not_started'
_SRP_AT=0.0
_SRP_SECONDS={}
_SRP_CACHE=None
_SRP_CACHE_COUNTS={'hit':0,'miss':0,'peak_entries':0}
_SRP_REFBACK=_ab_ref_back
_SRP_ACTUAL=[]

def _srp_budget():
    assert time.monotonic()-_SRP_START<295,('SRP_INCOMPLETE_BEFORE_295S',_SRP_PHASE)

def _srp_phase(name):
    global _SRP_PHASE,_SRP_AT
    elapsed=time.monotonic()-_SRP_START
    _SRP_SECONDS[_SRP_PHASE]=_SRP_SECONDS.get(_SRP_PHASE,0.)+elapsed-_SRP_AT
    _SRP_PHASE=name;_SRP_AT=elapsed

def _srp_back(front,weights,ids):
    cache=_SRP_CACHE
    signature=_gqb_signature(front) if cache is not None else None
    entry=cache.get(id(front)) if cache is not None else None
    if entry is not None:
        assert entry['front'] is front and entry['weights'] is weights,'SRP_REFERENCE_OWNER'
        assert entry['ids']._version==entry['ids_version'],'SRP_IDS_MUTATED'
        assert tuple(t._version for t in entry['result'])==entry['versions'],'SRP_RESULT_MUTATED'
    same=(entry is not None and entry['signature']==signature and
          ids.shape==entry['ids'].shape and ids.dtype==entry['ids'].dtype and
          ids.device==entry['ids'].device and torch.equal(ids,entry['ids']))
    if same:
        result=entry['result'];_SRP_CACHE_COUNTS['hit']+=1
    else:
        _srp_budget();result=_SRP_REFBACK(front,weights,ids);_srp_budget()
        if cache is not None:
            snapshot=ids.clone();_SRP_CACHE_COUNTS['miss']+=1
            cache[id(front)]={'front':front,'weights':weights,'signature':signature,
                'ids':snapshot,'ids_version':snapshot._version,'result':result,
                'versions':tuple(t._version for t in result)}
            assert len(cache)<=4,'SRP_REFERENCE_CACHE_BOUND'
            _SRP_CACHE_COUNTS['peak_entries']=max(_SRP_CACHE_COUNTS['peak_entries'],len(cache))
    assert all(bool(torch.isfinite(t).all()) for t in result),'SRP_REFERENCE_NONFINITE'
    return result

def _srp_native_resources(arms,budget):
    rows={}
    for key,t in (('tiny',1),('tiny2',8),('mid',16),('b4',32),('main',128)):
        vectors=[]
        for arm in arms:
            ext=_fork_native_moe(t,arm);v=list(ext.qual_native_resources());budget()
            assert len(v)==20,'SRP_RESOURCE_ABI'
            for i in (0,10):
                r=v[i:i+10]
                assert all(x>=0 for x in r) and 0<r[4]<=r[5] and r[6]>=1,'SRP_RESOURCE'
                assert r[7]==arm['nsm'] and 0<r[7]<=136 and r[8]==1 and r[9]==r[6]*r[7],'SRP_ADMISSION'
            vectors.append(v)
        assert all(v==vectors[0] for v in vectors),'SRP_RESOURCE_PARITY'
        rows[key]=vectors
    return rows

def _srp_occupied_empty(arms):
    # One captured graph sees occupied -> empty -> occupied -> empty metadata.
    # Output poison is captured; scratch/counters are never manually reset.
    checked=0
    for t in (1,8):
        for arm in arms:
            dev=arm['qkv'].device
            x=torch.randn((t,4096),device=dev,dtype=torch.bfloat16)
            occupied_ids=torch.arange(8,device=dev,dtype=torch.int32).expand(t,8).contiguous()
            ids=occupied_ids.clone()
            weights=torch.full((t,8),.125,device=dev,dtype=torch.float32)
            out=torch.full_like(x,17);ext=_fork_native_moe(t,arm)
            def call():
                out.fill_(17)
                ext.okc_run(arm['ma_w'],arm['ma_scratch'],x,ids,weights,out,0,32,arm['nsm'])
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            for occupied in (True,False,True,False):
                _srp_budget();ids.copy_(occupied_ids) if occupied else ids.fill_(-1)
                graph.replay();torch.cuda.synchronize()
                assert bool(torch.isfinite(out).all()),'SRP_OCCUPIED_EMPTY_NONFINITE'
                if occupied:
                    assert int(torch.count_nonzero(out))>0,'SRP_OCCUPIED_NOT_EXERCISED'
                else:
                    assert int(torch.count_nonzero(out))==0,'SRP_OCCUPIED_TO_EMPTY'
                    checked+=1
    assert checked==16
    return checked

def _srp_prepare(weights):
    global _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back
    global _SRP_START,_SRP_AT,_SRP_PHASE,_SRP_CACHE
    _SRP_START=time.monotonic();_SRP_AT=0.;_SRP_PHASE='normal_prepare'
    _SRP_SECONDS.clear();_SRP_ACTUAL.clear();_SRP_CACHE=None
    _SRP_CACHE_COUNTS.update(hit=0,miss=0,peak_entries=0)
    saved=(_ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back)
    previous=torch.backends.cuda.matmul.allow_tf32
    guard_mark=None;guarded_ids=None;audit=None
    try:
        _ma_ext=_SRP_OBSERVE_BUILD
        production=_WL_PREPARE(weights);_srp_phase('private');_srp_budget()
        guard_mark=len(_GUARDS);guarded_ids=set(_GUARDED_IDS)
        audit=_att7();audit.qual_att7_end()
        _fork_native_moe=_SRP_OBSERVE_NATIVE;_decode_attention7=_SRP_OBSERVE_ATTN
        _ab_ref_back=_srp_back;_AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(False),torch.no_grad():
            result,report=_wl_prepare(weights,production)
        assert result is production and not result.get('_srm_on',False),'SRP_OFF_JUDGED_OWNER'
        _srp_budget()
    finally:
        _srp_phase('cleanup')
        if _SRP_CACHE is not None:_SRP_CACHE.clear()
        _SRP_CACHE=None
        _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back=saved
        if audit is not None:audit.qual_att7_end()
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=previous
        if guard_mark is not None:
            del _GUARDS[guard_mark:]
            _GUARDED_IDS.clear();_GUARDED_IDS.update(guarded_ids)
        _WL_EXT_FLAGS.clear();_srp_phase('complete')
    report.update(cleanup_restored=True,reference_cache=dict(_SRP_CACHE_COUNTS),
        prepare_phase_seconds=dict(_SRP_SECONDS),prepare_s=time.monotonic()-_SRP_START)
    import json,zlib,base64
    blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':')).encode(),9)).decode()
    assert len(blob)<11000,'SRP_REPORT_LOG_OVERFLOW'
    print('SRM_COMMON_REPORT '+blob,flush=True);_srp_budget()
    return production
'''
