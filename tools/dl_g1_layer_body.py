"""CHECK ONLY. Native stage candidate judged; all paired arms own mutable backing storage."""
_WL_PREPARE=prepare
_WL_KERNEL=kernel_fn
_WL_NATIVE=_fork_native_moe
_WL_ATTN=_decode_attention7
_WL_MA_EXT=_ma_ext
_WL_EXT_FLAGS={}

def _ma_ext(defines=None):
    flags=tuple(MOE_DEFINES if defines is None else defines)
    ext=_WL_MA_EXT(defines)
    _WL_EXT_FLAGS[id(ext)]=flags
    return ext

_WL_SEEDS=(391031,391032,391033)
_WL_AFFECTED=(32,128,256)

def _fork_native_moe(M=None,state=None):
    ext=_WL_NATIVE(M,state)
    if state is not None and state.get('_wl_probe',False):
        assert state['_fork_native_eligible'] and state['ma_ok'] and not _pdl(M),'native/PDL drift'
        key='b4' if 16<M<=32 else 'main' if M>32 else 'small'
        flags=_WL_EXT_FLAGS[id(ext._module)]
        assert ('-DOKC_STAGE_RETIRE=1' in flags)==(bool(state['_wl_stage_on']) and key in ('main','b4')),'stage compiled flag drift'
        if key in ('main','b4'):assert ('-DOKC_B4FIN=1' in flags)==(key=='b4'),'compiled B4FIN drift'
        state['_wl_hits'].append((M,key,bool(state['_wl_stage_on'])))
        if key in ('main','b4'):assert ext is state['ma_ext'][key],'native module cache drift'
    return ext

def _decode_attention7(data,state,qkv):
    if state.get('_wl_probe',False):state['_wl_attn_hits']+=1
    return _WL_ATTN(data,state,qkv)

def _wl_run(data,state):
    state['_wl_probe']=True;state['_wl_hits']=[];state['_wl_attn_hits']=0
    try:out=_WL_KERNEL(data,state)
    finally:state['_wl_probe']=False
    t=data['config']['tokens']
    assert state['_wl_attn_hits']==1 and len(state['_wl_hits'])==1,('fallback path',t,state['_wl_hits'],state['_wl_attn_hits'])
    assert state['_wl_hits'][0][0]==t and (state['_wl_hits'][0][1]=='b4')==(t==32),('B4FIN dispatch drift',state['_wl_hits'])
    return out

def _wl_copy(dst,src):
    assert dst.keys()==src.keys()
    for key,v in src.items():
        if isinstance(v,torch.Tensor):
            assert dst[key].shape==v.shape and dst[key].dtype==v.dtype,('fresh shape',key)
            dst[key].copy_(v)
        else:assert dst[key]==v,('fresh static metadata',key)

def _wl_capture(data,state):
    saved=(data['stream'].clone(),data['attn_input'].clone())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1]);_wl_run(data,state)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=_wl_run(data,state)
    graph.replay();torch.cuda.synchronize()
    return graph,out

def _wl_refs(data,weights):
    front=_ab_ref_front(data,weights);front8=_ab_ref_front(data,weights,FP8)
    ids=front['biased'].topk(TOP_K,dim=-1).indices
    counts=torch.bincount((ids//LOCAL_EXPERTS).flatten(),minlength=8)
    return front,front8,counts

def _wl_check(base,data,out,weights,refs):
    assert len(out)==3
    ok,msg=_ab_validate(base,out,data,weights,*refs)
    assert ok,('WL_MATH',msg)
    assert all(bool(torch.isfinite(t).all()) for t in out),'WL_NONFINITE'
    return msg

def _wl_summary(rows,part):
    import math
    loads=sorted({r['tokens'] for r in rows})
    assert sorted(loads)==sorted(_WL_LOADS)
    decisions=[];summaries=[]
    for seed in _WL_SEEDS:
        group=[r for r in rows if r['seed']==seed];assert len(group)==2*len(loads),'missing paired context'
        gains=[0.]*4;raw_gains=[0.]*4;null=0.;regression=True
        for t in loads:
            pair={r['context']:r['times_us'] for r in group if r['tokens']==t};assert set(pair)=={'sliding','full'}
            cost=[46*pair['sliding'][i]+16*pair['full'][i] for i in range(4)]
            for c,v in ((0,2),(0,3),(1,2),(1,3)):
                regression &= all(pair[k][v]-pair[k][c]<=max(.5,.01*pair[k][c]) for k in pair)
            for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):raw_gains[k]+=math.log(cost[c]/cost[v])/6
            null+=max(abs(math.log(cost[0]/cost[1])),abs(math.log(cost[2]/cost[3])))/6
            if t in _WL_AFFECTED:
                for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):gains[k]+=math.log(cost[c]/cost[v])/6
        passed=min(gains)>max(math.log(1.002),2*null) and min(raw_gains)>max(0.,2*null) and regression
        summaries.append({'seed':seed,'sixload_log_contribution':gains,'raw_allrow_log_contribution':raw_gains,'duplicate_log_null_contribution':null,'regression_gate':regression,'partial_or_complete_gate':passed})
        decisions.append(passed)
    complete=part=='all'
    # Split parts NEVER independently assert a complete six-load gain. Combine
    # matching terminal reports with the offline merger first.
    return {'complete_sixload':complete,'gate':all(decisions) if complete else None,'seeds':summaries}

def _wl_weight_signature(weights):
    import hashlib
    h=hashlib.sha256()
    for name,t in sorted(weights.items()):
        assert isinstance(t,torch.Tensor) and t.is_contiguous(),'unreviewed weight object'
        raw=t.view(torch.uint8).reshape(-1);step=max(1,raw.numel()//128)
        h.update(repr((name,tuple(t.shape),str(t.dtype))).encode())
        h.update(bytes(raw[::step][:128].cpu().tolist()))
    return h.hexdigest()

def _wl_empty(arms):
    # Check the actual layer B4FIN and MAIN binaries, including empty metadata.
    checked=0
    for t in (32,128,256):
        for a in arms:
            dev=a['qkv'].device;x=torch.randn((t,4096),device=dev,dtype=torch.bfloat16)
            ids=torch.full((t,8),-1,device=dev,dtype=torch.int32)
            w=torch.full((t,8),.125,device=dev,dtype=torch.float32)
            out=torch.full_like(x,17)
            _fork_native_moe(t,a).okc_run(a['ma_w'],a['ma_scratch'],x,ids,w,out,0,32,a['nsm'])
            torch.cuda.synchronize()
            assert torch.count_nonzero(out)==0 and torch.isfinite(out).all(),'WL_EMPTY_NATIVE'
            checked+=1
    assert checked==12
    return checked

def _wl_judged_state(production, stage):
    # Keep the full preparation's plans and scratch, including routes outside
    # a partial diagnostic. Only the two immutable native modules change.
    for key in ('main','b4'):
        ext=stage['ma_ext'][key]
        flags=_WL_EXT_FLAGS[id(ext._module)]
        assert '-DOKC_STAGE_RETIRE=1' in flags and ('-DOKC_B4FIN=1' in flags)==(key=='b4')
        production['ma_ext'][key]=ext
    production['_wl_stage_on']=True
    return production

def _wl_prepare(weights):
    import statistics,random,json,zlib,base64
    assert _WL_CONFORMANCE is not None and _WL_CONFORMANCE['candidate_pin']==_WL_PIN,'G1b4 conformance receipt required before measurement'
    started=time.monotonic()
    def budget():assert time.monotonic()-started<300,'WL_INCOMPLETE_BEFORE_300S'
    state=_WL_PREPARE(weights);budget();weight_signature=_wl_weight_signature(weights)
    assert state['_fork_native_eligible'] and state['ma_sc6'] and state['ma_ok'] and PDL is False,'native SC6 conformance required first'
    assert state.get('bank') is None,'unused mutable Marlin bank unsupported'
    readonly=_wl_readonly(state,weights)
    mutable_bytes=sum(n for _,_,n in _wl_intervals(state,readonly))
    free,total=torch.cuda.mem_get_info()
    # Retain base plus4clones, graph tensors/caches and two FP32 reference banks.
    assert mutable_bytes*4<=8<<30 and free>mutable_bytes*4+(8<<30),('WL_MEMORY_PREFLIGHT',mutable_bytes,free,total)
    arms=[_wl_clone(state,readonly,i>=2) for i in range(4)]
    owned=_wl_check_owners([state,*arms],readonly)[1:];budget()
    resources={}
    for key,t in (('b4',32),('main',128)):
        rr=[]
        for i in (0,2):
            ext=_fork_native_moe(t,arms[i]);r=list(ext.wl_resources());budget()
            assert len(r)==12 and max(r[1],r[7])<=1024 and min(r[4],r[10])>=1,('WL_RESOURCE',key,r)
            rr.append(r)
        for off in (0,6):assert rr[0][off+2:off+6]==rr[1][off+2:off+6],('WL_RESOURCE_GEOMETRY',key,rr)
        resources[key]=rr
    for a in arms:
        for t in (32,128):_fork_native_moe(t,a)
    for key in ('main','b4'):
        assert arms[0]['ma_ext'][key]._module is arms[1]['ma_ext'][key]._module
        assert arms[2]['ma_ext'][key]._module is arms[3]['ma_ext'][key]._module
        assert arms[0]['ma_ext'][key]._module is not arms[2]['ma_ext'][key]._module
    empty_checks=_wl_empty(arms);budget()
    print('WL_PREP',_WL_PART,'mutable_bytes',mutable_bytes,'arm_owned',owned,'free',free,'two_flags_per_main_B4_dedup','resources',resources,flush=True)
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
    rows=[];math_checks=0;fresh_checks=0
    for tokens in _WL_LOADS:
        batch,length=({1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[tokens])
        for context in ('sliding','full'):
            for seed_idx,seed in enumerate(_WL_SEEDS):
                budget();base=_ab_inputs(batch,length,context,seed+tokens*1009+(context=='full')*100)
                front,front8,counts=_wl_refs(base,weights)
                rank=counts.argsort(descending=True);local_start=32*int(rank[(0,1,7)[seed_idx]])
                base['local_start'].fill_(local_start);front['local_start']=front8['local_start']=local_start
                refs=(front,front8)
                data=[{k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in base.items()} for _ in arms]
                intervals=[[_wl_storage(v) for v in d.values() if isinstance(v,torch.Tensor)] for d in data]
                assert not any(_wl_overlap(intervals[i],intervals[j]) for i in range(4) for j in range(i)),'input/cache storage overlap'
                order=list(range(4));random.Random(seed+tokens).shuffle(order);captures={}
                for i in order:captures[i]=_wl_capture(data[i],arms[i])
                _wl_check_owners(arms,readonly)
                outputs=[captures[i][1] for i in range(4)]
                for i in range(4):_wl_check(base,data[i],outputs[i],weights,refs);math_checks+=1
                out_intervals=[[_wl_storage(t) for t in out] for out in outputs]
                assert not any(_wl_overlap(out_intervals[i],out_intervals[j]) for i in range(4) for j in range(i)),'output overlap'
                fresh=_ab_inputs(batch,length,context,seed+tokens*1009+(context=='full')*100+17)
                fresh['local_start'].fill_(local_start);freshrefs=_wl_refs(fresh,weights)[:2]
                for i in range(4):_wl_copy(data[i],fresh)
                for _ in range(8):
                    for i in range(4):
                        data[i]['stream'].copy_(fresh['stream']);data[i]['attn_input'].copy_(fresh['attn_input']);captures[i][0].replay()
                torch.cuda.synchronize()
                for i in range(4):_wl_check(fresh,data[i],outputs[i],weights,freshrefs);fresh_checks+=1
                for i in range(4):_wl_copy(data[i],base);captures[i][0].replay()
                torch.cuda.synchronize()
                for i in range(4):_wl_check(base,data[i],outputs[i],weights,refs)
                samples=[[] for _ in arms]
                for rep in range(24):
                    order=list(range(4));random.Random(seed+tokens+rep*31).shuffle(order)
                    for i in order:
                        data[i]['stream'].copy_(base['stream']);data[i]['attn_input'].copy_(base['attn_input'])
                        flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                        a.record();captures[i][0].replay();z.record();z.synchronize();samples[i].append(a.elapsed_time(z)*1000)
                for i in range(4):_wl_check(base,data[i],outputs[i],weights,refs)
                rows.append({'tokens':tokens,'context':context,'seed':seed,'local_start':local_start,'reference_group_counts':counts.tolist(),
                    'times_us':[statistics.median(t) for t in samples],'actual_routes':[a['_wl_hits'] for a in arms]})
                budget()
    expected=4*len(_WL_LOADS)*2*3
    assert math_checks==fresh_checks==expected,'WL_INCOMPLETE_MATH_COVERAGE'
    _wl_check_owners(arms,readonly)
    for arm in arms:
        for raw,words in arm['_wl_guard_storage']:
            assert bool((raw.view(torch.int32)[words:]==SCRATCH_SENTINEL).all()),'WL_SCRATCH_GUARD'
    report={'schema':1,'part':_WL_PART,'candidate_pin':_WL_PIN,'g1b4_conformance':_WL_CONFORMANCE,'device_sha':_WL_DEVICE_SHA,'harness_core_sha':_WL_CORE_SHA,'weight_signature':weight_signature,'empty_native_checks':empty_checks,'seeds':_WL_SEEDS,'rows':rows,'resources':resources,
        'module_flags':{str(i)+':'+key:_WL_EXT_FLAGS[id(ext._module)] for i in (0,2) for key,ext in arms[i]['ma_ext'].items()},
        'math_checks':math_checks,'fresh_final_checks':fresh_checks,'fresh_replays_per_bank':8,'owners_bytes':owned,
        'judged_stage_on':True,'judged_official_tests':'external judge after prepare; require13/13 in owning receipt',
        'prepare_s':time.monotonic()-started,'decision':_wl_summary(rows,_WL_PART)}
    budget();blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':')).encode(),9)).decode()
    assert len(blob)<11000,'WL_FULL_REPORT_LOG_OVERFLOW'
    print('WL_STAGE_REPORT '+blob,flush=True)
    _wl_check_owners([state,*arms],readonly)
    _check_guards()
    return _wl_judged_state(state,arms[2])

def prepare(weights):
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        _AB_DEQ.clear()
        torch.backends.cuda.matmul.allow_tf32=False
        return _wl_prepare(weights)
    finally:
        _AB_DEQ.clear()
        torch.backends.cuda.matmul.allow_tf32=previous

def kernel_fn(data,state):
    return _WL_KERNEL(data,state)
