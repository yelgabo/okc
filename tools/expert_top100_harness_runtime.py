"""Embedded private runtime; imported as text by the offline generator only."""
BODY = r'''
_EH_PREPARE=prepare
_WL_KERNEL=kernel_fn
_EH_NATIVE=_fork_native_moe
_EH_BUILD=_ma_ext
_EH_START=None
_EH_BUILDS={}
_EH_CACHE=None
_EH_BACK=_ab_ref_back

def _eh_budget():
    assert time.monotonic()-_EH_START<295,'EXPERT_PREPARE_295S'

def _eh_build(defines=None):
    flags=tuple(MOE_DEFINES if defines is None else defines)
    ext=_EH_BUILD(defines)
    src=MOE_CUDA_SRC
    for flag in flags:
        if flag.startswith('@'):src=_MA_ALT_SRCS[flag[1:]]
    import hashlib
    _EH_BUILDS[id(ext)]=(flags,hashlib.sha256(src.encode()).hexdigest())
    return ext

class _ExpertCall:
    def __init__(self,ext,state,t,key):
        self.ext,self.state,self.t,self.key=ext,state,t,key
    def __getattr__(self,key):return getattr(self.ext,key)
    def okc_run(self,*args):
        result=self.ext.okc_run(*args)
        self.state['_wl_hits'].append((self.t,self.key,self.state['_expert_pair_arm']))
        self.state['_eh_args']=args
        return result

def _eh_native(M=None,state=None):
    ext=_EH_NATIVE(M,state)
    if state is None or not state.get('_wl_probe',False):return ext
    key=route_key(M)
    assert state['_fork_native_eligible'] and state['ma_ok'] and not _pdl(M),'EXPERT_FALLBACK'
    assert ext is state['ma_ext'][key],'EXPERT_CACHE_DRIFT'
    flags,digest=_EH_BUILDS[id(ext._module)]
    expected=_EH_SELECTED[str(int(state['_expert_pair_arm']))+':'+key]
    assert list(flags)==expected['flags'] and digest==expected['sha256'],'EXPERT_SELECTED_SOURCE'
    return _ExpertCall(ext,state,M,key)

def _wl_run(data,state):
    state['_wl_probe']=True;state['_wl_hits']=[]
    try:out=_WL_KERNEL(data,state)
    finally:state['_wl_probe']=False
    t=data['config']['tokens']
    assert state['_wl_hits']==[(t,route_key(t),state['_expert_pair_arm'])],'EXPERT_ACTUAL_CALL'
    return out

def _eh_back(front,weights,ids):
    # Reuse only exact IDs, immutable front versions and the identical weights owner.
    signature=_gqb_signature(front)
    entry=_EH_CACHE.get(id(front))
    if entry is not None:
        assert entry['front'] is front and entry['weights'] is weights,'EXPERT_REFERENCE_OWNER'
        assert entry['ids']._version==entry['ids_version'],'EXPERT_REFERENCE_IDS_STALE'
        assert tuple(t._version for t in entry['result'])==entry['versions'],'EXPERT_REFERENCE_RESULT_STALE'
    same=(entry is not None and entry['signature']==signature and ids.shape==entry['ids'].shape
          and ids.dtype==entry['ids'].dtype and ids.device==entry['ids'].device and torch.equal(ids,entry['ids']))
    if same:return entry['result']
    _eh_budget();result=_EH_BACK(front,weights,ids);_eh_budget()
    assert all(bool(torch.isfinite(t).all()) for t in result),'EXPERT_REFERENCE_NONFINITE'
    snapshot=ids.clone()
    _EH_CACHE[id(front)]={'front':front,'weights':weights,'signature':signature,
        'ids':snapshot,'ids_version':snapshot._version,'result':result,'versions':tuple(t._version for t in result)}
    assert len(_EH_CACHE)<=4,'EXPERT_REFERENCE_CACHE_BOUND'
    return result

def _wl_check(base,data,out,weights,refs):
    assert len(out)==3 and out[0].dtype==torch.bfloat16 and out[1].dtype==torch.float32,'EXPERT_OUTPUT_BOUNDARY'
    assert out[2].dtype in (torch.int32,torch.int64),'EXPERT_IDS_DTYPE'
    for front in refs:
        for value in front.values():
            if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all()),'EXPERT_FRONT_NONFINITE'
    ok,msg=_qual_oracle(base,out,data,weights,*refs)
    assert ok,('EXPERT_ORACLE',msg)
    assert all(bool(torch.isfinite(t).all()) for t in out),'EXPERT_OUTPUT_NONFINITE'
    return msg

def _eh_routes(state,out,start):
    args=state['_eh_args'];x,ids,wts,y=args[2:6]
    assert x.dtype==y.dtype==torch.bfloat16 and wts.dtype==torch.float32,'EXPERT_NATIVE_BOUNDARY'
    global_ids=out[2].long()
    local=(global_ids-start)%256
    want=torch.where(local<32,local,-1).to(ids.dtype)
    assert torch.equal(ids,want),'EXPERT_ACTUAL_LOCAL_IDS'
    sorted_ids=global_ids.sort(-1).values
    assert bool((sorted_ids[:,1:]!=sorted_ids[:,:-1]).all()),'EXPERT_DUPLICATE_IDS'
    assert bool(torch.isfinite(wts).all()) and bool((wts>=0).all()),'EXPERT_ROUTE_WEIGHTS'

def _eh_guards(arms):
    for arm in arms:
        for raw,words in arm['_wl_guard_storage']:
            assert bool((raw.view(torch.int32)[words:]==SCRATCH_SENTINEL).all()),'EXPERT_SCRATCH_GUARD'
    _check_guards()

def _eh_transitions(arms,weights):
    # First-empty and every occupied/empty replay are inspected. Distinct 0..7
    # and regular/tail IDs preserve the per-expert capacity invariant.
    count=0
    for t in (32,128,256):
        dev=arms[0]['qkv'].device
        gen=torch.Generator(device=dev).manual_seed(1102031+t)
        banks=torch.randn((4,t,4096),device=dev,dtype=torch.bfloat16,generator=gen)
        scores=torch.randn((4,t,8),device=dev,dtype=torch.float32,generator=gen).softmax(-1)
        routes=_release_routes(torch,t,dev)
        for arm in arms:
            empty=torch.full_like(banks[3],float('nan'))
            _EH_NATIVE(t,arm).okc_run(arm['ma_w'],arm['ma_scratch'],banks[3],routes[3],scores[3],empty,0,32,arm['nsm'])
            torch.cuda.synchronize()
            assert bool(torch.isfinite(empty).all()) and int(torch.count_nonzero(empty))==0,'EXPERT_FIRST_EMPTY'
        # Compare candidate explicitly against separately owned exact control
        # in addition to each arm's complete independent layer oracle checks.
        expected=[]
        for k in range(4):
            y=torch.empty_like(banks[k])
            _EH_NATIVE(t,arms[0]).okc_run(arms[0]['ma_w'],arms[0]['ma_scratch'],banks[k],routes[k],scores[k],y,0,32,arms[0]['nsm'])
            expected.append(y)
        torch.cuda.synchronize()
        for arm in arms:
            x=banks[3].clone();ids=routes[3].clone();tw=scores[3].clone();y=torch.empty_like(x)
            ext=_EH_NATIVE(t,arm)
            def call():
                y.fill_(float('nan'))
                ext.okc_run(arm['ma_w'],arm['ma_scratch'],x,ids,tw,y,0,32,arm['nsm'])
            call();torch.cuda.synchronize()
            assert bool(torch.isfinite(y).all()) and int(torch.count_nonzero(y))==0,'EXPERT_FIRST_EMPTY'
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            for k in (3,1,2,0,2,3,0,1):
                _eh_budget();x.copy_(banks[k]);ids.copy_(routes[k]);tw.copy_(scores[k])
                graph.replay();torch.cuda.synchronize()
                _release_verify(torch,y,expected[k])
                if k==3:assert int(torch.count_nonzero(y))==0,'EXPERT_EMPTY_REPLAY'
                count+=1;_eh_guards(arms)
            del graph
    assert count==96
    return count

def _eh_component(state,t):
    # Complete native expert endpoint: packing in ma_w is shared immutable
    # preparation; routing/activation producers remain in whole-layer timing.
    # Preserve their natural output values, not a synthetic routing replacement.
    args=list(state['_eh_args']);expected=args[5].clone()
    for index in (2,3,4,5):args[index]=args[index].clone()
    ext=_EH_NATIVE(t,state)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):ext.okc_run(*args)
    graph.replay();torch.cuda.synchronize()
    _release_verify(torch,args[5],expected)
    return graph,args,expected

def _eh_readonly_inventory(state,weights):
    readonly=_wl_readonly(state,weights)
    # Report every shared storage, its dimensions and the reviewed owning paths.
    records={}
    def walk(value,path,seen):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
            key=_wl_storage(value)
            if key in readonly:records.setdefault(key,[]).append([path,list(value.shape),str(value.dtype)])
        elif isinstance(value,dict):
            for k,v in value.items():walk(v,path+'/'+str(k),seen)
        elif isinstance(value,(tuple,list)):
            for k,v in enumerate(value):walk(v,path+'/'+str(k),seen)
    walk(weights,'weights',set());walk(state,'state',set())
    assert set(records)==readonly,'EXPERT_READONLY_INVENTORY'
    return readonly,[{'device':k[0],'bytes':k[2],'paths':v} for k,v in sorted(records.items())]

def _eh_private(weights,state):
    global _EH_CACHE
    import random,statistics
    _eh_budget()
    assert state['_fork_native_eligible'] and state['ma_sc6'] and state['ma_ok'] and PDL is False,'EXPERT_NATIVE_REQUIRED'
    assert state.get('bank') is None,'EXPERT_UNREVIEWED_BANK'
    readonly,inventory=_eh_readonly_inventory(state,weights)
    mutable=sum(n for _,_,n in _wl_intervals(state,readonly))
    free,total=torch.cuda.mem_get_info()
    assert mutable*4<=8<<30 and free>mutable*4+(8<<30),'EXPERT_MEMORY_PREFLIGHT'
    arms=[_wl_clone(state,readonly,i>=2) for i in range(4)]
    for i,arm in enumerate(arms):
        _expert_pair_bind(arm,i>=2)
        for t in LOADS:_EH_NATIVE(t,arm)
    owners=_wl_check_owners([state,*arms],readonly)[1:]
    for key in ('b4','main'):
        assert arms[0]['ma_ext'][key]._module is arms[1]['ma_ext'][key]._module
        assert arms[2]['ma_ext'][key]._module is arms[3]['ma_ext'][key]._module
        assert arms[0]['ma_ext'][key]._module is not arms[2]['ma_ext'][key]._module
    transitions=_eh_transitions(arms,weights)
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
    rows=[];checks=0;fresh_checks=0;restores=0
    for tokens in LOADS:
        batch,length={1:(1,1),8:(1,8),16:(16,1),32:(32,1),128:(16,8),256:(32,8)}[tokens]
        for context in ('sliding','full'):
            for seed_idx,seed in enumerate(BANKS):
                _eh_budget();_EH_CACHE={}
                base=_ab_inputs(batch,length,context,seed+tokens*1009+(context=='full')*100)
                front,front8,counts=_wl_refs(base,weights)
                rank=counts.argsort(descending=True);start=32*int(rank[(0,1,7)[seed_idx]])
                base['local_start'].fill_(start);front['local_start']=front8['local_start']=start
                refs=(front,front8)
                data=[{k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in base.items()} for _ in arms]
                intervals=[[_wl_storage(v) for v in d.values() if isinstance(v,torch.Tensor)] for d in data]
                assert not any(_wl_overlap(intervals[i],intervals[j]) for i in range(4) for j in range(i)),'EXPERT_INPUT_ALIAS'
                order=list(range(4));random.Random(seed+tokens).shuffle(order);captures={}
                for i in order:captures[i]=_wl_capture(data[i],arms[i])
                outputs=[captures[i][1] for i in range(4)]
                for i in range(4):
                    _wl_check(base,data[i],outputs[i],weights,refs);_eh_routes(arms[i],outputs[i],start);checks+=1
                assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]),'EXPERT_ROUTING_ARM_MISMATCH'
                out_intervals=[[_wl_storage(t) for t in out] for out in outputs]
                assert not any(_wl_overlap(out_intervals[i],out_intervals[j]) for i in range(4) for j in range(i)),'EXPERT_OUTPUT_ALIAS'
                fresh=_ab_inputs(batch,length,context,seed+tokens*1009+(context=='full')*100+17)
                fresh['local_start'].fill_(start);freshrefs=_wl_refs(fresh,weights)[:2]
                for i in range(4):_wl_copy(data[i],fresh)
                for rep in range(8):
                    for i in range(4):
                        _eh_budget()
                        data[i]['stream'].copy_(fresh['stream']);data[i]['attn_input'].copy_(fresh['attn_input'])
                        captures[i][0].replay();torch.cuda.synchronize()
                        _wl_check(fresh,data[i],outputs[i],weights,freshrefs)
                        _eh_routes(arms[i],outputs[i],start);fresh_checks+=1
                    assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]),'EXPERT_FRESH_ROUTING_MISMATCH'
                for i in range(4):_wl_copy(data[i],base);captures[i][0].replay()
                torch.cuda.synchronize()
                for i in range(4):_wl_check(base,data[i],outputs[i],weights,refs);restores+=1
                components=[_eh_component(arm,tokens) for arm in arms]
                samples=[[[] for _ in arms] for endpoint in range(2)]
                for endpoint in range(2):
                    for rep in range(24):
                        order=list(range(4));random.Random(seed+tokens+rep*31+endpoint*100003).shuffle(order)
                        for i in order:
                            _eh_budget()
                            data[i]['stream'].copy_(base['stream']);data[i]['attn_input'].copy_(base['attn_input'])
                            flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                            graph=captures[i][0] if endpoint==0 else components[i][0]
                            a.record();graph.replay();z.record();z.synchronize()
                            samples[endpoint][i].append(positive(a.elapsed_time(z)*1000))
                for graph,args,expected in components:_release_verify(torch,args[5],expected)
                for i in range(4):_wl_check(base,data[i],outputs[i],weights,refs)
                _wl_check_owners([state,*arms],readonly);_eh_guards(arms)
                rows.append(dict(tokens=tokens,context=context,seed=seed,local_start=start,reference_group_counts=counts.tolist(),
                    times_us=[statistics.median(v) for v in samples[0]],
                    component_us=[statistics.median(v) for v in samples[1]],actual_routes=[a['_wl_hits'] for a in arms]))
                _EH_CACHE.clear();_EH_CACHE=None
                # Release graph-owned inputs before allocating the next full KV bank.
                for arm in arms:arm.pop('_eh_args',None)
                del graph,args,expected,components,captures,outputs,data,base,fresh,refs,freshrefs,front,front8
    assert checks==restores==144 and fresh_checks==1152,'EXPERT_COVERAGE'
    return dict(schema=1,pins=_EH_PINS,rows=rows,math_checks=checks,fresh_checks=fresh_checks,restores=restores,
        transitions=transitions,owners_bytes=owners,readonly_inventory=inventory,
        weight_signature=_wl_weight_signature(weights),decision=decision(rows),official_arm='exact607')

def prepare(weights):
    global _EH_START,_EH_CACHE,_ma_ext,_fork_native_moe,_ab_ref_back
    import json,zlib,base64
    if _EH_OFFLINE_HOLD:raise RuntimeError('EXPERT_HARNESS_STOP_CPU_HOST_GATE')
    _EH_START=time.monotonic();_EH_CACHE=None;_EH_BUILDS.clear()
    saved=(_ma_ext,_fork_native_moe,_ab_ref_back)
    tf32=torch.backends.cuda.matmul.allow_tf32
    guard_mark=None;guard_ids=None
    try:
        _ma_ext=_eh_build
        state=_EH_PREPARE(weights);_eh_budget()
        assert not state.get('_expert_pair_arm',False),'EXPERT_OFFICIAL_ARM'
        guard_mark=len(_GUARDS);guard_ids=set(_GUARDED_IDS)
        _fork_native_moe=_eh_native;_ab_ref_back=_eh_back
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(False),torch.no_grad():report=_eh_private(weights,state)
        _eh_budget()
    finally:
        if _EH_CACHE is not None:_EH_CACHE.clear()
        _EH_CACHE=None;_AB_DEQ.clear()
        _ma_ext,_fork_native_moe,_ab_ref_back=saved
        torch.backends.cuda.matmul.allow_tf32=tf32
        if guard_mark is not None:
            del _GUARDS[guard_mark:];_GUARDED_IDS.clear();_GUARDED_IDS.update(guard_ids)
        _EH_BUILDS.clear()
    report.update(cleanup_restored=True,prepare_s=time.monotonic()-_EH_START)
    blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    assert len(blob)<11000,'EXPERT_REPORT_LOG_OVERFLOW'
    _eh_budget();print('EXPERT_TOP100_REPORT '+blob,flush=True)
    return state
'''
