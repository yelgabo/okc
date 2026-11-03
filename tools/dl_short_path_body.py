"""PRIVATE trace only. No synthetic IDs, PDL, fences or mutable production ownership changes."""
_SP_PREPARE=prepare
_SP_KERNEL=kernel_fn
_SP_POOL=_pool_router_fp32
_SP_NATIVE=_fork_native_moe
_SP_SEEDS=(401031,401032,401033)

def _pool_router_fp32(h,hr,T,state,ppick):
    if not state.get('_sp_active',False):return _SP_POOL(h,hr,T,state,ppick)
    assert T in (1,8) and h.dtype==torch.bfloat16 and hr.dtype==torch.float32 and ppick in ('bfp','bfpf'),'short producer fallback'
    if state['_sp_mode']=='fused':p,s=_SP_POOL(h,hr,T,state,ppick)
    else:p,s=_gle_prior_pool_router_fp32(h,hr,T,state,ppick)
    assert s==-1 and isinstance(p,tuple) and p[0].dtype==torch.bfloat16 and p[1].dtype==torch.float32,'precision/producer path drift'
    state['_sp_capture'].update(h=h,hr=hr,pooled=p[0],logits=p[1],ppick=ppick)
    return p,s

def _fork_native_moe(M=None,state=None):
    ext=_SP_NATIVE(M,state)
    if state is not None and state.get('_sp_active',False):
        assert M in (1,8) and state['_fork_native_eligible'] and not _pdl(M),'non-native short expert'
        state['_sp_native'].append(ext._module)
    return ext

def _sp_call(data,state):
    state['_sp_active']=True;state['_sp_capture']={};state['_sp_native']=[]
    try:out=_SP_KERNEL(data,state)
    finally:state['_sp_active']=False
    cap=state['_sp_capture']
    assert set(cap)>={'h','hr','pooled','logits','xm','ids','lids','wts','moe_backend'} and cap['moe_backend']=='agent','missing actual short route'
    assert len(state['_sp_native'])==1 and cap['xm'].dtype==torch.bfloat16 and cap['wts'].dtype==torch.float32
    assert cap['ids'].dtype==cap['lids'].dtype==torch.int32
    return out

def _sp_ns(ev):
    if hasattr(ev,'start_ns'):return ev.start_ns(),ev.end_ns()
    return round(ev.start_us()*1000),round((ev.start_us()+ev.duration_us())*1000)

def _sp_chain(events,mode):
    e=sorted(events);origin=min(x[0] for x in e)
    rid=[i for i,v in enumerate(e) if '_sl_tile_route' in v[3]]
    mid=[i for i,v in enumerate(e) if 'okc::moe_fused' in v[3]]
    bid=[i for i,v in enumerate(e) if 'glue_cluster_boundary' in v[3]]
    assert len(rid)==len(mid)==len(bid)==1,'selected short path missing'
    b,r,m=bid[0],rid[0],mid[0]
    assert b<r<m and m==r+1,'unexpected route/expert dependency'
    producers=e[b+1:r]
    assert len(producers)==(1 if mode=='fused' else 2),'producer coverage mismatch'
    if mode=='fused':assert 'gle' in producers[0][3],'not actual FRT'
    else:assert 'router_fp32_kernel' in producers[-1][3],'not serial FP32 router'
    assert len({v[2] for v in e[b:m+1]})==1,'unexpected multistream schedule'
    assert all(e[i][1]<=e[i+1][0] for i in range(b,m)),'ordinary pipeline overlap mismatch'
    return {'producer_kernels':[v[3] for v in producers],'producer_span_us':(producers[-1][1]-producers[0][0])/1000,
        'producer_to_route_gap_us':(e[r][0]-producers[-1][1])/1000,'route_us':(e[r][1]-e[r][0])/1000,
        'route_to_expert_gap_us':(e[m][0]-e[r][1])/1000,'expert_us':(e[m][1]-e[m][0])/1000,
        'PR_start_to_expert_end_us':(e[m][1]-producers[0][0])/1000,
        'graph_span_us':(max(v[1] for v in e)-origin)/1000,'activity_sum_not_saved_us':sum(v[1]-v[0] for v in e)/1000}

def _sp_graph(data,state):
    saved=(data['stream'].clone(),data['attn_input'].clone())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1]);_sp_call(data,state)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=_sp_call(data,state)
    g.replay();torch.cuda.synchronize()
    return g,out,saved

def _sp_transport(report):
    # Metadata only: keep full-precision median_us used by every scoring gate.
    # Preserve all rows, all events, all20samples at declared1ns print precision.
    import copy
    r=copy.deepcopy(report)
    r['transport']={'sample_time_precision_us':.001,'event_precision_us':.1,'score_medians':'unrounded','chain_intervals':'reconstruct_from_complete_events'}
    for row in r['rows']:
        row['times_us']=[round(v,3) for v in row['times_us']]
        trace=row['trace']
        for k,v in list(trace.items()):
            if isinstance(v,float):trace[k]=round(v,3)
        trace.pop('chain') # redundant; reconstruct every interval from retained events offline
    return r

def _sp_prepare(weights):
    import random,statistics
    from torch.profiler import profile,ProfilerActivity
    from torch.autograd import DeviceType
    started=time.monotonic()
    def budget():assert time.monotonic()-started<295,'SHORT_INCOMPLETE_BEFORE295'
    production=_SP_PREPARE(weights);budget()
    assert PDL is False and GLE_FRT and production['_fork_native_eligible'],'exact scored native path required'
    ro=_wl_readonly(production,weights)
    need=sum(n for _,_,n in _wl_intervals(production,ro));free,total=torch.cuda.mem_get_info()
    assert 4*need<=8<<30 and free>4*need+(8<<30),'short trace memory preflight'
    arms=[_wl_clone(production,ro,False) for _ in range(4)]
    for i,a in enumerate(arms):a['_sp_mode']='fused' if i<2 else 'split'
    _wl_check_owners(arms,ro)
    original_owner={k:id(v) for k,v in production.items()}
    original_storage=set(_wl_intervals(production,ro))
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
    report={'schema':1,'source_job':'607eb52bdaf24160','source_sha':'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab',
            'fixed_routing':'actual natural IDS/LIDS equal across all4 arms; no replacement',
            'schedule':'ordinary same-stream PDLFalse no fence changes','rows':[]}
    checks=0
    for length in (1,8):
        for context in ('sliding','full'):
            for seed in _SP_SEEDS:
                budget();base=_ab_inputs(1,length,context,seed+length*1009+(context=='full')*100)
                refs=(_ab_ref_front(base,weights),_ab_ref_front(base,weights,FP8))
                reference_ids=refs[0]['biased'].topk(TOP_K,dim=-1).indices
                group_counts=torch.bincount((reference_ids//LOCAL_EXPERTS).flatten(),minlength=8)
                local_start=LOCAL_EXPERTS*int(group_counts.argmax())
                base['local_start'].fill_(local_start)
                for front in refs:front['local_start']=local_start
                data=[{k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in base.items()} for _ in arms]
                for key in ('stream','attn_input','k_cache','v_cache'):assert len({_wl_storage(d[key]) for d in data})==4
                captured=[_sp_graph(d,a) for d,a in zip(data,arms)]
                _wl_check_owners(arms,ro)
                for i,(g,out,saved) in enumerate(captured):
                    ok,msg=_ab_validate(base,out,data[i],weights,*refs);assert ok,('SHORT_MATH',i,msg);checks+=1
                caps=[a['_sp_capture'] for a in arms]
                assert all(torch.equal(caps[0]['ids'],c['ids']) and torch.equal(caps[0]['lids'],c['lids']) for c in caps),'natural routing drift; cannot attribute expert timing'
                assert len({id(a['_sp_native'][0]) for a in arms})==1,'native expert binary drift'
                natural_ids=caps[0]['ids'].clone();natural_lids=caps[0]['lids'].clone()
                routes=caps[0]['lids'];assert int((routes>=0).sum())>0,'no occupied expert work in selected busiest range'
                route_counts={'local_start':local_start,'reference_group_slots':group_counts.tolist(),'local_slots':int((routes>=0).sum()),'local_experts':int(torch.unique(routes[routes>=0]).numel())}
                times=[[] for _ in arms]
                for repeat in range(20):
                    budget()
                    order=list(range(4));random.Random(seed+repeat).shuffle(order)
                    for i in order:
                        budget()
                        g,out,saved=captured[i];data[i]['stream'].copy_(saved[0]);data[i]['attn_input'].copy_(saved[1]);flush.zero_()
                        a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                        a.record();g.replay();z.record();z.synchronize();times[i].append(a.elapsed_time(z)*1000)
                        assert torch.equal(caps[i]['ids'],natural_ids) and torch.equal(caps[i]['lids'],natural_lids),'routing changed during timed replay'
                for i,(g,out,saved) in enumerate(captured):
                    traces=[]
                    for rep in range(3):
                        budget()
                        data[i]['stream'].copy_(saved[0]);data[i]['attn_input'].copy_(saved[1]);flush.zero_();torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CUDA]) as p:
                            g.replay();torch.cuda.synchronize()
                        events=[]
                        for ev in p.profiler.kineto_results.events():
                            if ev.device_type()==DeviceType.CUDA:
                                a,z=_sp_ns(ev);events.append((a,z,ev.device_resource_id(),ev.name()))
                        assert torch.equal(caps[i]['ids'],natural_ids) and torch.equal(caps[i]['lids'],natural_lids),'routing changed during traced replay'
                        trace=summarize(events);trace['chain']=_sp_chain(events,arms[i]['_sp_mode']);traces.append(trace)
                    ok,msg=_ab_validate(base,out,data[i],weights,*refs);assert ok,('SHORT_REPLAY_MATH',i,msg);checks+=1
                    report['rows'].append({'arm':('fused1','fused2','split1','split2')[i],'mode':arms[i]['_sp_mode'],
                        'length':length,'context':context,'seed':seed,'times_us':times[i],'median_us':statistics.median(times[i]),
                        'trace':sorted(traces,key=lambda t:t['span_us'])[1],'routing':route_counts,'correct':True,
                        'pool_choice':repr((arms[i].get('bfp') or {}).get('choice',{}).get(('pooler',length)))})
                del captured,data,caps,base,refs
                budget()
    assert checks==96 and len(report['rows'])==48,'short trace incomplete coverage'
    for arm in arms:
        for raw,words in arm['_wl_guard_storage']:
            assert bool((raw.view(torch.int32)[words:]==SCRATCH_SENTINEL).all()),'SHORT_SCRATCH_GUARD'
    _check_guards()
    assert original_owner=={k:id(v) for k,v in production.items()},'production state binding changed'
    assert original_storage==set(_wl_intervals(production,ro)),'production mutable owner changed'
    report['checks']=checks;report['prepare_s']=time.monotonic()-started
    budget()
    blob=encode_report(_sp_transport(report));budget()
    print('SHORT_PATH_TRACE '+blob,flush=True)
    for a in arms:a.pop('_sp_capture',None);a.pop('_sp_native',None);a['_sp_active']=False
    return production  # original all-shape preparation, never a private sliced owner

def prepare(weights):
    previous=torch.backends.cuda.matmul.allow_tf32
    _AB_DEQ.clear()
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _sp_prepare(weights)
    finally:
        _AB_DEQ.clear()
        torch.backends.cuda.matmul.allow_tf32=previous

def kernel_fn(data,state):return _SP_KERNEL(data,state)
