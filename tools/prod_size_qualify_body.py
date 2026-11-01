"""Single-arm startup qualification only; the existing timed entry is unchanged."""
BODY=r'''
_QUAL_NORMAL_PREPARE=prepare
_QUAL_BACK_ORIGINAL=_ab_ref_back

def _qual_budget(start):
    assert time.monotonic()-start<295,'QUAL_PREPARE_BUDGET'

def _qual_finite(value,label):
    assert bool(torch.isfinite(value.float()).all()),('QUAL_NONFINITE',label)
    assert bool(torch.isfinite(value.float().norm())),('QUAL_NONFINITE_NORM',label)

def _qual_ref_back(front,weights,ids):
    result=_QUAL_BACK_ORIGINAL(front,weights,ids)
    for i,value in enumerate(result):_qual_finite(value,'reference_back_'+str(i))
    return result

_ab_ref_back=_qual_ref_back

def _qual_storage(t):
    s=t.untyped_storage();return (str(t.device),s.data_ptr(),s.nbytes())

def _qual_overlap(a,b):
    return any(da==db and pa<pb+nb and pb<pa+na for da,pa,na in a for db,pb,nb in b if na and nb)

def _qual_owners(first,second,weights):
    def tensors(value,seen):
        if id(value) in seen:return []
        seen.add(id(value))
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,dict):items=value.values()
        elif isinstance(value,(tuple,list)):items=value
        elif isinstance(value,torch.nn.Module) or type(value).__module__.startswith('sglang'):items=vars(value).values()
        else:return []
        return [t for item in items for t in tensors(item,seen)]
    readonly={_qual_storage(t) for t in weights.values()}
    groups=[[_qual_storage(t) for t in tensors(state,set()) if _qual_storage(t) not in readonly] for state in (first,second)]
    assert not _qual_overlap(*groups),'QUAL_OWNED_STATE_ALIAS'
    workspace=[]
    for state in (first,second):
        assert set(state['dlplt'])=={('unpooler',128),('unpooler',256),('qkv_gate',256)},'QUAL_LT_COVERAGE'
        current=[]
        for fn in state['dlplt'].values():
            names=fn.__code__.co_varnames[1:fn.__code__.co_argcount]
            defaults=dict(zip(names,fn.__defaults__))
            assert isinstance(defaults['ws'],torch.Tensor),'QUAL_LT_WORKSPACE'
            current.append(_qual_storage(defaults['ws']))
        workspace.append(current)
    assert not _qual_overlap(*workspace),'QUAL_LT_ALIAS'

def _qual_inputs(batch,layer,seed,weights):
    data=_ab_inputs(batch,8,layer,seed)
    refs=(_ab_ref_front(data,weights),_ab_ref_front(data,weights,FP8))
    counts=torch.bincount((refs[0]['biased'].topk(8,dim=-1).indices//32).flatten(),minlength=8)
    local=32*int(counts.argmax());assert int(counts.max())>0,'QUAL_EMPTY_REFERENCE_ROUTES'
    data['local_start'].fill_(local)
    for front in refs:front['local_start']=local
    return data,refs

def _qual_restore(dst,src):
    for name,value in src.items():
        if isinstance(value,torch.Tensor):
            assert dst[name].shape==value.shape and dst[name].dtype==value.dtype,'QUAL_INPUT_ABI'
            dst[name].copy_(value)
        else:assert dst[name]==value,'QUAL_INPUT_METADATA'

def _qual_poison(data,output=None):
    if output is not None:
        for value in output:
            value.fill_(float('nan') if value.is_floating_point() else -777)
    rows=data['cache_loc'].long().reshape(-1)
    assert torch.unique(rows).numel()==rows.numel(),'QUAL_CACHE_LOC_ALIAS'
    for name in ('k_cache','v_cache'):
        data[name].view(torch.uint8).reshape(-1,2,64)[rows]=127

def _qual_validate(base,data,output,refs,weights):
    for i,value in enumerate(output[:2]):_qual_finite(value,'output_'+str(i))
    for front in refs:
        for name,value in front.items():
            if isinstance(value,torch.Tensor):_qual_finite(value,'front_'+name)
    ok,message=_qual_oracle(base,output,data,weights,*refs)
    assert ok,('QUAL_MODEL_ROUTER_CACHE',message)
    routes,distinct=_ab_local(output[2].long(),int(base['local_start']))
    assert routes>0 and distinct>0,'QUAL_NATIVE_EXPERT_WORK'

def _qual_diagnostics(state):
    assert int(torch.count_nonzero(state['verify_diag']))==0,'QUAL_WD_DIAGNOSTIC'
    assert int(state['moe1']['scratch']['ctr'][8])==0,'QUAL_EXPERT_DIAGNOSTIC'

def _qual_resources(state):
    owned=_VERIFY_HYBRID_OWNED
    moe=list(owned._moe1_ext().qual_moe_resources())
    one=list(owned._one_ext().qual_one_resources())
    assert len(moe)==10 and len(one)==20,'QUAL_RESOURCE_ABI'
    rows=[('moe51',moe),('one8_mode3',one[:10]),('one8_mode51',one[10:])]
    for name,r in rows:
        # regs,local,static,dynamic,threads,maxthreads,blocks/SM,SMs,coop,resident
        assert all(isinstance(v,int) and v>=0 for v in r),'QUAL_RESOURCE_VALUES'
        assert r[1]==0,'QUAL_RESOURCE_LOCAL'
        assert r[4]>0 and r[4]<=r[5] and r[6]>=1 and r[7]>0 and r[8]==1,'QUAL_RESOURCE_ADMISSION'
        assert r[9]==r[6]*r[7],'QUAL_RESOURCE_RESIDENT'
        assert r[3]<=227*1024,'QUAL_RESOURCE_SHARED'
    assert moe[4]==576 and moe[9]>=moe[7],'QUAL_MOE_GRID'
    assert state['one']['coop_launch'],'QUAL_ONE_COOPERATIVE_REQUIRED'
    assert 0<state['one']['resident'][8,3]<=one[9] and 0<state['one']['resident'][8,51]<=one[19],'QUAL_ONE_RESIDENCY'
    return rows

def _qual_empty_check(out,state,m):
    assert out.dtype==torch.bfloat16 and out.shape==(m,D_EXPERT),'QUAL_EMPTY_ABI'
    assert int(torch.count_nonzero(out))==0,'QUAL_EMPTY_OUTPUT'
    _qual_diagnostics(state)

def _qual_empty_capture(state,m,start):
    _qual_budget(start);device=state['verify_diag'].device
    gen=torch.Generator(device=device).manual_seed(970001+m)
    x=torch.randn((m,D_EXPERT),generator=gen,device=device).to(torch.bfloat16)
    ids=torch.arange(32,40,dtype=torch.int32,device=device).expand(m,8).contiguous()
    weights=torch.full((m,8),.125,dtype=torch.float32,device=device)
    local=torch.zeros(1,dtype=torch.int32,device=device)
    def run():return _VERIFY_HYBRID_OWNED._moe1_run(x,weights,ids,local,state,10,False)
    out=run();torch.cuda.synchronize();_qual_empty_check(out,state,m)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    graph._qual_inputs=(x,weights,ids,local)
    graph.replay();torch.cuda.synchronize();_qual_empty_check(out,state,m)
    _qual_budget(start)
    return graph,out

def _qual_prepare(weights,production,start):
    import copy
    _qual_budget(start)
    if 'owned' not in production:
        print('HYBRID_QUAL_DONE selected=0 fallback=unsupported_weights checks=0 captures=0 fresh=0',flush=True)
        _qual_budget(start);return production
    private_owned=_VERIFY_HYBRID_OWNED.prepare(weights)
    torch.cuda.synchronize();_qual_budget(start)
    _qual_owners(production['owned'],private_owned,weights)
    private={'G':production['G'],'owned':private_owned}
    resources=_qual_resources(private_owned)
    checks=0;captures=0;fresh_checks=0;covered=set();empty_checks=0;empty_captures=0;restored=0
    for batch,layer in ((16,'sliding'),(16,'full'),(32,'sliding'),(32,'full')):
        for seed in (938101,938123):
            _qual_budget(start)
            base,refs=_qual_inputs(batch,layer,seed,weights)
            fresh,fresh_refs=_qual_inputs(batch,layer,seed+101,weights)
            data=copy.deepcopy(base)
            assert _verify_override_bit(data)==(1<<((0 if batch==16 else 2)+(layer=='full'))),'QUAL_ACTUAL_DISPATCH'
            a=[_qual_storage(v) for v in data.values() if isinstance(v,torch.Tensor)]
            b=[_qual_storage(v) for v in base.values() if isinstance(v,torch.Tensor)]
            assert not _qual_overlap(a,b),'QUAL_INPUT_OWNER_ALIAS'
            empty_graph,empty_out=_qual_empty_capture(private_owned,batch*8,start)
            empty_checks+=2;empty_captures+=1
            warm=torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for i in range(2):
                    _qual_budget(start);_qual_restore(data,base);_qual_poison(data)
                    output=kernel_fn(data,private);torch.cuda.synchronize()
                    _qual_validate(base,data,output,refs,weights);_qual_diagnostics(private_owned);checks+=1
                    if i==0:restored+=1
            torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
            _qual_restore(data,base);_qual_poison(data)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=kernel_fn(data,private)
            graph._qual_inputs=data
            graph.replay();torch.cuda.synchronize()
            _qual_validate(base,data,output,refs,weights);_qual_diagnostics(private_owned);checks+=1;captures+=1
            for source,want in ((fresh,fresh_refs),(base,refs),(fresh,fresh_refs)):
                _qual_budget(start)
                empty_out.fill_(float('nan'));empty_graph.replay();torch.cuda.synchronize()
                _qual_empty_check(empty_out,private_owned,batch*8);empty_checks+=1
                for value in output:value.fill_(float('nan') if value.is_floating_point() else -777)
                # Restore after output poison, including any legitimate input/output alias.
                _qual_restore(data,source);_qual_poison(data)
                graph.replay();torch.cuda.synchronize()
                _qual_validate(source,data,output,want,weights);_qual_diagnostics(private_owned)
                checks+=1;fresh_checks+=1;restored+=1
            covered.add((batch,layer))
            del graph,output,data,base,fresh,refs,fresh_refs,source,want,empty_graph,empty_out,warm
            _qual_budget(start)
    assert len(covered)==4 and checks==48 and captures==8 and fresh_checks==24,'QUAL_INCOMPLETE'
    assert empty_checks==40 and empty_captures==8 and restored==32,'QUAL_EMPTY_RESTORE_COVERAGE'
    _qual_owners(production['owned'],private_owned,weights);_qual_diagnostics(private_owned)
    for name,vector in resources:print('HYBRID_QUAL_RESOURCE',name,vector,flush=True)
    _qual_budget(start)
    print('HYBRID_QUAL_DONE selected=4 fallback=0 checks=48 captures=8 fresh=24 empty_checks=40 empty_captures=8 restored=32',flush=True)
    _qual_budget(start)
    return production

def prepare(weights):
    start=time.monotonic()
    production=_QUAL_NORMAL_PREPARE(weights)
    _qual_budget(start)
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
        return _qual_prepare(weights,production,start)
    finally:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=old_tf32
'''
