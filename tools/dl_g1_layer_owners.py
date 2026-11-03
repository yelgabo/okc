"""Conservative private layer state clone: storage ownership, nested Lt, read-only inventory."""

def _wl_storage(t):
    s=t.untyped_storage();return (str(t.device),int(s.data_ptr()),int(s.nbytes()))

def _wl_tensors(x,seen=None):
    seen=set() if seen is None else seen
    if id(x) in seen:return []
    seen.add(id(x))
    if isinstance(x,torch.Tensor):return [x]
    if isinstance(x,dict):return [t for v in x.values() for t in _wl_tensors(v,seen)]
    if isinstance(x,(list,tuple)):return [t for v in x for t in _wl_tensors(v,seen)]
    if type(x).__name__=='_GraftLt':return _wl_tensors(x.workspace,seen)
    return []

def _wl_readonly(state,weights):
    # Only inventoried input/packed-weight and immutable lookup coefficient storage.
    keys=('ma_w','qkv','o','pr','unpool','unpool_slice','gate1','gate2','g_preffn','g_preexp',
          'g_postexp','g_postffn','g_next','bias','gs13','gs2','invf','invf0','router_fp32','graft')
    values=list(weights.values())+[state[k] for k in keys if k in state]
    values += [v for k,v in state.items() if isinstance(k,str) and k.startswith('graft_t_')]
    b=state.get('bfp') or {}
    values += [b[k] for k in ('raw','streams','tabs','base4','sk2','wtile','router_hi_t','router_lo_t','router_cat') if k in b]
    for plan in state.get('graft_lt',{}).values():values.append(plan['weight'])
    return {_wl_storage(t) for v in values for t in _wl_tensors(v)}

def _wl_clone(state,readonly,on):
    import types
    memo={};storage={}
    # TVM router module is compiled code only; its kernels receive all tensors explicitly.
    immutable_code_ids={id(state['router_fp32_ext'])} if state.get('router_fp32_ext') is not None else set()
    def clone(x):
        if id(x) in memo:return memo[id(x)]
        if id(x) in immutable_code_ids:return x
        if isinstance(x,torch.Tensor):
            key=_wl_storage(x)
            if key in readonly:return x
            if key not in storage:
                raw=torch.empty(0,device=x.device,dtype=torch.uint8).set_(x.untyped_storage(),0,(key[2],),(1,))
                storage[key]=raw.clone()
            y=torch.empty(0,device=x.device,dtype=x.dtype).set_(storage[key].untyped_storage(),x.storage_offset(),x.shape,x.stride())
            memo[id(x)]=y;return y
        if isinstance(x,dict):
            y={};memo[id(x)]=y
            for k,v in x.items():
                if k in ('layouts','algo') and 'lt' in x:y[k]=v # immutable Lt descriptors/algorithm
                else:y[k]=clone(v)
            return y
        if isinstance(x,list):
            y=[];memo[id(x)]=y;y.extend(clone(v) for v in x);return y
        if isinstance(x,tuple):
            y=tuple(clone(v) for v in x);memo[id(x)]=y;return y
        if type(x) is set:
            # Actual prepare owns compose_active and bfp.disabled sets. Their
            # members are immutable policy scalars/tuples, never tensor owners.
            # Exact types avoid opaque/custom hash/iterator side effects.
            active=set()
            def member(v):
                if type(v) in (str,int,float,bool,type(None)):return v
                if type(v) is tuple:
                    if id(v) in active:raise RuntimeError('cyclic policy set member')
                    active.add(id(v))
                    try:
                        for item in v:member(item)
                    finally:active.remove(id(v))
                    return v
                raise RuntimeError('unreviewed policy set member '+type(v).__name__)
            y=set();memo[id(x)]=y
            for item in x:y.add(member(item))
            return y
        if type(x).__name__=='_GraftLt':
            y=type(x).__new__(type(x));memo[id(x)]=y
            for name in ('lib','handle','desc','alpha','beta'):setattr(y,name,getattr(x,name))
            y.workspace=clone(x.workspace);return y
        if type(x).__name__=='_RouterExt' or isinstance(x,(types.ModuleType,types.FunctionType,types.BuiltinFunctionType)):return x
        if isinstance(x,(str,int,float,bool,type(None),torch.device,torch.dtype)):return x
        raise RuntimeError('unreviewed state owner '+type(x).__name__)
    arm=clone(state)
    arm['attention_scratch']={'rows':{},'cluster':{},'retained':[]}
    arm['ma_ext']=dict(state['ma_ext']);arm['ma_ext'].pop('main',None);arm['ma_ext'].pop('b4',None)
    arm['_wl_stage_on']=bool(on)
    arm['_wl_guard_storage']=[(storage[key],words) for full,words in _GUARDS if (key:=_wl_storage(full)) in storage]
    arm['_wl_cloned_bytes']=sum(t.numel() for t in storage.values())
    return arm

def _wl_intervals(state,readonly):
    return sorted(set(_wl_storage(t) for t in _wl_tensors(state) if _wl_storage(t) not in readonly))

def _wl_overlap(a,b):
    return any(da==db and pa<pb+nb and pb<pa+na for da,pa,na in a for db,pb,nb in b if na and nb)

def _wl_check_owners(arms,readonly):
    intervals=[_wl_intervals(a,readonly) for a in arms]
    for i,a in enumerate(arms):
        assert a['attention_scratch'] is not arms[(i+1)%len(arms)]['attention_scratch']
        for j in range(i):assert not _wl_overlap(intervals[i],intervals[j]),('mutable storage overlap',i,j)
        for plan in a.get('graft_lt',{}).values():
            assert _wl_storage(plan['lt'].workspace) not in readonly,'Lt workspace shared as weight'
    return [sum(n for _,_,n in v) for v in intervals]
