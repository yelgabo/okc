"""PRIVATE CHECK ONLY: repaired SC6 identical packs, ordinary dispatch stage-x pair."""
_ST_PREPARE=prepare
_ST_KERNEL=kernel_fn
_ST_AFFECTED=(32,128,256)
_ST_SEEDS=(291031,291032,291033)
_ST_ORIG_EXT=_ext
_ST_OBSERVE=False
_ST_SEEN=[]

def _ext(defines=None):
    if _ST_OBSERVE:_ST_SEEN.append(tuple(MAIN_DEFINES if defines is None else defines))
    return _ST_ORIG_EXT(defines)

def _st_call(data,arm):
    global _ST_OBSERVE,_ST_SEEN
    _ST_SEEN=[];_ST_OBSERVE=True
    try:out=_ST_KERNEL(data,arm)
    finally:_ST_OBSERVE=False
    expected=_st_defs(arm,data['hidden_states'].shape[0])[1]
    assert _ST_SEEN==[expected],('actual backend drift',_ST_SEEN,expected)
    return out


def _st_errors(out,ref):
    a,b=out.float(),ref.float();delta=a-b
    assert bool(torch.isfinite(a).all()),'ST_NONFINITE'
    ge=float(delta.norm()/b.norm().clamp_min(1e-12));rn=b.norm(dim=-1);floor=(.1*rn.mean()).clamp_min(1e-12)
    re=float((delta.norm(dim=-1)/torch.maximum(rn,floor)).max())
    assert ge<=.015 and re<=.075,('ST_MATH',ge,re)
    return ge,re

def _st_defs(arm,m):
    sfx=arm['sfx']
    if TINY_DEFINES and m<=TINY_MAX_M:return 'tiny',tuple(TINY_DEFINES)+arm['sfxs']
    if TINY2_DEFINES and m<=TINY2_MAX_M and (sfx or '-DOKC_WGS=1' not in tuple(TINY2_DEFINES)):return 'tiny2',tuple(TINY2_DEFINES)+arm['sfxs']
    if MID_DEFINES and m<=MID_MAX_M:return 'mid',tuple(MID_DEFINES)+arm['sfxs']
    return 'main',tuple(_main_defines(sfx))+arm['sfxm']

def _st_summary(rows):
    import math
    verdicts=[]
    assert len(rows)==18,'all six loads x three seeds required'
    for seed in _ST_SEEDS:
        group=[r for r in rows if r[1]==seed];assert sorted(r[0] for r in group)==[1,8,16,32,128,256]
        gains=[0.]*4;null=0.;regress=True
        transfer=[0.]*4
        for m,_,t in group:
            if m not in _ST_AFFECTED:continue
            for k,(c,v) in enumerate(((0,2),(0,3),(1,2),(1,3))):
                gains[k]+=math.log(t[c]/t[v])/6
                if m!=32:transfer[k]+=math.log(t[c]/t[v])/6
                regress &= t[v]-t[c]<=max(.5,.01*t[c])
            null+=max(abs(math.log(t[0]/t[1])),abs(math.log(t[2]/t[3])))/6
        passed=min(gains)>=math.log(1.01) and min(gains)>2*null and regress
        verdicts.append(passed)
        print('ST_SCORE',seed,'fourcross_affected_only',[math.exp(x) for x in gains],'null_log',null,
              'regression_gate',regress,'gate',passed,'exclude_nontransfer_B4FIN32',[math.exp(x) for x in transfer],flush=True)
    print('ST_DECISION',all(verdicts),'all3seeds_required','component_only_no_layer_score_claim',flush=True)
    return all(verdicts)

def _st_private(data):
    return {k:v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v for k,v in data.items()}

def _st_copy(dst,src):
    for k in ('hidden_states','topk_ids','topk_weights'):dst[k].copy_(src[k])

def _st_scratch_ptrs(x):
    if isinstance(x,torch.Tensor):return {x.data_ptr()} if x.numel() else set()
    if isinstance(x,dict):return set().union(*(_st_scratch_ptrs(v) for v in x.values())) if x else set()
    if isinstance(x,(tuple,list)):return set().union(*(_st_scratch_ptrs(v) for v in x)) if x else set()
    return set()

def _st_run(weights):
    import time,random,statistics
    began=time.monotonic()
    def budget():assert time.monotonic()-began<300,'ST_PREPARE_BUDGET_INCOMPLETE'
    state=_ST_PREPARE(weights);budget()
    assert state['lo']==0 and state['cnt']==32 and '-DOKC_SC6W=1' in state['sfxm'] and '-DOKC_PRMTSC=1' in state['sfx']
    assert not state.get('ps') and not PDL,'ordinary repaired route only'
    arms=[]
    for i in range(4):
        arm=dict(state);arm['scratch']=_make_scratch(weights['w13'].device)
        arm['sfxm']=tuple(state['sfxm'])+(('-DOKC_STAGE_RETIRE=1',) if i>=2 else ())
        assert arm['w'] is state['w'],'same SC6 pack required'
        arms.append(arm)
    for i in range(4):
        for j in range(i):assert not (_st_scratch_ptrs(arms[i]['scratch'])&_st_scratch_ptrs(arms[j]['scratch'])),'shared mutable scratch'
    resources=[];modules=[]
    for i in (0,2):
        kind,defs=_st_defs(arms[i],128)
        assert kind=='main' and '-DOKC_WGS=1' in defs and '-DOKC_SC6W=1' in defs
        ext=_ext(defs);ext.okc_set_pdl(0);modules.append(ext);r=list(ext.sc6_resources());budget()
        assert len(r)==12 and max(r[1],r[7])<=1024 and min(r[4],r[10])>=1,('ST_RESOURCE',r)
        resources.append(r)
    assert modules[0] is not modules[1],'distinct on/off binaries required'
    for offset in (0,6):assert resources[0][offset+2:offset+6]==resources[1][offset+2:offset+6],('ST_RESIDENCY',resources)
    reach=[]
    for m in (1,8,16,32,128,256):
        defs=[_st_defs(a,m) for a in arms]
        assert (defs[0][0]=='main')==(m in _ST_AFFECTED)
        if m in _ST_AFFECTED:
            assert defs[0][1]==defs[1][1] and defs[2][1]==defs[3][1] and defs[2][1]==defs[0][1]+('-DOKC_STAGE_RETIRE=1',)
        else:assert all(x==defs[0] for x in defs),'unaffected route drift'
        reach.append((m,defs[0][0],list(defs[0][1]),list(defs[2][1])))
    print('ST_RESOURCE',resources,'compiled_main_binaries',2,'private_placements',4,'null_kind_shared_binary_private_scratch_graph',flush=True)
    print('ST_REACH',reach,'B4FIN_standalone32',False,'layer32_B4FIN',True,'no_transfer32_claim',flush=True)
    cw={k:v[:32].clone() if k!='post_norm_w' else v.clone() for k,v in weights.items()}
    ns={'__name__':'_stage_private_sglang'};exec(_SC_BASELINE,ns);control=ns['prepare'](cw);budget()
    flush=torch.empty(64*1024*1024,device=weights['w13'].device,dtype=torch.int32)
    rows=[];counts={'base':0,'fresh':0,'remote':0,'restore':0,'negative':0};worst=[0.,0.];routes=[]
    def check(out,ref,tag):
        err=_st_errors(out,ref);counts[tag]+=1
        for j in range(2):worst[j]=max(worst[j],err[j])
    for m in (1,8,16,32,128,256):
        for seed in _ST_SEEDS:
            budget();original=_gen_input(m,seed*1009+m,weights['w13'].device)
            data=[_st_private(original) for _ in arms]
            for k in ('hidden_states','topk_ids','topk_weights'):assert len({d[k].data_ptr() for d in data})==4
            reference=ns['kernel_fn'](_st_private(original),control);torch.cuda.synchronize()
            graphs=[];outputs=[]
            for arm,d in zip(arms,data):
                for _ in range(2):_st_call(d,arm)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=_st_call(d,arm)
                graph.replay();graphs.append(graph);outputs.append(out)
            torch.cuda.synchronize();assert len({o.data_ptr() for o in outputs})==4
            for out in outputs:check(out,reference,'base')
            fresh=_gen_input(m,seed*2017+m,weights['w13'].device)
            freshref=ns['kernel_fn'](_st_private(fresh),control);torch.cuda.synchronize()
            for d in data:_st_copy(d,fresh)
            for _ in range(16):
                for graph in graphs:graph.replay()
            torch.cuda.synchronize()
            for out in outputs:check(out,freshref,'fresh')
            remote=_st_private(fresh);remote['topk_ids'].copy_(torch.arange(32,40,device=remote['topk_ids'].device,dtype=torch.int32)[None,:].expand(m,8))
            remoteref=ns['kernel_fn'](_st_private(remote),control);torch.cuda.synchronize()
            assert torch.count_nonzero(remoteref)==0,'SGLang remote oracle'
            for d in data:_st_copy(d,remote)
            for graph,out in zip(graphs,outputs):out.fill_(17);graph.replay()
            torch.cuda.synchronize()
            for out in outputs:
                assert out.dtype==torch.bfloat16 and bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,'ST_REMOTE_UNWRITTEN'
                counts['remote']+=1
            for d in data:_st_copy(d,original)
            for graph in graphs:graph.replay()
            torch.cuda.synchronize()
            for out in outputs:check(out,reference,'restore')
            if m==32 and seed==_ST_SEEDS[0]:
                for bad in (torch.zeros_like(reference),freshref,reference+1):
                    rejected=False
                    try:_st_errors(bad,reference)
                    except AssertionError:rejected=True
                    assert rejected,'ST_NEGATIVE_NOT_REJECTED';counts['negative']+=1
            samples=[[] for _ in arms]
            for rep in range(24):
                order=list(range(4));random.Random(seed+rep).shuffle(order)
                for i in order:
                    flush.add_(1);a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                    a.record();graphs[i].replay();z.record();z.synchronize();samples[i].append(a.elapsed_time(z)*1000)
            for out in outputs:_st_errors(out,reference)
            t=[statistics.median(x) for x in samples];rows.append((m,seed,t))
            local=torch.bincount(original['topk_ids'].flatten().long(),minlength=256)[:32]
            routes.append((m,seed,int(local.sum()),int((local>0).sum()),int(local.max())))
            print('ST_CASE',rows[-1],flush=True)
            budget()
    assert counts=={'base':72,'fresh':72,'remote':72,'restore':72,'negative':3},('ST_INCOMPLETE',counts)
    budget();_st_summary(rows)
    print('ST_CORE',counts,'worst_global_row',worst,'resources',resources,'routes',routes,flush=True)
    print('ST_DONE',time.monotonic()-began,'judged_primary_repaired_SC6_control','two_binaries_four_private_placements','no_submit',flush=True)
    state['_stage_control']=arms[0]
    return state

def prepare(weights):
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _st_run(weights)
    finally:torch.backends.cuda.matmul.allow_tf32=previous

def kernel_fn(data,state):
    return _ST_KERNEL(data,state['_stage_control'])
