"""Fixed four-placement outgoing seam validation, embedded only in private checks."""
SUPPORT = r'''
_OS_ARMS=("rawOFF1","rawOFF2","adapterOFF","adapterON")
_OS_GRAPHS=[]
_OS_SECONDARY={}
_OS_BUILDS=[]
_OS_ORIGINAL_LOAD=load_inline

def _os_load_inline(*args,**kwargs):
    import hashlib
    def digest(value):
        if isinstance(value,(list,tuple)):value="\n".join(value)
        return hashlib.sha256((value or "").encode()).hexdigest()
    started=time.monotonic()
    extension=_OS_ORIGINAL_LOAD(*args,**kwargs)
    _OS_BUILDS.append(dict(name=kwargs.get("name",args[0] if args else None),cuda_sha256=digest(kwargs.get("cuda_sources")),cpp_sha256=digest(kwargs.get("cpp_sources")),flags=kwargs.get("extra_cuda_cflags",[]),functions=kwargs.get("functions",[]),host_load_seconds=time.monotonic()-started))
    return extension

def _os_score(rows):
    import math
    cells={(row["seed"],row["tokens"],row["context"]):row["times_us"] for row in rows}
    expected={(seed,tokens,context) for seed in (993811,993817,993821) for tokens in (1,8,16,32,128,256) for context in ("sliding","full")}
    assert len(rows)==len(cells)==36 and set(cells)==expected,"OS_SCORE_COVERAGE"
    banks=[]
    for seed in (993811,993817,993821):
        net=[0.,0.];short=[0.,0.];adapt=[0.,0.];isolated=0.;null=0.;regress=[True,True]
        for tokens in (1,8,16,32,128,256):
            pair=[cells[seed,tokens,context] for context in ("sliding","full")]
            assert all(len(times)==4 and all(type(value) in (int,float) and math.isfinite(value) and value>0 for value in times) for times in pair),"OS_SCORE_TIME"
            cost=[46*pair[0][arm]+16*pair[1][arm] for arm in range(4)]
            null+=abs(math.log(cost[0]/cost[1]))/6
            isolated+=math.log(cost[2]/cost[3])/6
            for control in (0,1):
                gain=math.log(cost[control]/cost[3])/6
                net[control]+=gain;adapt[control]+=math.log(cost[control]/cost[2])/6
                if tokens in (1,8):short[control]+=gain
                regress[control]&=all(times[3]-times[control]<=max(.5,.01*times[control]) for times in pair)
        threshold=max(math.log(1.01),2*null)
        banks.append(dict(seed=seed,net_log_gains=net,short_log_contributions=short,adapter_off_log_gains=adapt,isolated_pdl_log_gain=isolated,duplicate_log_null=null,threshold_log_gain=threshold,context_regression_gates=regress,arithmetic_gate=min(net)>threshold and all(regress)))
    return dict(arms=list(_OS_ARMS),banks=banks,gate=all(bank["arithmetic_gate"] for bank in banks),complete_sixload=True)

def _os_edges(rows,primary,consumer,on):
    assert rows and rows[0]==[2,0,1,1],"OS_EDGE_ABI"
    nodes={row[1]:row[2] for row in rows[1:] if row[0]==0}
    kernels=[row for row in rows[1:] if row[0]==0]
    edges=[row[1:] for row in rows[1:] if row[0]==1]
    assert len(nodes)==len(kernels) and len(rows)==1+len(kernels)+len(edges),"OS_EDGE_ROWS"
    assert all(len(row)==5 and row[2:] in ([0,0,0],[1,0,1]) for row in edges),"OS_EDGE_DATA"
    producers=[node for node,pointer in nodes.items() if pointer in primary]
    consumers=[node for node,pointer in nodes.items() if pointer==consumer]
    assert len(producers)==len(consumers)==1,"OS_NODE_IDENTITY"
    producer=producers[0];target=consumers[0]
    seam=[row for row in edges if row[:2]==[producer,target]]
    incoming=[row for row in edges if row[1]==producer]
    outgoing=[row for row in edges if row[0]==target]
    programmatic=[row for row in edges if row[4]==1]
    assert len(seam)==1 and seam[0][2:]==([1,0,1] if on else [0,0,0]),"OS_SEAM"
    assert programmatic==(seam if on else []),"OS_EXTRA_PROGRAMMATIC"
    assert incoming and outgoing and all(row[2:]==[0,0,0] for row in incoming+outgoing),"OS_BOUNDARY_COMPLETION"
    assert all(row[0]==producer or row[2:]==[0,0,0] for row in edges if row[1]==target),"OS_CONSUMER_INPUT"
    return dict(producer=[producer,nodes[producer]],consumer=[target,nodes[target]],primary=list(primary),incoming=incoming,seam=seam,outgoing=outgoing,neighbor_kernels=[[node,nodes.get(node,0)] for node in sorted({row[0] for row in incoming}|{row[1] for row in outgoing})],edge_data_nonnull=True,programmatic_edges=len(programmatic))

def _os_secondary_gate(arms,budget):
    original=bfp__get_ext()
    for tokens in (1,8):
        vectors=[]
        for arm in arms:
            scoped=arm.get("_os_unpool_bfp",arm["bfp"])
            pmax=scoped["sk2"]["slice8_r192"][3]
            ext=scoped["_os_launch"][0] if "_os_launch" in scoped else original
            vector=list(ext.seam_unpool_resources(pmax));budget()
            assert len(vector)==8 and all(type(value)is int and value>=0 for value in vector),"OS_SECONDARY_RESOURCE_ABI"
            assert vector[0]>0 and vector[1]>0 and vector[5]>=416 and vector[6]>=1 and vector[7]==416,"OS_SECONDARY_RESOURCE"
            vectors.append(vector)
        assert vectors[0]==vectors[1] and vectors[2]==vectors[3],"OS_SECONDARY_CONTROL_IDENTITY"
        raw,adapt=vectors[0],vectors[2]
        assert adapt[1]<=raw[1] and adapt[2]<=raw[2] and adapt[3:6]==raw[3:6] and adapt[6]>=raw[6],"OS_SECONDARY_RESOURCE_LOSS"
        _OS_SECONDARY[tokens]=vectors
    return dict(_OS_SECONDARY)

def _os_graph(graph,data,state):
    tokens=data["config"]["tokens"]
    if tokens not in (1,8):return
    placement=state["_os_placement"]
    query=state.get("_os_unpool_bfp",{}).get("_os_launch",(bfp__get_ext(),0))[0]
    raw=list(query.seam_graph_edges(graph.raw_cuda_graph()))
    proof=_os_edges(raw,state["_os_primary"],_OS_SECONDARY[tokens][placement][0],placement==3)
    _OS_GRAPHS.append([tokens,data["config"]["window"],state["_os_bank"],placement,proof])

_OS_RAW_SK2_SOURCE=bfp__sk2_src
def bfp__sk2_src(*args,**kwargs):
    return _OS_RAW_SK2_SOURCE(*args,**kwargs)+_OS_QUERY_SOURCE

def _wl_summary(rows,part):
    assert part=="all"
    return _os_score(rows)

def _srm_resource_gate(arms,budget):
    result={}
    for key,tokens in (("tiny",1),("tiny2",8),("mid",16),("b4",32),("main",128)):
        vectors=[]
        for arm in arms:
            ext=_fork_native_moe(tokens,arm);vector=list(ext.wl_resources());budget()
            assert len(vector)==12 and min(vector[4],vector[10])>=1,"OS_PRIMARY_RESOURCE"
            assert "-DOKC_SHORT_FINAL_RELAXED=1" not in _WL_EXT_FLAGS[id(ext._module)],"OS_RMW_CHANGED"
            assert ext._module is arms[0]["ma_ext"][key]._module,"OS_PRIMARY_MODULE_DRIFT"
            vectors.append(vector)
        assert all(vector==vectors[0] for vector in vectors),"OS_PRIMARY_RESOURCE_PARITY"
        result[key]=vectors
    return result
'''
