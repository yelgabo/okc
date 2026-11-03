"""Private four-cell verify comparison; normal judged entry remains SGLang."""
_VERIFY_SEEDS = (781051, 781061)
_VERIFY_CASES = ((16,"sliding"),(16,"full"),(32,"sliding"),(32,"full"))


def _verify_inputs(batch, layer, seed):
    data = _ab_inputs(batch,8,layer,seed)
    front = _ab_ref_front(data,_PAIR_WEIGHTS)
    front8 = _ab_ref_front(data,_PAIR_WEIGHTS,FP8)
    counts = torch.bincount((front["biased"].topk(8,dim=-1).indices//32).flatten(),minlength=8)
    local = 32*int(counts.argmax())
    data["local_start"].fill_(local)
    front["local_start"] = front8["local_start"] = local
    return data,(front,front8)


def _pair_owned_owners(first, second, weights):
    def tensors(value, seen):
        if id(value) in seen:return []
        seen.add(id(value))
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,dict):items=value.values()
        elif isinstance(value,(tuple,list)):items=value
        elif isinstance(value,torch.nn.Module) or type(value).__module__.startswith("sglang"):
            items=vars(value).values()
        else:return []
        return [t for item in items for t in tensors(item,seen)]
    readonly={_wl_storage(t) for t in weights.values()}
    intervals=[[_wl_storage(t) for t in tensors(state,set()) if _wl_storage(t) not in readonly]
               for state in (first,second)]
    assert not _wl_overlap(*intervals),"VERIFY_OWNED_MUTABLE_ALIAS"
    workspaces=[]
    for state in (first,second):
        assert set(state["dlplt"])=={("unpooler",128),("unpooler",256),("qkv_gate",256)}
        current=[]
        for run in state["dlplt"].values():
            names=run.__code__.co_varnames[1:run.__code__.co_argcount]
            defaults=dict(zip(names,run.__defaults__))
            assert isinstance(defaults["ws"],torch.Tensor)
            current.append(_wl_storage(defaults["ws"]))
        workspaces.append(current)
    assert not _wl_overlap(*workspaces),"VERIFY_OWNED_LT_ALIAS"


def _verify_report(rows, checks, seconds):
    import math
    assert len(rows)==8 and checks==96
    assert {(r["batch"],r["layer"],r["seed"]) for r in rows}=={
        (b,l,s) for b,l in _VERIFY_CASES for s in _VERIFY_SEEDS}
    for r in rows:
        assert len(r["times_us"])==4 and len(r["samples_us"])==4
        assert all(math.isfinite(x) and x>0 for x in r["times_us"])
        assert all(len(v)==24 and all(math.isfinite(x) and x>0 for x in v) for v in r["samples_us"])
        b,l=r["batch"],r["layer"]
        w=46 if l=="sliding" else 16
        base=_VERIFY_G_COSTS[b]
        # These are conditional transfer scenarios. Actual diagnostic event
        # ratios must reproduce in a real hybrid, whose other11 cells stay fixed.
        r["conditional_cell_hybrid_factors"]=[
            (base/(base+w*_VERIFY_G_CELLS[b,l]*(r["times_us"][o]/r["times_us"][g]-1)))**(1/6)
            for o in (0,1) for g in (2,3)]
    grouped=[]
    for seed in _VERIFY_SEEDS:
        by={(r["batch"],r["layer"]):r["times_us"] for r in rows if r["seed"]==seed}
        costs={b:[46*by[b,"sliding"][i]+16*by[b,"full"][i] for i in range(4)] for b in (16,32)}
        ratios=[math.sqrt((costs[16][o]/costs[16][g])*(costs[32][o]/costs[32][g]))
                for o in (0,1) for g in (2,3)]
        grouped.append({"seed":seed,"two_load_cross_ratios":ratios})
    return {"schema":"VERIFY_LAYER_PAIR_V1","scope":"verify16/32, sliding/full, two of six loads",
            "source_pins":_PAIR_PINS,"rows":rows,"math_checks":checks,
            "identical_natural_global_ids":True,"prepare_s":seconds,
            "weighted_two_load_comparisons":grouped,"whole_six_load_score_claim":False}


def _verify_prepare(weights):
    import copy
    import statistics
    import random
    import base64
    import zlib
    global _PAIR_WEIGHTS,_GUARDS
    started=time.monotonic()
    def budget():assert time.monotonic()-started<295,"VERIFY_PREPARE_BUDGET"
    _PAIR_WEIGHTS=weights
    baseline=_PAIR_MODULES["baseline"].prepare(weights)
    budget()
    owned=_PAIR_MODULES["owned"].prepare(weights)
    budget()
    graft=_PAIR_MODULES["G"].prepare(weights)
    _GUARDS=_PAIR_MODULES["G"]._GUARDS
    budget()
    owned2=_PAIR_MODULES["owned"].prepare(weights)
    _pair_owned_owners(owned,owned2,weights)
    readonly=_wl_readonly(graft,weights)
    graft2=_wl_clone(graft,readonly,False)
    _wl_check_owners([graft,graft2],readonly)
    states=[owned,owned2,graft,graft2]
    modules=[_PAIR_MODULES["owned"]]*2+[_PAIR_MODULES["G"]]*2
    if torch.cuda.mem_get_info()[0]<12<<30:raise RuntimeError("VERIFY_MEMORY_PREFLIGHT")
    flush=torch.empty(256<<20,device=weights["qkv_gate"].device,dtype=torch.uint8)
    rows=[];checks=0
    for batch,layer in _VERIFY_CASES:
        for seed in _VERIFY_SEEDS:
            budget()
            base,refs=_verify_inputs(batch,layer,seed)
            fresh,fresh_refs=_verify_inputs(batch,layer,seed+101)
            data=[copy.deepcopy(base) for _ in states]
            mutable=[{(str(t.device),t.untyped_storage().data_ptr()) for t in d.values()
                      if isinstance(t,torch.Tensor)} for d in data]
            assert all(not mutable[i]&mutable[j] for i in range(4) for j in range(i))
            captures=[]
            for module,inputs,state in zip(modules,data,states):
                budget()
                warm=torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(2):
                        _pair_restore(inputs,base);_pair_run(module,inputs,state)
                torch.cuda.current_stream().wait_stream(warm)
                torch.cuda.synchronize()
                _pair_restore(inputs,base)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=_pair_run(module,inputs,state)
                graph._verify_inputs=inputs
                graph.replay();torch.cuda.synchronize()
                _pair_validate(base,inputs,output,refs);checks+=1
                captures.append((graph,output))
            assert all(torch.equal(captures[0][1][2],x[1][2]) for x in captures[1:]),"VERIFY_NATURAL_ROUTING_DRIFT"
            for inputs in data:_pair_restore(inputs,fresh)
            for _ in range(4):
                budget()
                for inputs,(graph,_) in zip(data,captures):
                    inputs["stream"].copy_(fresh["stream"])
                    inputs["attn_input"].copy_(fresh["attn_input"])
                    graph.replay()
            torch.cuda.synchronize()
            for inputs,(_,output) in zip(data,captures):
                _pair_validate(fresh,inputs,output,fresh_refs);checks+=1
            assert all(torch.equal(captures[0][1][2],x[1][2]) for x in captures[1:]),"VERIFY_FRESH_ROUTING_DRIFT"
            for inputs,(graph,_) in zip(data,captures):
                _pair_restore(inputs,base);graph.replay()
            torch.cuda.synchronize()
            samples=[[] for _ in states]
            for repeat in range(24):
                budget()
                order=list(range(4));random.Random(seed+repeat).shuffle(order)
                for index in order:
                    inputs,(graph,_)=data[index],captures[index]
                    inputs["stream"].copy_(base["stream"])
                    inputs["attn_input"].copy_(base["attn_input"])
                    flush.zero_()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    samples[index].append(start.elapsed_time(end)*1000)
            for inputs,(_,output) in zip(data,captures):
                _pair_validate(base,inputs,output,refs);checks+=1
            assert all(int(s["verify_diag"][0])==0 for s in (owned,owned2)),"VERIFY_WD_DIAGNOSTIC"
            assert all(int(s["moe1"]["scratch"]["ctr"][8])==0 for s in (owned,owned2)),"VERIFY_OWNED_MOE_SPIN"
            rows.append({"batch":batch,"layer":layer,"seed":seed,
                         "times_us":[statistics.median(v) for v in samples],"samples_us":samples})
            # Captures explicitly retain their private input allocations; drop the
            # completed case before allocating the next large cache/input banks.
            del captures,data,base,fresh,refs,fresh_refs,graph,output,inputs
    _pair_owned_owners(owned,owned2,weights)
    _wl_check_owners([graft,graft2],readonly)
    _PAIR_MODULES["G"]._check_guards()
    for storage,words in graft2["_wl_guard_storage"]:
        assert bool((storage.view(torch.int32)[words:]==_PAIR_MODULES["G"].SCRATCH_SENTINEL).all()),"VERIFY_G_CLONE_GUARD"
    report=_verify_report(rows,checks,time.monotonic()-started)
    payload=base64.b64encode(zlib.compress(json.dumps(report,separators=(",",":"),allow_nan=False).encode(),9)).decode()
    assert len(payload)<11000,"VERIFY_LOG_BUDGET"
    budget()
    print("VERIFY_LAYER_PAIR_ZLIB "+payload,flush=True)
    return baseline


def prepare(weights):
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
        return _verify_prepare(weights)
    finally:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=old_tf32


def kernel_fn(data,state):
    return _PAIR_MODULES["baseline"].kernel_fn(data,state)
