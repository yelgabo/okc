"""Appended to the private comparator; no file reads or input caching at runtime."""
_SC_PREPARE = prepare
_SC_KERNEL = kernel_fn


def _sc_errors(out, ref):
    a, b = out.float(), ref.float()
    assert bool(torch.isfinite(a).all()), 'SC6_NONFINITE'
    delta = a-b
    ge = float(delta.norm()/b.norm().clamp_min(1.e-12))
    rn = b.norm(dim=-1)
    floor = (.1*rn.mean()).clamp_min(1.e-12)
    re = float((delta.norm(dim=-1)/torch.maximum(rn,floor)).max())
    assert ge <= .015 and re <= .075, ('SC6_MATH', ge, re)
    return ge,re


def _sc_summary(rows):
    import math
    totals = [[],[]]
    nulls = []
    all_cases = True
    for m,seed,t in rows:
        ratios = (t[0]/t[2], t[1]/t[3])
        null = max(abs(math.log(t[0]/t[1])),abs(math.log(t[2]/t[3])))
        nulls.append(null)
        for k in range(2):
            totals[k].append(math.log(ratios[k]))
        regress = max(t[2]-t[0],t[3]-t[1])
        all_cases &= regress <= max(.5,.01*min(t[0],t[1]))
        print('SC6PAIR',m,seed,[round(x,4) for x in t],*[round(x,7) for x in ratios],flush=True)
    gains = [math.exp(sum(v)/len(v)) for v in totals]
    null = math.exp(sum(nulls)/len(nulls))-1
    passed = min(gains)>=1.01 and min(gains)-1 > 2*null and all_cases
    print('SC6SIXLOAD',gains,'absolute_log_null',null,'regression_gate',all_cases,'gate',passed,flush=True)
    return gains,null,passed


def prepare(weights):
    import math
    import time
    import statistics
    began = time.monotonic()
    assert weights['w13'].shape[0]==256 and weights['w2'].shape[0]==256, 'Actual standalone256 bank required'
    torch.backends.cuda.matmul.allow_tf32 = False  # common, serial setup before either arm
    assert not torch.backends.cuda.matmul.allow_tf32, 'Keep full input precision'
    state = _SC_PREPARE(weights)
    assert state['lo']==0 and state['cnt']==32 and 'w_ns' in state
    assert '-DOKC_PRMTSC=1' in state['sfx'] and '-DOKC_SC6W=1' in state['sfxm']
    assert not state.get('ps'), 'Dedicated issuer excluded from controlled pair'
    # Read-only packed weights may be shared; every computing arm has private
    # mutable scratch, including duplicate placement controls.
    arms = []
    for compressed in (False,False,True,True):
        arm = dict(state)
        arm['w'] = state['w'] if compressed else state['w_ns']
        arm['scratch'] = _make_scratch(weights['w13'].device)
        arm['sfxm'] = tuple(d for d in state['sfxm'] if not d.startswith('-DOKC_SC6W')) + (('-DOKC_SC6W=1',) if compressed else ())
        arms.append(arm)
    assert len({a['scratch'][0].data_ptr() for a in arms})==4
    ext_off = _ext(tuple(_main_defines(state['sfx']))+arms[0]['sfxm'])
    ext_on = _ext(tuple(_main_defines(state['sfx']))+arms[2]['sfxm'])
    resources = []
    for ext in (ext_off,ext_on):
        torch.cuda.synchronize()
        r = ext.sc6_resources()
        assert len(r)==12 and all(r[i]==0 for i in (1,7)) and min(r[4],r[10])>=1, ('SC6_RESOURCE',r)
        resources.append(r)
    assert resources[0][2:6]==resources[1][2:6] and resources[0][8:12]==resources[1][8:12], ('SC6_RESIDENCY',resources)
    for name,defs in (('tiny',TINY_DEFINES),('tiny2',TINY2_DEFINES),('mid',MID_DEFINES)):
        torch.cuda.synchronize()
        r=_ext(tuple(defs)+state['sfxs']).sc6_resources()
        assert len(r)==12 and r[1]==r[7]==0 and min(r[4],r[10])>=1,('SC6_RESOURCE',name,r)
        print('SC6CONTROLRESOURCE',name,r,flush=True)
    print('SC6RESOURCE byte/sc6',resources,flush=True)
    # Baseline loader mutates layouts: provide a PRIVATE cloned local bank only.
    control_weights={k:(v[:32].clone() if k!='post_norm_w' else v.clone()) for k,v in weights.items()}
    ns={'__name__':'_sc6_private_sglang_baseline'}
    exec(_SC_BASELINE,ns)
    control = ns['prepare'](control_weights)
    assert not torch.backends.cuda.matmul.allow_tf32
    def ref(data): return ns['kernel_fn'](data,control)
    def budget(): assert time.monotonic()-began<280, 'SC6_PREPARE_BUDGET'
    flush = torch.empty(64*1024*1024,device=weights['w13'].device,dtype=torch.int32)
    rows=[]; checks=0; fresh_checks=0; empty_checks=0; worst=[0.,0.]; route_stats=[]
    for m in (1,8,16,32,128,256):
        for seed in (901,902):
            budget()
            data = _gen_input(m,1009*seed+m,weights['w13'].device)
            assert data['topk_ids'].min()>=0 and data['topk_ids'].max()<256
            original={k:data[k].clone() for k in ('hidden_states','topk_ids','topk_weights')}
            reference=ref(data)
            torch.cuda.synchronize()
            graphs=[]; outputs=[]
            for arm in arms:
                _SC_KERNEL(data,arm)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): output=_SC_KERNEL(data,arm)
                for _ in range(3): graph.replay()
                graphs.append(graph); outputs.append(output)
            torch.cuda.synchronize()
            for output in outputs:
                err=_sc_errors(output,reference); checks+=1
                worst=[max(worst[i],err[i]) for i in range(2)]
            # Fresh x, global IDs and routing weights enter the SAME captured
            # pointers; stale input/output caching cannot satisfy this gate.
            fresh=_gen_input(m,2017*seed+m,weights['w13'].device)
            fresh_ref=ref(fresh)
            for key in original: data[key].copy_(fresh[key])
            for _ in range(16):
                for graph in graphs: graph.replay()
            torch.cuda.synchronize()
            for output in outputs:
                err=_sc_errors(output,fresh_ref); fresh_checks+=1
                worst=[max(worst[i],err[i]) for i in range(2)]
            # Real global expert IDs32..39 are all nonlocal, not synthetic -1 IDs.
            data['topk_ids'].copy_(torch.arange(32,40,device=data['topk_ids'].device,dtype=torch.int32)[None,:].expand(m,8))
            for graph,output in zip(graphs,outputs):
                output.fill_(17)
                graph.replay()
            torch.cuda.synchronize()
            for output in outputs:
                assert output.dtype==torch.bfloat16 and bool(torch.isfinite(output).all()) and int(torch.count_nonzero(output))==0,'SC6_EMPTY'
                empty_checks+=1
            for key in original: data[key].copy_(original[key])
            for graph in graphs: graph.replay()
            torch.cuda.synchronize()
            for output in outputs: _sc_errors(output,reference)
            # Cold L2, same input/weights, interleaved and reversed arm order.
            samples=[[] for _ in arms]
            for rep in range(20):
                order=range(4) if (rep+seed)%2==0 else range(3,-1,-1)
                for a in order:
                    flush.add_(1)
                    e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
                    e0.record();graphs[a].replay();e1.record();e1.synchronize()
                    samples[a].append(e0.elapsed_time(e1)*1000)
            torch.cuda.synchronize()
            for output in outputs: _sc_errors(output,reference)
            t=[statistics.median(x) for x in samples]
            rows.append((m,seed,t))
            counts=torch.bincount(original['topk_ids'].flatten().long(),minlength=256)[:32]
            route_stats.append((m,seed,int(counts.sum()),int((counts>0).sum()),int(counts.max())))
            print('SC6CASE',rows[-1],flush=True)
    budget()
    state['scarms']=arms
    print('SC6CORE',checks,fresh_checks,empty_checks,'worst',worst,'resources',resources,'routes',route_stats,flush=True)
    _sc_summary(rows)
    print('SC6DONE prepare_seconds',time.monotonic()-began,'judged_path byte fully_repaired','no_submit',flush=True)
    return state


def kernel_fn(data,state):
    return _SC_KERNEL(data,state['scarms'][0])
