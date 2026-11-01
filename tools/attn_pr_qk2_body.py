"""Appended private CHECK-ONLY paired test. Production judged route is frozen control."""
import types as _q2_types
import time as _q2_time
import random as _q2_random
import statistics as _q2_stats


def _q2_clone(data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else dict(v) if isinstance(v, dict) else v
            for k, v in data.items()}


def _q2_copy(dst, src):
    assert dst.keys() == src.keys()
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            assert dst[key].shape == value.shape and dst[key].dtype == value.dtype, ('Q2 input shape',key)
            dst[key].copy_(value)
        else:
            assert dst[key] == value, ('Q2 static metadata',key)


def _q2_math(data, out):
    # Existing judge-style independent FP32 references include actual stored new
    # KV and official output/row/cache limits. No candidate-fitted tolerance.
    okay, message = _ob_check(data, out)
    assert okay, ('Q2 official math', message)
    assert bool(torch.isfinite(out).all()), 'Q2 nonfinite'


def _q2_gate(rows):
    import math
    assert len(rows) == 6
    decisions=[]
    for seed in sorted({r[0] for r in rows}):
        pair=[r for r in rows if r[0]==seed]
        assert sorted(r[1] for r in pair)==['full','sliding']
        loads=[sum((46 if context=='sliding' else 16)*times[i]
                   for _,context,times in pair)/62 for i in range(4)]
        null=max(abs(math.log(loads[0]/loads[1])), abs(math.log(loads[2]/loads[3])))
        crosses=[math.log(loads[c]/loads[k])/6 for c in (0,1) for k in (2,3)]
        # Only decode1 changes: the other five scored loads stay fixed. The
        # sixth-root factors here are the actual scoring counterfactual.
        win=min(crosses)>max(2*null/6,math.log(1.002))
        regression=all(times[k]-times[c] <= max(.1,.01*times[c])
                       for _,_,times in pair for c in (0,1) for k in (2,3))
        decisions.append(win and regression)
        print('Q2_SCORE',seed,'loads',loads,'cross_factors',[math.exp(x) for x in crosses],
              'duplicate_log_null',null,'win',win,'context_regression_gate',regression,flush=True)
    result=all(decisions) and len(decisions)==3
    print('Q2_DECISION',result,'all_three_seeds_required',flush=True)
    return result


def _q2_run():
    started=_Q2_IMPORT_STARTED
    def budget(): assert _q2_time.monotonic()-started<280,'Q2 incomplete before280s'
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        # Original diagnostic control is already built during the module's import.
        mods=[_q2_types.SimpleNamespace(**globals())]
        for i in range(1,4):
            budget()
            src=_Q2_STARTED_SOURCE
            old='name="okc_attn_one_v1"'
            assert src.count(old)==1
            src=src.replace(old,'name="okc_attn_q2_private_%d"'%i)
            # Only ONE_SRC macro default changes. Export/resource helpers are
            # common to all diagnostic arms, including both independent controls.
            assert src.count('#define ONE_PR_QK2 0')==1
            if i>=2: src=src.replace('#define ONE_PR_QK2 0','#define ONE_PR_QK2 1')
            mod=_q2_types.ModuleType('_q2_private_%d'%i)
            exec(compile(src,'_q2_private_%d.py'%i,'exec'),mod.__dict__)
            assert mod._ONE is not None,'Q2 extension build failed'
            mods.append(mod)
        resources=[]
        for mod in mods:
            torch.cuda.synchronize()
            r=list(mod._ONE.attn_q2_resources()); assert len(r)%8==0
            r={tuple(r[j:j+2]):r[j+2:j+8] for j in range(0,len(r),8)}
            assert (1,11) in r and (1,13) in r
            for key in ((1,11),(1,13)):
                assert r[key][1]==0 and r[key][-1]>=1,('Q2 local/residency',key,r[key])
            resources.append(r)
        assert all(r.keys()==resources[0].keys() for r in resources)
        for key in resources[0]:
            for r in resources:
                # Keep static/dynamic shared bytes, launched threads and actual
                # blocks/SM. Register count may change with two live chains.
                assert r[key][2:]==resources[0][key][2:],('Q2 resource geometry',key,resources)
                if key not in ((1,11),(1,13)):
                    assert r[key]==resources[0][key],('Q2 unaffected resources',key)
        print('Q2_RESOURCE',[{str(k):v for k,v in r.items()} for r in resources],flush=True)
        rows=[]; eager_checks=0; fresh_checks=0
        # All native scored paths are checked, not merely the two timed paths.
        for batch in (1,16,32):
            for length in (1,8):
                for kind in ('sliding','full'):
                    budget()
                    original=_ob_inputs(batch,80000,'tp8',kind,length,'fixed' if batch==1 else 'agentic',
                                        'soft' if kind=='sliding' else 'coherent',1,1.,seed=713101)
                    for mod in mods:
                        data=_q2_clone(original); output=mod.kernel_fn(data)
                        torch.cuda.synchronize(); _q2_math(data,output); eager_checks+=1
                    del data,output,original
        flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8)
        for kind in ('sliding','full'):
            initial=_ob_inputs(1,80000,'tp8',kind,1,'fixed',
                               'soft' if kind=='sliding' else 'coherent',1,1.,seed=713201)
            data=[_q2_clone(initial) for _ in mods]
            assert len({d['qkv'].data_ptr() for d in data})==4
            assert len({d['k_cache'].data_ptr() for d in data})==4
            assert len({d['v_cache'].data_ptr() for d in data})==4
            graphs=[]; outputs=[]
            # Independent module/state, inputs, caches, graph and output per arm.
            for mod,d in zip(mods,data):
                for _ in range(3): mod.kernel_fn(d)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): out=mod.kernel_fn(d)
                graph.replay(); torch.cuda.synchronize()
                _q2_math(d,out); fresh_checks+=1
                graphs.append(graph); outputs.append(out)
                assert mod._one_plan(d['config'],d['page_table'].shape[1],mod._one_state(d['qkv'].device)['resident']) == ((32,1,13) if kind=='sliding' else (252,8,11))
            assert len({out.data_ptr() for out in outputs})==4
            for seed in (713301,713302,713303):
                budget()
                fresh=_ob_inputs(1,80000,'tp8',kind,1,'fixed',
                                 'soft' if kind=='sliding' else 'coherent',1,1.,seed=seed)
                for d in data: _q2_copy(d,fresh)
                for graph in graphs: graph.replay()
                torch.cuda.synchronize()
                for d,out in zip(data,outputs): _q2_math(d,out); fresh_checks+=1
                # Keep one known-old result for a stale-output negative control.
                stale=outputs[0].clone()
                d=data[0]
                oldgate=d['qkv'][...,1280:].clone()
                d['qkv'][...,1280:].fill_(-20)
                graphs[0].replay(); torch.cuda.synchronize()
                rejected=False
                try: _q2_math(d,stale)
                except AssertionError: rejected=True
                assert rejected,'Q2 stale-output negative control failed'
                d['qkv'][...,1280:].copy_(oldgate)
                for graph in graphs: graph.replay()
                torch.cuda.synchronize()
                # Replays mutate each arm's own scratch/cache. All are queued on
                # the same stream, and no arm shares cross-call generation state.
                for _ in range(128):
                    for graph in graphs: graph.replay()
                torch.cuda.synchronize()
                for d,out in zip(data,outputs): _q2_math(d,out); fresh_checks+=1
                times=[[] for _ in mods]
                for repeat in range(80):
                    order=list(range(4)); _q2_random.Random(seed+repeat).shuffle(order)
                    for arm in order:
                        flush.zero_()
                        a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                        a.record(); graphs[arm].replay();z.record();z.synchronize()
                        times[arm].append(a.elapsed_time(z)*1000)
                med=[_q2_stats.median(t) for t in times]
                for d,out in zip(data,outputs): _q2_math(d,out); fresh_checks+=1
                rows.append((seed,kind,med))
                print('Q2_CASE',seed,kind,med,flush=True)
                del fresh,stale
            del graphs,outputs,data,initial
        budget()
        print('Q2_CORE',eager_checks,fresh_checks,'all_math_passed',flush=True)
        _q2_gate(rows)
        print('Q2_DONE',_q2_time.monotonic()-started,'judged_frozen_control_no_submit',flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32=saved_tf32


_q2_run()
