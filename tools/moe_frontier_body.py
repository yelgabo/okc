"""Private off-board WGS phase timing. Appended by moe_make_frontier.py."""
_frontier_prepare_base = prepare


def prepare(weights):
    import statistics
    import time
    began = time.monotonic()
    assert weights['w13'].shape[0] == weights['w2'].shape[0] == 256
    torch.backends.cuda.matmul.allow_tf32 = False
    state = _frontier_prepare_base(weights)
    grid, lo = state['grid'], state['lo']
    assert lo == 0 and state['cnt'] == 32
    flags = tuple(_main_defines(state['sfx'])) + tuple(state['sfxm'])
    assert '-DOKC_WGS=1' in flags and '-DOKC_SC6W=1' in flags
    base = _ext(flags)
    stamped = _ext(flags + ('-DOKC_FRONTIER=1',))
    resources = [module.sc6_resources() for module in (base, stamped)]
    for r in resources:
        assert len(r) == 12 and min(r[4],r[10]) >= 1
    resource_gate = (resources[0][7:12] == resources[1][7:12] and resources[0][7] == 0)
    assert resource_gate, ('WGSFRONT_RESOURCE_MISMATCH',resources)
    baseline = {'__name__': '_frontier_private_sglang_baseline'}
    exec(_SC_BASELINE, baseline)
    ref_weights = {k:(v.clone() if k == 'post_norm_w' else v[:32].clone()) for k,v in weights.items()}
    reference_state = baseline['prepare'](ref_weights)
    del ref_weights
    dev = weights['w13'].device
    flush = torch.empty(64 * 1024 * 1024, device=dev, dtype=torch.int32)
    scratch = [state['scratch'], _make_scratch(dev), _make_scratch(dev)]
    assert all(len(s) == 3 for s in scratch)
    assert len({s[0].data_ptr() for s in scratch}) == 3
    records = []
    capsule = None
    for m in (128,256):
        for seed in (4511,4517,4523):
            assert time.monotonic()-began < 270, 'frontier prepare budget'
            data = _gen_input(m,seed+m,dev)
            x, ids, tw = (data[k] for k in ('hidden_states','topk_ids','topk_weights'))
            assert ids.dtype == torch.int32 and ids.is_contiguous() and x.is_contiguous()
            trace = torch.zeros(trace_words(grid,m),device=dev,dtype=torch.int64)
            slots = [*scratch[2], torch.empty(0,device=dev,dtype=torch.int64), trace]
            # All workspaces and graph outputs are independent. Only frozen weights
            # and the current input buffers are shared. The fourth slot is an
            # unused placeholder for the original optional debug buffer.
            arms = [(base,scratch[0]),(base,scratch[1]),(stamped,slots)]
            outputs = [torch.empty_like(x) for _ in arms]
            graphs = []
            ref = baseline['kernel_fn'](data,reference_state)
            for i,(module,ws) in enumerate(arms):
                args = (state['w'],ws,x,ids,tw,outputs[i],lo,32,grid)
                module.okc_run(*args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    module.okc_run(*args)
                graph.replay()
                graphs.append(graph)
            torch.cuda.synchronize()
            for out in outputs:
                _sc_errors(out,ref)
            # Rotate all three graphs; flush and clear trace outside timing.
            laps = [[] for _ in arms]
            for lap in range(33):
                for offset in range(3):
                    i = (lap+offset)%3
                    if i == 2:
                        trace.zero_()
                    flush.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[i].replay()
                    end.record()
                    end.synchronize()
                    laps[i].append(start.elapsed_time(end)*1000)
            med = [statistics.median(ts) for ts in laps]
            for out in outputs:
                _sc_errors(out,ref)
            # Compare a fresh route and activation through the same captured pointers.
            fresh = _gen_input(m,5*seed+m,dev)
            fresh_ref = baseline['kernel_fn'](fresh,reference_state)
            for key in ('topk_ids','topk_weights','hidden_states'):
                data[key].copy_(fresh[key])
            for graph,out in zip(graphs,outputs):
                out.fill_(float('nan'))
                graph.replay()
            torch.cuda.synchronize()
            for out in outputs:
                _sc_errors(out,fresh_ref)
            if seed == 4511:
                remote = torch.arange(32,40,device=dev,dtype=torch.int32)
                ids.copy_(remote.expand(m,-1))
                trace.zero_()
                for graph,out in zip(graphs,outputs):
                    out.fill_(float('nan'))
                    graph.replay()
                torch.cuda.synchronize()
                for out in outputs:
                    assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out)) == 0
                empty = analyze(trace.cpu().tolist(),ids.cpu().tolist(),lo,grid)
                assert empty['unique_experts'] == 0, 'spurious producer on empty routing'
            # Restore the exact timed input using a new generator call. Each
            # sample uses a cleared trace and a separate cache flush.
            original = _gen_input(m,seed+m,dev)
            for key in ('topk_ids','topk_weights','hidden_states'):
                data[key].copy_(original[key])
            routes = ids.cpu().tolist()
            samples = []
            for repeat in range(5):
                trace.zero_()
                flush.zero_()
                graphs[2].replay()
                torch.cuda.synchronize()
                _sc_errors(outputs[2],ref)
                marks = trace.cpu().tolist()
                samples.append((analyze(marks,routes,lo,grid),marks))
            samples.sort(key=lambda pair: pair[0]['ideal_saving_us'])
            result,marks = samples[2]
            result['saving_samples_us'] = [p[0]['ideal_saving_us'] for p in samples]
            records.append((m,seed,med,result))
            if m == 256 and seed == 4517:
                capsule = encode_trace(marks,routes,lo,grid)
            del graph,graphs,outputs
    assert len(records) == 6 and capsule is not None
    print('WGSFRONT_RESOURCE',resources,'same_residency_no_spill',resource_gate,flush=True)
    for row in records:
        print('WGSFRONT_CASE',row,flush=True)
    print('WGSFRONT_TRACE',capsule,flush=True)
    print('WGSFRONT_DONE',len(records),'prepare_s',time.monotonic()-began,flush=True)
    return state
