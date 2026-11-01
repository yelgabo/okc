"""Private, selected-load G-versus-owned experiment. The official entry is SGLang."""
_PAIR_SEEDS = (781031, 781033)


def _pair_inputs(seed):
    data = _ab_inputs(32, 8, "sliding", seed)
    front = _ab_ref_front(data, _PAIR_WEIGHTS)
    front8 = _ab_ref_front(data, _PAIR_WEIGHTS, FP8)
    counts = torch.bincount((front["biased"].topk(8, dim=-1).indices // 32).flatten(), minlength=8)
    local = 32 * int(counts.argmax())
    data["local_start"].fill_(local)
    front["local_start"] = front8["local_start"] = local
    return data, (front, front8)


def _pair_restore(dst, src):
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            assert dst[key].shape == value.shape and dst[key].dtype == value.dtype
            dst[key].copy_(value)
        else:
            assert dst[key] == value


def _pair_run(module, data, state):
    return module.kernel_fn(data, state)


def _pair_validate(base, data, output, refs):
    ok, message = _ab_validate(base, output, data, _PAIR_WEIGHTS, *refs)
    assert ok, ("PAIR_MATH", message)


def _pair_owned_owners(first, second, weights):
    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [t for item in value.values() for t in tensors(item)]
        if isinstance(value, (list, tuple)):
            return [t for item in value for t in tensors(item)]
        return []
    readonly = {_wl_storage(t) for t in weights.values()}
    intervals = [[_wl_storage(t) for t in tensors(state) if _wl_storage(t) not in readonly]
                 for state in (first, second)]
    assert not _wl_overlap(*intervals), "PAIR_OWNED_MUTABLE_ALIAS"
    workspaces = []
    for state in (first, second):
        current = []
        assert set(state["dlplt"]) == {("qkv_gate", 256), ("unpooler", 256)}
        for run in state["dlplt"].values():
            names = run.__code__.co_varnames[1:run.__code__.co_argcount]
            defaults = dict(zip(names, run.__defaults__))
            assert isinstance(defaults["ws"], torch.Tensor)
            current.append(_wl_storage(defaults["ws"]))
        workspaces.append(current)
    assert not _wl_overlap(*workspaces), "PAIR_OWNED_LT_ALIAS"


def _pair_prepare(weights):
    import copy
    import statistics
    import random
    global _PAIR_WEIGHTS, _GUARDS
    started = time.monotonic()
    def budget():
        assert time.monotonic() - started < 295, "PAIR_PREPARE_BUDGET"
    _PAIR_WEIGHTS = weights
    baseline = _PAIR_MODULES["baseline"].prepare(weights)
    budget()
    owned = _PAIR_MODULES["owned"].prepare(weights)
    budget()
    graft = _PAIR_MODULES["G"].prepare(weights)
    _GUARDS = _PAIR_MODULES["G"]._GUARDS
    budget()
    modules = [_PAIR_MODULES["owned"]] * 2 + [_PAIR_MODULES["G"]] * 2
    # Source-specific deep copies retain immutable compiled modules and copy
    # tensor backing allocations and nested Lt workspaces.
    # The owned Lt runners close over their workspaces, so deepcopy would share
    # those closures. A second preparation creates distinct runners/workspaces.
    owned2 = _PAIR_MODULES["owned"].prepare(weights)
    _pair_owned_owners(owned, owned2, weights)
    readonly = _wl_readonly(graft, weights)
    graft2 = _wl_clone(graft, readonly, False)
    _wl_check_owners([graft, graft2], readonly)
    states = [owned, owned2, graft, graft2]
    if torch.cuda.mem_get_info()[0] < 12 << 30:
        raise RuntimeError("PAIR_MEMORY_PREFLIGHT")
    flush = torch.empty(256 << 20, device=weights["qkv_gate"].device, dtype=torch.uint8)
    rows = []
    checks = 0
    for seed in _PAIR_SEEDS:
        budget()
        base, refs = _pair_inputs(seed)
        fresh, fresh_refs = _pair_inputs(seed + 101)
        data = [copy.deepcopy(base) for _ in states]
        mutable = [{(str(t.device), t.untyped_storage().data_ptr()) for t in d.values()
                    if isinstance(t, torch.Tensor)} for d in data]
        assert all(not mutable[i] & mutable[j] for i in range(4) for j in range(i))
        captures = []
        for module, inputs, state in zip(modules, data, states):
            budget()
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(2):
                    _pair_restore(inputs, base)
                    _pair_run(module, inputs, state)
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            _pair_restore(inputs, base)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _pair_run(module, inputs, state)
            graph.replay()
            torch.cuda.synchronize()
            _pair_validate(base, inputs, output, refs)
            checks += 1
            captures.append((graph, output))
        assert all(torch.equal(captures[0][1][2], item[1][2]) for item in captures[1:]), "PAIR_NATURAL_ROUTING_DRIFT"
        for inputs in data:
            _pair_restore(inputs, fresh)
        for _ in range(4):
            for inputs, (graph, _) in zip(data, captures):
                inputs["stream"].copy_(fresh["stream"])
                inputs["attn_input"].copy_(fresh["attn_input"])
                graph.replay()
        torch.cuda.synchronize()
        for inputs, (_, output) in zip(data, captures):
            _pair_validate(fresh, inputs, output, fresh_refs)
            checks += 1
        assert all(torch.equal(captures[0][1][2], item[1][2]) for item in captures[1:]), "PAIR_FRESH_ROUTING_DRIFT"
        for inputs, (graph, _) in zip(data, captures):
            _pair_restore(inputs, base)
            graph.replay()
        torch.cuda.synchronize()
        times = [[] for _ in states]
        for repeat in range(24):
            budget()
            order = list(range(4))
            random.Random(seed + repeat).shuffle(order)
            for index in order:
                inputs, (graph, _) = data[index], captures[index]
                inputs["stream"].copy_(base["stream"])
                inputs["attn_input"].copy_(base["attn_input"])
                flush.zero_()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times[index].append(start.elapsed_time(end) * 1000)
        for inputs, (_, output) in zip(data, captures):
            _pair_validate(base, inputs, output, refs)
            checks += 1
        medians = [statistics.median(sample) for sample in times]
        rows.append({"seed": seed, "times_us": medians, "samples_us": times,
                     "cross_ratios": [medians[c] / medians[g] for c in (0, 1) for g in (2, 3)]})
    assert checks == 24 and len(rows) == 2
    _pair_owned_owners(owned, owned2, weights)
    _wl_check_owners([graft, graft2], readonly)
    _PAIR_MODULES["G"]._check_guards()
    for storage, words in graft2["_wl_guard_storage"]:
        assert bool((storage.view(torch.int32)[words:] ==
                     _PAIR_MODULES["G"].SCRATCH_SENTINEL).all()), "PAIR_G_CLONE_GUARD"
    report = {"scope": "verify32 sliding, weight46, one of twelve cells",
              "source_pins": _PAIR_PINS, "rows": rows, "math_checks": checks,
              "identical_natural_global_ids": True, "prepare_s": time.monotonic() - started,
              "aggregate_score_claim": False}
    print("SELECTED_LAYER_PAIR " + json.dumps(report, separators=(",", ":")), flush=True)
    return baseline


def prepare(weights):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        _AB_DEQ.clear()
        torch.backends.cuda.matmul.allow_tf32 = False
        return _pair_prepare(weights)
    finally:
        _AB_DEQ.clear()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def kernel_fn(data, state):
    return _PAIR_MODULES["baseline"].kernel_fn(data, state)
