def _flip_time(fn, flush, iters=20):
    """_graph_time with the flush optional (None: no L2 eviction between replays)."""
    if flush is not None:
        return _graph_time(fn, flush, iters)
    tiny = torch.empty(1, dtype=torch.uint8, device="cuda")
    return _graph_time(fn, tiny, iters)


def _flip_stats(ids, start):
    loc = ids.long().flatten() - start
    loc = loc[(loc >= 0) & (loc < LOCAL_EXPERTS)]
    cnt = torch.bincount(loc, minlength=LOCAL_EXPERTS)
    return f"{int(cnt.sum())}/{int((cnt > 0).sum())}/{int(cnt.max())}"


def _flip_probe(state):
    """Second probe (the first, check 0cb15c11accf4190, found the layer's own routing flips
    the ranking): separates the routing from the stream structure. For each seed, the
    routing the layer's router gives a random BF16 FFN input (captured eagerly from
    _moe_front) and the bench's synthetic routing, each timed with the experts alone
    (graph replay after an L2 flush): w4-run (metadata outside the graph, as _moe1_tune),
    w4-full (route_align inside), and the fused main variants; then the whole front plus
    experts on the router routing. Routing stats: local routes / active experts / max routes
    per expert."""
    if not (state.get("moe1_ok") and state.get("w4_ok") and state.get("moe_ours")):
        print("FLIP skipped: moe1/w4 not enabled")
        return
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    mains = [v for v, (_, kind, _) in enumerate(_MOE1_VARIANTS) if kind == "main"]
    seeds = (11, 12, 13, 14, 15, 16)
    for bs in (128, 256):
        gen = torch.Generator(device="cuda").manual_seed(31 + bs)
        local_start = torch.tensor([96], dtype=torch.int32, device="cuda")
        variant = _front_variant(bs, state)
        cands = ["w4-run", "w4-full"] + mains
        tabs = {src: {c: [] for c in cands} for src in ("router", "syn")}
        front = {c: [] for c in [None] + mains}
        stats = {"router": [], "syn": []}
        for seed in seeds:
            ffn_input = torch.randn((bs, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            opts0 = dict(_moe_opts(bs, state), fused=None)
            ei, tw_r, ids_r, meta0 = _moe_front(ffn_input, None, local_start, state, opts0,
                                                variant)
            torch.cuda.synchronize()
            x_r = ei.clone() if meta0 is not None else _pre_norm(ei, state).clone()
            tw_r = tw_r.float().contiguous().clone()
            ids_r = ids_r.to(torch.int32).contiguous().clone()
            xs, tws, idss, st = _moe_routing(bs, 977 * seed + bs, 0, False)
            inputs = {"router": (x_r, tw_r, ids_r, local_start),
                      "syn": (xs, tws, idss, st)}
            stats["router"].append(_flip_stats(ids_r, 96))
            stats["syn"].append(_flip_stats(idss, 0))
            for src, (x, tw, ids, start) in inputs.items():
                for c in cands:
                    fused = c if isinstance(c, int) else None
                    opts = dict(_moe_opts(bs, state), fused=fused)
                    if c == "w4-full":
                        fn = (lambda o=opts, x=x, tw=tw, ids=ids, start=start:
                              _experts(x, tw, ids, start, state, o))
                    else:
                        meta = _route_meta(tw, ids, start, bs, state, opts)
                        fn = (lambda m=meta, x=x: _experts_run(x, m, state))
                    tabs[src][c].append(_flip_time(fn, flush))
            for c in front:
                opts = dict(_moe_opts(bs, state), fused=c)

                def run(o=opts):
                    e2, tw2, ids2, meta = _moe_front(ffn_input, None, local_start, state, o,
                                                     variant)
                    if meta is not None:
                        return _experts_run(e2, meta, state)
                    return _experts(_pre_norm(e2, state), tw2, ids2, local_start, state, o)

                front[c].append(_flip_time(run, flush))
        n = len(seeds)
        for src, tab in tabs.items():
            line = " ".join(f"{c if isinstance(c, str) else _moe1_name(c)}:"
                            f"{sum(v) / n:.1f}" for c, v in tab.items())
            print(f"FLIP2 experts-only {src} T={bs} {line} us (mean of {n} seeds); routing "
                  f"local/active/max per seed " + " ".join(stats[src]))
            for v in mains:
                d = [a - b for a, b in zip(tab[v], tab["w4-run"])]
                d2 = [a - b for a, b in zip(tab[v], tab["w4-full"])]
                print(f"FLIP2 experts-only {src} T={bs} {_moe1_name(v)} minus w4-run "
                      + " ".join(f"{x:+.1f}" for x in d)
                      + "; minus w4-full " + " ".join(f"{x:+.1f}" for x in d2))
        line = " ".join(f"{_moe1_name(c)}:{sum(v) / n:.1f}" for c, v in front.items())
        print(f"FLIP2 front+experts router T={bs} ({variant}) {line} us (mean of {n} seeds)")
        for v in mains:
            d = [a - b for a, b in zip(front[v], front[None])]
            print(f"FLIP2 front+experts router T={bs} {_moe1_name(v)} minus w4 "
                  + " ".join(f"{x:+.1f}" for x in d))


