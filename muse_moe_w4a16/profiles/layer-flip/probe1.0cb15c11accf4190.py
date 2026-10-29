

def _flip_time(fn, flush, iters=20):
    """_graph_time with the flush optional (None: no L2 eviction between replays)."""
    if flush is not None:
        return _graph_time(fn, flush, iters)
    tiny = torch.empty(1, dtype=torch.uint8, device="cuda")
    return _graph_time(fn, tiny, iters)


def _flip_probe(state):
    """Why the layer's MoE bench keeps the W4 chain at 128 tokens while mode 16 wins
    standalone: same process, same inputs, interleaved per seed.
    syn-*: the bench's synthetic routing (_moe_routing, start 0, as _moe1_tune);
      w4-run times only _experts_run with the routing metadata built outside the graph (what
      _moe1_tune times), w4-full includes route_align (what the standalone bench times);
      cold = L2 flushed before each replay, warm = no flush.
    front: the layer's own FFN front (_moe_front: pooler+router GEMM, pre-expert norm on the
      main stream, routing and metadata on the side stream, join) plus the experts, on the
      layer's router and a random BF16 FFN input, as kernel_fn runs it."""
    if not (state.get("moe1_ok") and state.get("w4_ok") and state.get("moe_ours")):
        print("FLIP skipped: moe1/w4 not enabled")
        return
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    mains = [v for v, (_, kind, _) in enumerate(_MOE1_VARIANTS) if kind == "main"]
    seeds = (11, 12, 13, 14, 15, 16)
    for bs in (128, 256):
        cands = ["w4-run", "w4-full"] + mains
        cold = {c: [] for c in cands}
        warm = {c: [] for c in ["w4-run"] + mains}
        for seed in seeds:
            x, tw, ids, start = _moe_routing(bs, 977 * seed + bs, 0, False)
            for c in cands:
                fused = c if isinstance(c, int) else None
                opts = dict(_moe_opts(bs, state), fused=fused)
                if c == "w4-full":
                    fn = (lambda o=opts: _experts(x, tw, ids, start, state, o))
                else:
                    meta = _route_meta(tw, ids, start, bs, state, opts)
                    fn = (lambda m=meta: _experts_run(x, m, state))
                cold[c].append(_flip_time(fn, flush))
                if c in warm:
                    warm[c].append(_flip_time(fn, None))
        for label, tab in (("cold", cold), ("warm", warm)):
            line = " ".join(f"{c if isinstance(c, str) else _moe1_name(c)}:"
                            f"{sum(v) / len(v):.1f}" for c, v in tab.items())
            print(f"FLIP syn-{label} T={bs} {line} us (mean of {len(seeds)} seeds)")
        for v in mains:
            d_run = [a - b for a, b in zip(cold[v], cold["w4-run"])]
            d_full = [a - b for a, b in zip(cold[v], cold["w4-full"])]
            print(f"FLIP syn-cold T={bs} {_moe1_name(v)} minus w4-run per seed "
                  + " ".join(f"{d:+.1f}" for d in d_run)
                  + "; minus w4-full " + " ".join(f"{d:+.1f}" for d in d_full))
        meta_cost = [a - b for a, b in zip(cold["w4-full"], cold["w4-run"])]
        print(f"FLIP syn-cold T={bs} w4-full minus w4-run per seed "
              + " ".join(f"{d:+.1f}" for d in meta_cost))

        gen = torch.Generator(device="cuda").manual_seed(31 + bs)
        local_start = torch.tensor([96], dtype=torch.int32, device="cuda")
        variant = _front_variant(bs, state)
        fcands = [None] + mains
        front = {c: [] for c in fcands}
        front_warm = {c: [] for c in fcands}
        hot = []
        for seed in seeds:
            ffn_input = torch.randn((bs, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            router_input = ffn_input.float() if bs <= ROUTER_FP32_MAX_TOKENS else None
            seen = {}
            for c in fcands:
                opts = dict(_moe_opts(bs, state), fused=c)

                def run(o=opts):
                    ei, tw2, ids2, meta = _moe_front(ffn_input, router_input, local_start,
                                                     state, o, variant)
                    seen["ids"] = ids2
                    if meta is not None:
                        return _experts_run(ei, meta, state)
                    return _experts(_pre_norm(ei, state), tw2, ids2, local_start, state, o)

                front[c].append(_flip_time(run, flush))
                front_warm[c].append(_flip_time(run, None))
            run()
            torch.cuda.synchronize()
            loc = seen["ids"].long().flatten() - 96
            loc = loc[(loc >= 0) & (loc < LOCAL_EXPERTS)]
            cnt = torch.bincount(loc, minlength=LOCAL_EXPERTS)
            hot.append(f"{int((cnt > 0).sum())}/{int(cnt.max())}/{int((cnt > 64).sum())}")
        for label, tab in (("cold", front), ("warm", front_warm)):
            line = " ".join(f"{_moe1_name(c)}:{sum(v) / len(v):.1f}" for c, v in tab.items())
            print(f"FLIP front-{label} T={bs} ({variant}) {line} us (mean of {len(seeds)} seeds)")
        for v in mains:
            d = [a - b for a, b in zip(front[v], front[None])]
            print(f"FLIP front-cold T={bs} {_moe1_name(v)} minus w4 per seed "
                  + " ".join(f"{x:+.1f}" for x in d))
        print(f"FLIP front T={bs} local routes active experts/max/experts over 64 per seed: "
              + " ".join(hot))
    pol = state.get("moe1_policy") or {}
    print("FLIP echo MOE1POLICY {" + ", ".join(f"{b}: {_moe1_name(v)}" for b, v in pol.items())
          + "}; front " + str(state.get("front_policy")))

