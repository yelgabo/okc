

def _flip_time(fn, flush, iters=20):
    return _graph_time(fn, flush, iters)


def _flip_stats(ids, start):
    loc = ids.long().flatten() - start
    loc = loc[(loc >= 0) & (loc < LOCAL_EXPERTS)]
    cnt = torch.bincount(loc, minlength=LOCAL_EXPERTS)
    return f"{int(cnt.sum())}/{int((cnt > 0).sum())}/{int(cnt.max())}/{int((cnt > 64).sum())}"


def _flip_stream(tokens, q_len, gen):
    """reference.py's residual stream (STREAM_SEED basis: a fixed mean direction, rank-4 shared
    structure, noise; verify draft rows keep 0.6 of the shared part), copied from its
    generate_input."""
    g = torch.Generator().manual_seed(5354)
    mean = torch.randn((HIDDEN,), generator=g).cuda()
    shared = torch.randn((4, HIDDEN), generator=g).cuda()
    coef = torch.randn((tokens, 4), generator=gen, device="cuda")
    scale = torch.ones((tokens // q_len, q_len, 1), device="cuda")
    scale[:, 1:] = 0.6
    scale = scale.reshape(tokens, 1)
    return (scale * (0.7 * mean + (0.9 / 4) ** 0.5 * coef @ shared)
            + (1 - 0.9 * scale * scale).sqrt()
            * torch.randn((tokens, HIDDEN), generator=gen, device="cuda"))


def _flip_probe(state):
    """Third probe: the judge's own layer routing, approximated. The FFN input is the attention
    boundary (the layer's own module) applied to reference.py's residual stream and a random
    attention output of RMS `amp`; the router is the judge's (prepare's weights); local_start
    is the 32-expert range the routing sends most pairs to, as reference.py's case_fields sets
    it. Experts alone, graph replay after an L2 flush, per seed: w4-run (metadata outside the
    graph, as _moe1_tune), w4-full, and the fused main variants. Routing stats: local routes /
    active experts / max routes per expert / experts over 64."""
    if not (state.get("moe1_ok") and state.get("w4_ok") and state.get("moe_ours")):
        print("FLIP skipped: moe1/w4 not enabled")
        return
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    mains = [v for v, (_, kind, _) in enumerate(_MOE1_VARIANTS) if kind == "main"]
    seeds = (11, 12, 13, 14, 15, 16)
    for amp in (0.25, 1.0):
        for bs in (128, 256):
            gen = torch.Generator(device="cuda").manual_seed(77 + bs)
            cands = ["w4-run", "w4-full"] + mains
            tab = {c: [] for c in cands}
            stats = []
            for seed in seeds:
                stream = _flip_stream(bs, 8, gen)
                attn = (amp * torch.randn((bs, HIDDEN), generator=gen, device="cuda")).to(
                    torch.bfloat16)
                ffn_input, _, _ = state["attn_boundary"](stream, attn, branch_dtype=torch.bfloat16,
                                                         router_dtype=None)
                logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
                logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
                tw, ids = state["routing"](None, logits, TOP_K, True)
                ranks = torch.bincount((ids.long() // 32).flatten(), minlength=N_EXPERTS // 32)
                start = int(ranks.argmax()) * 32
                local_start = torch.tensor([start], dtype=torch.int32, device="cuda")
                x = _pre_norm(ffn_input @ state["dense"]["pooler"].t(), state).contiguous()
                tw = tw.float().contiguous()
                ids = ids.to(torch.int32).contiguous()
                stats.append(_flip_stats(ids, start))
                for c in cands:
                    fused = c if isinstance(c, int) else None
                    opts = dict(_moe_opts(bs, state), fused=fused)
                    if c == "w4-full":
                        fn = (lambda o=opts: _experts(x, tw, ids, local_start, state, o))
                    else:
                        meta = _route_meta(tw, ids, local_start, bs, state, opts)
                        fn = (lambda m=meta: _experts_run(x, m, state))
                    tab[c].append(_flip_time(fn, flush))
            n = len(seeds)
            line = " ".join(f"{c if isinstance(c, str) else _moe1_name(c)}:"
                            f"{sum(v) / n:.1f}" for c, v in tab.items())
            print(f"FLIP3 judge-like amp={amp} T={bs} {line} us (mean of {n} seeds); routing "
                  f"local/active/max/over64 per seed " + " ".join(stats))
            for v in mains:
                d = [a - b for a, b in zip(tab[v], tab["w4-run"])]
                d2 = [a - b for a, b in zip(tab[v], tab["w4-full"])]
                print(f"FLIP3 judge-like amp={amp} T={bs} {_moe1_name(v)} minus w4-run "
                      + " ".join(f"{x:+.1f}" for x in d)
                      + "; minus w4-full " + " ".join(f"{x:+.1f}" for x in d2))

