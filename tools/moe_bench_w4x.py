
# Appended to bench.py by `python3 tools/moe_make_bench.py tools/moe_bench_w4x.py`: the W4
# GEMMs with weights by bulk copy (TMA, mode 7) and with large token tiles for heavily routed
# experts, against the policy prepare() tuned, per batch size, GEMM and routing seed. Every
# time is the median of 20 CUDA graph replays, each after a write that evicts L2, and includes
# the graph replay overhead (XB empty), the SwiGLU reduce (GEMM1 with K splits) or the
# finalizer (GEMM2), as the tuner times them.

XB_SEEDS = (("normal", 11), ("normal", 12), ("normal", 13), ("skewed", 21), ("skewed", 22))
XB_BS = (1, 8, 16, 32, 128, 256)
# Registers, shared memory and occupancy per config were printed by check 4a8852d513874318;
# off to keep the log tail inside its 16,000 characters.
XB_INFO = False


def _xb_cfg(t):
    return W4_CONFIGS.index(t)


# (label, small config, large-tile config or None, threshold in routes, smallest bs)
XB_G1 = [
    ("r8a", (1, 1, 8, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("r16w8l1", (1, 2, 16, 4, 8, 4, 0, 1, 0), None, 0, 1),
    ("tA", (1, 1, 8, 4, 4, 4, 7, 1, 0), None, 0, 1),
    ("tB", (1, 1, 8, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tC", (1, 1, 8, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tD", (1, 1, 8, 2, 8, 4, 7, 1, 0), None, 0, 1),
    ("tE", (1, 1, 8, 2, 4, 8, 7, 1, 0), None, 0, 1),
    ("tF", (1, 2, 16, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tG", (1, 2, 16, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tH", (1, 1, 1, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("u32", (1, 4, 32, 4, 4, 4, 0, 0, 0), None, 0, 32),
    ("u64", (1, 8, 64, 4, 8, 4, 0, 0, 0), None, 0, 32),
    ("h8+32", (1, 1, 8, 4, 4, 4, 0, 0, 0), (1, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+32", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+64", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 8, 64, 4, 8, 4, 0, 0, 0), 64, 32),
    ("h16+64d2", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 8, 64, 4, 4, 2, 0, 0, 0), 64, 32),
    ("h8s2+32", (1, 1, 8, 2, 8, 4, 0, 0, 0), (1, 4, 32, 2, 8, 4, 0, 0, 0), 32, 32),
]
XB_G2 = [
    ("r8a", (2, 1, 8, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("r16", (2, 2, 16, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("tA", (2, 1, 8, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tB", (2, 1, 8, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tC", (2, 1, 8, 2, 4, 8, 7, 1, 0), None, 0, 1),
    ("tD", (2, 2, 16, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tE", (2, 2, 16, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tF", (2, 1, 1, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("u32", (2, 4, 32, 4, 4, 4, 0, 0, 0), None, 0, 32),
    ("u64", (2, 8, 64, 4, 8, 4, 0, 0, 0), None, 0, 32),
    ("h8+32", (2, 1, 8, 4, 4, 4, 0, 0, 0), (2, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+32", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+64", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 8, 64, 4, 8, 4, 0, 0, 0), 64, 32),
    ("h16+64d2", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 8, 64, 4, 4, 2, 0, 0, 0), 64, 32),
]


def _xb_fmt(t):
    return f"{t:.0f}" if t >= 100 else f"{t:.1f}"


def _xb_info():
    ext = _w4_ext()
    seen = set()
    for lst in (XB_G1, XB_G2):
        for _, small, big, _, _ in lst:
            for t in (small, big):
                if t is None or t in seen:
                    continue
                seen.add(t)
                regs, smem, blocks, threads, local = ext.w4_info(_xb_cfg(t))
                print(f"XINFO {t}: regs {regs} smem {smem} blocks/SM {blocks} "
                      f"warps/SM {blocks * threads // 32} local {local}")


def _xb_run_bench(state):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    print(f"XB empty graph replay {_empty_graph_us(flush):.1f} us")
    if XB_INFO:
        _xb_info()
    w4 = state["w4"]
    ext = _w4_ext()
    policy = state["w4_policy"]
    worst = {}
    for bs in XB_BS:
        g1p, g2p = policy[_w4_key(bs)][:2]
        numel = bs * TOP_K
        bsm = _opts_for(bs)["bsm"]
        max_padded = (numel * bsm if numel < N_EXPERTS + 1
                      else numel + (N_EXPERTS + 1) * (bsm - 1))
        max_padded = (max_padded + 3) // 4 * 4
        times = [{}, {}]
        route = []
        for dist, seed in XB_SEEDS:
            data = _make_inputs(bs, 977 * seed + bs, dist)
            topk_ids = data["topk_ids"].to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            x = data["hidden_states"].contiguous()
            ids = topk_ids.view(-1)
            cnt = torch.bincount(ids[ids < 32].long(), minlength=32).tolist()
            live_c = [c for c in cnt if c]
            route.append(f"{dist[0]}{seed}:{len(live_c)}e/max{max(live_c)}/"
                         + "/".join(f"i{r}={sum((c + r - 1) // r for c in live_c)}"
                                    for r in (8, 16, 32, 64)))

            def lists(small, big, at, gemm):
                rows = 8 * small[1]
                brow = 8 * big[1] if big is not None else 0
                r = _ext().route_align(topk_ids, 0, 32, bsm, max_padded,
                                    rows if gemm == 1 else 0, rows if gemm == 2 else 0,
                                    brow if gemm == 1 else 0, brow if gemm == 2 else 0,
                                    max(at, 1), max(at, 1))
                sids, lids = r[0], r[3]
                its, n = (r[4], r[5]) if gemm == 1 else (r[6], r[7])
                bits, bn = (r[8], r[9]) if gemm == 1 else (r[10], r[11])
                b = (_xb_cfg(big), bits, bn) if big is not None else None
                return sids, lids, its, n, b

            inter_ref = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)
            cur1 = ("cur", W4_CONFIGS[g1p], None, 0, 1)
            cur2 = ("cur", W4_CONFIGS[g2p], None, 0, 1)
            for gemm, variants in ((1, [cur1] + XB_G1), (2, [cur2] + XB_G2)):
                ref = None
                for label, small, big, at, min_bs in variants:
                    if bs < min_bs or not _w4_fits(_xb_cfg(small), bs):
                        continue
                    if big is not None and not _w4_fits(_xb_cfg(big), bs):
                        continue
                    sids, lids, its, n, b = lists(small, big, at, gemm)
                    live = lids.view(-1) >= 0
                    ci = _xb_cfg(small)
                    res = {}
                    if gemm == 1:
                        out = inter_ref if label == "cur" else torch.zeros_like(inter_ref)

                        def run():
                            _w4_gemm1(w4, ci, x, sids, its, n, lids, numel, out, False, b)
                            res["o"] = out
                    else:
                        def run():
                            part = _w4_gemm2(w4, ci, inter_ref, sids, its, n, numel, False, b)
                            res["o"] = ext.w4_finalize(part, tw, state["norm_w"], lids.view(-1),
                                                       POST_NORM_EPS, False, 1)
                    try:
                        t = _graph_time(run, flush)
                        run()
                        got = res["o"][live] if gemm == 1 else res["o"]
                    except Exception as exc:
                        print(f"XB fail g{gemm} bs={bs} {label}: {type(exc).__name__}: {exc}")
                        continue
                    times[gemm - 1].setdefault(label, []).append(t)
                    if label == "cur":
                        ref = got.clone()
                    elif ref is not None:
                        err = _rel(got, ref) if torch.isfinite(got).all() else float("inf")
                        key = (gemm, label)
                        worst[key] = max(worst.get(key, 0.0), err)
        print(f"XROUTE bs={bs} " + " ".join(route))
        for k, cur in ((0, g1p), (1, g2p)):
            line = " ".join(f"{lab}:" + "/".join(_xb_fmt(t) for t in ts)
                            for lab, ts in times[k].items())
            print(f"XB g{k + 1} bs={bs} cur={cur} {line}")
        print(f"XERR bs={bs} " + " ".join(f"g{g}:{lab}:{e:.0e}" for (g, lab), e in worst.items()))
        worst = {}


def _w4_key(bs):
    return next((b for b in TUNE_BS if b >= bs), TUNE_BS[-1])


def _run_bench(state):
    if not state.get("w4_ok"):
        print("XB skipped: W4 path off")
        return
    _xb_run_bench(state)
