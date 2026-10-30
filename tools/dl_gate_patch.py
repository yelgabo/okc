"""Apply the fresh-input replay gate to a layer file (user decision 2026-10-30, recorded in
docs/runs-decode-layer.md and docs/release-manifest.md).

    python3 tools/dl_gate_patch.py IN.py OUT.py [--m1]

Always: the MoE replay self-tests (_moe1_stress, _moe1_pdl_self_test) give every replay new
inputs, compare each replay with that input's reference on the device, fail on any order-1
error, and require the candidate's error spread to stay inside its control's envelope, in
place of the 1e-4 bound on the last replay. Every other check is unchanged.

--m1 (IN must be the M1 file, 6377b5a or a descendant): the same gate for M1's replay stress
(M1 off as the control, M1 on as the candidate), M1 timing that always reports the
attention -> o -> boundary and boundary -> front chains and the whole step at T = 8 / 16 / 32
(sliding and full), the stop rule on the whole step at 8 tokens sliding, and a
fence.proxy.async.global after griddepcontrol.wait in the thread that issues the x copies
(docs/steering/2026-10-30-m1-ptx-ordering.md).
"""
import sys

GATE_SRC = '''
# Fresh-input replay gate (user decision 2026-10-30; docs/runs-decode-layer.md). The 1e-4 bound
# on one replay's output failed on FP32-atomics merge order alone (checks 6372154fcd7a4fa9,
# 3e2aa83358dd4f60). A read of stale rows gives order-1 errors, so any element error of
# GATE_ORDER1 fails outright; below that the candidate must stay inside GATE_SPREAD times its
# control's measured spread (under GATE_BOUND always passes, as the old bound did).
GATE_ORDER1 = 1e-2
GATE_BOUND = 1e-4
GATE_SPREAD = 2.0
GATE_RATE_FLOOR = 0.01
GATE_BANK = 4
GATE_CTRL_CALLS = 400


def _gate_refs(outs):
    """Stacked references over the bank for one output: values, L2 norms, and the per-element
    scale |r| + row RMS. A one-ULP BF16 flip then counts at most 2^-7 (7.8e-3) and about
    4e-3 on most elements, and a read of another input's rows, even one row or one tile of a
    row, comes out near 0.3-1."""
    r = torch.stack([o.float() for o in outs])
    norm = r.flatten(1).norm(dim=1).clamp(min=1e-12)
    scale = r.abs() + r.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-12)
    return r, norm, scale


def _gate_acc(n, dev):
    return {"rel": torch.zeros(n, device=dev), "el": torch.zeros(n, device=dev),
            "cnt": torch.zeros(4, dtype=torch.long, device=dev)}


def _gate_zero(acc):
    for t in acc.values():
        t.zero_()


def _gate_add(acc, outs, refs, i, ids=None):
    """Device-only (graph-capturable): per output the relative L2 error and the largest element
    error against bank entry i; counts calls, calls over GATE_BOUND, order-1 calls (element or
    relative error >= GATE_ORDER1, or non-finite), and calls whose ids differ."""
    rels, els = [], []
    for out, (r, norm, scale) in zip(outs, refs):
        d = out.float() - r.index_select(0, i)[0]
        rel = d.norm() / norm.index_select(0, i)[0]
        el = (d.abs() / scale.index_select(0, i)[0]).amax()
        rels.append(torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf"))))
        els.append(torch.where(torch.isfinite(el), el, torch.full_like(el, float("inf"))))
    rel, el = torch.stack(rels), torch.stack(els)
    torch.maximum(acc["rel"], rel, out=acc["rel"])
    torch.maximum(acc["el"], el, out=acc["el"])
    worst = torch.maximum(rel.max(), el.max())
    flags = [torch.ones_like(acc["cnt"][0]), (rel.max() > GATE_BOUND).long(),
             (worst >= GATE_ORDER1).long()]
    if ids is None:
        flags.append(torch.zeros_like(acc["cnt"][0]))
    else:
        flags.append((ids[0] != ids[1].index_select(0, i)[0]).any().long())
    acc["cnt"].add_(torch.stack(flags))


def _gate_host(acc):
    cnt = acc["cnt"].tolist()
    return {"rel": acc["rel"].tolist(), "el": acc["el"].tolist(), "n": cnt[0], "over": cnt[1],
            "o1": cnt[2], "ids": cnt[3], "rate": cnt[1] / max(cnt[0], 1)}


def _gate_text(h):
    return (f"rel {'/'.join(f'{v:.1e}' for v in h['rel'])} el "
            f"{'/'.join(f'{v:.1e}' for v in h['el'])} >1e-4 {h['over']}/{h['n']} o1 {h['o1']}"
            + (f" ids!= {h['ids']}" if h["ids"] else ""))


def _gate_short(cand, ctrl):
    """The SUMMARY form: candidate|control relative error per output, largest element error,
    calls over 1e-4 and order-1 calls."""
    c, k = _gate_host(cand), _gate_host(ctrl)
    return (f"rel {'/'.join(f'{v:.0e}' for v in c['rel'])}|"
            f"{'/'.join(f'{v:.0e}' for v in k['rel'])} el {max(c['el']):.0e}|{max(k['el']):.0e}"
            f" >1e-4 {c['over']}/{c['n']}|{k['over']}/{k['n']} o1 {c['o1']}|{k['o1']}")


def _gate_verdict(cand, ctrl):
    """(ok, text). Fails on any order-1 or ids-differing call in either arm; per output the
    candidate's largest relative error must be under GATE_BOUND or within GATE_SPREAD times the
    control's, and its share of calls over GATE_BOUND within GATE_SPREAD times the control's
    share (at least GATE_RATE_FLOOR). The text records both arms' envelopes."""
    c, k = _gate_host(cand), _gate_host(ctrl)
    why = []
    if c["o1"] or k["o1"]:
        why.append(f"order-1 cand {c['o1']} ctrl {k['o1']}")
    if c["ids"] or k["ids"]:
        why.append(f"ids differ cand {c['ids']} ctrl {k['ids']}")
    for j, (cr, kr) in enumerate(zip(c["rel"], k["rel"])):
        lim = max(GATE_BOUND, GATE_SPREAD * kr)
        if not cr <= lim:
            why.append(f"out{j} rel {cr:.1e} > {lim:.1e}")
    rate_lim = max(GATE_RATE_FLOOR, GATE_SPREAD * k["rate"])
    if not c["rate"] <= rate_lim:
        why.append(f"share over 1e-4 {c['rate']:.3f} > {rate_lim:.3f}")
    text = f"cand {_gate_text(c)} | ctrl {_gate_text(k)}"
    return not why, text + ("" if not why else " FAIL " + "; ".join(why))


def _moe1_stress(state, sizes, variant, replays):
    """The MoE calls for `sizes` captured back to back in one graph (as a sliding and a full
    layer run one after the other on the shared scratch) and replayed `replays` times. Before
    each replay's calls a device counter picks one of GATE_BANK input sets per call and copies
    them into the captured inputs; after the calls each output is compared on the device with
    that set's eager reference (_gate_add). The control is the same step made eagerly
    GATE_CTRL_CALLS times. The grid barrier's generation must advance once per replayed launch,
    the scratch invariant must hold, and _gate_verdict must pass. Raises on any failure."""
    s = state["moe1"]["scratch"]
    dev = s["ctr"].device
    calls, banks, refs = [], [], []
    for i, tokens in enumerate(sizes):
        bank = [_moe_routing(tokens, 4000 + tokens + 7 * i + 1000 * k, 96, True)
                for k in range(GATE_BANK)]
        live = [t.clone() for t in bank[0]]
        opts = dict(_moe_opts(tokens, state), fused=variant)
        calls.append((live, opts))
        banks.append([torch.stack([b[j] for b in bank]) for j in range(3)])
        outs = []
        for b in bank:
            outs.append(_experts(*b[:4], state, opts).clone())
        refs.append(_gate_refs(outs))
    torch.cuda.synchronize()
    sel = torch.zeros(1, dtype=torch.long, device=dev)

    def step(acc):
        sel.add_(1)
        i = sel % GATE_BANK
        for (live, _), bank in zip(calls, banks):
            for dst, src in zip(live[:3], bank):
                dst.copy_(src.index_select(0, i)[0])
        outs = [_experts(*live, state, opts) for live, opts in calls]
        _gate_add(acc, outs, refs, i)

    cand, ctrl = _gate_acc(len(sizes), dev), _gate_acc(len(sizes), dev)
    graph = _graph_build(lambda: step(cand))
    _gate_zero(cand)
    torch.cuda.synchronize()
    gen0 = int(s["ctr"][3])
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    gens = int(s["ctr"][3]) - gen0
    del graph
    for _ in range(GATE_CTRL_CALLS):
        step(ctrl)
    torch.cuda.synchronize()
    clean, msg = _moe1_check(state)
    ok, text = _gate_verdict(cand, ctrl)
    state.setdefault("moe1_gate", {})[_moe1_name(variant)] = (
        f"{'+'.join(map(str, sizes))} {_gate_short(cand, ctrl)}")
    msg = (f"T={'+'.join(map(str, sizes))} {_moe1_name(variant)} fresh-input gate: {replays} "
           f"replays (one host sync after them) against {GATE_CTRL_CALLS} eager calls, "
           f"{GATE_BANK} input sets per call, per call size {text} generations={gens} {msg}")
    if not (ok and clean and gens == len(sizes) * replays):
        raise RuntimeError(f"moe fused stress: {msg}")
    return msg
'''

PDL_SRC = '''def _moe1_pdl_self_test(state):
    """Every fused variant that passed its FP32 self-test, launched as the route kernel's PDL
    secondary (the preamble initialises the ring's mbarriers before griddepcontrol.wait and the
    routing section skips them), against the plain launch, under the fresh-input gate: per
    token count GATE_BANK (logits, pooled) sets, each run once with the plain launch as its
    reference; the candidate is the PDL launch, eagerly on every set and as a CUDA graph
    replayed ROUTE_PDL_REPLAYS times on the next set each replay, checked on the device; the
    control is the plain launch made eagerly as often. _gate_verdict decides, and the scratch
    must stay clean. A variant the policy uses that fails raises (the route variants are then
    off); any other failing variant is reported."""
    dev = state["pre_expert_w"].device
    gen = torch.Generator(device=dev).manual_seed(78)
    ctr = state["moe1"]["scratch"]["ctr"]
    used = set((state.get("moe1_policy") or {}).values())
    local_start = torch.tensor([96], dtype=torch.int32, device=dev)
    bad, lines = [], []
    for variant in state.get("moe1_passed", []):
        name, kind, _ = _MOE1_VARIANTS[variant]
        texts, shorts, ok = [], [], True
        for tokens in ((8, 16) if kind == "skinny" else (32, 128, 256)):
            bank_l = 4.0 * torch.randn((GATE_BANK, tokens, N_EXPERTS), generator=gen, device=dev)
            bank_p = torch.randn((GATE_BANK, tokens, D_EXPERT), generator=gen, device=dev)
            bank_p = bank_p.to(torch.bfloat16)
            logits, pooled = bank_l[0].clone(), bank_p[0].clone()

            def run(pdl):
                x, ids, w = _route_fwd(logits, 0, pooled, state)
                return _moe1_run(x, w, ids, local_start, state, variant, pdl)

            outs = []
            for k in range(GATE_BANK):
                logits.copy_(bank_l[k])
                pooled.copy_(bank_p[k])
                outs.append(run(False).clone())
            refs = [_gate_refs(outs)]
            sel = torch.zeros(1, dtype=torch.long, device=dev)

            def step(acc, pdl):
                sel.add_(1)
                i = sel % GATE_BANK
                logits.copy_(bank_l.index_select(0, i)[0])
                pooled.copy_(bank_p.index_select(0, i)[0])
                _gate_add(acc, [run(pdl)], refs, i)

            cand, ctrl = _gate_acc(1, dev), _gate_acc(1, dev)
            for _ in range(GATE_BANK):
                step(cand, True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step(cand, True)
            for _ in range(ROUTE_PDL_REPLAYS):
                graph.replay()
            torch.cuda.synchronize()
            del graph
            for _ in range(ROUTE_PDL_REPLAYS + GATE_BANK):
                step(ctrl, False)
            torch.cuda.synchronize()
            good, text = _gate_verdict(cand, ctrl)
            ok = ok and good
            texts.append(f"T={tokens} {text}")
            shorts.append(f"T{tokens} {_gate_short(cand, ctrl)}")
        clean, msg = _moe1_check(state)
        ok = ok and clean
        lines.append(f"{name} {'ok' if ok else 'FAIL ' + msg} [{'; '.join(texts)}]")
        state.setdefault("moe1_pdl_gate", []).append(
            f"{name} {'ok' if ok else 'FAIL'} {'; '.join(shorts)}")
        if not ok:
            ctr[8] = 0
            bad.append(variant)
            _FAILED.append(f"moe1 pdl {name} ({'; '.join(texts)[-120:]}, {msg[:60]})")
    state["moe1_pdl_report"] = ", ".join(f"{_MOE1_VARIANTS[v][0]} FAIL" for v in bad) or "all ok"
    print("MOE1PDL vs plain launch after the route kernel (fresh-input gate): "
          + " || ".join(lines))
    if any(v in used for v in bad):
        raise RuntimeError("a fused variant the policy uses fails as the route kernel's PDL "
                           "secondary")


'''

M1_SRC = '''def _m1_fresh(state, data, policy, bank, refs, replays):
    """kernel_fn under `policy`, captured with a device counter that copies the next of the
    bank's (stream, attn_input) pairs into the captured inputs before each replay; each
    replay's hidden and stream outputs and ids are compared on the device with that pair's
    M1-off eager reference (_gate_add)."""
    dev = data["stream"].device
    bank_s, bank_a = bank
    acc = _gate_acc(2, dev)
    sel = torch.zeros(1, dtype=torch.long, device=dev)
    saved = dict(state.get("m1_policy", {}))
    state["m1_policy"] = policy

    def run():
        sel.add_(1)
        i = sel % bank_s.shape[0]
        data["stream"].copy_(bank_s.index_select(0, i)[0])
        data["attn_input"].copy_(bank_a.index_select(0, i)[0])
        out = kernel_fn(data, state)
        _gate_add(acc, out[:2], refs[:2], i, ids=(out[2], refs[2]))

    try:
        graph = _graph_build(run)
        _gate_zero(acc)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        del graph
    finally:
        state["m1_policy"] = saved
    return acc


def _m1_stress(state):
    """The fresh-input gate (user decision 2026-10-30) at every scored T <= 32 shape: five
    (stream, attn_input) pairs, each run eagerly with M1 off as its reference; then M1 off (the
    control) and M1 on (o and front) captured and replayed M1_STRESS_REPLAYS times each, every
    replay on the next pair and checked on the device. _gate_verdict decides (order-1 errors,
    ids, and the candidate inside the control's envelope). Returns per-shape lines with both
    envelopes; raises on a failure."""
    lines = []
    saved = dict(state.get("m1_policy", {}))
    on = {b: "o+front" for b in M1_BUCKETS}
    try:
        for batch, length, layer in ((1, 1, "sliding"), (1, 1, "full"), (1, 8, "sliding"),
                                     (1, 8, "full"), (16, 1, "sliding"), (16, 1, "full"),
                                     (32, 1, "sliding"), (32, 1, "full")):
            data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            dev = data["stream"].device
            gen = torch.Generator(device=dev).manual_seed(777 + batch * length)
            bank_s = torch.randn((5, batch * length, HIDDEN), generator=gen, device=dev)
            bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                      ).to(torch.bfloat16)
            state["m1_policy"] = {}
            hs, ss, ids = [], [], []
            for k in range(5):
                data["stream"].copy_(bank_s[k])
                data["attn_input"].copy_(bank_a[k])
                out = kernel_fn(data, state)
                hs.append(out[0].clone())
                ss.append(out[1].clone())
                ids.append(out[2].clone())
            refs = [_gate_refs(hs), _gate_refs(ss), torch.stack(ids)]
            ctrl = _m1_fresh(state, data, {}, (bank_s, bank_a), refs, M1_STRESS_REPLAYS)
            cand = _m1_fresh(state, data, on, (bank_s, bank_a), refs, M1_STRESS_REPLAYS)
            ok, text = _gate_verdict(cand, ctrl)
            lines.append(f"{batch}x{length}{layer[0]} {'ok' if ok else 'FAIL'} {text}")
            if not ok:
                raise RuntimeError("M1 fresh-input gate: " + lines[-1])
    finally:
        state["m1_policy"] = saved
    return lines


def _m1_pair(state, graphs, flush, rounds):
    """Interleaved rounds of cold-L2 graph medians; the median per key in us."""
    times = {k: [] for k in graphs}
    for r in range(rounds):
        keys = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
        for k in keys:
            times[k].append(_graph_median(graphs[k], flush, 10))
    return {k: sorted(v)[len(v) // 2] for k, v in times.items()}


def _m1_chain(state, batch, length, layer, flush, rounds=9):
    """attention -> o -> attention boundary as a CUDA graph, o plain against o as a PDL
    secondary, cold L2; returns (plain, pdl) in us."""
    data = _timeline_case(state, batch, length, layer, 9500 + batch * length)
    cfg = data["config"]
    tokens = cfg["tokens"]
    qkv = _proj(data["attn_input"], "qkv_gate", state["dense"]["qkv_gate"].shape[0], state)
    path = _attn_path(cfg, state)

    def chain():
        o = path(qkv, data, state)
        a = _proj(o, "o", HIDDEN, state)
        if state["abnd_ours"]:
            _abnd(data["stream"], a, tokens <= ROUTER_FP32_MAX_TOKENS, state,
                  _abnd_geo(tokens, state))
        else:
            state["attn_boundary"](data["stream"], a, branch_dtype=torch.bfloat16,
                                   router_dtype=None)

    saved = dict(state.get("m1_policy", {}))
    graphs = {}
    try:
        for mode in ("off", "o"):
            state["m1_policy"] = {b: mode for b in M1_BUCKETS}
            graphs[mode] = _graph_build(chain)
    finally:
        state["m1_policy"] = saved
    med = _m1_pair(state, graphs, flush, rounds)
    return med["off"], med["o"]


def _m1_front_chain(state, batch, length, layer, flush, rounds=9):
    """attention boundary -> MoE front (the skf pooler + router and, for skf-r, the route
    kernel) as a CUDA graph, the front plain against the front as a PDL secondary, cold L2;
    (plain, pdl) in us, or None where the front variant is not skf."""
    data = _timeline_case(state, batch, length, layer, 9600 + batch * length)
    cfg = data["config"]
    tokens = cfg["tokens"]
    variant = _front_variant(tokens, state)
    if variant not in ("skf", "skf-r"):
        return None
    qkv = _proj(data["attn_input"], "qkv_gate", state["dense"]["qkv_gate"].shape[0], state)
    a = _proj(_attn_path(cfg, state)(qkv, data, state), "o", HIDDEN, state)
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    moe_opts = _moe_opts(tokens, state) if state["moe_ours"] else None

    def chain():
        if state["abnd_ours"]:
            ffn_input, _, router_input = _abnd(data["stream"], a, not split_router, state,
                                               _abnd_geo(tokens, state))
        else:
            ffn_input, _, router_input = state["attn_boundary"](
                data["stream"], a, branch_dtype=torch.bfloat16,
                router_dtype=None if split_router else torch.float32)
        _moe_front(ffn_input, router_input, data["local_start"], state, moe_opts, variant)

    saved = dict(state.get("m1_policy", {}))
    graphs = {}
    try:
        for mode in ("off", "o+front"):
            state["m1_policy"] = {b: mode for b in M1_BUCKETS}
            graphs[mode] = _graph_build(chain)
    finally:
        state["m1_policy"] = saved
    med = _m1_pair(state, graphs, flush, rounds)
    return med["off"], med["o+front"]


def _m1_tune(state):
    """M1 per token bucket: the self-test with the bit set and the fresh-input gate first (a
    failure blocks M1), then always the attention -> o -> boundary and boundary -> front chains
    and the whole step (kernel_fn, 46 x sliding + 16 x full, cold L2, interleaved rounds) for
    off, o and o + front per bucket; the fastest load wins per bucket. Stop rule: under
    M1_CHAIN_GATE_US saved on the whole step at 8 tokens sliding closes M1 (policy empty)."""
    state["m1_report"] = []
    rep = state["m1_report"]
    checked = _m1_selftest(state)
    rep.append(f"selftest PDL = plain bitwise on {len(checked)} launches")
    state["m1_ok"] = True
    for line in _m1_stress(state):
        rep.append("gate " + line)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    shapes = ((8, 1, 8), (16, 16, 1), (32, 32, 1))
    for layer in ("sliding", "full"):
        parts = []
        for t, batch, length in shapes:
            a, b = _m1_chain(state, batch, length, layer, flush)
            fr = _m1_front_chain(state, batch, length, layer, flush)
            parts.append(f"T={t} attn-o-bnd {a:.1f}/{b:.1f} ({a - b:+.2f})"
                         + ("" if fr is None else
                            f" bnd-front {fr[0]:.1f}/{fr[1]:.1f} ({fr[0] - fr[1]:+.2f})"))
        rep.append(f"chain {layer[0]} plain/pdl us: " + "; ".join(parts))
    policy, v1s_saving = {}, 0.0
    for bucket, (_, batch, length) in zip(M1_BUCKETS, shapes):
        datas = {layer: _timeline_case(state, batch, length, layer, 9700 + batch * length)
                 for layer in ("sliding", "full")}
        graphs = {}
        saved = dict(state.get("m1_policy", {}))
        try:
            for mode in ("off", "o", "o+front"):
                state["m1_policy"] = {bucket: mode}
                for layer, data in datas.items():
                    graphs[(mode, layer)] = _graph_build(lambda d=data: kernel_fn(d, state))
        finally:
            state["m1_policy"] = saved
        med = _m1_pair(state, graphs, flush, 9)
        del graphs
        load = {mode: (46 * med[(mode, "sliding")] + 16 * med[(mode, "full")]) / 62
                for mode in ("off", "o", "o+front")}
        best = min(load, key=load.get)
        policy[bucket] = best
        if bucket == M1_BUCKETS[0]:
            v1s_saving = med[("off", "sliding")] - min(med[(m, "sliding")] for m in ("o", "o+front"))
        rep.append(f"T={bucket} step s/f " + " ".join(
            f"{m} {med[(m, 'sliding')]:.1f}/{med[(m, 'full')]:.1f}" for m in load)
            + " load " + " ".join(f"{m} {t:.1f}" for m, t in load.items()) + f" -> {best}")
    if v1s_saving < M1_CHAIN_GATE_US:
        state["m1_policy"] = {}
        rep.append(f"closed: whole step at 8 tokens sliding saves {v1s_saving:.2f} us < "
                   f"{M1_CHAIN_GATE_US} (policy would have been {policy})")
        return
    state["m1_policy"] = policy
    rep.append(f"open: whole step at 8 tokens sliding saves {v1s_saving:.2f} us")


'''

SUMMARY_OLD = '''    lines.append("moe pdl " + state.get("moe1_pdl_report", "not run"))'''
SUMMARY_NEW = SUMMARY_OLD + '''
    for name, item in state.get("moe1_gate", {}).items():
        lines.append(f"moe gate {name} {item}")
    for item in state.get("moe1_pdl_gate", []):
        lines.append(f"moe pdl gate {item}")'''

FENCE_OLD = '''        asm volatile("griddepcontrol.wait;" ::: "memory");
        if (tid == 0) {
            for (int st = 0; st < npre; ++st) {
                issue_x(st);'''
FENCE_NEW = '''        asm volatile("griddepcontrol.wait;" ::: "memory");
        if (tid == 0) {
            // The predecessor wrote x with generic stores; the copies below read it through
            // the async proxy. The wait gives the cross-grid order; this fence puts the proxy
            // fence the memory model asks for between them in this thread
            // (docs/steering/2026-10-30-m1-ptx-ordering.md).
            asm volatile("fence.proxy.async.global;" ::: "memory");
            for (int st = 0; st < npre; ++st) {
                issue_x(st);'''


PRODUCER_OLD = '''    if (warp == NW) {
        if (lane == 0) {
            bool dead = false;
            for (int st = npre; st < nst_tot; ++st) {'''
PRODUCER_NEW = '''    if (warp == NW) {
        if (lane == 0) {
            if constexpr (PDL) {
                // This thread's own x copies (stages npre on) also read the predecessor's
                // generic stores through the async proxy, so it fences after its wait too.
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
            bool dead = false;
            for (int st = npre; st < nst_tot; ++st) {'''


def cut(text, start, end, new):
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + new + text[b:]


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    text = cut(text, "def _moe1_stress(state, sizes, variant, replays):", "def _moe1_tune(",
               GATE_SRC.lstrip("\n") + "\n\n")
    text = cut(text, "def _moe1_pdl_self_test(state):", "def _front_tune(", PDL_SRC)
    if text.count(SUMMARY_OLD) != 1:
        raise SystemExit("the SUMMARY moe pdl line is not where expected")
    text = text.replace(SUMMARY_OLD, SUMMARY_NEW)
    if "--m1" in sys.argv:
        if text.count(FENCE_OLD) != 1:
            raise SystemExit("the SK_PDL x-issue block is not where expected")
        text = text.replace(FENCE_OLD, FENCE_NEW)
        if text.count(PRODUCER_OLD) != 1:
            raise SystemExit("the sk producer loop is not where expected")
        text = text.replace(PRODUCER_OLD, PRODUCER_NEW)
        text = cut(text, "def _m1_stress(state):", "def _m1_enable(state):", M1_SRC)
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
