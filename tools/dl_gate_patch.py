"""Apply the fresh-input replay gate to a layer file (user decision 2026-10-30, recorded in
docs/runs-decode-layer.md and docs/release-manifest.md).

    python3 tools/dl_gate_patch.py IN.py OUT.py [--m1 [--m1-runtime]]

Always: the MoE replay self-tests (_moe1_stress, _moe1_pdl_self_test) and the route-variant
replay check (_route_variant_check) give every replay new inputs, compare each replay with that
input's reference on the device, fail on an order-1 relative error, and require the
candidate's error spread to stay inside its control's envelope, in place of the 1e-4 bound on
the last replay; negative controls test the comparator on real outputs (report-only) and a
manifest line counts failures by class. Every other check is unchanged.

--m1 (IN must be the M1 file, 6377b5a or a descendant): the same gate for M1's replay stress
(M1 off as the control, M1 on as the candidate), M1 timing that always reports the
attention -> o -> boundary and boundary -> front chains and the whole step at T = 8 / 16 / 32
(sliding and full), the stop rule on the whole step at 8 tokens sliding, and a
fence.proxy.async.global after griddepcontrol.wait in the thread that issues the x copies
(docs/steering/2026-10-30-m1-ptx-ordering.md).

Always, too: W4 GEMM2's no-work early exit executes griddepcontrol.wait before returning.
"""
import sys

GATE_SRC = '''
# Fresh-input replay gate (user decisions 2026-10-30; docs/runs-decode-layer.md). The 1e-4 bound
# on one replay's output failed on FP32-atomics merge order alone (checks 6372154fcd7a4fa9,
# 3e2aa83358dd4f60). Order-1: an output's relative L2 error of GATE_ORDER1 or more fails
# outright; the element bound of the first version failed the M1-off layer itself
# (60cf58b69b75491e). Relative L2 cannot see one stale element, so a second per-call check
# bounds the element error in units of the row's largest magnitude by the control's envelope.
# Under GATE_BOUND (the old bound) and under one BF16 ULP of the row maximum always pass.
GATE_ORDER1 = 1e-2
GATE_BOUND = 1e-4
GATE_EL_FLOOR = 2.0 ** -7
GATE_SPREAD = 2.0
GATE_RATE_FLOOR = 0.01
GATE_BANK = 4
GATE_NEG = ("none", "tile", "1%", "0.1%", "row", "elem")


def _gate_refs(outs):
    """Stacked references over the bank for one output: values, L2 norms, and each row's
    largest magnitude (the element-error unit)."""
    r = torch.stack([o.float() for o in outs])
    norm = r.flatten(1).norm(dim=1).clamp(min=1e-12)
    rmax = r.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    return r, norm, rmax


def _gate_acc(n, dev):
    return {"rel": torch.zeros(n, device=dev), "el": torch.zeros(n, device=dev),
            "cnt": torch.zeros(5, dtype=torch.long, device=dev)}


def _gate_zero(acc):
    for t in acc.values():
        t.zero_()


def _gate_add(acc, outs, refs, i, ids=None):
    """Device-only (graph-capturable): per output the relative L2 error and the largest element
    error (in units of its row's largest reference magnitude) against bank entry i; counts
    calls, calls over GATE_BOUND, order-1 calls, non-finite calls and calls whose ids differ."""
    rels, els, bad = [], [], []
    for out, (r, norm, rmax) in zip(outs, refs):
        d = out.float() - r.index_select(0, i)[0]
        rel = d.norm() / norm.index_select(0, i)[0]
        el = (d.abs() / rmax.index_select(0, i)[0]).amax()
        bad.append(~torch.isfinite(rel) | ~torch.isfinite(el))
        rels.append(torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf"))))
        els.append(torch.where(torch.isfinite(el), el, torch.full_like(el, float("inf"))))
    rel, el = torch.stack(rels), torch.stack(els)
    torch.maximum(acc["rel"], rel, out=acc["rel"])
    torch.maximum(acc["el"], el, out=acc["el"])
    one = torch.ones_like(acc["cnt"][0])
    if ids is None:
        idf = torch.zeros_like(one)
    else:
        idf = (ids[0] != ids[1].index_select(0, i)[0]).any().long()
    acc["cnt"].add_(torch.stack((one, (rel.max() > GATE_BOUND).long(),
                                 (rel.max() >= GATE_ORDER1).long(),
                                 torch.stack(bad).any().long(), idf)))


def _gate_host(acc):
    cnt = acc["cnt"].tolist()
    return {"rel": acc["rel"].tolist(), "el": acc["el"].tolist(), "n": cnt[0], "over": cnt[1],
            "o1": cnt[2], "nf": cnt[3], "ids": cnt[4], "rate": cnt[1] / max(cnt[0], 1),
            "idr": cnt[4] / max(cnt[0], 1)}


def _gate_text(h):
    return (f"rel {'/'.join(f'{v:.1e}' for v in h['rel'])} el "
            f"{'/'.join(f'{v:.1e}' for v in h['el'])} >1e-4 {h['over']}/{h['n']} o1 {h['o1']}"
            + (f" nonfinite {h['nf']}" if h["nf"] else "")
            + (f" ids!= {h['ids']}" if h["ids"] else ""))


def _gate_short(cand, ctrl):
    """The SUMMARY form: candidate|control relative error per output, largest element error,
    calls over 1e-4, order-1 calls and ids flips."""
    c, k = _gate_host(cand), _gate_host(ctrl)
    return (f"rel {'/'.join(f'{v:.0e}' for v in c['rel'])}|"
            f"{'/'.join(f'{v:.0e}' for v in k['rel'])} el {max(c['el']):.0e}|{max(k['el']):.0e}"
            f" >1e-4 {c['over']}/{c['n']}|{k['over']}/{k['n']} o1 {c['o1']}|{k['o1']}"
            + (f" ids {c['ids']}|{k['ids']}" if c["ids"] or k["ids"] else ""))


def _gate_call_reasons(c, k):
    """The per-call criteria (those one call can fail), as (class, reason) pairs: non-finite,
    order-1 relative L2, and the element error over the control's envelope."""
    why = []
    if c["nf"]:
        why.append(("nonfinite", f"nonfinite {c['nf']}"))
    if c["o1"]:
        why.append(("replay", f"order-1 {c['o1']}"))
    el_lim = max(GATE_EL_FLOOR, GATE_SPREAD * max(k["el"]))
    if not max(c["el"]) <= el_lim:
        why.append(("replay", f"el {max(c['el']):.1e} > {el_lim:.1e}"))
    return why


def _gate_verdict(cand, ctrl):
    """(ok, text, classes). The per-call criteria on the candidate and the control alike (a
    control that fails them means the path itself is broken); then the envelope: the
    candidate's largest relative error under GATE_BOUND or within GATE_SPREAD times the
    control's, its share of calls over GATE_BOUND within GATE_SPREAD times the control's share
    (at least GATE_RATE_FLOOR), and its share of calls with differing ids within GATE_SPREAD
    times the control's plus two calls. Both arms make the same number of calls (a control
    with fewer samples has a lower maximum: 60cf58b69b75491e)."""
    c, k = _gate_host(cand), _gate_host(ctrl)
    why = [(cls, "cand " + r) for cls, r in _gate_call_reasons(c, k)]
    why += [(cls, "ctrl " + r) for cls, r in _gate_call_reasons(k, k)]
    lim = max(GATE_BOUND, GATE_SPREAD * max(k["rel"]))
    if not max(c["rel"]) <= lim:
        why.append(("replay", f"rel {max(c['rel']):.1e} > {lim:.1e}"))
    rate_lim = max(GATE_RATE_FLOOR, GATE_SPREAD * k["rate"])
    if not c["rate"] <= rate_lim:
        why.append(("replay", f"share over 1e-4 {c['rate']:.3f} > {rate_lim:.3f}"))
    if not c["ids"] <= GATE_SPREAD * k["idr"] * c["n"] + 2:
        why.append(("replay", f"ids flips {c['ids']}/{c['n']} against {k['ids']}/{k['n']}"))
    text = f"cand {_gate_text(c)} | ctrl {_gate_text(k)}"
    classes = sorted({cls for cls, _ in why})
    return (not why, text + ("" if not why else " FAIL " + "; ".join(r for _, r in why)),
            classes)


def _gate_negctl(obs, refs, i, j, ctrl, seed, elem_floor=None):
    """Negative controls of the comparator: obs is a real fresh output for bank entry i
    (rows x cols); each injection replaces part of it with entry j's reference, as a read of a
    previous generation would, and is judged by the per-call criteria against the measured
    control envelope. 'none' injects nothing (false rejection). With elem_floor, the single
    element is drawn among those where entries i and j differ by at least elem_floor of the
    row maximum: a stale element equal to the fresh one is harmless and no comparator sees it,
    and drawing among all elements misses with a few percent chance per output, which an
    enforced gate cannot afford. Returns (text, misses): kinds that should be rejected but
    pass, or 'none' rejected."""
    r, norm, rmax = refs
    dev = obs.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    stale = r[j]
    rows, cols = obs.shape
    k = _gate_host(ctrl)
    idx = torch.tensor([i], device=dev)
    items, misses = [], []
    for kind in GATE_NEG:
        o = obs.float().clone()
        flat, sflat = o.view(-1), stale.reshape(-1)
        if kind == "tile":
            c0 = (cols // 2) // 128 * 128
            o[:min(rows, 64), c0:c0 + 128] = stale[:min(rows, 64), c0:c0 + 128]
        elif kind == "row":
            o[rows // 2] = stale[rows // 2]
        elif kind == "elem":
            if elem_floor is None:
                e = int(torch.randint(0, flat.numel(), (1,), generator=gen, device=dev))
            else:
                far = ((stale.float() - r[i]).abs() >= elem_floor * rmax[i]).reshape(-1)
                cand = far.nonzero().flatten()
                pick = torch.randint(0, max(int(cand.numel()), 1), (1,), generator=gen,
                                     device=dev)
                e = int(cand[pick]) if cand.numel() else 0
            flat[e] = sflat[e]
        elif kind != "none":
            m = max(1, int((0.01 if kind == "1%" else 0.001) * flat.numel()))
            pos = torch.randperm(flat.numel(), generator=gen, device=dev)[:m]
            flat[pos] = sflat[pos]
        acc = _gate_acc(1, dev)
        _gate_add(acc, [o], [refs], idx)
        h = _gate_host(acc)
        why = _gate_call_reasons(h, k)
        rejected = bool(why)
        items.append(f"{kind}:{'R' if rejected else 'pass'}({h['rel'][0]:.0e},{h['el'][0]:.0e})")
        if rejected == (kind == "none"):
            misses.append(kind)
    return " ".join(items), misses


def _gate_fail(cls, text):
    """A gate failure in the manifest's classes: reference, replay, nonfinite, workspace,
    sensitivity (a negative control the comparator missed, report-only)."""
    _FAILED.append(f"[{cls}] {text}")


GATE_CLASSES = ("reference", "replay", "nonfinite", "workspace", "sensitivity")


def _gate_class(item):
    """A failure's class: the tag a gate wrote, else a keyword guess for older messages."""
    for start in range(len(item)):
        if item[start] == "[":
            end = item.find("]", start)
            tags = item[start + 1:end].split(",") if end > start else []
            if tags and all(t in GATE_CLASSES for t in tags):
                return tags
    low = item.lower()
    if "fp32" in low or "self-test" in low or "reference" in low:
        return ["reference"]
    if "scratch" in low or "generation" in low or "workspace" in low:
        return ["workspace"]
    if "non-finite" in low or "nonfinite" in low:
        return ["nonfinite"]
    return ["other"]


def _gate_manifest(state):
    """One line: MoE variants checked and rejected, route gates passed, failures per class."""
    built = [v[0] for v in _MOE1_VARIANTS] if state.get("moe1") else []
    passed = {_MOE1_VARIANTS[v][0] for v in state.get("moe1_passed", [])}
    pdl_bad = [x.split(" ")[0] for x in state.get("moe1_pdl_gate", []) if " FAIL" in x]
    rejected = [n for n in built if n not in passed] + [f"{n}(pdl)" for n in pdl_bad]
    routes = state.get("route_gate", {})
    rok = sum(1 for v in routes.values() if v.startswith("ok"))
    counts = {c: 0 for c in GATE_CLASSES + ("other",)}
    for item in _FAILED:
        for c in _gate_class(item):
            counts[c] += 1
    return (f"manifest moe checked {len(built)} rejected {rejected or 'none'}; route gates ok "
            f"{rok}/{len(routes)}; failures " + " ".join(f"{c} {n}" for c, n in counts.items()))


def _route_fresh_gate(front_moe, res, ffn_input, router_input, label, state):
    """The route variant's front + MoE under the fresh-input gate: GATE_BANK FFN inputs, each
    run once eagerly as its reference; the candidate is a CUDA graph of the step replayed
    ROUTE_STRESS times on the next input each replay (the PDL launch under graph replay, as
    judged), the control the same step made eagerly as often; MoE output and ids compared on
    the device. Restores the caller's input. Returns None or the failure text."""
    dev = ffn_input.device
    saved = ffn_input.clone()
    gen = torch.Generator(device=dev).manual_seed(4242 + ffn_input.shape[0])
    bank = torch.randn((GATE_BANK,) + tuple(ffn_input.shape), generator=gen, device=dev)
    bank = bank.to(ffn_input.dtype)

    def load(src):
        ffn_input.copy_(src)
        if router_input is not None:
            router_input.copy_(src.float())

    ys, ids = [], []
    for k in range(GATE_BANK):
        load(bank[k])
        front_moe()
        ys.append(res["y"].clone())
        ids.append(res["ids"].clone())
    refs = [_gate_refs(ys)]
    ref_ids = torch.stack(ids)
    sel = torch.zeros(1, dtype=torch.long, device=dev)

    def step(acc):
        sel.add_(1)
        i = sel % GATE_BANK
        load(bank.index_select(0, i)[0])
        front_moe()
        _gate_add(acc, [res["y"]], refs, i, ids=(res["ids"], ref_ids))

    cand, ctrl = _gate_acc(1, dev), _gate_acc(1, dev)
    graph = _graph_build(lambda: step(cand))
    _gate_zero(cand)
    for _ in range(ROUTE_STRESS):
        graph.replay()
    torch.cuda.synchronize()
    del graph
    for _ in range(ROUTE_STRESS):
        step(ctrl)
    load(bank[0])
    front_moe()
    obs = res["y"].clone()
    load(saved)
    torch.cuda.synchronize()
    neg, misses = _gate_negctl(obs, refs[0], 0, 1, ctrl, 91 + ffn_input.shape[0])
    state.setdefault("gate_negctl", {})[f"route {label}"] = neg
    if misses:
        _gate_fail("sensitivity", f"route {label}: negative controls {misses}")
    ok, text, classes = _gate_verdict(cand, ctrl)
    state.setdefault("route_gate", {})[label] = (
        f"{'ok' if ok else 'FAIL'} {_gate_short(cand, ctrl)}")
    return None if ok else f"[{','.join(classes)}] fresh-input gate: {text}"


def _moe1_stress(state, sizes, variant, replays):
    """The MoE calls for `sizes` captured back to back in one graph (as a sliding and a full
    layer run one after the other on the shared scratch) and replayed `replays` times. Before
    each replay's calls a device counter picks one of GATE_BANK input sets per call and copies
    them into the captured inputs; after the calls each output is compared on the device with
    that set's eager reference (_gate_add). The control is the same step made eagerly as many
    times. The grid barrier's generation must advance once per replayed launch,
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
    for _ in range(replays):
        step(ctrl)
    torch.cuda.synchronize()
    big = max(range(len(sizes)), key=lambda c: sizes[c])
    for (live, _), bank in zip(calls, banks):
        for dst, src in zip(live[:3], bank):
            dst.copy_(src[0])
    obs = _experts(*calls[big][0], state, calls[big][1])
    neg, misses = _gate_negctl(obs, refs[big], 0, 1, ctrl, 77 + variant)
    torch.cuda.synchronize()
    clean, msg = _moe1_check(state)
    ok, text, classes = _gate_verdict(cand, ctrl)
    if not clean or gens != len(sizes) * replays:
        classes = sorted(set(classes) | {"workspace"})
    name = _moe1_name(variant)
    state.setdefault("moe1_gate", {})[name] = (
        f"{'+'.join(map(str, sizes))} {_gate_short(cand, ctrl)}")
    state.setdefault("gate_negctl", {})[f"moe {name} T={sizes[big]}"] = neg
    if misses:
        _gate_fail("sensitivity", f"moe {name} T={sizes[big]}: negative controls {misses}")
    msg = (f"T={'+'.join(map(str, sizes))} {name} fresh-input gate: {replays} "
           f"replays (one host sync after them) against as many eager calls, "
           f"{GATE_BANK} input sets per call, per call size {text} generations={gens} {msg}")
    if not (ok and clean and gens == len(sizes) * replays):
        raise RuntimeError(f"[{','.join(classes)}] moe fused stress: {msg}")
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
        texts, shorts, ok, classes = [], [], True, set()
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
            good, text, cls = _gate_verdict(cand, ctrl)
            classes |= set(cls)
            ok = ok and good
            texts.append(f"T={tokens} {text}")
            shorts.append(f"T{tokens} {_gate_short(cand, ctrl)}")
        clean, msg = _moe1_check(state)
        if not clean:
            classes.add("workspace")
        ok = ok and clean
        lines.append(f"{name} {'ok' if ok else 'FAIL ' + msg} [{'; '.join(texts)}]")
        state.setdefault("moe1_pdl_gate", []).append(
            f"{name} {'ok' if ok else 'FAIL'} {'; '.join(shorts)}")
        if not ok:
            ctr[8] = 0
            bad.append(variant)
            _gate_fail(",".join(sorted(classes)),
                       f"moe1 pdl {name} ({'; '.join(texts)[-120:]}, {msg[:60]})")
    state["moe1_pdl_report"] = ", ".join(f"{_MOE1_VARIANTS[v][0]} FAIL" for v in bad) or "all ok"
    print("MOE1PDL vs plain launch after the route kernel (fresh-input gate): "
          + " || ".join(lines))
    if any(v in used for v in bad):
        raise RuntimeError("a fused variant the policy uses fails as the route kernel's PDL "
                           "secondary")


'''

M1_SRC = '''# The single-element negative control in the M1 gate is drawn among elements where the stale
# and fresh references differ by at least this fraction of the row maximum (_gate_negctl).
M1_NEG_ELEM = 0.125


def _m1_fresh(state, data, policy, bank, refs, replays):
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
    ids, and the candidate inside the control's envelope). Then the negative controls on an
    eager M1-on output of pair 0, hidden and stream, judged against that output's control
    envelope: every injection must be rejected and the uninjected output must pass, or the
    comparator has not shown it can see a stale read at this shape and M1 stays off. Returns
    per-shape lines with both envelopes; raises on a failure."""
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
            ok, text, classes = _gate_verdict(cand, ctrl)
            state["m1_policy"] = on
            data["stream"].copy_(bank_s[0])
            data["attn_input"].copy_(bank_a[0])
            out = [t.clone() for t in kernel_fn(data, state)[:2]]
            state["m1_policy"] = {}
            negs = []
            for k, name in enumerate(("hidden", "stream")):
                one = _gate_acc(2, dev)
                for c in ("rel", "el"):
                    one[c].copy_(ctrl[c][k].expand(2))
                one["cnt"].copy_(ctrl["cnt"])
                neg, misses = _gate_negctl(out[k].reshape(out[k].shape[0], -1), refs[k], 0, 1,
                                           one, 55 + 7 * k + batch * length, M1_NEG_ELEM)
                negs.append(f"{name} {neg}")
                if misses:
                    ok = False
                    classes = sorted(set(classes) | {"sensitivity"})
                    text += f" FAIL negative controls {name} {misses}"
            label = f"{batch}x{length}{layer[0]}"
            state.setdefault("gate_negctl", {})[f"m1 {label}"] = "; ".join(negs)
            lines.append(f"{label} {'ok' if ok else 'FAIL'} {text}")
            if not ok:
                raise RuntimeError(f"[{','.join(classes)}] M1 fresh-input gate: " + lines[-1])
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
        lines.append(f"moe pdl gate {item}")
    for name, item in state.get("route_gate", {}).items():
        lines.append(f"route gate {name} {item}")
    for name, item in state.get("gate_negctl", {}).items():
        lines.append(f"gate negctl {name}: {item}")
    lines.append(_gate_manifest(state))'''

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

# --m1-runtime: the PDL issue order (all first-ring weights before griddepcontrol.wait, x after
# it) only for launches made as PDL secondaries; a plain launch in the SK_PDL build keeps the
# release's order (stage 0 weights and x first) and skips the wait. Bench fdf75ba5a8124544
# measured M1's 3-5 us whole-step saving cancelled by the SK_PDL build's cost with M1 off. The
# flag rides in bit 30 of M, which is at most 32, so no launch_sk signature changes.
RUNTIME_REPS = [
    ("const int32_t* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, "
     "int* __restrict__ diag,",
     "const int32_t* __restrict__ ptab, int pmax, uint32_t base4, int Mf, int N, int K, "
     "int* __restrict__ diag,", 1),
    ("    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;\n"
     "    constexpr int NW = C::NW",
     "    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;\n"
     "    const int M = Mf & 0x3fffffff;\n"
     "    constexpr int NW = C::NW", 1),
    ("    const int npre = PDL ? (nst_tot < NSTAGE ? nst_tot : NSTAGE) : 1;",
     "    const bool pdl_run = PDL && (Mf >> 30) != 0;\n"
     "    const int npre = pdl_run ? (nst_tot < NSTAGE ? nst_tot : NSTAGE) : 1;", 1),
    ("    if (PDL && tid == 0) {", "    if (pdl_run && tid == 0) {", 1),
    ("    if constexpr (PDL) {\n        // Every thread waits",
     "    if (pdl_run) {\n        // Every thread waits", 1),
    ("            if constexpr (PDL) {\n                // This thread",
     "            if (pdl_run) {\n                // This thread", 1),
    ("            C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xmap, wsp, op, tp, pmax, base4, M, N, "
     "K, diag, rws, rpart,",
     "            C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xmap, wsp, op, tp, pmax, base4,\n"
     "                                              M | (1 << 30), N, K, diag, rws, rpart,", 1),
]


# The route-variant check's replay part moves to the fresh-input gate (user decision 2026-10-30);
# its comparisons against the base variant (top-8 sets, 1e-3 bounds) stay.
ROUTE_REPS = [
    ('''                if variant in ROUTE_VARIANTS and res.get("y") is not None:
                    for _ in range(ROUTE_STRESS):
                        graph.replay()
                    torch.cuda.synchronize()
                    outs[variant].append(res["y"].clone())
''', '''                if variant in ROUTE_VARIANTS and res.get("y") is not None:
                    outs[variant].append(_route_fresh_gate(
                        front_moe, res, ffn_input, router_input, f"{variant}@{tokens}", state))
'''),
    ('''    if len(got) > 3 and not _rel(got[3], y) < 1e-4:
        return f"MoE output moved {_rel(got[3], y):.1e} over {ROUTE_STRESS} replays"
''', '''    if len(got) > 3 and got[3] is not None:
        return got[3]
'''),
    ('''    1e-3, and after ROUTE_STRESS more replays of its graph the same output and a clean MoE
    scratch (no spin timeout)."""''', '''    1e-3, the fresh-input replay gate (_route_fresh_gate, ROUTE_STRESS replays against as many
    eager calls) and a clean MoE scratch (no spin timeout)."""'''),
    ('''# Extra replays of a route variant's front + MoE graph in the tuner, checked for a stable
# output and a clean MoE scratch: the PDL launch runs there under graph replay, as judged.''',
     '''# Replays of a route variant's front + MoE graph in the tuner under the fresh-input gate, and
# a clean MoE scratch after them: the PDL launch runs there under graph replay, as judged.'''),
]


# W4 GEMM2 is a PDL secondary that waits only before reading GEMM1's output. A CTA with no work
# (no local routes: about a third of 1-token calls) returned before that wait, so the kernel
# could complete without its predecessor having completed, which the layer's PDL chain
# assumes of every secondary (docs/review-pdl-secondaries-2026-10-30.md). Nothing is written
# on that path, so the wait only restores the chain.
W4_WAIT_OLD = """    int q = P ? blockIdx.x : blockIdx.y;
    if (q >= total) {
        return;
    }
"""
W4_WAIT_NEW = """    int q = P ? blockIdx.x : blockIdx.y;
    if (q >= total) {
        if constexpr (KIND == 2) {
            // Complete only after the predecessor has: later kernels rely on that chain.
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        return;
    }
"""


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
    for old, new in ROUTE_REPS:
        if text.count(old) != 1:
            raise SystemExit(f"route-gate patch does not apply: {old[:60]!r}")
        text = text.replace(old, new)
    if text.count(W4_WAIT_OLD) != 1:
        raise SystemExit("the W4 GEMM early exit is not where expected")
    text = text.replace(W4_WAIT_OLD, W4_WAIT_NEW)
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
        if "--m1-runtime" in sys.argv:
            for old, new, n in RUNTIME_REPS:
                if text.count(old) != n:
                    raise SystemExit(f"expected {n} of {old[:60]!r}, found {text.count(old)}")
                text = text.replace(old, new)
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
