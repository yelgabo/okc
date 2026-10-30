"""Build the M1 stage diagnostic from a gated M1 file (tools/dl_gate_patch.py --m1 output).

    python3 tools/dl_m1_stage_diag.py GATED_M1.py OUT.py

Check 6ce5be0ba29642ea stopped M1 at its first gate shape (decode 1 sliding): the M1-off control
replays differed from their eager references (relative 2.7e-3 / 2.5e-3, reason cut off in the
log tail) while the M1-on replays were bitwise equal to them. This build never raises in the
gate: per T <= 32 shape it replays kernel_fn's stages (qkv, attention, o, the boundary's FFN
input, the MoE input, the routing weights, the MoE output, the unpooler, the next input, the
stream) on fresh inputs for four arms in order: M1 off, M1 off again, M1 on, and eager M1-off
calls, each against the same eager M1-off references, and prints per stage the largest
relative error, order-1 calls, ids mismatches and the first replay index of each. Then the M1
timing (chains and whole step) runs as in the candidate, and the policy is left empty.
"""
import sys

STAGES = '''
M1_DIAG_REPLAYS = 1000
M1_DIAG_STAGES = ("qkv", "attn", "o", "ffn", "xmoe", "tw", "moe", "unp", "hid", "str")


def _kfn_stages(data, state):
    """kernel_fn's body, returning its intermediates (no copies, so the launch order and the
    PDL adjacency are the timed path's) and the ids. The residual stream is updated in place,
    so only its final value is a stage."""
    cfg = data["config"]
    tokens = cfg["tokens"]
    dense = state["dense"]
    qkv = _proj(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], state)
    if state["prefetch_ours"] and tokens <= SLICED_UNPOOL_MAX_TOKENS:
        side_stream = state["side_stream"]
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _ext().prefetch_l2(*_prefetch_set(tokens, state))
    o = _attn_path(cfg, state)(qkv, data, state)
    attn_output = _proj(o, "o", HIDDEN, state)
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    if state["abnd_ours"]:
        ffn_input, stream, router_input = _abnd(data["stream"], attn_output, not split_router,
                                                state, _abnd_geo(tokens, state))
    else:
        ffn_input, stream, router_input = state["attn_boundary"](
            data["stream"], attn_output, branch_dtype=torch.bfloat16,
            router_dtype=None if split_router else torch.float32)
    moe_opts = _moe_opts(tokens, state) if state["moe_ours"] else None
    expert_input, topk_weights, topk_ids, meta = _moe_front(
        ffn_input, router_input, data["local_start"], state, moe_opts,
        _front_variant(tokens, state))
    if meta is not None:
        expert_output = _experts_run(expert_input, meta, state)
    else:
        expert_output = _experts(_pre_norm(expert_input, state), topk_weights, topk_ids,
                                 data["local_start"], state, moe_opts)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _proj(expert_output, "unpooler", UNPOOL_SLICE, state)
        if state["bnd_ours"]:
            nxt, stream = _bnd(stream, unpooled, state)
        else:
            nxt, stream, _ = state["ffn_boundary"](
                stream, F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE)), branch_dtype=torch.bfloat16)
    else:
        unpooled = _proj(expert_output, "unpooler", HIDDEN, state)
        nxt, stream, _ = state["ffn_boundary"](stream, unpooled, branch_dtype=torch.bfloat16)
    outs = [qkv, o, attn_output, ffn_input, expert_input, topk_weights, expert_output, unpooled,
            nxt, stream]
    return outs, topk_ids


def _stage_acc(n, dev):
    big = torch.full((n,), 1 << 40, dtype=torch.long, device=dev)
    return {"rel": torch.zeros(n, device=dev), "o1": torch.zeros(n, dtype=torch.long, device=dev),
            "over": torch.zeros(n, dtype=torch.long, device=dev), "f_over": big.clone(),
            "f_o1": big.clone(), "ids": torch.zeros(2, dtype=torch.long, device=dev)}


def _stage_add(acc, outs, refs, ids, ref_ids, i, sel):
    rels, o1s = [], []
    for out, (r, norm, scale) in zip(outs, refs):
        d = out.float() - r.index_select(0, i)[0]
        rel = d.norm() / norm.index_select(0, i)[0]
        rel = torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf")))
        el = (d.abs() / scale.index_select(0, i)[0]).amax()
        el = torch.where(torch.isfinite(el), el, torch.full_like(el, float("inf")))
        rels.append(rel)
        o1s.append(torch.maximum(rel, el) >= GATE_ORDER1)
    rel = torch.stack(rels)
    o1 = torch.stack(o1s)
    over = rel > GATE_BOUND
    torch.maximum(acc["rel"], rel, out=acc["rel"])
    acc["o1"].add_(o1.long())
    acc["over"].add_(over.long())
    now = sel.expand_as(acc["f_over"])
    torch.minimum(acc["f_over"], torch.where(over, now, acc["f_over"]), out=acc["f_over"])
    torch.minimum(acc["f_o1"], torch.where(o1, now, acc["f_o1"]), out=acc["f_o1"])
    bad = (ids != ref_ids.index_select(0, i)[0]).any().long()
    acc["ids"].add_(torch.stack((bad, torch.ones_like(bad))))


def _stage_text(acc, sel0):
    rel = acc["rel"].tolist()
    o1 = acc["o1"].tolist()
    over = acc["over"].tolist()
    fo = [v - sel0 if v < 1 << 40 else -1 for v in acc["f_over"].tolist()]
    f1 = [v - sel0 if v < 1 << 40 else -1 for v in acc["f_o1"].tolist()]
    ids, n = acc["ids"].tolist()
    first = min([v for v in fo if v >= 0], default=-1)
    first1 = min([v for v in f1 if v >= 0], default=-1)
    o1s = ",".join(f"{s}:{c}" for s, c in zip(M1_DIAG_STAGES, o1) if c)
    return (f"n{n} rel " + "/".join(f"{v:.0e}" for v in rel) + f" over(hid) {over[8]} "
            f"o1[{o1s}] ids {ids} first>1e-4 {first} "
            f"({','.join(s for s, v in zip(M1_DIAG_STAGES, fo) if v == first and first >= 0)})"
            f" first-o1 {first1}")


def _m1_stress(state):
    """Diagnostic (never raises): see the module docstring of tools/dl_m1_stage_diag.py."""
    lines = [f"stages {'/'.join(M1_DIAG_STAGES)}; arms off, off2, on (graph, "
             f"{M1_DIAG_REPLAYS} replays each), eager (M1 off, 200 calls); first = replay index"]
    saved = dict(state.get("m1_policy", {}))
    on = {b: "o+front" for b in M1_BUCKETS}
    for batch, length, layer in ((1, 1, "sliding"), (1, 1, "full"), (1, 8, "sliding"),
                                 (1, 8, "full"), (16, 1, "sliding"), (16, 1, "full"),
                                 (32, 1, "sliding"), (32, 1, "full")):
        tag = f"{batch}x{length}{layer[0]}"
        try:
            data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            dev = data["stream"].device
            gen = torch.Generator(device=dev).manual_seed(777 + batch * length)
            bank_s = torch.randn((5, batch * length, HIDDEN), generator=gen, device=dev)
            bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                      ).to(torch.bfloat16)
            state["m1_policy"] = {}
            per, ids_l = [[] for _ in M1_DIAG_STAGES], []
            for k in range(5):
                data["stream"].copy_(bank_s[k])
                data["attn_input"].copy_(bank_a[k])
                outs, ids = _kfn_stages(data, state)
                for j, t in enumerate(outs):
                    per[j].append(t.clone())
                ids_l.append(ids.clone())
            refs = [_gate_refs(p) for p in per]
            ref_ids = torch.stack(ids_l)
            sel = torch.zeros(1, dtype=torch.long, device=dev)

            def step(acc):
                sel.add_(1)
                i = sel % 5
                data["stream"].copy_(bank_s.index_select(0, i)[0])
                data["attn_input"].copy_(bank_a.index_select(0, i)[0])
                outs, ids = _kfn_stages(data, state)
                _stage_add(acc, outs, refs, ids, ref_ids, i, sel)

            for arm, policy in (("off", {}), ("off2", {}), ("on", on)):
                acc = _stage_acc(len(M1_DIAG_STAGES), dev)
                state["m1_policy"] = policy
                try:
                    graph = _graph_build(lambda: step(acc))
                finally:
                    state["m1_policy"] = saved
                for t in acc.values():
                    t.zero_()
                acc["f_over"].fill_(1 << 40)
                acc["f_o1"].fill_(1 << 40)
                torch.cuda.synchronize()
                sel0 = int(sel)
                for _ in range(M1_DIAG_REPLAYS):
                    graph.replay()
                torch.cuda.synchronize()
                del graph
                lines.append(f"{tag} {arm} {_stage_text(acc, sel0)}")
            acc = _stage_acc(len(M1_DIAG_STAGES), dev)
            state["m1_policy"] = {}
            sel0 = int(sel)
            for _ in range(200):
                step(acc)
            torch.cuda.synchronize()
            lines.append(f"{tag} eager {_stage_text(acc, sel0)}")
        except Exception as exc:
            lines.append(f"{tag} error {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            state["m1_policy"] = saved
    return lines


'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    a = text.index("def _m1_stress(state):")
    b = text.index("def _m1_pair(", a)
    text = text[:a] + STAGES.lstrip("\n") + text[b:]
    reps = [
        # The timeline lines pushed the gate lines out of the judge's 16 KB log tail in 6ce5.
        ("TIMELINE_SHAPES = (", "TIMELINE_SHAPES = () and (", 1),
        ('    state["m1_policy"] = policy\n    rep.append(f"open: whole step',
         '    state["m1_policy"] = {}\n    rep.append(f"diagnostic, policy left empty; would be '
         '{policy}")\n    rep.append(f"open: whole step', 1),
        ('        rep.append(f"closed: whole step', '        rep.append(f"diagnostic; closed: whole step', 1),
        ('    for item in state.get("moe1_pdl_gate", []):\n        lines.append(f"moe pdl gate {item}")',
         '    for item in state.get("moe1_pdl_gate", []):\n        if " FAIL " in item:\n'
         '            lines.append(f"moe pdl gate {item}")\n'
         '    lines.append(f"moe pdl gate {len(state.get(\'moe1_pdl_gate\', []))} variants, '
         'failing ones listed")', 1),
        ("# okc: problem=muse_decode_layer\n",
         "# okc: problem=muse_decode_layer\n# M1 stage diagnostic (not a candidate)\n", 1),
    ]
    for old, new, n in reps:
        if text.count(old) != n:
            raise SystemExit(f"expected {n} of {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    open(out, "w").write(text)
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
