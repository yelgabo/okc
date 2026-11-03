#!/usr/bin/env python3
"""T = 8 with all six PDL edges, gated on a race test: writes muse_decode_layer/cand/g1b4_pdl8e.py.

The file is g1b4_pdl8d.py (tools/dl_pdl8d_patch.py, 7cb74240) with a different T = 8 gate; no CUDA source and no
T = 1 / 16 / 32 launch changes. pdl8d's fixed-input stress compares PDL-on against the PDL-off arm's own drift, and
at verify 1 full the off arm happens to repeat bitwise while the coop attention's FP32 workspace merge (252 splits
into 8 copies, red.add in arrival order) changes with the CTA start pattern. The attention-board bench (check
2c84c321b3bb4dbb, docs/runs-attention.md) separates that order effect from a race. This gate keeps abort-to-off and
tests the two attention edges where order cannot hide a race:

  det    the attention runs with n_splits = nc = 8 (one split per workspace copy: an order-free merge) in both arms,
         fixed inputs and two inputs alternating by replay inside one graph; PDL-on must stay within the off arm's
         own drift (bound 1e-4), as pdl8d's rule. A read of partly written or previous-replay qkv or attention
         output shows here at any size.
  fresh  the production plan with the two alternating inputs; both arms within PDL8E_FRESH_BOUND of the PDL-off
         output of the same input, ids equal (a read of the other input's data moves outputs by far more).
  fixed  the production plan, fixed inputs: all six edges within PDL8E_FRESH_BOUND and ids equal.

All six edges turn on at T = 8 only if every stage passes at verify 1 full and sliding and the paired A/B saving is
positive; otherwise pdl8d's set (b1 + pool + route + fin) under pdl8d's rule, else T = 8 off.

    python3 tools/dl_pdl8e_patch.py
"""
import hashlib
import importlib.util
import py_compile
import sys

spec = importlib.util.spec_from_file_location("pdl8d", __file__.replace("dl_pdl8e_patch.py", "dl_pdl8d_patch.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
B = D.B

DST = B.A.ROOT / "muse_decode_layer/cand/g1b4_pdl8e.py"
PDL8D_PIN = "7cb74240d6ab5edd1c72e869777637157d17c67b615d508565a6084496028904"

HEAD = r'''
_PDL8E_DET = False
PDL8E_DET_PLAN = dict(n_splits=8, nc=8, coop=1)
# one BF16 ulp at the output's largest element is at most 2^-7 = 7.8e-3 of it
PDL8E_FRESH_BOUND = 1e-2
PDL8E_PDL8D_SET = ("b1", "pool", "route", "fin")
'''

T8E = r'''def _pdl8e_drift2(graph, outs, refs, flag, n):
    dev = refs[0][0].device
    acc = torch.zeros(len(refs[0]), device=dev)
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    for k in range(n):
        flag.fill_(bool(k & 1))
        graph.replay()
        for j, (a, b) in enumerate(zip(outs, refs[k & 1])):
            if a.is_floating_point():
                d = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
                acc[j] = torch.maximum(acc[j], torch.nan_to_num(d, nan=1e30))
            else:
                bad += (a != b).sum()
    torch.cuda.synchronize()
    return [float(x) for x in acc.tolist()], int(bad)


def _pdl8e_t8(st, gen, dev, cells):
    import time as _pdl8e_time
    st["pdl8_on"][8] = False
    t_end = (_PDL8_T0 or _pdl8e_time.time()) + PDL8D_BUDGET_S
    E6, E4 = PDL8D_EDGES, PDL8E_PDL8D_SET
    off = dict(st, pdl8_on={})
    spent = []

    def budget(tag):
        if _pdl8e_time.time() > t_end:
            raise RuntimeError(f"PDL8E prepare budget {PDL8D_BUDGET_S:.0f} s reached before {tag}")

    def graph(data, state, det=False, fresh=None):
        global _PDL8E_DET
        if fresh is None:
            fn = lambda: kernel_fn(data, state)
        else:
            xs, flag = fresh

            def fn():
                data["attn_input"].copy_(torch.where(flag, xs[1], xs[0]))
                return kernel_fn(data, state)
        _PDL8E_DET = det
        try:
            return _pdl8_graph(fn)
        finally:
            _PDL8E_DET = False

    def refs_of(g, o, fresh):
        if fresh is None:
            g.replay()
            torch.cuda.synchronize()
            return [t.clone() for t in o]
        out = []
        for v in (False, True):
            fresh[1].fill_(v)
            g.replay()
            torch.cuda.synchronize()
            out.append([t.clone() for t in o])
        return out

    def drift(g, o, ref, fresh):
        if fresh is None:
            return _pdl8_drift(g, o, ref, PDL8_STRESS)
        return _pdl8e_drift2(g, o, ref, fresh[1], PDL8_STRESS)

    def stage(layer, data, edges, tag, det, fresh, rule):
        budget(tag)
        t0 = _pdl8e_time.time()
        g_off, o_off = graph(data, off, det, fresh)
        ref = refs_of(g_off, o_off, fresh)
        off_max, off_bad = drift(g_off, o_off, ref, fresh)
        on = dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(edges)})
        g_on, o_on = graph(data, on, det, fresh)
        on_max, on_bad = drift(g_on, o_on, ref, fresh)
        if rule == "own":
            ok = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
        else:
            ok = on_bad == 0 and off_bad == 0 and all(v <= PDL8E_FRESH_BOUND for v in on_max + off_max)
        del g_off, o_off, g_on, o_on, ref
        spent.append(_pdl8e_time.time() - t0)
        st.setdefault("pdl8_log", []).append((8, layer, tag, tuple(edges), off_max, off_bad, on_max, on_bad, ok))
        print(f"PDL8E_STRESS T=8 {layer} {tag} edges={'+'.join(edges)}: off {['%.2e' % v for v in off_max]} "
              f"(int mismatches {off_bad}), on {['%.2e' % v for v in on_max]} (int mismatches {on_bad}); rule "
              f"{'within the off arm' if rule == 'own' else 'bound %.0e' % PDL8E_FRESH_BOUND}: "
              f"{'clean' if ok else 'DRIFT'} ({spent[-1]:.1f} s)", flush=True)
        return ok

    datas = {}
    try:
        for layer in ("full", "sliding"):
            data = _pdl8_data(1, 8, layer, gen, dev)
            ids = _pdl8_flat(kernel_fn(data, off))[-1]
            if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 64:
                counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
                data["local_start"].fill_(32 * int(counts.argmax()))
            kernel_fn(data, dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(E6)}))
            torch.cuda.synchronize()
            x0 = data["attn_input"].clone()
            s = torch.randn(x0.shape, generator=gen, device=dev)
            x1 = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x0.dtype)
            datas[layer] = (data, ((x0, x1), torch.zeros((1, 1), dtype=torch.bool, device=dev)))
        ok6 = True
        for layer in ("full", "sliding"):
            data, fresh = datas[layer]
            for tag, det, fr, rule in (("det", True, None, "own"), ("det_fresh", True, fresh, "own"),
                                       ("fresh", False, fresh, "bound"), ("fixed", False, None, "bound")):
                ok6 = stage(layer, data, E6, tag, det, fr, rule) and ok6
                if not ok6:
                    break
            data["attn_input"].copy_(fresh[0][0])
            if not ok6:
                break
        chosen = E6
        if not ok6:
            chosen = E4
            for layer in ("full", "sliding"):
                if not stage(layer, datas[layer][0], E4, "pdl8d_set", False, None, "own"):
                    chosen = ()
                    break
        print(f"PDL8E T=8 chosen edges {'+'.join(chosen) or 'none'}", flush=True)
        if not chosen:
            return
        for layer in ("sliding", "full"):
            data = datas[layer][0]
            on = dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(chosen)})
            g_off, o_off = graph(data, off)
            g_on, o_on = graph(data, on)
            rows = _pdl8_time([g_off, g_on])
            d = sorted(r[0] - r[1] for r in rows)
            moff = sorted(r[0] for r in rows)[len(rows) // 2]
            mon = sorted(r[1] for r in rows)[len(rows) // 2]
            cells[(8, layer)] = d[len(d) // 2]
            print(f"PDL8_AB T=8 {layer} edges={'+'.join(chosen)}: off {moff:.2f} on {mon:.2f} us, median paired "
                  f"saving {d[len(d) // 2]:.2f}, wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
            del g_off, o_off, g_on, o_on
        w = (46 * cells[(8, "sliding")] + 16 * cells[(8, "full")]) / 62
        if w > 0:
            st["pdl8_edges"] = {8: frozenset(chosen)}
            st["pdl8_on"][8] = True
        print(f"PDL8E T=8 {'on' if st['pdl8_on'][8] else 'off'}: weighted saving {w:.2f} us, edges "
              f"{'+'.join(chosen)}; stage times {['%.1f' % v for v in spent]} s", flush=True)
    finally:
        datas.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


'''

ATT_OLD = ('"k_scale": data["k_scale"], "v_scale": data["v_scale"]}, attention_state=state, prefetch=prefetch, '
           '**pdl)\n    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)\n\n_PDL8_GLUE = None')
ATT_NEW = ('"k_scale": data["k_scale"], "v_scale": data["v_scale"]}, attention_state=state, prefetch=prefetch, '
           '**pdl,\n                   **(PDL8E_DET_PLAN if _PDL8E_DET and cfg["batch"] == 1 and cfg["q_len"] == 8 '
           'else {}))\n    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)\n\n_PDL8_GLUE = None')

EDITS = [
    ("at T = 8 per edge, the edge set chosen in prepare (tools/dl_pdl8d_patch.py).",
     "at T = 8 all six edges when the det / fresh race stages pass in prepare (tools/dl_pdl8e_patch.py), else "
     "pdl8d's four.", 1),
    ("\n_PDL8_T0 = None\n", "\n_PDL8_T0 = None\n" + HEAD, 1),
    (ATT_OLD, ATT_NEW, 1),
    ("def _pdl8_setup(st):\n", T8E + "def _pdl8_setup(st):\n", 1),
    ("        _pdl8d_t8(st, gen, dev, cells)\n", "        _pdl8e_t8(st, gen, dev, cells)\n", 1),
    ('print(f"PDL8D T=8 off: {repr(e)[:400]}", flush=True)', 'print(f"PDL8E T=8 off: {repr(e)[:400]}", flush=True)', 1),
]


def main():
    s, body = B.make_block()
    if hashlib.sha256((s.rstrip("\n") + "\n" + body).encode()).hexdigest() != D.PDL8B_PIN:
        sys.exit("tools/dl_pdl8b_patch.py no longer reproduces g1b4_pdl8b.py 7f5651e5")
    cuda_before = body[:body.index("_PDL8_GRAFT_GLUE_EDITS")]
    cuda_before = cuda_before[cuda_before.index("_PDL8_EDITS = "):]
    for old, new, n in D.EDITS:
        body = B.A.sub(body, old, new, n)
    if hashlib.sha256((s.rstrip("\n") + "\n" + body).encode()).hexdigest() != PDL8D_PIN:
        sys.exit("tools/dl_pdl8d_patch.py no longer reproduces g1b4_pdl8d.py 7cb74240")
    for old, new, n in EDITS:
        body = B.A.sub(body, old, new, n)
    if cuda_before not in body or body.count("launch_pdl=_PDL8)") != 2:
        sys.exit("pdl8e changed a CUDA edit or a T = 16 / 32 launch")
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
