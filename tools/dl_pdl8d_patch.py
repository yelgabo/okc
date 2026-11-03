#!/usr/bin/env python3
"""Per-edge PDL at T = 8: writes muse_decode_layer/cand/g1b4_pdl8d.py from the pinned release.

The file is g1b4_pdl8b.py (tools/dl_pdl8b_patch.py, 7f5651e5) with T = 8 put back under PDL one edge at a time.
pdl8's six T = 8 edges are named by their consumer launch: attn (qkv sk2 -> attention), o (attention -> o sk2),
b1 (o sk2 -> glue_cluster_boundary), pool (b1 -> fused pooler / router sk2), route (pooler / router -> route),
fin (unpooler sk2 -> finish). An edge that is off keeps its consumer's griddepcontrol code, its .cg / TMA reads and
its producer's trigger; only the programmatic launch attribute is dropped, so the consumer starts after the
producer completes. prepare() runs the 2000-replay stress at verify 1 full with all edges, the halves and each edge
alone, keeps the edges that are clean alone (greedily rebuilt if their union is not), then runs a fresh stress of
that set at verify 1 full and sliding plus the paired A/B. T = 8 turns on only if both stresses are within the
off arm's own drift and the weighted saving is positive. The edge set is fixed in prepare; kernel_fn only looks it up.
No CUDA source differs from pdl8b. At T = 1 / 16 / 32 the edge mask is None and every launch is pdl8b's.

    python3 tools/dl_pdl8d_patch.py
"""
import hashlib
import importlib.util
import py_compile
import sys

spec = importlib.util.spec_from_file_location("pdl8b", __file__.replace("dl_pdl8d_patch.py", "dl_pdl8b_patch.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

DST = B.A.ROOT / "muse_decode_layer/cand/g1b4_pdl8d.py"
PDL8B_PIN = "7f5651e55f397158c7c5928013a11ba330bf3c50c8559b9dcf7b0666cdde79de"

HEAD_EDGES = r'''
_PDL8 = False
_PDL8_EDGES = None
PDL8D_EDGES = ("attn", "o", "b1", "pool", "route", "fin")
# pdl8b's prepare took 108-122 s of the spec's 300 s (prepare_s, 8f6acdda / 0cb69c8b / eba613c0)
PDL8D_BUDGET_S = 230.0
_PDL8_T0 = None


def _pdl8_edge(name):
    return _PDL8 and (_PDL8_EDGES is None or name in _PDL8_EDGES)
'''

T8 = r'''def _pdl8d_t8(st, gen, dev, cells):
    import time as _pdl8d_time
    st["pdl8_on"][8] = False
    t_end = (_PDL8_T0 or _pdl8d_time.time()) + PDL8D_BUDGET_S
    E = PDL8D_EDGES
    arms = {}
    for layer in ("full", "sliding"):
        data = _pdl8_data(1, 8, layer, gen, dev)
        off = dict(st, pdl8_on={})
        ids = _pdl8_flat(kernel_fn(data, off))[-1]
        if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 64:
            counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
            data["local_start"].fill_(32 * int(counts.argmax()))
        kernel_fn(data, dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(E)}))
        torch.cuda.synchronize()
        g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
        g_off.replay()
        torch.cuda.synchronize()
        ref = [t.clone() for t in o_off]
        off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
        arms[layer] = [data, off, g_off, o_off, ref, off_max, off_bad]
        print(f"PDL8D_OFF T=8 {layer}: {PDL8_STRESS} replays, max rel diff to the PDL-off output "
              f"{['%.2e' % v for v in off_max]} (int mismatches {off_bad})", flush=True)
    spent = []

    def stress(layer, edges, tag):
        if _pdl8d_time.time() > t_end:
            raise RuntimeError(f"PDL8D prepare budget {PDL8D_BUDGET_S:.0f} s reached before {tag}")
        t0 = _pdl8d_time.time()
        data, off, g_off, o_off, ref, off_max, off_bad = arms[layer]
        on = dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(edges)})
        g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
        on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
        within = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
        del g_on, o_on
        spent.append(_pdl8d_time.time() - t0)
        st.setdefault("pdl8_log", []).append((8, layer, tag, tuple(edges), off_max, off_bad, on_max, on_bad, within))
        print(f"PDL8D_STRESS T=8 {layer} {tag} edges={'+'.join(edges) or '-'}: on {['%.2e' % v for v in on_max]} "
              f"(int mismatches {on_bad}) against off {['%.2e' % v for v in off_max]}: "
              f"{'clean' if within else 'DRIFT'} ({spent[-1]:.1f} s)", flush=True)
        return within

    chosen = []
    try:
        if stress("full", E, "all"):
            chosen = list(E)
        else:
            for tag, half in (("half1", E[:3]), ("half2", E[3:])):
                stress("full", half, tag)
            alone = [e for e in E if stress("full", (e,), "single")]
            if len(alone) > 1 and not stress("full", alone, "union"):
                chosen = []
                for e in alone:
                    if stress("full", tuple(chosen) + (e,), "greedy"):
                        chosen.append(e)
            else:
                chosen = alone
        carriers = [e for e in E if e not in chosen]
        print(f"PDL8D T=8 chosen edges {'+'.join(chosen) or 'none'}; left serialized {'+'.join(carriers) or 'none'}",
              flush=True)
        if not chosen:
            return
        data, off, g_off, o_off, ref, off_max, off_bad = arms["full"]
        rep_max, rep_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
        print(f"PDL8D_OFF T=8 full repeat: {['%.2e' % v for v in rep_max]} (int mismatches {rep_bad}), "
              f"informational", flush=True)
        ok = stress("full", chosen, "final") and stress("sliding", chosen, "final")
        for layer in ("sliding", "full"):
            data, off, g_off, o_off = arms[layer][:4]
            on = dict(st, pdl8_on={8: True}, pdl8_edges={8: frozenset(chosen)})
            g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
            rows = _pdl8_time([g_off, g_on])
            d = sorted(r[0] - r[1] for r in rows)
            moff = sorted(r[0] for r in rows)[len(rows) // 2]
            mon = sorted(r[1] for r in rows)[len(rows) // 2]
            cells[(8, layer)] = d[len(d) // 2]
            print(f"PDL8_AB T=8 {layer} edges={'+'.join(chosen)}: off {moff:.2f} on {mon:.2f} us, median paired "
                  f"saving {d[len(d) // 2]:.2f}, wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
            del g_on, o_on
        w = (46 * cells[(8, "sliding")] + 16 * cells[(8, "full")]) / 62
        if ok and w > 0:
            st["pdl8_edges"] = {8: frozenset(chosen)}
            st["pdl8_on"][8] = True
        print(f"PDL8D T=8 {'on' if st['pdl8_on'][8] else 'off'}: final stress {'clean' if ok else 'DRIFT'}, "
              f"weighted saving {w:.2f} us, edges {'+'.join(chosen)}; stress times {['%.1f' % v for v in spent]} s",
              flush=True)
    finally:
        for a in arms.values():
            del a[:]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


'''

KFN_OLD = '''        global _PDL8, bfp_SK2_PDL
        T = data["config"]["tokens"]
        _PDL8 = T in PDL8_T and bool(state.get("pdl8_on", {}).get(T, False))
        try:
            return _pdl8_parent_kernel(data, state)
        finally:
            _PDL8 = False
            bfp_SK2_PDL = 0
'''
KFN_NEW = '''        global _PDL8, _PDL8_EDGES, bfp_SK2_PDL
        T = data["config"]["tokens"]
        _PDL8 = T in PDL8_T and bool(state.get("pdl8_on", {}).get(T, False))
        _PDL8_EDGES = state.get("pdl8_edges", {}).get(T) if _PDL8 else None
        try:
            return _pdl8_parent_kernel(data, state)
        finally:
            _PDL8 = False
            _PDL8_EDGES = None
            bfp_SK2_PDL = 0
'''

EDITS = [
    ("PDL on for the non-cooperative edges of the T = 1, 16 and 32 chains (T = 8 off).",
     "PDL on for the non-cooperative edges of the T = 1, 16 and 32 chains; at T = 8 per edge, the edge set chosen "
     "in prepare (tools/dl_pdl8d_patch.py).", 1),
    ("PDL8_T = (1, 16, 32)\n", "PDL8_T = (1, 8, 16, 32)\n", 1),
    ("\n_PDL8 = False\n", HEAD_EDGES, 1),
    ("or _PDL8} if ATT_HAS_PDL", 'or _pdl8_edge("attn")} if ATT_HAS_PDL', 1),
    ("mod.glue_set_pdl(1 if _PDL8 else 0)", 'mod.glue_set_pdl(1 if _pdl8_edge("b1") else 0)', 1),
    ('((1 << 29) if role in ("qkv", "un") else 0)',
     '((1 << 29) if role in ("qkv", "un") or not _pdl8_edge(role) else 0)', 1),
    ("num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_PDL8)", 'num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))', 2),
    ("num_warps=nw,launch_pdl=_PDL8)", 'num_warps=nw,launch_pdl=_pdl8_edge("fin"))', 1),
    ("    value = {1: 0.212, 16: 0.105, 32: 0.087}\n    est = 0.0\n    for T in (1, 16, 32):\n",
     "    try:\n        _pdl8d_t8(st, gen, dev, cells)\n    except Exception as e:\n"
     "        st[\"pdl8_on\"][8] = False\n        st.pop(\"pdl8_edges\", None)\n"
     "        print(f\"PDL8D T=8 off: {repr(e)[:400]}\", flush=True)\n        torch.cuda.synchronize()\n"
     "    value = {1: 0.212, 8: 0.144, 16: 0.105, 32: 0.087}\n    est = 0.0\n    for T in (1, 8, 16, 32):\n", 1),
    ("def _pdl8_setup(st):\n", T8 + "def _pdl8_setup(st):\n", 1),
    (KFN_OLD, KFN_NEW, 1),
    ("    def prepare(weights):\n        st = _pdl8_prior_prepare(weights)\n",
     "    def prepare(weights):\n        global _PDL8_T0\n        import time as _pdl8d_time\n"
     "        _PDL8_T0 = _pdl8d_time.time()\n        st = _pdl8_prior_prepare(weights)\n", 1),
]


def main():
    s, body = B.make_block()
    if hashlib.sha256((s.rstrip("\n") + "\n" + body).encode()).hexdigest() != PDL8B_PIN:
        sys.exit("tools/dl_pdl8b_patch.py no longer reproduces g1b4_pdl8b.py 7f5651e5")
    cuda_before = body[:body.index("_PDL8_GRAFT_GLUE_EDITS")]
    cuda_before = cuda_before[cuda_before.index("_PDL8_EDITS = "):]
    for old, new, n in EDITS:
        body = B.A.sub(body, old, new, n)
    if cuda_before not in body or body.count("launch_pdl=_PDL8)") != 2:
        sys.exit("pdl8d changed a CUDA edit or a T = 16 / 32 launch")
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
