#!/usr/bin/env python3
"""T = 1 / 16 / 32 under pdl8e's staged gate: writes muse_decode_layer/cand/g1b4_pdl8f.py.

The file is g1b4_pdl8e.py (tools/dl_pdl8e_patch.py, 27b5f252) with pdl8b's single fixed-input stress at T = 1 / 16 / 32
replaced by the four stages pdl8e runs at T = 8. In check 61c009066a5d40a6 that single stress turned T = 16 off on merge
order noise (on s2 4.40e-3 against off 3.70e-3) although the T = 16 code is pdl8d's, which passed it in 4a737387f88b4d7d.
Stages, on each T's stressed layer (T = 1 sliding, T = 16 sliding, T = 32 full, as pdl8b):

  det        attention with an order-free merge in both arms (B = 1: n_splits = nc = 8, coop; B > 1: per-request,
             n_splits = nc = 8, no flat kernel), fixed inputs.
  det_fresh  the same with two inputs alternating per replay inside one graph.
  fresh      the production plan, alternating inputs.
  fixed      the production plan, fixed inputs.

Every stage, at every T including T = 8, passes when ids are equal and every PDL-on value is within
max(PDL8F_BOUND = 2e-2, the off arm's own drift). The box dry run of the first pdl8f showed "within the off arm's own
drift" failing on downstream order noise even with the order-free attention (T = 8 det_fresh, pdl8d's four-edge
set at sliding), and the 1e-2 bound failing on the PDL-off arm alone (T = 32 fresh s2 1.00e-2).

A T stays on only if every stage passes; any failure or error turns that T off, as before. No CUDA source, no launch
attribute and no T = 8 code changes; the order-free plan exists only inside the stress graphs.

    python3 tools/dl_pdl8f_patch.py
"""
import hashlib
import importlib.util
import py_compile
import sys

spec = importlib.util.spec_from_file_location("pdl8e", __file__.replace("dl_pdl8f_patch.py", "dl_pdl8e_patch.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
B = E.B
D = E.D

DST = B.A.ROOT / "muse_decode_layer/cand/g1b4_pdl8f.py"
PDL8E_PIN = "27b5f25236ceda79934798dac5595cc37395e071f1c8f3cd65f93f5c064b130f"

GATE = r'''# The layer output carries order noise downstream of the attention too (box: PDL-off drift up to 1.00e-2 on
# fixed inputs, MoE / norm FP32 order), so a PDL-on arm is judged by error size: ids equal and every value within
# max(PDL8F_BOUND, the off arm's own drift). 2e-2 is 2.56 BF16 ulps of the largest output element; a read of the
# other input's or partly written data under alternating inputs moves outputs by far more.
PDL8F_BOUND = 2e-2


def _pdl8f_det_plan(cfg):
    if cfg["batch"] == 1:
        return PDL8E_DET_PLAN
    return dict(n_splits=8, nc=8, flat=0, coop=0)


def _pdl8f_gate(st, T, layer, data, gen, dev):
    """pdl8e's four stages for a T whose edges are all on; returns whether every stage passed."""
    off = dict(st, pdl8_on={})
    on = dict(st, pdl8_on={T: True})
    x0 = data["attn_input"].clone()
    s = torch.randn(x0.shape, generator=gen, device=dev)
    x1 = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x0.dtype)
    flag = torch.zeros((1, 1), dtype=torch.bool, device=dev)

    def graph(state, det, fresh):
        global _PDL8E_DET
        if fresh:
            def fn():
                data["attn_input"].copy_(torch.where(flag, x1, x0))
                return kernel_fn(data, state)
        else:
            fn = lambda: kernel_fn(data, state)
        _PDL8E_DET = det
        try:
            return _pdl8_graph(fn)
        finally:
            _PDL8E_DET = False

    def refs(g, o, fresh):
        if not fresh:
            g.replay()
            torch.cuda.synchronize()
            return [t.clone() for t in o]
        out = []
        for v in (False, True):
            flag.fill_(v)
            g.replay()
            torch.cuda.synchronize()
            out.append([t.clone() for t in o])
        return out

    def drift(g, o, ref, fresh):
        return _pdl8e_drift2(g, o, ref, flag, PDL8_STRESS) if fresh else _pdl8_drift(g, o, ref, PDL8_STRESS)

    ok = True
    try:
        for tag, det, fresh in (("det", True, False), ("det_fresh", True, True), ("fresh", False, True),
                                ("fixed", False, False)):
            g_off, o_off = graph(off, det, fresh)
            ref = refs(g_off, o_off, fresh)
            off_max, off_bad = drift(g_off, o_off, ref, fresh)
            g_on, o_on = graph(on, det, fresh)
            on_max, on_bad = drift(g_on, o_on, ref, fresh)
            good = on_bad == 0 and all(v <= max(PDL8F_BOUND, c) for v, c in zip(on_max, off_max))
            del g_off, o_off, g_on, o_on, ref
            st.setdefault("pdl8_log", []).append((T, layer, tag, off_max, off_bad, on_max, on_bad, good))
            print(f"PDL8F_STRESS T={T} {layer} {tag}: off {['%.2e' % v for v in off_max]} (int mismatches {off_bad}), "
                  f"on {['%.2e' % v for v in on_max]} (int mismatches {on_bad}); bound max({PDL8F_BOUND:.0e}, off): "
                  f"{'clean' if good else 'DRIFT, T off'}", flush=True)
            if not good:
                ok = False
                break
    finally:
        data["attn_input"].copy_(x0)
        torch.cuda.synchronize()
    return ok


'''

T8_OLD = '        if rule == "own":\n            ok = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))\n        else:\n            ok = on_bad == 0 and off_bad == 0 and all(v <= PDL8E_FRESH_BOUND for v in on_max + off_max)\n'
T8_NEW = '        ok = on_bad == 0 and all(v <= max(PDL8F_BOUND, c) for v, c in zip(on_max, off_max))\n'
T8_PRINT_OLD = 'f"{\'within the off arm\' if rule == \'own\' else \'bound %.0e\' % PDL8E_FRESH_BOUND}: "'
T8_PRINT_NEW = 'f"bound max({PDL8F_BOUND:.0e}, off): "'

OLD_STRESS = '''        if stress:
            g_off.replay()
            torch.cuda.synchronize()
            ref = [t.clone() for t in o_off]
            off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
            on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
            strict = on_bad == 0 and all(v <= PDL8_BOUND for v in on_max)
            within = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
            if not within:
                st["pdl8_on"][T] = False
            st.setdefault("pdl8_log", []).append((T, layer, off_max, off_bad, on_max, on_bad, strict, within))
            print(f"PDL8_STRESS T={T} {layer}: {PDL8_STRESS} replays per arm, max rel diff to the PDL-off output "
                  f"off {['%.2e' % v for v in off_max]} (int mismatches {off_bad}), on {['%.2e' % v for v in on_max]} "
                  f"(int mismatches {on_bad}); bound {PDL8_BOUND:.0e}: strict {'pass' if strict else 'FAIL'}, "
                  f"within the off arm's own drift {'pass' if within else 'FAIL, T off'}", flush=True)
        del g_off, g_on, o_off, o_on, data
'''
NEW_STRESS = '''        del g_off, g_on, o_off, o_on
        if stress:
            try:
                if not _pdl8f_gate(st, T, layer, data, gen, dev):
                    st["pdl8_on"][T] = False
            except Exception as e:
                st["pdl8_on"][T] = False
                print(f"PDL8F T={T} off: {repr(e)[:400]}", flush=True)
                torch.cuda.synchronize()
        del data
'''

EDITS = [
    ("pdl8d's four.", "pdl8d's four; T = 1 / 16 / 32 under the same stages (tools/dl_pdl8f_patch.py).", 1),
    ('**(PDL8E_DET_PLAN if _PDL8E_DET and cfg["batch"] == 1 and cfg["q_len"] == 8 else {}))',
     '**(_pdl8f_det_plan(cfg) if _PDL8E_DET else {}))', 1),
    ("def _pdl8_setup(st):\n", GATE + "def _pdl8_setup(st):\n", 1),
    (OLD_STRESS, NEW_STRESS, 1),
    (T8_OLD, T8_NEW, 1),
    (T8_PRINT_OLD, T8_PRINT_NEW, 1),
]


def main():
    s, body = B.make_block()
    for old, new, n in D.EDITS:
        body = B.A.sub(body, old, new, n)
    for old, new, n in E.EDITS:
        body = B.A.sub(body, old, new, n)
    if hashlib.sha256((s.rstrip("\n") + "\n" + body).encode()).hexdigest() != PDL8E_PIN:
        sys.exit("tools/dl_pdl8e_patch.py no longer reproduces g1b4_pdl8e.py 27b5f252")
    cuda_before = body[body.index("_PDL8_EDITS = "):body.index("_PDL8_GRAFT_GLUE_EDITS")]
    for old, new, n in EDITS:
        body = B.A.sub(body, old, new, n)
    if cuda_before not in body or body.count("launch_pdl=_PDL8)") != 2:
        sys.exit("pdl8f changed a CUDA edit or a T = 16 / 32 launch")
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
