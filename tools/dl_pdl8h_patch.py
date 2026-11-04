#!/usr/bin/env python3
"""PDL enabled at build time at every T, with a coarse in-prepare guard: writes muse_decode_layer/cand/g1b4_pdl8h.py.

The file is g1b4_rs.py (tools/dl_rs_patch.py on pdl8d, 01426b89) with only the PDL enable decision and its gate changed.
pdl8d's gate turned a T on only when the PDL-on stress drifted no more than the PDL-off arm's own drift on one input.
Routing ids and output values move between PDL-off replays with FP32 merge order, so that gate turned T = 8, 16 or 32 off
at random in the last judge checks (61c00906, 9ae1d02f, f8a55f10, 9d80ae24). The user decided (final-day ledger,
2026-11-03) to enable PDL at T = 1, 8 (all six edges), 16 and 32 at build time, on the source audit (every consumer reads
its predecessor's output after griddepcontrol.wait through .cg / __ldcg), the bitwise attention bench 2c84c321b3bb4dbb
and the clean T = 8 stages of 61c00906 / 9ae1d02f. The stress stays only as a guard against gross errors, on the same
inputs and replay counts as pdl8d (T = 1 sliding, T = 16 sliding, T = 32 full, T = 8 full and sliding; 2000 replays per
arm): a T stays on when its PDL-on routing ids mismatch the PDL-off output on at most 2 x (the off arm's mismatches) + 16
entries and every PDL-on output is within max(2e-2, the off arm's own drift) of it. A failed guard or any error turns
that T off. No CUDA source, launch attribute, edge wiring or routing-slack code changes.

    python3 tools/dl_pdl8h_patch.py                  # onto g1b4_rs (01426b89), writes g1b4_pdl8h.py
    python3 tools/dl_pdl8h_patch.py --base pdl8d     # onto g1b4_pdl8d (7cb74240), writes g1b4_pdl8hn.py

The organizers disqualified routing slack (final-day ledger, 2026-11-03 17:45 PT), so --base pdl8d applies the same
edits to pdl8d: g1b4_pdl8hn.py is g1b4_pdl8h.py with tools/dl_rs_patch.py's two hunks reversed.
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASES = {"rs": ("g1b4_rs.py", "01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07", "g1b4_pdl8h.py"),
         "pdl8d": ("g1b4_pdl8d.py", "7cb74240d6ab5edd1c72e869777637157d17c67b615d508565a6084496028904", "g1b4_pdl8hn.py")}
BASE = BASES[sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "rs"]
SRC = ROOT / "muse_decode_layer/cand" / BASE[0]
DST = ROOT / "muse_decode_layer/cand" / BASE[2]

GUARD = r'''# A stale or partly written read moves outputs by 0.18 to 1.6 and flips 19 to 1165 ids (tools/dl_pdl8f_box_fault.py);
# merge-order noise moves PDL-off outputs up to about 1e-2 and flips up to a few thousand ids at T = 32 by itself
PDL8H_BOUND = 2e-2
PDL8H_IDS_MULT = 2
PDL8H_IDS_SLACK = 16


def _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad):
    limit = PDL8H_IDS_MULT * off_bad + PDL8H_IDS_SLACK
    ok = on_bad <= limit and all(v <= max(PDL8H_BOUND, c) for v, c in zip(on_max, off_max))
    st.setdefault("pdl8_log", []).append((T, layer, off_max, off_bad, on_max, on_bad, ok))
    print(f"PDL8H_GUARD T={T} {layer}: {PDL8_STRESS} replays per arm; ids mismatches off {off_bad} on {on_bad} "
          f"(limit {limit}); max rel diff to the PDL-off output off {['%.2e' % v for v in off_max]} on "
          f"{['%.2e' % v for v in on_max]} (bound max({PDL8H_BOUND:.0e}, off)): "
          f"{'pass, T stays on' if ok else 'FAIL, T off'}", flush=True)
    return ok


def _pdl8h_t8(st, gen, dev, cells):
    import time as _pdl8h_time
    t_end = (_PDL8_T0 or _pdl8h_time.time()) + PDL8D_BUDGET_S
    E = frozenset(PDL8D_EDGES)
    off = dict(st, pdl8_on={})
    on = dict(st, pdl8_on={8: True}, pdl8_edges={8: E})
    ok = False
    try:
        for layer in ("full", "sliding"):
            if _pdl8h_time.time() > t_end:
                raise RuntimeError(f"PDL8H prepare budget {PDL8D_BUDGET_S:.0f} s reached before T=8 {layer}")
            data = _pdl8_data(1, 8, layer, gen, dev)
            ids = _pdl8_flat(kernel_fn(data, off))[-1]
            if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 64:
                counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
                data["local_start"].fill_(32 * int(counts.argmax()))
            kernel_fn(data, on)
            torch.cuda.synchronize()
            g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
            g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
            rows = _pdl8_time([g_off, g_on])
            d = sorted(r[0] - r[1] for r in rows)
            moff = sorted(r[0] for r in rows)[len(rows) // 2]
            mon = sorted(r[1] for r in rows)[len(rows) // 2]
            cells[(8, layer)] = d[len(d) // 2]
            print(f"PDL8_AB T=8 {layer} edges={'+'.join(PDL8D_EDGES)}: off {moff:.2f} on {mon:.2f} us, median paired "
                  f"saving {d[len(d) // 2]:.2f}, wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
            g_off.replay()
            torch.cuda.synchronize()
            ref = [t.clone() for t in o_off]
            off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
            on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
            ok = _pdl8h_guard(st, 8, layer, off_max, off_bad, on_max, on_bad)
            del g_off, g_on, o_off, o_on, ref, data
            torch.cuda.empty_cache()
            if not ok:
                break
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    st["pdl8_on"][8] = ok
    if ok:
        st["pdl8_edges"] = {8: E}
    else:
        st.pop("pdl8_edges", None)
    print(f"PDL8H T=8 {'on, all six edges' if ok else 'off'}", flush=True)


'''

OLD_STRESS = '''            strict = on_bad == 0 and all(v <= PDL8_BOUND for v in on_max)
            within = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
            if not within:
                st["pdl8_on"][T] = False
            st.setdefault("pdl8_log", []).append((T, layer, off_max, off_bad, on_max, on_bad, strict, within))
            print(f"PDL8_STRESS T={T} {layer}: {PDL8_STRESS} replays per arm, max rel diff to the PDL-off output "
                  f"off {['%.2e' % v for v in off_max]} (int mismatches {off_bad}), on {['%.2e' % v for v in on_max]} "
                  f"(int mismatches {on_bad}); bound {PDL8_BOUND:.0e}: strict {'pass' if strict else 'FAIL'}, "
                  f"within the off arm's own drift {'pass' if within else 'FAIL, T off'}", flush=True)
'''
NEW_STRESS = '''            if not _pdl8h_guard(st, T, layer, off_max, off_bad, on_max, on_bad):
                st["pdl8_on"][T] = False
'''

EDITS = [
    ("at T = 8 per edge, the edge set chosen in prepare (tools/dl_pdl8d_patch.py).",
     "at T = 8 all six edges; enabled at build time at every T behind a coarse in-prepare guard "
     "(tools/dl_pdl8h_patch.py).", 1),
    ("PDL8_HARD = {1: True, 8: False, 16: True, 32: True}\n", "PDL8_HARD = {1: True, 8: True, 16: True, 32: True}\n", 1),
    ("def _pdl8_setup(st):\n", GUARD + "def _pdl8_setup(st):\n", 1),
    (OLD_STRESS, NEW_STRESS, 1),
    ("        _pdl8d_t8(st, gen, dev, cells)\n", "        _pdl8h_t8(st, gen, dev, cells)\n", 1),
    ('print(f"PDL8D T=8 off: {repr(e)[:400]}", flush=True)', 'print(f"PDL8H T=8 off: {repr(e)[:400]}", flush=True)', 1),
]


def sub(s, old, new, n):
    if s.count(old) != n:
        sys.exit(f"anchor {old[:70]!r}: {s.count(old)} found, {n} expected")
    return s.replace(old, new)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE[1]:
        sys.exit(f"{BASE[0]} is not {BASE[1][:8]}")
    s = raw.decode()
    for old, new, n in EDITS:
        s = sub(s, old, new, n)
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
