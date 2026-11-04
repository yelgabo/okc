#!/usr/bin/env python3
"""The box-revival combined candidate on g1b4_pdl8h (33c2353b): writes muse_decode_layer/cand/rv_k_combo2.py.

1. tools/rv_cgfe_rs_patch.py --base pdl8h: the release + m = 1 cgemv prefix of g1b4_cgfe, pdl8h's PDL tail, cgfe's
   cgemv PDL block (wait before x, x through __ldcg, trigger at the CTA's end).
2. the cgemv offered for the unpooler slice only, and a cgemv that passed CG_GATE restored as the m = 1 unpooler
   choice (the isolated tuner splits 7.4 to 7.9 us and picked qkv_gate cg on some prepares, a measured loss).
3. sk2 weight stages issued before griddepcontrol.wait: 1 at T = 32 (the default 2 elsewhere).
4. attention triggers its dependents at entry, after its griddepcontrol.wait (OKC_PDLTRIG 0) instead of at the merge.
Every edit is an exact string that must occur once. docs/box-revival.md has the A/B for each step.

    python3 tools/rv_combo_patch.py
"""
import hashlib
import pathlib
import py_compile
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = ROOT / "muse_decode_layer/cand"
EXPECT = "7f0a3e360f085c4f21c1b760d10eaf251d3d379464a1f0556d52828cf92d5f37"
EDITS = [
    ('CG_LAYERS = ("qkv_gate", "o", "unpooler")', 'CG_LAYERS = ("unpooler",)'),
    ('            print("CG_TUNE", layer, v, "->", pick, flush=True)\n    return state',
     '            print("CG_TUNE", layer, v, "->", pick, flush=True)\n'
     '    # the isolated tuner splits 7.4 to 7.9 us cg / sk2 and sometimes picks sk2, while in the chain cg saves 1 to 2 us\n'
     '    # per decode 1 cell (docs/box-revival.md): a cgemv that passed CG_GATE is the m = 1 unpooler choice\n'
     '    if "unpooler" in state.get("cg", {}):\n        state["cg_pick"]["unpooler"] = ("cg",)\n    return state'),
    ('PCX_CFG = {}   #', 'PCX_CFG = {(r, 32): (1, 0, 0, 0) for r in ("qkv", "o", "un", "pool", "router")}   #'),
    ('#define OKC_PDLTRIG 1   // pdl8', '#define OKC_PDLTRIG 0   // pdl8'),
]

subprocess.run([sys.executable, str(ROOT / "tools/rv_cgfe_rs_patch.py"), "--base", "pdl8h"], check=True)
s = (C / "rv_cgfe_h.py").read_text()
for old, new in EDITS:
    if s.count(old) != 1:
        raise SystemExit(f"{old[:60]!r} occurs {s.count(old)} times")
    s = s.replace(old, new)
dst = C / "rv_k_combo2.py"
dst.write_text(s)
py_compile.compile(str(dst), doraise=True)
h = hashlib.sha256(s.encode()).hexdigest()
print(dst.relative_to(ROOT), h, "matches the measured bytes" if h == EXPECT else "DIFFERS from the measured 7f0a3e36")
