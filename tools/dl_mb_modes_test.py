"""Offline test of the M-B switch logic in a layer built by tools/dl_mb_chain_patch.py: which
launches get the PDL bit, the trigger bit and the attention PDL flag per M1 mode and token
count, and that the census counter sees every PDL launch the switches add. CPU only, no torch.

    python3 tools/dl_mb_modes_test.py LAYER.py
"""
import ast
import sys

layer = open(sys.argv[1]).read()
keep = {"M1_BUCKETS", "M1_MODES", "M1_GATE_MODE", "_m1_mode", "_m1_has", "_m1_pdl", "_attn_pdl"}
body = [n for n in ast.parse(layer).body
        if (isinstance(n, ast.FunctionDef) and n.name in keep)
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in keep)]
ns = {"_M1_PDL_LAUNCHES": [0]}
exec(compile(ast.Module(body=body, type_ignores=[]), sys.argv[1], "exec"), ns)
bad = 0


def check(name, got, want):
    global bad
    ok = got == want
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))


def state(mode):
    return {"m1_ok": True, "m1_policy": {b: mode for b in ns["M1_BUCKETS"]}}


used = {s for mode in ns["M1_MODES"] if mode != "off" for s in mode.split("+")}
check("gate mode covers every switch the modes use",
      used <= set(ns["M1_GATE_MODE"].split("+")), True)
for mode in ns["M1_MODES"]:
    st = state(mode)
    for t in (1, 8, 16, 32, 128):
        sw = set(mode.split("+")) if t <= 32 and mode != "off" else set()
        check(f"{mode} T={t} o pdl", ns["_m1_pdl"]("o", t, st), "o" in sw)
        check(f"{mode} T={t} unpooler pdl", ns["_m1_pdl"]("unpooler", t, st), "unp" in sw)
        check(f"{mode} T={t} qkv pdl", ns["_m1_pdl"]("qkv_gate", t, st), False)
        check(f"{mode} T={t} trig", ns["_m1_has"](t, st, "trig"), "trig" in sw)
        n0 = ns["_M1_PDL_LAUNCHES"][0]
        got = ns["_attn_pdl"](t, st)
        check(f"{mode} T={t} attn pdl", got, "attn" in sw and 8 < t)
        check(f"{mode} T={t} attn counted", ns["_M1_PDL_LAUNCHES"][0] - n0, int(got))
off = {"m1_ok": False, "m1_policy": {8: ns["M1_GATE_MODE"]}}
check("M1 not ok: o pdl", ns["_m1_pdl"]("o", 8, off), False)
check("M1 not ok: attn pdl", ns["_attn_pdl"](16, off), False)
print("all cases as expected" if not bad else f"{bad} cases failed")
sys.exit(1 if bad else 0)
