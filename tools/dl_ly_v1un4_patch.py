#!/usr/bin/env python3
"""ly_v1un4: ly_v1un with the T = 8 unpooler-slice sk2 issuing all 4 of its weight stages before griddepcontrol.wait.

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). At T = 8 the slice sk2 (sk2_kernel<1, 8, 1,
1, 8, 2, 4, ...>, 128 CTAs) owns 4 stages of 12 KB weights per CTA, the whole K; ly_v1un makes it a PDL secondary
of the MoE with the default P = BFP_PDL_N = 2 stages issued before the wait, so the other 2 stages are issued by the
producer warp after it. One change: PCX_CFG[("un", 8)] = (4, 0, 0, 0), so P = min(4, NSTAGE 4, nst 4) = 4 and every
weight byte of the CTA is requested while the MoE's last CTAs finish. The value is read only through _pcx_val, which
the PDL8 path calls for role "un" at T = 8 only when the edge is on (with PDL off at T = 8 the launcher passes 0 and
P = 1, as in the base). The kernel's PDL path is the base's: weights and escape table before the wait, every thread's
griddepcontrol.wait, fence.proxy.async.global, then the x tiles; the producer loop is empty (st from P = nst) and
its launch_dependents runs at once, so the finish may launch earlier and wait on this grid as before.

    python3 tools/dl_ly_v1un4_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_v1un4.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

OLD = ('PCX_CFG = {(r, 32): (1, 0, 0, 0) for r in ("qkv", "o", "un", "pool", "router")}   # (role, T or 0) -> '
       '(P, d0, g[, e]); see harness/make_pcx.py\n')
NEW = OLD + 'PCX_CFG[("un", 8)] = (4, 0, 0, 0)   # ly_v1un4: the T = 8 slice issues its whole K of weights before the wait\n'
HEADER = ("# ly_v1un4: ly_v1un with the T = 8 slice's 4 weight stages all before griddepcontrol.wait "
          "(tools/dl_ly_v1un4_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if text.count(OLD) != 1:
        sys.exit(f"edit anchor found {text.count(OLD)} times, expected 1")
    text = HEADER + text.replace(OLD, NEW)
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
