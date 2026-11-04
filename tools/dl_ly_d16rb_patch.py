#!/usr/bin/env python3
"""ly_d16rb: the T = 16 / 32 row boundary kernel issues its Stream row load right after its Branch row load.

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). Target: decode 16 / 32 (docs/runs-decode-layer.md,
"Decode 16 / 32 sliding timeline (box)"). `_graft_row_boundary_pre_cg` (Gluon, one program of 32 warps per token row)
is b2 at T = 16 and both b1 and b2 at 16 < T <= 32, and is launched nowhere else. After its griddepcontrol.wait it
loads the Branch row (the producer's output), reduces it over the CTA (bar.sync), and only then loads the Stream row;
the `.cg` loads compile to LDG.E.*.STRONG.GPU, which ptxas does not move across the barrier, so the Stream row's
memory latency is serial behind the first reduction (SASS of the box's cached cubin: Stream LDG at 0x06d0, after
BAR.SYNC at 0x04a0 and MUFU.RSQ at 0x0580). Its critical path after the producer's end is 2.7 to 2.8 us for a
0.35 / 0.71 us byte floor.

Change (one line moved): `s = _gl.load(Stream + row * H + c, cache_modifier='.cg')` now sits right after the Branch
load, both after `_gdc_wait()`; the math, its order and every store are unchanged, so the output is bitwise the
base's. Stream is an earlier kernel's output (b1's s1, or the layer input at b1), so it stays after the wait as every
predecessor output does. No fence, barrier, wait, trigger or launch attribute changes.

    python3 tools/dl_ly_d16rb_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_d16rb.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

OLD = """    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
"""
NEW = """    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
"""
EDITS = [(OLD, NEW, 1)]
HEADER = ("# ly_d16rb: ly_v1un with the T = 16 / 32 row boundary's Stream load issued with its Branch load "
          "(tools/dl_ly_d16rb_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old!r}")
        text = text.replace(old, new)
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
