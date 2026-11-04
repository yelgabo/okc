#!/usr/bin/env python3
"""ly_b4rf: the decode 32 B4 WGS build gets ly_rfin's cut-unit finish (-DOKC_RFIN=1 -DOKC_RFINQ=2), as MAIN has.

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). Target: decode 32 sliding / full (B4, 16 < M <= 32;
docs/runs-decode-layer.md "Decode 16 / 32 sliding MoE (box)"). ly_rfin (docs/runs-decode-layer.md "Verify segment
switches (box)") put the define pair on the MAIN build only; B4 compiles the same MOE_CUDA_SRC with OKC_B4FIN=1 and still
runs the base walk: phase A reordered (OKC_WGSREORD), cut units summed by red.add with an atom.acq_rel round trip per
share (OKC_WGSDFIN 2). With RFIN the ranges run in order, the share that does not hold stage 0 stores its partials to its
own CTA's fixup slot and releases its count (kind 2: fence.proxy.async.global, bar.sync, red.release.gpu, as before);
the share holding stage 0 acquires the count (ld.acquire loop, bar.sync, fence.acq_rel.gpu), reads the slots with
ld.global.cg, zeroes them and runs the whole-unit epilogue and its deferred release. No fence, barrier or release is
removed. The RFIN path is the generic WGS text (no OKC_B4FIN branch in it); B4FIN's per-tile sum-of-squares partials
are written by the same wgs_epi_tile the finisher calls. The finisher zeroes the slots and its counter, so the scratch
invariant B4 and MAIN share (zero slots) holds with both builds on RFIN.

Change (one line, `_fork_native_moe`): the "b4" key's defines gain MAIN_EXTRA_DEFINES.

    python3 tools/dl_ly_b4rf_patch.py [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_b4rf.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ('        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)\n',
     '        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",) + MAIN_EXTRA_DEFINES\n', 1),
]
HEADER = "# ly_b4rf: ly_stk1 with ly_rfin's cut-unit finish on the decode 32 B4 build (tools/dl_ly_b4rf_patch.py).\n"


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
