#!/usr/bin/env python3
"""Decode 1 qkv_gate candidates on ly_v1un (88fbae86, checked): one change each, reached only at M = 1 on qkv_gate.

Base kernel at T = 1: sk2_kernel<1, 8, 3, 1, 4, 2, 6, ...> (`qkv8_r176`: 96 CTAs of 24 rows, 4 consumer warps x 2
units x 3 n8 tiles per 18 KB stage, 6-slot ring, 4 escape warps that walk every stage of the ring and hand every slot
back, one accumulator chain per tile). Stamps (docs/runs-decode-layer.md, "Decode 1 qkv_gate in the layer (box)"):
the consumers set the pace, about 0.38 us per stage with data waiting in shared memory, and the escape warps
observe stages late and reach cbar about 0.5 us after the consumer loop ends.

--mode n  (ly_q1n): T = 1 qkv_gate runs the file's own narrow build, the one it already runs at T = 2..8
          (`_de_source`: NACC 2, escape warps off the ring reading their x elements from global memory after
          griddepcontrol.wait, empty barriers count the 4 consumer warps only, 144 KB ring = 5 stages). One
          condition: `not 2<=m<=8` -> `not 1<=m<=8` in the narrow `bfp__run_sk2`.
--mode w8 (ly_q1w8): T = 1 qkv_gate runs the base sk2 kernel with 8 consumer warps of 1 unit each (WSK 8, UPW 1)
          instead of 4 of 2 (new config `qkv1w8_r176`, same 24-row CTAs, same 18 KB stage bytes in the same
          stream order, same 6-slot ring, same escape warps, NACC 1). Three edits: the config row, its dispatch line
          at tokr 8, and `bfp__sk2_name` returning it for (qkv_gate, m == 1).

Neither mode touches a fence, a barrier, griddepcontrol, the launch attributes or any other layer, T or kernel.

    python3 tools/dl_ly_q1_patch.py --mode n|w8 [--base F] [--pin SHA] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODE = sys.argv[sys.argv.index("--mode") + 1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / f"muse_decode_layer/cand/ly_q1{MODE}.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

HEADER = {
    "n": "# ly_q1n: ly_v1un with qkv_gate at T = 1 through the file's narrow sk2 build (NACC 2, escape warps off the ring), as at T = 2..8 (tools/dl_ly_q1_patch.py --mode n).\n",
    "w8": "# ly_q1w8: ly_v1un with qkv_gate at T = 1 on 8 consumer warps of 1 unit (sk2 WSK 8, UPW 1; same stream, ring and escapes) (tools/dl_ly_q1_patch.py --mode w8).\n",
}
EDITS = {
    "n": [
        (" if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n",
         " if layer!='qkv_gate' or not 1<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n"),
    ],
    "w8": [
        ("    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),\n]\n",
         "    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),\n"
         "    ('qkv1w8_r176', 'qkv_gate', 2304, 24, 3, 1, 8, 1, 1, 1),   # M = 1 only (bfp__sk2_name)\n]\n"),
        ("            if bfp_SK2_PICK.get((layer, tokr)) != name:\n                continue\n            hl, nstage = bfp__sk2_geom(cfg, tokr)\n            mt = 2 if tokr == 32 else 1\n            ot = ",
         "            if bfp_SK2_PICK.get((layer, tokr)) != name and (name, tokr) != (\"qkv1w8_r176\", 8):\n                continue\n            hl, nstage = bfp__sk2_geom(cfg, tokr)\n            mt = 2 if tokr == 32 else 1\n            ot = "),
        ("def bfp__sk2_name(layer, m):\n    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]\n",
         "def bfp__sk2_name(layer, m):\n    if layer == \"qkv_gate\" and m == 1:\n        return \"qkv1w8_r176\"\n"
         "    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]\n"),
    ],
}


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    s = raw.decode()
    for old, new in EDITS[MODE]:
        n = s.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:90]!r}")
        s = s.replace(old, new)
    s = HEADER[MODE] + s
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print("base", SRC.name, sha)
    print(DST.name, hashlib.sha256(s.encode()).hexdigest(), len(s))


if __name__ == "__main__":
    main()
