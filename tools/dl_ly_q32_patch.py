#!/usr/bin/env python3
"""qkv_gate at T = 16 / 32 candidates on ly_stk2 (4bbf4010, checked).

Base at T = 16 / 32: the wide sk2 build, `qkv16_r176` = sk2_kernel<1, 16, 3, 1, 4, 2, 5, ...> and `qkv32_r208` =
sk2_kernel<2, 32, 3, 1, 4, 2, 4, ...>, NACC 1, 4 escape warps that walk every ring stage, read their x elements from
the stage's x tile and hand every slot back (the empty barriers count 4 consumer + 4 escape warps).

--mode n  (ly_q32n): T = 9..32 qkv_gate runs the file's narrow build (`_de_source`, already used at T = 1..8): NACC 2,
          escape warps off the ring (each walks every fourth escape entry once, reading x from global memory after
          griddepcontrol.wait; empty barriers count the consumer warps only). Two edits: the narrow extension gains
          dispatch lines for tokr 16 and 32 (same templates as the wide picks, same ring caps r176 / r208, NACC 2;
          the tokr 8 launch is the unchanged fallback line), and the narrow `bfp__run_sk2` condition widens from
          `not 1<=m<=8` to `not 1<=m<=32`. Same packed streams and escape tables (the configs share them).
--mode u  (ly_q32u): the unpooler at T = 9..32 through the narrow build with the wide picks' templates, rings and
          NACC (`unpooler16h_r100`, `unpooler32_r144`, keyed on N 8192, K 4096): the escape protocol only.

No fence, barrier, griddepcontrol, launch attribute or other T changes.

    python3 tools/dl_ly_q32_patch.py --mode n|u [--base F] [--pin SHA] [--out F]   (--out is relative to the repo)
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODE = sys.argv[sys.argv.index("--mode") + 1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk2.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / f"muse_decode_layer/cand/ly_q32{MODE}.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

NACC = 2
HEADER = {
    "n": "# {name}: {base} with qkv_gate at T = 9..32 through the file's narrow sk2 build (NACC 2, escape warps off the ring), as at T = 1..8 (tools/dl_ly_q32_patch.py --mode n).\n",
    "u": "# {name}: {base} with the unpooler at T = 9..32 through the file's narrow sk2 build (escape warps off the ring; the wide picks' templates, rings and NACC) (tools/dl_ly_q32_patch.py --mode u).\n",
}
OLD_DISPATCH = (" dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>"
                "(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'\n")
NEW_DISPATCH = (
    " _,ns16=bfp__sk2_geom(('qkv16n_r176','qkv_gate',2304,24,3,1,4,2,%d,1),16);_,ns32=bfp__sk2_geom(('qkv32n_r208','qkv_gate',2304,24,3,1,4,2,%d,1),32)\n"
    " dispatch=(f'if (tokr == 16) {{ launch_sk2<1, 16, 3, 1, 4, 2, {ns16}, true, false, %d, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}\\n  '\n"
    "  f'if (tokr == 32) {{ launch_sk2<2, 32, 3, 1, 4, 2, {ns32}, true, false, %d, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}\\n  '\n"
    "  f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;')\n"
) % (NACC, NACC, NACC, NACC)
EDITS = [
    (OLD_DISPATCH, NEW_DISPATCH),
    (" if layer!='qkv_gate' or not 1<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n",
     " if layer!='qkv_gate' or not 1<=m<=32:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n"),
]
if MODE == "u":
    EDITS = [
        (OLD_DISPATCH,
         " _,nu16=bfp__sk2_geom(next(c for c in bfp_SK2_CONFIGS if c[0]=='unpooler16h_r100'),16);_,nu32=bfp__sk2_geom(next(c for c in bfp_SK2_CONFIGS if c[0]=='unpooler32_r144'),32)\n"
         " dispatch=(f'if (N == 8192 && K == 4096 && tokr == 16) {{ launch_sk2<1, 16, 4, 1, 4, 1, {nu16}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}\\n  '\n"
         "  f'if (N == 8192 && K == 4096 && tokr == 32) {{ launch_sk2<2, 32, 4, 2, 2, 2, {nu32}, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}\\n  '\n"
         "  f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;')\n"),
        (" if layer!='qkv_gate' or not 1<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n",
         " if not ((layer=='qkv_gate' and 1<=m<=8) or (layer=='unpooler' and 9<=m<=32)):return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n"),
    ]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    s = raw.decode()
    for old, new in EDITS:
        n = s.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:90]!r}")
        s = s.replace(old, new)
    s = HEADER[MODE].format(name=DST.stem, base=SRC.stem) + s
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print("base", SRC.name, sha)
    print(DST.name, hashlib.sha256(s.encode()).hexdigest(), len(s))


if __name__ == "__main__":
    main()
