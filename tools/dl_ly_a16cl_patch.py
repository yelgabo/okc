#!/usr/bin/env python3
"""ly_a16cl: the decode 16 / 32 sliding attention merges each request's splits through DSMEM in one thread-block
cluster (the kernel's existing MODE 2 build, attn_kernel<1, false, 2>: partials in shared memory, cluster barrier,
each CTA sums its rows over the peers and finalizes them) instead of red.global.add into the workspace, the
acq_rel ticket and the last CTA's finalize (MODE 1).

Base ly_stk3 (a1c48810). Round 2 stamps on the base: reds issued 5.17 / 7.66 us after qkv's end, ticket 6.72 / 8.59,
finalize stores 7.95 / 9.57, o's wait 8.51 / 10.10 (decode 16 / 32): 3.3 / 2.4 us between the last reds and o's
wait, all L2 round trips. Host only: _TUNED7 entries (decode 16: 16 splits in one cluster of 16, non-portable size,
the attribute the launcher already sets; decode 32: 8 splits in one cluster of 8), nc 1; --smem sets OKC_CLSM (0:
the cluster launch keeps sm_n, about three CTAs per SM; 81920: two per SM, as ly_a16o2 does for MODE 1). The cluster
barrier (barrier.cluster arrive.release / wait.acquire) orders the DSMEM partials; no global atomic remains in
this merge. Every other shape keeps its plan.

    python3 tools/dl_ly_a16cl_patch.py [--smem 0|81920] [--base F] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
SM = int(sys.argv[sys.argv.index("--smem") + 1]) if "--smem" in sys.argv else 0
DST = ROOT / f"muse_decode_layer/cand/{'ly_a16cl' if SM == 0 else 'ly_a16cl2'}.py"
if "--base" in sys.argv:
    SRC, PIN = ROOT / sys.argv[sys.argv.index("--base") + 1], None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = [
    ("#define OKC_CLSM 0 ", f"#define OKC_CLSM {SM} "),
    ("_TUNED7 = {\n",
     "_TUNED7 = {\n    (16, 1, True, 1): dict(cs=16, ncl=1, nc=1),   # ly_a16cl: DSMEM merge, one cluster per request\n"
     "    (32, 1, True, 1): dict(cs=8, ncl=1, nc=1),    # ly_a16cl\n"),
]


def main():
    text = SRC.read_text()
    if PIN and hashlib.sha256(text.encode()).hexdigest() != PIN:
        sys.exit("base SHA256 mismatch")
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"anchor found {text.count(old)} times: {old!r}")
        text = text.replace(old, new)
    text = (f"# {DST.stem}: ly_stk3 with the decode 16 / 32 sliding attention merged through DSMEM in one cluster per "
            f"request (cluster 16 / 8, OKC_CLSM {SM}; tools/dl_ly_a16cl_patch.py).\n") + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
