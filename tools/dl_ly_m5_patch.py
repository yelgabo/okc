#!/usr/bin/env python3
"""moe lane round 5 candidates on ly_stk3 (a1c48810), one change each.

    python3 tools/dl_ly_m5_patch.py mid|b4 [--base FILE --pin SHA256|none] --out FILE

mid (-DOKC_RB7=1 -DOKC_F1HOIST=1): round 4's rb7 (routing's first three barriers skip the FAST1 warp, see
tools/dl_ly_m4_patch.py) plus the FAST1 warp's four topk loads issued together before its count loop instead of one
per iteration, so its job scan and first weight copies start one load latency after entry rather than four. The
count loop reads the same values from registers; the atomics and everything after are unchanged. Works on the
instrumented copy too (the stamp lines sit between the anchors and are kept).
b4: see wwarm below.
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dl_ly_m4_patch import END, HMARK, MID_ADD, START, arg, rb7, rep, rwarm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
ZERO = "    S.fcnt[ln] = 0;\n    __syncwarp();\n"
LOAD = "      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n"


def f1hoist(whole):
    head, rest = whole.split(HMARK)
    end = rest.index('\n"""')
    h, tail = rest[:end], rest[end:]
    h = rep(h, ZERO, """#ifndef OKC_F1HOIST
#define OKC_F1HOIST 0
#endif
#if OKC_F1HOIST
    // all four loads in flight at once (M <= 16: M x 8 <= 128 entries); the loop below reads them back by i0
    int f1v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) f1v[k] = k * 32 + ln < p.M * 8 ? __ldg(p.topk_ids + k * 32 + ln) : 0;
#endif
""" + ZERO)
    h = rep(h, LOAD, """#if OKC_F1HOIST
      if (i < p.M * 8) {
        e = (i0 == 0 ? f1v[0] : i0 == 32 ? f1v[1] : i0 == 64 ? f1v[2] : i0 == 96 ? f1v[3] : __ldg(p.topk_ids + i)) - p.lo;
        if (e < 0 || e >= p.nloc) e = -1;
      }
#else
""" + LOAD + "#endif\n")
    tail = rep(tail, MID_ADD, MID_ADD + 'MOE_MID_DEFINES += ("-DOKC_F1HOIST=1",)\n')
    return head + HMARK + h + tail


WJOIN = "  WS_JOIN();  // routing tables and this CTA's stage ranges are ready\n  if ((threadIdx.x & 31) != 0) return;\n"
XDRY = """#if OKC_RWARM
  wgs_ranges(p, S, 1);  // dry run while routing is in flight: the real call below runs from a warm instruction cache
  __syncwarp();
#endif
"""


def wwarm(whole):
    """b4 (-DOKC_RWARM=1): round 4's dry wgs_ranges moved from the x producer to the weights producer warp, which is
    idle from its entry to WS_JOIN. The x producer reaches barrier 7 as early as in ly_stk3 and runs the real call
    behind the dry one on code the dry run has already fetched. The dry run stores nothing, so it cannot race the real
    call's tables (wcp, wcpB, wtot, wtotB, wgr), which still reach the weights producer only through WS_JOIN."""
    whole = rwarm(whole)
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    cu = rep(cu, XDRY, "")
    w0, x0 = cu.index("__device__ void wgs_wproducer("), cu.index("__device__ void wgs_xproducer(")
    k = cu.index(WJOIN, w0)
    if not w0 < k < x0 or cu.count(WJOIN) != 2:
        sys.exit("weights producer join anchor")
    cu = cu[:k] + """#if OKC_RWARM
  wgs_ranges(p, S, 1);  // dry run (stores nothing) while routing is in flight: warms the code for the x producer's call
  __syncwarp();
#endif
""" + cu[k:]
    cu = rep(cu, "#define RW_ARG0 , 0\n", "#define RW_ARG0 , 0\n#define RW_DRY dry\n")
    cu = rep(cu, "#define RW_ARG0\n", "#define RW_ARG0\n#define RW_DRY 0\n")
    cu = rep(cu, "    if (j < nj) { S.wcp[j] =", "    if (j < nj && !RW_DRY) { S.wcp[j] =")
    cu = rep(cu, "  if (lane == 0) { S.wcp[nj] =", "  if (lane == 0 && !RW_DRY) { S.wcp[nj] =")
    cu = rep(cu, "  if (lane < 4) S.wgr[lane] = res;\n", "  if (lane < 4 && !RW_DRY) S.wgr[lane] = res;\n")
    return head + START + cu + END + tail


def main():
    which = sys.argv[1]
    src = ROOT / arg("--base", "muse_decode_layer/cand/ly_stk3.py")
    pin = arg("--pin", PIN)
    dst = ROOT / arg("--out")
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if pin != "none" and sha != pin:
        sys.exit(f"{src.name} is {sha[:8]}, not {pin[:8]}")
    out = {"mid": lambda s: f1hoist(rb7(s)), "b4": wwarm}[which](raw.decode())
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(src.name, sha[:8], "->", dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
