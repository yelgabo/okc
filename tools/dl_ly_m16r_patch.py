#!/usr/bin/env python3
"""ly_m16r: the decode 16 MID MoE prologue issues its global loads before it uses them.

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). Target: the MID ramp at decode 16 (first ring
slot full 3.7 to 4.1 us after the first CTA's entry on the mean CTA, 5.0 to 5.3 on the latest; route() done at 2.73;
docs/runs-decode-layer.md "Decode 16 / 32 sliding MoE (box)"). Two serialized global round trips in helper0's prologue,
both seen in the MID SASS (nvcc 13.0, tools/ly_micro_sass.py):

1. The FAST1 warp's OKC_SPLIT count loop runs one topk_ids load per 32 entries and consumes it (MATCH.ANY, ATOMS) before
   the next load issues: at M = 16 (128 entries) four dependent L2 round trips before the warp can compute its first
   units and issue their bulk copies, and before it reaches route()'s first CTA barrier, which every thread waits on.
   Here the warp issues all four loads first (registers), then runs the same match / count steps on them.
2. Thread 0 reads the launch generation (a volatile load of p.ctr[3]) and stores S.gen / S.ssp from it before route(),
   so its own route() topk load issues only after that round trip returns, and route()'s first CTA barrier waits for it.
   Here thread 0 issues the same load at the same point but stores S.gen / S.ssp inside route() right after its first
   CTA barrier; route()'s later CTA barriers order the stores before every reader (the ss zeroing after route(), the
   epilogues behind WS_JOIN), as the old barrier did. The generation changes only at this launch's grid barrier, after
   every read here.

Values, counts, job lists, unit ranges, the copies issued and every fence, barrier and wait are unchanged; the output is
the base's. Only MID gets -DOKC_F1HOIST=1, so TINY / TINY2 preprocess to the base text.

    python3 tools/dl_ly_m16r_patch.py [--base FILE] [--pin SHA256] [--out FILE]
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_m16r.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
MID_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'


def once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times: {old[:80]!r}")
    return s.replace(old, new)


def guard(new, old):
    return "#if OKC_F1HOIST\n" + new + "#else\n" + old + "#endif\n"


F1_OLD = (
    "    S.fcnt[ln] = 0;\n"
    "    __syncwarp();\n"
    "    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {\n"
    "      const int i = i0 + ln;\n"
    "      int e = -1;\n"
    "      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n"
    "      const unsigned peers = __match_any_sync(0xffffffffu, e);\n"
    "      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n"
    "    }\n")
F1_NEW = (
    "    int f1v[4];  // MID: M <= 16, so at most 4 x 32 entries; every load is in flight before the first use\n"
    "#pragma unroll\n"
    "    for (int k = 0; k < 4; ++k) { const int i = k * 32 + ln; f1v[k] = i < p.M * 8 ? __ldg(p.topk_ids + i) : 0; }\n"
    "    S.fcnt[ln] = 0;\n"
    "    __syncwarp();\n"
    "#pragma unroll\n"
    "    for (int k = 0; k < 4; ++k) {\n"
    "      const int i = k * 32 + ln;\n"
    "      int e = -1;\n"
    "      if (i < p.M * 8) { e = f1v[k] - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n"
    "      const unsigned peers = __match_any_sync(0xffffffffu, e);\n"
    "      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n"
    "    }\n"
    "    for (int i0 = 128; i0 < p.M * 8; i0 += 32) {\n"
    "      const int i = i0 + ln;\n"
    "      int e = -1;\n"
    "      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n"
    "      const unsigned peers = __match_any_sync(0xffffffffu, e);\n"
    "      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n"
    "    }\n")
GEN_OLD = (
    "    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)\n"
    "    S.gen = gen;\n"
    "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n")
GEN_NEW = "    f1h_gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation; route() stores S.gen / S.ssp\n"
TID0 = "  if (tid == 0) {\n#if !FAST1_ON\n"
ROUTE_DEF = "__device__ void route(const Params& p, Smem& S) {\n"
ROUTE_SYNC1 = "#endif\n  CTA_SYNC();\n  for (int base = i0; base < i1; base += 32) {\n"
ROUTE_CALL = "  route(p, S);\n#if OKC_WGS && OKC_WGSPR\n"


def patch_helper0(h):
    h = once(h, F1_OLD, guard(F1_NEW, F1_OLD), "FAST1 count loop")
    h = once(h, GEN_OLD, guard(GEN_NEW, GEN_OLD), "generation read")
    h = once(h, TID0, guard("  int f1h_gen = 0;\n", "") + TID0, "thread 0 block")
    h = once(h, ROUTE_DEF, guard("__device__ void route(const Params& p, Smem& S, int gen) {\n", ROUTE_DEF), "route def")
    h = once(h, ROUTE_SYNC1, "#endif\n  CTA_SYNC();\n" + guard(
        "  if (tid == 0) { S.gen = gen; S.ssp = (gen & 1) ? p.ss1 : p.ss; }  // ordered by route()'s later CTA barriers\n",
        "") + "  for (int base = i0; base < i1; base += 32) {\n", "route sync 1")
    h = once(h, ROUTE_CALL, guard("  route(p, S, f1h_gen);\n", "  route(p, S);\n") + "#if OKC_WGS && OKC_WGSPR\n",
             "route call")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(SRC))
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--out", default=str(DST))
    a = ap.parse_args()
    src, dst = ROOT / a.base, ROOT / a.out
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != a.pin:
        sys.exit(f"{src.name} is {sha}, expected {a.pin[:8]}")
    whole = raw.decode()
    if whole.count(MARK) != 1:
        sys.exit("helper0 mark")
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    helper = patch_helper0(helper)
    tail = once(tail, MID_ADD, MID_ADD + 'MOE_MID_DEFINES += ("-DOKC_F1HOIST=1",)\n', "MID define")
    out = ("# ly_m16r: ly_stk1 with the decode 16 MID MoE prologue's global loads issued before their uses "
           "(tools/dl_ly_m16r_patch.py).\n" + head + MARK + helper + tail)
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
