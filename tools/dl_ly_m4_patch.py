#!/usr/bin/env python3
"""moe lane round 4 candidates on ly_stk2 (4bbf4010), one change each.

    python3 tools/dl_ly_m4_patch.py rb7|rwarm [--base FILE --pin SHA256|none] --out FILE

rb7 (MID, -DOKC_RB7=1): at decode 16 the FAST1 warp (warp NW - 1) holds none of routing's E = 8 M <= 224 entries, but
routing's first three CTA barriers waited for it to finish its four topk loads, job scan, barrier inits and first
weight copies (about 1.2 us after warp 0 is ready, prologue stamps). Those three barriers become named barrier 3 over
warps 0..NW-2; warp 0 zeroes the FAST1 warp's count row in its place, and the FAST1 warp joins only routing's last
CTA barrier, which orders every routing table before any use exactly as before.
rwarm (B4, -DOKC_RWARM=1): the x producer warp runs wgs_ranges once on a fixed two-job table (no routing input read)
before it waits for routing; the real call after routing then runs from a warm instruction cache (first call 2.66 us,
an immediate second call 0.90 us). The dry call's shared writes (wcp, wcpB, wtot, wtotB, wgr) are read by no one
before the real call rewrites them, and the real call is unchanged.
Both work on the instrumented copies too (anchors avoid the stamp lines).
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846"
HMARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
START, END = "MOE_CUDA_SRC = r'''\n", "\n'''\nMOE_CPP_SRC"
MID_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'
B4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",'


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rep(s, old, new, count=1):
    n = s.count(old)
    if n != count:
        sys.exit(f"anchor found {n} times, {count} expected: {old[:100]!r}")
    return s.replace(old, new)


def rb7(whole):
    head, rest = whole.split(HMARK)
    end = rest.index('\n"""')
    h, tail = rest[:end], rest[end:]
    h = rep(h, "__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {\n",
            """__device__ void early_issue(const Params& p, Smem& S);
#ifndef OKC_RB7
#define OKC_RB7 0
#endif
#if OKC_RB7 && !FAST1_ON
#error "OKC_RB7: the last warp's routing exemption assumes the FAST1 warp"
#endif
#if OKC_RB7
// routing's first three CTA barriers over warps 0..NW-2 only (named barrier 3) when the FAST1 warp holds no entries
#define RT_SYNC() do { if (rb7) asm volatile("bar.sync 3, %0;\\n" ::"n"(NT - 32) : "memory"); else CTA_SYNC(); } while (0)
#else
#define RT_SYNC() CTA_SYNC()
#endif
__device__ void route(const Params& p, Smem& S) {
""")
    h = rep(h, "  const int i0 = warp * CH, i1 = min(E, i0 + CH);\n  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;\n",
            """  const int i0 = warp * CH, i1 = min(E, i0 + CH);
#if OKC_RB7
  // E <= (NW - 1) x 32: the FAST1 warp loads, counts and places no entry, so it only joins the last barrier; warp 0
  // zeroes its count row (read by warp 0's scan after barrier 2)
  const bool rb7 = E <= (NW - 1) * 32;
  if (rb7 && warp == NW - 1) { CTA_SYNC(); return; }
  if (rb7 && tid < 32) S.wc[NW - 1][tid] = 0;
#endif
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
""")
    r0 = h.index("__device__ void route(const Params& p, Smem& S) {")
    r1 = h.index("\n}\n", r0)
    body = h[r0:r1]
    if body.count("  CTA_SYNC();\n") not in (3, 4):  # (4 in the stamped copies: a stamp follows the last one)
        sys.exit("route barrier count")
    body = body.replace("  CTA_SYNC();\n", "  RT_SYNC();\n", 3)  # the first three; the last stays a CTA barrier
    h = h[:r0] + body + h[r1:]
    tail = rep(tail, MID_ADD, MID_ADD + 'MOE_MID_DEFINES += ("-DOKC_RB7=1",)\n')
    return head + HMARK + h + tail


def rwarm(whole):
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    cu = rep(cu, "__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)\n",
             """#ifndef OKC_RWARM
#define OKC_RWARM 0
#endif
#if OKC_RWARM
#define RW_PARAM , int dry
#define RW_PARAMD , int dry = 0
#define RW_ARG , dry
#define RW_CNT(v) (dry ? 8 : (v))
#else
#define RW_PARAM
#define RW_PARAMD
#define RW_ARG
#define RW_CNT(v) (v)
#endif
__device__ void wgs_ranges(const Params& p, Smem& S RW_PARAM);  // below (needs the cost tables)
""")
    cu = rep(cu, "__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n",
             "__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x RW_PARAMD) {\n")
    cu = rep(cu, "  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;\n",
             "  const uint32_t cs = (uint32_t)wgs_cost(pb, RW_CNT(S.jC[lo])), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;\n")
    cu = rep(cu, "__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\n  const int lane = threadIdx.x & 31;\n  const int nj = S.njF + S.njS;\n",
             """__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S RW_PARAM) {
  const int lane = threadIdx.x & 31;
#if OKC_RWARM
  const int nj = dry ? 2 : S.njF + S.njS;  // dry: two 8-token jobs, no routing table read
#else
  const int nj = S.njF + S.njS;
#endif
""")
    cu = rep(cu, "      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;\n      ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;\n",
             "      ia = ((uint32_t)wgs_cost(0, RW_CNT(S.jC[j])) * 16u + OKC_WGSU) << 5;\n      ib = ((uint32_t)wgs_cost(1, RW_CNT(S.jC[j])) * 16u + OKC_WGSUB) << 4;\n")
    cu = rep(cu, "  else res = wgs_stage_at(S, nj, pbl, tot * (uint32_t)(c + kk) / (uint32_t)G);\n",
             "  else res = wgs_stage_at(S, nj, pbl, tot * (uint32_t)(c + kk) / (uint32_t)G RW_ARG);\n")
    cu = rep(cu, '  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");\n',
             '''#if OKC_RWARM
  wgs_ranges(p, S, 1);  // dry run while routing is in flight: the real call below runs from a warm instruction cache
  __syncwarp();
#endif
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
''')
    cu = rep(cu, "  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)\n",
             "  wgs_ranges(p, S RW_ARG0);  // all 32 lanes (warp-parallel)\n")
    cu = rep(cu, "#define RW_CNT(v) (dry ? 8 : (v))\n", "#define RW_CNT(v) (dry ? 8 : (v))\n#define RW_ARG0 , 0\n")
    cu = rep(cu, "#define RW_CNT(v) (v)\n", "#define RW_CNT(v) (v)\n#define RW_ARG0\n")
    tail = rep(tail, B4, B4 + ' "-DOKC_RWARM=1",')
    return head + START + cu + END + tail


def main():
    which = sys.argv[1]
    src = ROOT / arg("--base", "muse_decode_layer/cand/ly_stk2.py")
    pin = arg("--pin", PIN)
    dst = ROOT / arg("--out")
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if pin != "none" and sha != pin:
        sys.exit(f"{src.name} is {sha[:8]}, not {pin[:8]}")
    out = {"rb7": rb7, "rwarm": rwarm}[which](raw.decode())
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(src.name, sha[:8], "->", dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
