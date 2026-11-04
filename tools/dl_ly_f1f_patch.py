#!/usr/bin/env python3
"""ly_f1f: the decode 1 TINY FAST1 warp computes its first copies' unit range and experts warp-parallel.

Stamps on ly_stk3 (tools/dl_ly_mtf2_patch.py) put 0.50 us between the touched-expert mask and the second copy's
issue: 0.12 for popc + two muldiv_g, then about 0.19 per copy, mostly the per-copy __fns on lane 0. Under OKC_F1FAST
(TINY only) every lane computes the CTA's unit range for k = lane + 1 touched experts while the topk loads are in
flight, the range for k = popc(m) arrives by one shuffle, and each first unit's expert (the ((v >> 9) + 1)-th set bit
of m, as __fns gave) is one ballot over lanes. Lane 0 then runs the existing init, fences and copy loop unchanged.
Same unit ranges, experts and addresses as the control; no fence, barrier or ordering changed.

    python3 tools/dl_ly_f1f_patch.py --base muse_decode_layer/cand/ly_stk3.py --out muse_decode_layer/cand/ly_f1f.py [--fast1 3] [--fence-early]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
FAST1 = sys.argv[sys.argv.index("--fast1") + 1] if "--fast1" in sys.argv else None
FENCE_EARLY = "--fence-early" in sys.argv
FENCE = '      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n'
PFENCE = '      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n'
EARLY_EDITS = [
    # the proxy fence right after lane 0's inits, while the topk loads are in flight; it still sits between the
    # inits (the only generic-proxy shared writes the copies' barriers depend on) and the first cp.async.bulk
    (FENCE + "    }\n    if (ln < E8)",
     FENCE + "#if OKC_F1FAST\n" + PFENCE + "#endif\n    }\n    if (ln < E8)"),
    ("#endif\n" + PFENCE + "#if OKC_SPLIT || OKC_F1FAST\n", "#endif\n#if !OKC_F1FAST\n" + PFENCE + "#endif\n#if OKC_SPLIT || OKC_F1FAST\n"),
]
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"

SEG_EDITS = [
    ("#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits\n",
     "#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits\n"
     "#ifndef OKC_F1FAST\n#define OKC_F1FAST 0  // FAST1 unit range and experts warp-parallel (no per-copy __fns on lane 0)\n#endif\n"),
    ("    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n",
     "    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n"
     "#if OKC_F1FAST\n"
     "    // while the loads are in flight: lane l holds this CTA's static unit range for k = l + 1 touched experts\n"
     "    const int fkS = (ln + 1) * RB_A - dyn_pool((ln + 1) * RB_A, gridDim.x);\n"
     "    const float frG = __fdividef(1.f, (float)gridDim.x);\n"
     "    const int fv0 = muldiv_g(fkS, blockIdx.x, gridDim.x, frG), fv1 = muldiv_g(fkS, blockIdx.x + 1, gridDim.x, frG);\n"
     "#endif\n"),
    ("    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n",
     "    m = __reduce_or_sync(0xffffffffu, m);\n"
     "#if OKC_F1FAST\n"
     "    const int fk = __popc(m);\n"
     "    const int fv0k = __shfl_sync(0xffffffffu, fv0, fk > 0 ? fk - 1 : 0), fv1k = __shfl_sync(0xffffffffu, fv1, fk > 0 ? fk - 1 : 0);\n"
     "    const int fn = fk > 0 ? min(OKC_FAST1, fv1k - fv0k) : 0;\n"
     "    const bool fbit = (m >> ln) & 1u;\n"
     "    const int frank = __popc(m & ((1u << ln) - 1u));\n"
     "    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];\n"
     "#pragma unroll\n"
     "    for (int i = 0; i < OKC_FAST1; ++i)  // the ((v >> RB_A_LOG) + 1)-th set bit of m, as __fns(m, 0, .) gives\n"
     "      fex[i] = __ffs(__ballot_sync(0xffffffffu, fbit && frank == ((fv0k + i) >> RB_A_LOG))) - 1;\n"
     "#endif\n#endif\n"),
    ("#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#else\n",
     "#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#elif OKC_F1FAST\n      const int v0 = fv0k, n = fn;\n#else\n"),
    ("#if OKC_SPLIT\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n",
     "#if OKC_SPLIT || OKC_F1FAST\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n"),
]
TOP_EDITS = [
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_F1FAST=1",)   # ly_f1f: FAST1 range and experts warp-parallel (decode 1 issue latency)\n'),
]


def sub(seg, old, new):
    if seg.count(old) != 1:
        sys.exit(f"anchor found {seg.count(old)} times: {old!r}")
    return seg.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base is {sha}, expected {PIN}")
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    b = text.index('\n"""\n', a)
    seg = text[a:b]
    for old, new in SEG_EDITS + (EARLY_EDITS if FENCE_EARLY else []):
        seg = sub(seg, old, new)
    text = text[:a] + seg + text[b:]
    for old, new in TOP_EDITS:
        text = sub(text, old, new)
    if FAST1:
        text = sub(text, '"-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2",', f'"-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1={FAST1}",')
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
