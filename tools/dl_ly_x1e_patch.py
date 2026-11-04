#!/usr/bin/env python3
"""ly_x1e: at decode 1 (TINY MoE build, M = 1) the compute warps load phase A's x registers right after the entry
griddepcontrol.wait instead of after route() / WS_JOIN.

At M = 1 every routing entry i < 8 belongs to token 0 (S.ent[st] >> 3 == 0), so a one-entry job's x registers are
the same for every phase A job: lanes 0..3 hold row 0's slice for their warp, the other lanes zeros (load_x_regs with
cnt == 1). xm is written by the route kernel, the MoE's immediate stream predecessor; the entry griddepcontrol.wait
returns only after that grid has completed and its writes are visible, and the base's own x prefetch.global.L2
already reads xm right there. The skinny loop skips load_x_regs for a phase A job while the registers hold that form
(p.M == 1 and S.jC[j] == 1, checked per job) and reloads otherwise. Phase B, the helper warp, every fence, barrier,
mbarrier, acquire, atomic, wait and trigger are unchanged; the x values and every arithmetic operation are the same.
Gated by -DOKC_X1E=1 in MOE_TINY_DEFINES only: TINY2 and MID preprocess to the base's device code.

    python3 tools/dl_ly_x1e_patch.py [--base muse_decode_layer/cand/ly_stk3.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
DST = ROOT / "muse_decode_layer/cand/ly_x1e.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

MAIN_EDITS = [
    ("#ifndef OKC_TINY\n#define OKC_TINY 0\n#endif\n",
     "#ifndef OKC_TINY\n#define OKC_TINY 0\n#endif\n#ifndef OKC_X1E\n#define OKC_X1E 0\n#endif\n"),
    ("constexpr int X1P = 512 + 8;",
     "#if OKC_X1E\n"
     "// load_x_regs<false> for a one-entry job whose entry is token 0\n"
     "__device__ __forceinline__ void load_x_tok0(const Params& p, int lane, int warp, uint4 (&x)[2 * NCH]) {\n"
     "  const int g = lane >> 2, t = lane & 3;\n"
     "  if (g == 0) {\n"
     "    const uint4* src = reinterpret_cast<const uint4*>(p.x + warp * (64 * NCH) + 16 * t);\n"
     "#pragma unroll\n"
     "    for (int c = 0; c < NCH; ++c) {\n"
     "      x[2 * c] = __ldg(src + c * 8);\n"
     "      x[2 * c + 1] = __ldg(src + c * 8 + 1);\n"
     "    }\n"
     "  } else {\n"
     "#pragma unroll\n"
     "    for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n"
     "  }\n"
     "}\n"
     "#endif\n\n"
     "constexpr int X1P = 512 + 8;"),
    ("template <bool PB>\n__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n",
     "template <bool PB>\n"
     "#if OKC_X1E\n"
     "__device__ __forceinline__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n"
     "                                               uint4 (&x0)[2 * NCH], bool xt0) {\n"
     "#else\n"
     "__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n"
     "#endif\n"),
    ("  const int seq0 = seq;\n  uint4 x0[2 * NCH];\n  int curj = -1;\n",
     "  const int seq0 = seq;\n#if !OKC_X1E\n  uint4 x0[2 * NCH];\n#endif\n  int curj = -1;\n"),
    ("        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n    }\n    const int q = seq0 + i;\n",
     "        __syncwarp();\n      }\n"
     "#if OKC_X1E\n"
     "      const bool t0 = !PB && p.M == 1 && S.jC[j] == 1;   // x0 already holds this job's rows (entry load)\n"
     "      if (!(t0 && xt0))\n"
     "#endif\n"
     "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n"
     "#if OKC_X1E\n"
     "      xt0 = t0;\n"
     "#endif\n"
     "    }\n    const int q = seq0 + i;\n"),
    ("__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n",
     "#if OKC_X1E\n"
     "__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf, uint4 (&x0)[2 * NCH], bool xt0) {\n"
     "#else\n"
     "__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n"
     "#endif\n"),
    ("  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n",
     "#if OKC_X1E\n"
     "  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq, x0, xt0);\n"
     "#else\n"
     "  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
     "#endif\n"),
    ("  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n",
     "#if OKC_X1E\n"
     "  // xm is the route kernel's output: complete and visible once the griddepcontrol.wait above returns\n"
     "  uint4 x1e[2 * NCH];\n"
     "  const bool xt0 = p.M == 1;\n"
     "  if (xt0) load_x_tok0(p, tid & 31, WARP_IDX(tid), x1e);\n"
     "#endif\n"
     "  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n"),
    ("#else\n  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n#endif\n",
     "#elif OKC_X1E\n"
     "  run_phase<false, BIG>(p, S, xbuf, x1e, xt0);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf, x1e, false);\n  DBG(3);\n"
     "#else\n  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n#endif\n"),
]
TAIL = ('\n# ly_x1e: the decode 1 (TINY) MoE loads phase A\'s token-0 x registers right after its entry wait\n'
        'MOE_TINY_DEFINES += ("-DOKC_X1E=1",)\n')


def sub(seg, old, new):
    if seg.count(old) != 1:
        sys.exit(f"anchor found {seg.count(old)} times: {old[:90]!r}")
    return seg.replace(old, new)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit(f"base {SRC} is not {PIN}")
    text = raw.decode()
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    b = text.index('\n"""\n', a)
    seg = text[a:b]
    for old, new in MAIN_EDITS:
        seg = sub(seg, old, new)
    text = text[:a] + seg + text[b:]
    text = "# ly_x1e: ly_stk3 + the decode 1 MoE's x registers loaded at entry (tools/dl_ly_x1e_patch.py).\n" + text
    text = text.rstrip("\n") + "\n" + TAIL
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
