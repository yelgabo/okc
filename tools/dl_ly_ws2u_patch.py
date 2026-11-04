#!/usr/bin/env python3
"""ly_ws2u: ly_ws2t's 16-token MID jobs with the one-tile unit loop kept as the control's own function body.

ly_ws2t (tools/dl_ly_ws2t_patch.py) removed the decode 16 split re-stream but put the two-tile and one-tile unit
loops in one function body, and every unit retired slower (1.18 -> 1.25 us, registers 152 -> 164); its cost split
also ran in the routing prologue (route stamp 2.8 -> 4.1 us). Here, from rv_k_combo2n.py (3ac83fb4, SHA checked):

- skinny_ring_ws and ws_helper stay as the control's text. ly_ws2t's two-tile versions are added beside them as
  skinny_ring_ws2 / ws_helper2, and a CTA whose unit range holds no job of more than 8 tokens runs the control's
  function (CTA-uniform choice from S.jC, made after the WS_JOIN that publishes the job table). ws_helper's leading
  WS_JOIN moves into the selecting wrapper, at the same program point. red1 goes to the end of Smem so every other
  member keeps its offset.
- variant a: ly_ws2t's cost split as built (ws_ranges after route() on warp 0, and again in the FAST1 warp).
- variant b: the cost split is a closed form computed once, in the FAST1 warp, which already holds the per-expert
  counts and runs beside warp 0's route(); it writes S.wsr before the WS_JOIN and warp 0 computes nothing after
  route(). The closed form needs one job per local expert, which holds for MID (M <= 16, jobs of up to 16 tokens).

All edits are under `#if OKC_SKT > 8` (MID only gets -DOKC_SKT=16), so TINY / TINY2 preprocess to the original
code. No fence, barrier, mbarrier protocol, griddepcontrol or launch attribute is removed or changed.

    python3 tools/dl_ly_ws2u_patch.py [--variant a|b] [--tl] [--out NAME]
      default: variant b -> muse_decode_layer/cand/ly_ws2u.py; --variant a -> ly_ws2u_a.py
      --tl: the same edits on ly_tl16.py (rebuild it first with tools/ly_moe_tl16.py) -> ly_ws2u{a,b}_tl.py
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dl_ly_ws2t_patch as T  # noqa: E402

ROOT = T.ROOT
SRC = T.SRC
PIN = T.PIN
MARK = T.MARK
MID_ADD = T.MID_ADD
once, guard = T.once, T.guard

WS_A = "template <bool PB>\n__device__ void skinny_ring_ws(const Params& p"
WS_B = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {"
HP_A = "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
HP_JOIN = "  WS_JOIN();  // routing tables and this CTA's ranges are ready\n"
HP_END = "  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n"
SM_END = "  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n"
CALL = "  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
RNG = T_RNG = ("  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n"
               "    const int G = gridDim.x, c = blockIdx.x;\n"
               "    const float rG = __fdividef(1.f, (float)G);\n"
               "    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n"
               "    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n"
               "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n"
               "  }\n")
F1_END = "      S.preissued = n > 0 ? n : 0;\n    }\n"
F1 = "    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n"

X1A = (
    "// rows 8..15 of a two-tile job into this warp's x1 rows, asynchronously (zero-filled past the job's tokens);\n"
    "// the caller waits (cp.async.wait_all, __syncwarp) before the first MMA that reads them\n"
    "template <bool PB>\n"
    "__device__ __forceinline__ void load_x1_async(const Params& p, const Smem& S, int j, int lane, int warp, __nv_bfloat16* x1s) {\n"
    "  const int cnt = S.jC[j], st = S.jS[j];\n"
    "#pragma unroll\n"
    "  for (int m = 0; m < 16; ++m) {\n"
    "    const int r = m >> 1, c = 32 * (m & 1) + lane;\n"
    "    const bool ok = 8 + r < cnt;\n"
    "    const int rr = ok ? r : 0;   // row 8 exists: two-tile jobs have more than 8 tokens\n"
    "    const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + rr) * ID : p.x + (size_t)(S.ent[st + 8 + rr] >> 3) * HD;\n"
    "    const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n"
    "    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16, %2;\\n\" ::\"r\"(smem_u32(x1s + r * X1P + c * 8)), \"l\"(src), \"r\"(ok ? 16 : 0) : \"memory\");\n"
    "  }\n"
    "}\n")
ANY2 = (
    "// OKC_WS2INL 0: the two-tile functions as separate calls (kernel 160 registers, 272 B stack, local loads). Inlined\n"
    "// (166 registers) the one-tile loop has the control's instruction count (tools/ly_ws2u_sass_cmp.py).\n"
    "#ifndef OKC_WS2INL\n#define OKC_WS2INL 1\n#endif\n"
    "#if OKC_WS2INL\n#define WS2_FN\n#else\n#define WS2_FN __noinline__\n#endif\n"
    "// a CTA-uniform choice: does unit range [v0, v1) of one phase (rbs units per job) hold a two-tile job?\n"
    "__device__ __forceinline__ bool ws_any2(const Smem& S, int v0, int v1, int rbs) {\n"
    "  if (v1 <= v0) return false;\n"
    "  const int njF = S.njF;\n"
    "  for (int jj = v0 / rbs; jj <= (v1 - 1) / rbs; ++jj)\n"
    "    if (S.jC[njF + jj] > 8) return true;\n"
    "  return false;\n"
    "}\n")

# closed-form cost split: one job per local expert (lane), jobs in expert order, a unit of a job of more than 8
# tokens costs SPW2, else SPW1 (default 4 / 3; box timeline sweep of 5/4, 4/3, 7/5, 3/2, 5/3). Costs stay below 2^24 for muldiv_g:
# RB_A * SPW2 * 32 experts * 132 CTAs <= 512 * 7 * 32 * 132 = 15.1e6.
SPLIT = (
    "#ifndef OKC_SPW1\n#define OKC_SPW1 3\n#endif\n#ifndef OKC_SPW2\n#define OKC_SPW2 4\n#endif\n"
    "#ifndef OKC_SPB1\n#define OKC_SPB1 OKC_SPW1\n#endif\n#ifndef OKC_SPB2\n#define OKC_SPB2 OKC_SPW2\n#endif\n"
    "constexpr int SPW1 = OKC_SPW1, SPW2 = OKC_SPW2, SPB1 = OKC_SPB1, SPB2 = OKC_SPB2;  // phase A / phase B unit costs\n"
    "template <int K0, int K1, int SPW1, int SPW2>\n"
    "__device__ __forceinline__ void ws_split(int tot, int lane, int c, int G, int (&r)[4]) {\n"
    "  const unsigned mT = __ballot_sync(0xffffffffu, tot > 0), m2 = __ballot_sync(0xffffffffu, tot > 8);\n"
    "  const unsigned lt = (1u << lane) - 1u;\n"
    "  const int jb = __popc(mT & lt), n2b = __popc(m2 & lt), J = __popc(mT), N2 = __popc(m2);\n"
    "  const int w = tot > 8 ? SPW2 : SPW1, pw = SPW1 * jb + (SPW2 - SPW1) * n2b;\n"
    "  const int cw = SPW1 * J + (SPW2 - SPW1) * N2;\n"
    "  const float rG = __fdividef(1.f, (float)G);\n"
    "#pragma unroll\n"
    "  for (int k = K0; k < K1; ++k) {\n"
    "    const int rbs = k < 2 ? RB_A : RB_B;\n"
    "    const int t = muldiv_g(rbs * cw, c + (k & 1), G, rG);\n"
    "    const int P = rbs * pw;\n"
    "    const unsigned bal = __ballot_sync(0xffffffffu, tot > 0 && P + rbs * w > t);\n"
    "    const int e = bal ? __ffs(bal) - 1 : 0;\n"
    "    const int o = t - P;\n"
    "    const int u = jb * rbs + (w == SPW1 ? (o + SPW1 - 1) / SPW1 : (o + SPW2 - 1) / SPW2);\n"
    "    const int ue = __shfl_sync(0xffffffffu, u, e);\n"
    "    r[k] = bal ? ue : J * rbs;\n"
    "  }\n"
    "}\n")


def patch_helper(h, variant):
    ht = T.patch_helper(h)
    a, b = ht.index(WS_A), ht.index(WS_B)
    ws2 = ht[a:b].replace("__device__ void skinny_ring_ws(", "__device__ WS2_FN void skinny_ring_ws2(", 1)
    # rows 8..15 by cp.async, waited for after the unit's ring-slot wait: the blocking loader (load_x1_ws) held a
    # two-tile CTA's first unit 3.7 us (box timeline, loads issued while the weight stream fills HBM's queue)
    ws2 = once(ws2, "  bool two = false;\n", "  bool two = false, x1p = false;\n", "ws2 x1p")
    ws2 = once(ws2, "      if (two) load_x1_ws<PB>(p, S, j, lane, warp, x1s);\n",
               "      if (two) { load_x1_async<PB>(p, S, j, lane, warp, x1s); x1p = true; }\n", "ws2 x1 async")
    ws2 = once(ws2, "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n",
               "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n"
               "    if (x1p) { asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\"); __syncwarp(); x1p = false; }\n",
               "ws2 x1 wait")
    c = ht.index(HP_A)
    d = ht.index(HP_END, c) + len(HP_END)
    hp2 = ht[c:d]
    if hp2.count(HP_JOIN) != 1:
        sys.exit("ws_helper2 join")
    hp2 = hp2.replace(HP_JOIN, "").replace("__device__ void ws_helper(", "__device__ WS2_FN void ws_helper2(", 1)

    # the ws2t blocks that do not touch the unit loops: SKT define, WS assert, x1 loader + ws_ranges, RED_W
    h = once(h, "#define OKC_SKT 8\n", "#ifndef OKC_SKT\n#define OKC_SKT 8\n#endif\n", "SKT define")
    h = once(h, 'static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\n',
             'static_assert(!OKC_WS || OKC_SKT == 8 || OKC_SKT == 16, "OKC_WS: skinny jobs of one or two n-tiles");\n',
             "WS assert")
    x1 = ht[ht.index("#if OKC_SKT > 8\ntemplate <bool PB>\n__device__ __forceinline__ void load_x1_ws("):]
    x1 = x1[: x1.index("#else\n#endif\n") + len("#else\n#endif\n")]
    anchor = "#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n"
    h = once(h, anchor, x1 + guard(X1A + ANY2 + SPLIT, "") + anchor, "x1ws")
    h = once(h, "constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\n",
             "constexpr int RED_W = (OKC_SKT > 8 && !OKC_WS) ? 8 : 4;   // WS two-tile builds: tile 1 in red1\n", "RED_W")
    h = once(h, SM_END, SM_END[: -len("};\n")]
             + "#if OKC_WS && OKC_SKT > 8\n  alignas(16) float red1[NRB][NW][32][4];  // tile 1 (tokens 8..15) of two-tile units (ws2 only; float4 access)\n#endif\n};\n",
             "red1")

    # two-tile copies beside the control's functions
    b = h.index(WS_B)
    h = h[:b] + guard(ws2, "") + h[b:]
    c = h.index(HP_A)
    d = h.index(HP_END, c) + len(HP_END)
    hp1 = h[c:d]
    if hp1.count(HP_JOIN) != 1:
        sys.exit("ws_helper join")
    hp1_nojoin = hp1.replace(HP_JOIN, "").replace("void ws_helper(", "void ws_helper1(", 1)
    sel = ("__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
           + HP_JOIN +
           "  if (ws_any2(S, S.wsr[0], S.wsr[1], RB_A) || ws_any2(S, S.wsr[2], S.wsr[3], RB_B)) ws_helper2(p, S, ring);\n"
           "  else ws_helper1(p, S, ring);\n"
           "}\n")
    h = h[:c] + guard(hp1_nojoin + hp2 + sel, hp1) + h[d:]

    h = once(h, CALL, guard(
        "  v0 = S.wsr[PB ? 2 : 0]; v1 = S.wsr[PB ? 3 : 1];\n"
        "  if (ws_any2(S, v0, v1, rbs)) skinny_ring_ws2<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
        "  else skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n",
        CALL), "run_phase dispatch")

    if variant == "a":
        h = once(h, RNG, guard(
            "  if (tid < 32) {  // this CTA's unit ranges, split by cost (ws_ranges); run_phase and FAST1 use the same split\n"
            "    const int G = gridDim.x, c = blockIdx.x, tot = tid < p.nloc ? S.cnt[tid] : 0;\n"
            "    int r[4];\n"
            "    ws_ranges<4>(tot, tid, c, G, r);\n"
            "    if (tid == 0) { S.wsr[0] = r[0]; S.wsr[1] = r[1]; S.wsr[2] = r[2]; S.wsr[3] = r[3]; }\n"
            "  }\n", RNG), "ws ranges")
        h = once(h, F1, guard(
            "    (void)nS1; (void)rG1;\n"
            "    int wr[4];\n"
            "    ws_ranges<2>(S.fcnt[ln], ln, blockIdx.x, gridDim.x, wr);\n"
            "    const int v01 = wr[0], v11 = wr[1];\n", F1), "FAST1 ranges")
    else:
        # the FAST1 warp (tid >= NT - 32) runs before route() and the WS_JOIN; its S.wsr stores are published by it
        h = once(h, RNG, guard("", RNG), "ws ranges")
        h = once(h, F1, guard(
            "    (void)nS1; (void)rG1;\n"
            "    int wr[4];\n"
            "    ws_split<0, 2, SPW1, SPW2>(S.fcnt[ln], ln, blockIdx.x, gridDim.x, wr);\n"
            "    const int v01 = wr[0], v11 = wr[1];\n", F1), "FAST1 ranges")
        # phase B ranges after the preissue, so the first units go out as early as in the control
        h = once(h, F1_END, F1_END + guard(
            "    ws_split<2, 4, SPB1, SPB2>(S.fcnt[ln], ln, blockIdx.x, gridDim.x, wr);\n"
            "    if (ln == 0) { S.wsr[0] = wr[0]; S.wsr[1] = wr[1]; S.wsr[2] = wr[2]; S.wsr[3] = wr[3]; }\n", ""),
            "FAST1 B ranges")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("a", "b"), default="b")
    ap.add_argument("--tl", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--spw", default="", help="W1:W2 unit costs of the variant b split (default 3:4)")
    ap.add_argument("--spb", default="", help="W1:W2 phase B unit costs (default: --spw)")
    a = ap.parse_args()
    cand = ROOT / "muse_decode_layer/cand"
    if a.tl:
        src, dst = cand / "ly_tl16.py", cand / f"ly_ws2u{a.variant}_tl.py"
    else:
        src, dst = SRC, cand / ("ly_ws2u.py" if a.variant == "b" else "ly_ws2u_a.py")
    if a.out:
        dst = cand / a.out
    raw = src.read_bytes()
    print(src.name, hashlib.sha256(raw).hexdigest())
    if src == SRC and hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("rv_k_combo2n.py is not 3ac83fb4")
    whole = raw.decode()
    if whole.count(MARK) != 1:
        sys.exit("helper0 mark")
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    helper = patch_helper(helper, a.variant)
    extra = ""
    if a.spw:
        w1, w2 = a.spw.split(":")
        extra = f', "-DOKC_SPW1={w1}", "-DOKC_SPW2={w2}"'
        dst = dst.with_name(dst.stem + f"_s{w1}{w2}.py")
    if a.spb:
        w1, w2 = a.spb.split(":")
        extra += f', "-DOKC_SPB1={w1}", "-DOKC_SPB2={w2}"'
        dst = dst.with_name(dst.stem + f"_b{w1}{w2}.py")
    tail = once(tail, MID_ADD, MID_ADD + f'MOE_MID_DEFINES += ("-DOKC_SKT=16"{extra},)\n', "MID define")
    out = head + MARK + helper + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
