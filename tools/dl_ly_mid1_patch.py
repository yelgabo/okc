#!/usr/bin/env python3
"""ly_mid1: the decode 16 MID MoE streams every expert's weights once.

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032deb505..., checked). The MID build (9 to 16 tokens, decode 16)
cuts every local expert with more than 8 tokens into 8-token jobs (OKC_SPLIT), and each job streams and decodes the
expert's 768 weight units again. Here MID gets -DOKC_SKT=16: one job per local expert, up to 16 tokens, and every unit
runs two n-tiles on one decode of the weights: tokens 0..7 from the register fragments as before, tokens 8..15 from
rows this warp stages in shared memory by cp.async (zero-filled past the job's tokens, so a job of 8 or fewer tokens
runs the same code with a zero second tile). Every CTA runs the same unit loop, which is what the ly_ws2u timeline
asked for (its two-tile CTAs stalled on their first unit while the one-tile CTAs ran the control's loop).

Numerics: tile 0 keeps the control's four accumulator chains and their combine order, so tokens 0..7 of every job are
bitwise the control's; tile 1 has its own four chains combined the same way, so tokens 8..15 are bitwise what the
control computes for them as a separate 8-token job. The tile-1 partials go to a new red1 buffer at the end of Smem
(every other member keeps its offset), written before the same fence.proxy.async, CTA barrier and mbarrier hand-over
as tile 0; the helper warp reads them before it frees the buffer and runs the epilogue for tokens 8..15 when the job
has more than 8. Unit ranges stay the control's count split (every unit now costs the same). No fence, barrier,
mbarrier protocol, griddepcontrol or launch attribute changes. Every edit is inside the helper0 source under
`#if OKC_SKT > 8` with the original text in the `#else` arm, so TINY / TINY2 (OKC_SKT 8) preprocess to the base.

    python3 tools/dl_ly_mid1_patch.py [--base FILE] [--pin SHA256] [--out FILE]
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
DST = ROOT / "muse_decode_layer/cand/ly_mid1.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
MID_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'


def once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times: {old[:80]!r}")
    return s.replace(old, new)


def guard(new, old):
    return "#if OKC_SKT > 8\n" + new + "#else\n" + old + "#endif\n"


def block(s, start, end, what):
    for a in (start, end):
        if s.count(a) != 1:
            sys.exit(f"{what}: anchor found {s.count(a)} times: {a[:60]!r}")
    i = s.index(start)
    return s[i:s.index(end, i)]


WS_A = "template <bool PB>\n__device__ void skinny_ring_ws(const Params& p"
WS_B = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {"
HP_A = "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
HP_END = "  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n"
SM_END = "  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n"
WS_ANCHOR = "#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n"

# rows 8..15 of the job into this warp's x1 rows (K range [512 warp, +512)), zero-filled past the job's tokens; the
# caller waits (cp.async.wait_all, __syncwarp) before the first MMA that reads them. The source address of a
# zero-filled row is the job's first token row, which always exists (src-size 0 reads nothing).
X1A = (
    "template <bool PB>\n"
    "__device__ __forceinline__ void load_x1_async(const Params& p, const Smem& S, int j, int lane, int warp, __nv_bfloat16* x1s) {\n"
    "  const int cnt = S.jC[j], st = S.jS[j];\n"
    "#pragma unroll\n"
    "  for (int m = 0; m < 16; ++m) {\n"
    "    const int r = m >> 1, c = 32 * (m & 1) + lane;\n"
    "    const bool ok = 8 + r < cnt;\n"
    "    const int tk = ok ? st + 8 + r : st;\n"
    "    const __nv_bfloat16* row = PB ? p.inter + (size_t)tk * ID : p.x + (size_t)(S.ent[tk] >> 3) * HD;\n"
    "    const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n"
    "    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16, %2;\\n\" ::\"r\"(smem_u32(x1s + r * X1P + c * 8)), \"l\"(src), \"r\"(ok ? 16 : 0) : \"memory\");\n"
    "  }\n"
    "}\n")

MMA_P = ("        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2l(r0, sgp), bmul2h(r1, sgp), "
         "bmul2l(r2, sgp), bmul2h(r3, sgp), b[2 * st], b[2 * st + 1]);\n")
MMA_N = ("        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), "
         "bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n")
MMA2_P = ("        const uint32_t a0 = bmul2l(r0, sgp), a1 = bmul2h(r1, sgp), a2 = bmul2l(r2, sgp), a3 = bmul2h(r3, sgp);\n"
          "        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n"
          "        mma_bf16(st == 0 ? acc5 : st == 1 ? acc6 : st == 2 ? acc7 : acc8, a0, a1, a2, a3, bb2[2 * st], bb2[2 * st + 1]);\n")
MMA2_N = ("        const uint32_t a0 = bmul2(r0, sg), a1 = bmul2(r1, sg8), a2 = bmul2(r2, sg), a3 = bmul2(r3, sg8);\n"
          "        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n"
          "        mma_bf16(st == 0 ? acc5 : st == 1 ? acc6 : st == 2 ? acc7 : acc8, a0, a1, a2, a3, bb2[2 * st], bb2[2 * st + 1]);\n")
SUM1 = "#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
SUM2 = "#pragma unroll\n    for (int k = 0; k < 4; ++k) { acc[k] += acc2[k] + acc3[k] + acc4[k]; acc5[k] += acc6[k] + acc7[k] + acc8[k]; }\n"
ST0 = "    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n"
ST1 = "    *reinterpret_cast<float4*>(&S.red1[bb][warp][lane][0]) = make_float4(acc5[0], acc5[1], acc5[2], acc5[3]);\n"
RED1_SUM = ("    float4 sm1 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
            "    if (two) {\n#pragma unroll\n      for (int w = 0; w < NW; ++w) {\n"
            "        const float4 t = *reinterpret_cast<const float4*>(&S.red1[bb][w][lane][0]);\n"
            "        sm1.x += t.x; sm1.y += t.y; sm1.z += t.z; sm1.w += t.w;\n      }\n    }\n")


def tile_loop(old, indent, tk_old, tk_new):
    """An epilogue block over tile 0 (partials sm) becomes a loop over both tiles (sv = sm, sm1; tk offset by 8)."""
    body = once(old, tk_old, tk_new, "tile loop tk").replace("sm.", "smt.")
    return (f"{indent}for (int nt = 0; nt < (two ? 2 : 1); ++nt) {{\n"
            f"{indent}  const float4 smt = nt ? sm1 : sm;\n" + body + f"{indent}}}\n")


def patch_ring(ws):
    ws = once(ws, "  uint4 x0[2 * NCH];\n", guard(
        "  bool x1p = false;\n"
        "  __nv_bfloat16* x1s = reinterpret_cast<__nv_bfloat16*>(ring + OKC_NS * UNIT_BYTES) + warp * 8 * X1P;\n", "")
        + "  uint4 x0[2 * NCH];\n", "ws decl")
    ws = once(ws, "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n",
              "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n" + guard(
                  "      load_x1_async<PB>(p, S, j, lane, warp, x1s); x1p = true;\n", ""), "ws x1")
    ws = once(ws, "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n",
              "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n" + guard(
                  "    if (x1p) { asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\"); __syncwarp(); x1p = false; }\n", ""),
              "ws x1 wait")
    mma_old = block(ws, "    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n",
                    "    // WSB: the compute warps keep the base loop's CTA barrier per unit", "ws mma")
    loop_old = block(mma_old, "#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n", SUM1, "ws loop")
    head = mma_old[: mma_old.index(loop_old)]
    rest = mma_old[mma_old.index(loop_old) + len(loop_old) + len(SUM1):]
    two_loop = once(loop_old, "      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n",
                    "      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n"
                    "      const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n"
                    "      const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n"
                    "      const uint32_t bb2[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n", "ws ww")
    two_loop = once(two_loop, MMA_P, MMA2_P, "ws mma prmt")
    two_loop = once(two_loop, MMA_N, MMA2_N, "ws mma lut")
    mma_new = head + guard(
        "    float acc5[4] = {0.f, 0.f, 0.f, 0.f}, acc6[4] = {0.f, 0.f, 0.f, 0.f};\n"
        "    float acc7[4] = {0.f, 0.f, 0.f, 0.f}, acc8[4] = {0.f, 0.f, 0.f, 0.f};\n"
        "    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n"
        + two_loop + SUM2, loop_old + SUM1) + rest
    ws = ws.replace(mma_old, mma_new)
    ws = once(ws, ST0, ST0 + guard(ST1, ""), "ws store")
    # the WSBLK last-unit epilogue (phase B, warps 0..K-1)
    blk_old = block(ws, "    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n", "  seq = seq0 + nU;\n", "ws blk")
    hstart = "#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n"
    if blk_old.count(hstart) != 1 or not blk_old.endswith("    }\n  }\n"):
        sys.exit("ws blk h")
    hloop = blk_old[blk_old.index(hstart): len(blk_old) - len("  }\n")]
    pre = blk_old[: blk_old.index(hstart)]
    cnt_line = "    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n"
    blk_new = (once(pre, cnt_line, cnt_line + "    const bool two = cnt > 8;\n" + RED1_SUM, "ws blk cnt")
               + tile_loop(hloop, "    ", "const int tk = 2 * tt + h;", "const int tk = nt * 8 + 2 * tt + h;") + "  }\n")
    return ws.replace(blk_old, guard(blk_new, blk_old))


def patch_helper_warp(hp):
    sync = "    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n"
    hp = once(hp, sync, guard("    const bool two = S.jC[j] > 8;\n" + RED1_SUM, "") + sync, "helper sum")
    # phase A: the SiLU of both rows and the paired bf16x2 store (wspack form)
    a_start = "      float sv[2];\n"
    a_end = "      if (q == nA - 1) {"
    ha = block(hp, a_start, a_end, "helper A")
    ha_new = tile_loop(ha, "      ", "        const int tk = 2 * tt + od;\n", "        const int tk = nt * 8 + 2 * tt + od;\n")
    hp = hp.replace(ha, guard(ha_new, ha))
    # phase B: the ybuf stores and the sum of squares
    hb0 = "#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float v0f"
    hb_end = "        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n"
    i = hp.index(hb0)
    j = hp.index(hb_end, i) + len(hb_end)
    hb = hp[i:j]
    if hp.count(hb) != 1:
        sys.exit("helper B")
    hb_new = tile_loop(hb, "      ", "const int tk = 2 * tt + h;", "const int tk = nt * 8 + 2 * tt + h;")
    return hp.replace(hb, guard(hb_new, hb))


def patch_helper0(h):
    h = once(h, "#define OKC_SKT 8\n", "#ifndef OKC_SKT\n#define OKC_SKT 8\n#endif\n", "SKT define")
    h = once(h, 'static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\n',
             'static_assert(!OKC_WS || OKC_SKT == 8 || OKC_SKT == 16, "OKC_WS: skinny jobs of one or two n-tiles");\n',
             "WS assert")
    h = once(h, "constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\n",
             "constexpr int RED_W = (OKC_SKT > 8 && !OKC_WS) ? 8 : 4;   // WS two-tile builds: tile 1 in red1\n", "RED_W")
    h = once(h, SM_END, SM_END[: -len("};\n")]
             + "#if OKC_WS && OKC_SKT > 8\n  alignas(16) float red1[NRB][NW][32][4];  // tile 1 (tokens 8..15) partial sums\n#endif\n};\n",
             "red1")
    h = once(h, WS_ANCHOR, guard(X1A, "") + WS_ANCHOR, "x1 loader")
    a, b = h.index(WS_A), h.index(WS_B)
    h = h[:a] + patch_ring(h[a:b]) + h[b:]
    c = h.index(HP_A)
    d = h.index(HP_END, c) + len(HP_END)
    return h[:c] + patch_helper_warp(h[c:d]) + h[d:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(SRC))
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--out", default=str(DST))
    a = ap.parse_args()
    src, dst = ROOT / a.base, ROOT / a.out
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print(src.name, sha)
    if sha != a.pin:
        sys.exit(f"{src.name} is not {a.pin[:8]}")
    whole = raw.decode()
    if whole.count(MARK) != 1:
        sys.exit("helper0 mark")
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    helper = patch_helper0(helper)
    tail = once(tail, MID_ADD, MID_ADD + 'MOE_MID_DEFINES += ("-DOKC_SKT=16",)\n', "MID define")
    out = head + MARK + helper + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
