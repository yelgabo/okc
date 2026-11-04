#!/usr/bin/env python3
"""ly_mid2: ly_mid1's single weight read at decode 16 with the second n-tile's x held in registers.

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). ly_mid1 (tools/dl_ly_mid1_patch.py, box loss
+10.6 us at decode 16, docs/runs-decode-layer.md "MID single weight read at decode 16") gave the MID build (decode 16,
helper0's warp-specialized ring) -DOKC_SKT=16: one job per local expert, every unit two n-tiles on one decode of the
weights, so a local expert with 9 to 16 tokens streams its 768 units once instead of twice. Its tokens 8..15 came from
per-warp shared-memory rows (cp.async at the job switch, 16 LDS.128 per unit plus their address arithmetic); its unit
loop grew 692 -> 1006 instructions and every unit retired 23 to 35% slower. Here tokens 8..15 are B fragments in
registers, loaded at the job switch exactly as tokens 0..7 are (load_x_regs with the row offset 8, zeros past the job's
tokens), so the loop's only additions are the second mma per k16 step and its four accumulators. Everything else is
ly_mid1's: tile 1 partials in red1 behind the same fence.proxy.async / CTA barrier / mbarrier hand-over as tile 0, the
helper warp's and the WSBLK last-unit epilogues looped over both tiles, the count split of unit ranges. No fence,
barrier, mbarrier protocol, griddepcontrol or launch attribute changes; every edit is under `#if OKC_SKT > 8`, and only
MID gets -DOKC_SKT=16, so TINY / TINY2 preprocess to the base. Numerics: tile 0 keeps the control's four accumulator
chains and combine order (tokens 0..7 of every job are what the control computes); tile 1 has its own four chains
combined the same way.

    python3 tools/dl_ly_mid2_patch.py [--base FILE] [--pin SHA256] [--out FILE]
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dl_ly_mid1_patch as m1  # noqa: E402

ROOT = m1.ROOT
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_mid2.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
once, guard, block, tile_loop = m1.once, m1.guard, m1.block, m1.tile_loop

X1R = (
    "template <bool PB>\n"
    "__device__ __forceinline__ void load_x1_regs(const Params& p, const Smem& S, int j, int lane, int warp,\n"
    "                                             uint4 (&x)[2 * NCH]) {\n"
    "  const int g = lane >> 2, t = lane & 3;\n"
    "  const int cnt = S.jC[j], st = S.jS[j];\n"
    "  const int r = 8 + g;\n"
    "  if (r < cnt) {\n"
    "    const __nv_bfloat16* row;\n"
    "    if (PB) row = p.inter + (size_t)(st + r) * ID;\n"
    "    else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;\n"
    "    const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n"
    "#pragma unroll\n"
    "    for (int c = 0; c < NCH; ++c) {\n"
    "      x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);\n"
    "      x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);\n"
    "    }\n"
    "  } else {\n"
    "#pragma unroll\n"
    "    for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n"
    "  }\n"
    "}\n")


def patch_ring(ws):
    ws = once(ws, "  uint4 x0[2 * NCH];\n", "  uint4 x0[2 * NCH];\n" + guard("  uint4 x1[2 * NCH];\n", ""), "ws decl")
    ws = once(ws, "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n",
              "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n" + guard(
                  "      load_x1_regs<PB>(p, S, j, lane, warp, x1);\n", ""), "ws x1")
    mma_old = block(ws, "    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n",
                    "    // WSB: the compute warps keep the base loop's CTA barrier per unit", "ws mma")
    loop_old = block(mma_old, "#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n", m1.SUM1, "ws loop")
    head = mma_old[: mma_old.index(loop_old)]
    rest = mma_old[mma_old.index(loop_old) + len(loop_old) + len(m1.SUM1):]
    b0 = ("      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n"
          "                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n")
    two_loop = once(loop_old, b0, b0 +
                    "      const uint32_t bb2[8] = {x1[2 * c].x, x1[2 * c].y, x1[2 * c].z, x1[2 * c].w,\n"
                    "                               x1[2 * c + 1].x, x1[2 * c + 1].y, x1[2 * c + 1].z, x1[2 * c + 1].w};\n",
                    "ws bb2")
    two_loop = once(two_loop, m1.MMA_P, m1.MMA2_P, "ws mma prmt")
    two_loop = once(two_loop, m1.MMA_N, m1.MMA2_N, "ws mma lut")
    mma_new = head + guard(
        "    float acc5[4] = {0.f, 0.f, 0.f, 0.f}, acc6[4] = {0.f, 0.f, 0.f, 0.f};\n"
        "    float acc7[4] = {0.f, 0.f, 0.f, 0.f}, acc8[4] = {0.f, 0.f, 0.f, 0.f};\n"
        + two_loop + m1.SUM2, loop_old + m1.SUM1) + rest
    ws = ws.replace(mma_old, mma_new)
    ws = once(ws, m1.ST0, m1.ST0 + guard(m1.ST1, ""), "ws store")
    blk_old = block(ws, "    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n", "  seq = seq0 + nU;\n", "ws blk")
    hstart = "#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n"
    if blk_old.count(hstart) != 1 or not blk_old.endswith("    }\n  }\n"):
        sys.exit("ws blk h")
    hloop = blk_old[blk_old.index(hstart): len(blk_old) - len("  }\n")]
    pre = blk_old[: blk_old.index(hstart)]
    cnt_line = "    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n"
    blk_new = (once(pre, cnt_line, cnt_line + "    const bool two = cnt > 8;\n" + m1.RED1_SUM, "ws blk cnt")
               + tile_loop(hloop, "    ", "const int tk = 2 * tt + h;", "const int tk = nt * 8 + 2 * tt + h;") + "  }\n")
    return ws.replace(blk_old, guard(blk_new, blk_old))


def patch_helper0(h):
    h = once(h, "#define OKC_SKT 8\n", "#ifndef OKC_SKT\n#define OKC_SKT 8\n#endif\n", "SKT define")
    h = once(h, 'static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\n',
             'static_assert(!OKC_WS || OKC_SKT == 8 || OKC_SKT == 16, "OKC_WS: skinny jobs of one or two n-tiles");\n',
             "WS assert")
    h = once(h, "constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\n",
             "constexpr int RED_W = (OKC_SKT > 8 && !OKC_WS) ? 8 : 4;   // WS two-tile builds: tile 1 in red1\n", "RED_W")
    h = once(h, m1.SM_END, m1.SM_END[: -len("};\n")]
             + "#if OKC_WS && OKC_SKT > 8\n  alignas(16) float red1[NRB][NW][32][4];  // tile 1 (tokens 8..15) partial sums\n#endif\n};\n",
             "red1")
    h = once(h, m1.WS_ANCHOR, guard(X1R, "") + m1.WS_ANCHOR, "x1 loader")
    a, b = h.index(m1.WS_A), h.index(m1.WS_B)
    h = h[:a] + patch_ring(h[a:b]) + h[b:]
    c = h.index(m1.HP_A)
    d = h.index(m1.HP_END, c) + len(m1.HP_END)
    return h[:c] + m1.patch_helper_warp(h[c:d]) + h[d:]


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
    if whole.count(m1.MARK) != 1:
        sys.exit("helper0 mark")
    head, rest = whole.split(m1.MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    helper = patch_helper0(helper)
    tail = once(tail, m1.MID_ADD, m1.MID_ADD + 'MOE_MID_DEFINES += ("-DOKC_SKT=16",)\n', "MID define")
    out = ("# ly_mid2: ly_stk1 with one weight read per local expert at decode 16, the second n-tile's x in registers "
           "(tools/dl_ly_mid2_patch.py).\n" + head + m1.MARK + helper + tail)
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
