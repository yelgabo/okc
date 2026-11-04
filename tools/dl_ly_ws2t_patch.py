#!/usr/bin/env python3
"""ly_ws2t: decode 16 MID MoE with two-n-tile skinny jobs on the warp-specialized ring.

Writes muse_decode_layer/cand/ly_ws2t.py from rv_k_combo2n.py (3ac83fb4). The layer's MID build (9 to 16 tokens,
decode 16) splits every local expert with more than 8 tokens into 8-token jobs, and each job streams and decodes the
expert's 768 weight units again (on 34 of 40 box inputs one expert holds 10 to 12 of the 16 tokens; docs/runs-decode-
layer.md "Decode 16 / 32 sliding: floor and gap"). Here MID gets -DOKC_SKT=16: a job holds up to 16 tokens, and a job
with more than 8 runs the second n-tile (tokens 8..15, x rows staged per warp in shared memory by the existing
load_x1_smem) on the same dequantized weight fragments, so the expert's units are decoded once.

Edits, all inside the helper0 source and all under `#if OKC_SKT > 8` with the original text in the `#else` arm, so the
TINY / TINY2 builds (OKC_SKT 8) preprocess to the original code:
- `#define OKC_SKT 8` becomes overridable; the WS static_assert admits OKC_SKT 16.
- skinny_ring_ws: per job `two = cnt > 8` and the x1 rows; per unit the 2-tile MMA (chains acc / acc3 for tokens 0..7,
  acc2 / acc4 for 8..15); the tile-1 partial sums go to S.red[..][4..7] (RED_W is 8 when OKC_SKT > 8) before the same
  fence.proxy.async, CTA barrier and mbarrier hand-over as tile 0; the WSBLK last-unit epilogue loops over both tiles.
- ws_helper: reads the tile-1 partials before it frees the buffer and runs the epilogue for tokens 8..15.
No fence, barrier, mbarrier protocol, griddepcontrol or launch attribute changes; the job builder, FAST1 preissue and unit
ranges already take SK_T.

    python3 tools/dl_ly_ws2t_patch.py
    python3 tools/dl_ly_ws2t_patch.py --tl   # the same edits on ly_tl16.py (tools/ly_moe_tl16.py): ly_ws2t_tl.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/rv_k_combo2n.py"
DST = ROOT / "muse_decode_layer/cand/ly_ws2t.py"
PIN = "3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
MID_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'


def once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times")
    return s.replace(old, new)


def guard(new, old):
    return "#if OKC_SKT > 8\n" + new + "#else\n" + old + "#endif\n"


def block(s, start, end, what):
    """The text from `start` (inclusive) to `end` (exclusive), both unique in s."""
    for a in (start, end):
        if s.count(a) != 1:
            sys.exit(f"{what}: anchor found {s.count(a)} times: {a[:60]!r}")
    i = s.index(start)
    j = s.index(end, i)
    return s[i:j]


def tile_loop(old, indent):
    """Wrap an epilogue h-loop over tile 0 into a loop over both tiles (tk += 8 * nt, partials from sv)."""
    body = old.replace("const int tk = 2 * tt + h;", "const int tk = nt * 8 + 2 * tt + h;")
    body = body.replace("sm.", "sv.")
    return (f"{indent}for (int nt = 0; nt < (two ? 2 : 1); ++nt) {{\n"
            f"{indent}  const float4 sv = nt ? sm1 : sm;\n" + body + f"{indent}}}\n")


def patch_helper(h):
    h = once(h, "#define OKC_SKT 8\n", "#ifndef OKC_SKT\n#define OKC_SKT 8\n#endif\n", "SKT define")
    h = once(h, 'static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\n',
             'static_assert(!OKC_WS || OKC_SKT == 8 || OKC_SKT == 16, "OKC_WS: skinny jobs of one or two n-tiles");\n',
             "WS assert")
    # rows 8..15 of a two-tile job: all of a warp's loads in flight before its stores (load_x1_smem issues one 16 B
    # load per iteration and waits on each before its store)
    x1ws = (
        "template <bool PB>\n"
        "__device__ __forceinline__ void load_x1_ws(const Params& p, const Smem& S, int j, int lane, int warp,\n"
        "                                           __nv_bfloat16* x1s) {\n"
        "  const int cnt = S.jC[j], st = S.jS[j];\n"
        "#pragma unroll\n"
        "  for (int hb = 0; hb < 2; ++hb) {\n"
        "    uint4 v[8];\n"
        "#pragma unroll\n"
        "    for (int k = 0; k < 8; ++k) {\n"
        "      const int m = 8 * hb + k, r = m >> 1, c = 32 * (m & 1) + lane;\n"
        "      v[k] = make_uint4(0, 0, 0, 0);\n"
        "      if (8 + r < cnt) {\n"
        "        const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;\n"
        "        const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n"
        "        v[k] = PB ? __ldcg(src) : __ldg(src);\n"
        "      }\n"
        "    }\n"
        "#pragma unroll\n"
        "    for (int k = 0; k < 8; ++k) {\n"
        "      const int m = 8 * hb + k, r = m >> 1, c = 32 * (m & 1) + lane;\n"
        "      *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v[k];\n"
        "    }\n"
        "  }\n"
        "  __syncwarp();\n"
        "}\n"
        "// two-tile units retire about 1.27x slower than one-tile units (box MID timeline): ranges split by cost, not count\n"
        "#ifndef OKC_WSW2\n#define OKC_WSW2 20\n#endif\n"
        "constexpr int WS_W1 = 16, WS_W2 = OKC_WSW2;\n"
        "// r[k], k < K: first unit of CTA c (k = 0, 2) / c + 1 (k = 1, 3) in phase A (k < 2) / B: the first unit whose cost\n"
        "// prefix reaches floor(T * c / G). Warp-wide, lane-uniform result; lane = local expert, tot = its tokens; jobs in\n"
        "// expert order, ceil(tot / 16) per expert, a job of more than 8 tokens costs WS_W2 per unit. Costs fit 32 bits:\n"
        "// T * rbs * c <= 32 * 2 * 24 * 512 * 132.\n"
        "template <int K>\n"
        "__device__ __forceinline__ void ws_ranges(int tot, int lane, int c, int G, int (&r)[4]) {\n"
        "  const int nfull = tot >> 4, rem = tot & 15, nj = nfull + (rem > 0 ? 1 : 0);\n"
        "  const int ce = nfull * WS_W2 + (rem > 0 ? (rem > 8 ? WS_W2 : WS_W1) : 0);\n"
        "  int ci = ce, ji = nj;\n"
        "#pragma unroll\n"
        "  for (int d = 1; d < 32; d <<= 1) {\n"
        "    const int a = __shfl_up_sync(0xffffffffu, ci, d), b = __shfl_up_sync(0xffffffffu, ji, d);\n"
        "    if (lane >= d) { ci += a; ji += b; }\n"
        "  }\n"
        "  const int T = __shfl_sync(0xffffffffu, ci, 31), J = __shfl_sync(0xffffffffu, ji, 31);\n"
        "  const double rG = 1.0 / (double)G;\n"
        "#pragma unroll\n"
        "  for (int k = 0; k < K; ++k) {\n"
        "    const int rbs = k < 2 ? RB_A : RB_B, cc = c + (k & 1);\n"
        "    if (cc >= G) { r[k] = J * rbs; continue; }\n"
        "    const unsigned X = (unsigned)(T * rbs) * (unsigned)cc;\n"
        "    int t = (int)((double)X * rG);\n"
        "    while ((unsigned)t * (unsigned)G > X) --t;\n"
        "    while ((unsigned)(t + 1) * (unsigned)G <= X) ++t;\n"
        "    const unsigned bal = __ballot_sync(0xffffffffu, ci * rbs > t);\n"
        "    if (!bal) { r[k] = 0; continue; }\n"
        "    const int e = __ffs(bal) - 1;\n"
        "    int u = 0;\n"
        "    if (lane == e) {\n"
        "      int o = t - (ci - ce) * rbs;\n"
        "      u = (ji - nj) * rbs;\n"
        "      for (int q = 0; q < nj; ++q) {\n"
        "        const int w = (q < nfull ? 16 : rem) > 8 ? WS_W2 : WS_W1;\n"
        "        if (o < rbs * w) { u += w == WS_W1 ? (o + WS_W1 - 1) / WS_W1 : (o + WS_W2 - 1) / WS_W2; break; }\n"
        "        o -= rbs * w; u += rbs;\n"
        "      }\n"
        "    }\n"
        "    r[k] = __shfl_sync(0xffffffffu, u, e);\n"
        "  }\n"
        "}\n")
    anchor = "#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n"
    h = once(h, anchor, guard(x1ws, "") + anchor, "x1ws")
    h = once(h, "constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\n",
             "constexpr int RED_W = (OKC_SKT > 8 && !OKC_WS) ? 8 : 4;   // WS two-tile builds: tile 1 in red1\n", "RED_W")
    red = "  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums\n"
    h = once(h, red, red + "#if OKC_WS && OKC_SKT > 8\n  float red1[NRB][NW][32][4];      // tile 1 (tokens 8..15) of two-tile units\n#endif\n",
             "red1")
    a = h.index("__device__ void skinny_ring_ws(const Params& p")
    b = h.index("__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {")
    ws = h[a:b]

    ws = once(ws, "  int curj = -1;\n", "  int curj = -1;\n" + guard(
        "  bool two = false;\n"
        "  __nv_bfloat16* x1s = reinterpret_cast<__nv_bfloat16*>(ring + OKC_NS * UNIT_BYTES) + warp * 8 * X1P;\n", ""),
        "ws decl")
    ws = once(ws, "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n",
              "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n" + guard(
                  "      two = S.jC[j] > 8;\n"
                  "      if (two) load_x1_ws<PB>(p, S, j, lane, warp, x1s);\n", ""), "ws x1")

    mma_old = block(ws, "    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n",
                    "    // WSB: the compute warps keep the base loop's CTA barrier per unit", "ws mma")
    loop_old = block(mma_old, "#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n",
                     "#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n", "ws loop")
    head = mma_old[: mma_old.index(loop_old)]
    one_mma = ("        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), "
               "bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n")
    if loop_old.count(one_mma) != 1:
        sys.exit("ws mma line")
    two_loop = loop_old.replace(
        "      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n",
        "      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n"
        "      const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n"
        "      const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n"
        "      const uint32_t bb2[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n").replace(
        one_mma,
        "        const uint32_t a0 = bmul2(r0, sg), a1 = bmul2(r1, sg8), a2 = bmul2(r2, sg), a3 = bmul2(r3, sg8);\n"
        "        mma_bf16((st & 1) ? acc3 : acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n"
        "        mma_bf16((st & 1) ? acc4 : acc2, a0, a1, a2, a3, bb2[2 * st], bb2[2 * st + 1]);\n")
    tail_old = mma_old[mma_old.index(loop_old) + len(loop_old):]
    sum_one = "#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
    if not tail_old.startswith(sum_one):
        sys.exit("ws sum line")
    rest = tail_old[len(sum_one):]
    mma_new = head + guard(
        "    if (two) {\n"
        "      const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n"
        + two_loop +
        "#pragma unroll\n    for (int k = 0; k < 4; ++k) { acc[k] += acc3[k]; acc2[k] += acc4[k]; }\n"
        "    } else {\n" + loop_old + sum_one + "    }\n",
        loop_old + sum_one) + rest
    ws = ws.replace(mma_old, mma_new)

    st0 = "    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n"
    ws = once(ws, st0, st0 + guard(
        "    if (two) *reinterpret_cast<float4*>(&S.red1[bb][warp][lane][0]) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);\n",
        ""), "ws store")

    blk_old = block(ws, "    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n", "  seq = seq0 + nU;\n", "ws blk")
    hstart = "#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n"
    if blk_old.count(hstart) != 1 or not blk_old.endswith("    }\n  }\n"):
        sys.exit("ws blk h")
    hloop = blk_old[blk_old.index(hstart): len(blk_old) - len("  }\n")]
    pre = blk_old[: blk_old.index(hstart)]
    blk_new = (pre.replace("    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n",
                           "    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const bool two = cnt > 8;\n"
                           "    float4 sm1 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
                           "    if (two) {\n#pragma unroll\n      for (int w = 0; w < NW; ++w) {\n"
                           "        const float4 t = *reinterpret_cast<const float4*>(&S.red1[bb][w][lane][0]);\n"
                           "        sm1.x += t.x; sm1.y += t.y; sm1.z += t.z; sm1.w += t.w;\n      }\n    }\n")
               + tile_loop(hloop, "    ") + "  }\n")
    if blk_new.count("const bool two = cnt > 8;") != 1:
        sys.exit("ws blk cnt line")
    ws = ws.replace(blk_old, guard(blk_new, blk_old))
    h = h[:a] + ws + h[b:]

    c = h.index("__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {")
    d = h.index("  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n", c)
    hp = h[c:d]
    sync = "    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n"
    hp = once(hp, sync, guard(
        "    const bool two = S.jC[j] > 8;\n"
        "    float4 sm1 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
        "    if (two) {\n#pragma unroll\n      for (int w = 0; w < NW; ++w) {\n"
        "        const float4 t = *reinterpret_cast<const float4*>(&S.red1[bb][w][lane][0]);\n"
        "        sm1.x += t.x; sm1.y += t.y; sm1.z += t.z; sm1.w += t.w;\n      }\n    }\n", "") + sync, "helper sum")
    ha = block(hp, "#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        if (tk < cnt) {\n          const float gt",
               "      if (q == nA - 1) {", "helper A")
    hp = hp.replace(ha, guard(tile_loop(ha, "      "), ha))
    hb0 = "#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float v0f"
    i = hp.index(hb0)
    j = hp.index("        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n", i)
    j += len("        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n")
    hb = hp[i:j]
    if hp.count(hb) != 1:
        sys.exit("helper B")
    hp = hp.replace(hb, guard(tile_loop(hb, "      "), hb))
    h = h[:c] + hp + h[d:]

    rng = ("  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n"
           "    const int G = gridDim.x, c = blockIdx.x;\n"
           "    const float rG = __fdividef(1.f, (float)G);\n"
           "    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n"
           "    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n"
           "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n"
           "  }\n")
    h = once(h, rng, guard(
        "  if (tid < 32) {  // this CTA's unit ranges, split by cost (ws_ranges); run_phase and FAST1 use the same split\n"
        "    const int G = gridDim.x, c = blockIdx.x, tot = tid < p.nloc ? S.cnt[tid] : 0;\n"
        "    int r[4];\n"
        "    ws_ranges<4>(tot, tid, c, G, r);\n"
        "    if (tid == 0) { S.wsr[0] = r[0]; S.wsr[1] = r[1]; S.wsr[2] = r[2]; S.wsr[3] = r[3]; }\n"
        "  }\n", rng), "ws ranges")
    call = "  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n"
    h = once(h, call, guard("  v0 = S.wsr[PB ? 2 : 0]; v1 = S.wsr[PB ? 3 : 1];\n", "") + call, "run_phase ranges")
    f1 = "    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n"
    h = once(h, f1, guard(
        "    (void)nS1; (void)rG1;\n"
        "    int wr[4];\n"
        "    ws_ranges<2>(S.fcnt[ln], ln, blockIdx.x, gridDim.x, wr);\n"
        "    const int v01 = wr[0], v11 = wr[1];\n", f1), "FAST1 ranges")
    return h


def main():
    src, dst = SRC, DST
    if "--tl" in sys.argv:
        src, dst = ROOT / "muse_decode_layer/cand/ly_tl16.py", ROOT / "muse_decode_layer/cand/ly_ws2t_tl.py"
    w2 = next((a.split("=")[1] for a in sys.argv if a.startswith("--w2=")), None)
    if w2:
        dst = dst.with_name(dst.stem + f"_w{w2}.py")
    raw = src.read_bytes()
    if src == SRC and hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("rv_k_combo2n.py is not 3ac83fb4")
    whole = raw.decode()
    if whole.count(MARK) != 1:
        sys.exit("helper0 mark")
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    helper = patch_helper(helper)
    extra = f', "-DOKC_WSW2={w2}"' if w2 else ""
    tail = once(tail, MID_ADD, MID_ADD + f'MOE_MID_DEFINES += ("-DOKC_SKT=16"{extra},)\n', "MID define")
    out = head + MARK + helper + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
