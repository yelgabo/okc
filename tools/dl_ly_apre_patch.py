#!/usr/bin/env python3
"""ly_apre: the flat q_len 1 attention kernel (decode 16 / 32 full) issues its first STAGES tiles of K/V right after
the partition barrier, before the segment machinery starts.

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 37654700..., pinned). Per-CTA stamps on the box (docs/runs-decode-layer.md,
"Full-context attention limit (box)"): the producer warps issue tile 0's copies 3.3 to 3.9 us after kernel entry, of
which the dependent latencies (seq_lens, page indices) are under 1 us; the rest is the 64-bit partition divisions,
thread 0's segment list, a block barrier, setmaxnreg, the segment prologue and the page-index TMA ring setup. The
consumers see their first tile at 5.0 / 5.4 us (decode 16 / 32 full), and the steady state after it already runs at
the DRAM rate, so the startup is the largest measured piece of the distance to the byte floor.

Change (attn_kernel<1, true, 0> only, the flat producer-warpgroup build with the EARLYBAR stage barriers): once the
partition barrier has passed and this CTA's key range [cs, ce) is known, the producer threads find the first segment
(the same scan thread 0 performs for the segment list), load that segment's first min(STAGES, tiles) tiles' page
indices with plain loads, issue exactly the copies the producer loop would have issued for those tiles (same
cp.async instructions, same zero-fill for rows past the range, same swizzled stage addresses) and arrive on the same
full barriers. The producer loop of segment 0 then skips the empty wait, the copies and the arrival for those tiles
and keeps the page-index TMA ring accounting unchanged (fetches, waits and slot barriers). No fence, barrier or
arrival count changes; the stage barriers were initialized by the EARLYBAR code before the partition barrier, which
is what the stock first arrival relies on too. Consumers are untouched, so the output is bitwise the base's.

    python3 tools/dl_ly_apre_patch.py [--base muse_decode_layer/cand/ly_fabq.py] [--pin SHA256] [--out muse_decode_layer/cand/ly_apre.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_apre.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

PRE_BLOCK = r"""  // apre: the producer warps issue the first segment's first STAGES tiles now, from plain page-index loads, so the
  // DRAM stream starts while thread 0 builds the segment list and the segment prologue sets up its TMA ring
  constexpr bool apre = L == 1 && FLAT && Cfg<L, FLAT>::WS && OKC_TMAPG && !OKC_CONT && !OKC_PTR && OKC_EARLYBAR &&
                        !OKC_PDLKV && !OKC_DYN && !fpart;
  if constexpr (apre) {
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      constexpr int NP = Cfg<L, FLAT>::NP, PR = NP / 8, PK = TILE / PR, STG = Cfg<L, FLAT>::STAGES;
      const int q = threadIdx.x - Cfg<L, FLAT>::NT, ch = q & 7, r0 = q >> 3;
      int b0 = 0;   // the first segment: the same scan as the segment list below
      while (S.cum[b0 + 1] <= cs) ++b0;
      const int seq0 = __ldg(p.seq_lens + b0);
      const int lo0 = p.window ? max(0, seq0 + 1 - p.window) : 0;
      const int s0 = max(cs, S.cum[b0]) - S.cum[b0] + lo0, s1 = min(ce, S.cum[b0 + 1]) - S.cum[b0] + lo0;
      const int cend = min(s1, seq0);
      const int pre = min(STG, (s1 - s0 + TILE - 1) / TILE);
      const int* ptr = p.page_table + (ix_t)b0 * p.max_pages;
      const bool pfast = OKC_OFF32 && p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      int pg[STG][PK];
#pragma unroll
      for (int x = 0; x < STG; ++x)
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = s0 + x * TILE + r0 + PR * k;
          pg[x][k] = (x < pre && pos < cend) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
#pragma unroll
      for (int x = 0; x < STG; ++x) {
        if (x >= pre) break;
        const uint32_t kb = smem_u32(S.k + x * STAGE_BYTES), vb = smem_u32(S.v + x * STAGE_BYTES);
        if (pfast && s0 + x * TILE + TILE <= cend) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[x][k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = s0 + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cend) {
            if (p.off32) {
              const uint32_t off = ((((uint32_t)pg[x][k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[x][k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= s1) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[x]);
      }
      if (q == 0) S.npre = pre;   // read by the producers after the segment-list barrier below
    }
  }
"""

EDITS = [
    # shared: the number of pre-issued tiles of the first segment
    ("  int nseg;\n  int fcs, fce;", "  int nseg;\n  int npre;                 // apre: tiles of segment 0 issued before the segment loop\n  int fcs, fce;", 1),
    # segment(): the pre-issued tile count
    ("                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,\n"
     "                                        const bool skip_init_ = false) {\n",
     "                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,\n"
     "                                        const bool skip_init_ = false, const int pre_ = 0) {\n", 1),
    ("  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);\n#undef UNI\n",
     "  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);\n"
     "  const int pre = UNI(pre_);   // apre: tiles already issued (and arrived on their full barriers) by the kernel prologue\n#undef UNI\n", 1),
    # producer loop: pre-issued tiles keep only the page-ring accounting
    ("        if constexpr (TMAPG_ON) {\n          fetch_pages(x + PD);\n          load_pages_t(x, pg);\n        } else {\n"
     "          load_pages(x + 1, pgn);   // in flight while this tile's copies issue\n        }\n        PROF_MARK(2);\n",
     "        if constexpr (TMAPG_ON) {\n          fetch_pages(x + PD);\n          load_pages_t(x, pg);\n        } else {\n"
     "          load_pages(x + 1, pgn);   // in flight while this tile's copies issue\n        }\n"
     "        if (x < pre) {   // apre: issued and arrived in the prologue; the page ring still advances\n"
     "          if constexpr (TMAPG_ON) {\n            asm volatile(\"bar.sync 2, %0;\\n\" ::\"n\"(NP) : \"memory\");\n          } else {\n"
     "#pragma unroll\n            for (int k = 0; k < PK; ++k) pg[k] = pgn[k];\n          }\n          continue;\n        }\n"
     "        PROF_MARK(2);\n", 1),
    # the prologue block, after the empty-CTA return and before the page-ring / segment-list code
    ("  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {\n",
     PRE_BLOCK + "  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {\n", 1),
    # run_segments: segment 0 of the producer role gets the count
    ("                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);\n",
     "                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0,\n"
     "                            (apre && R == 1 && k == 0) ? S.npre : 0);\n", 1),
]


def build(src):
    out = src
    for old, new, n in EDITS:
        c = out.count(old)
        assert c == n, (c, n, old[:80])
        out = out.replace(old, new)
    return out


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    out = build(raw.decode())
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
