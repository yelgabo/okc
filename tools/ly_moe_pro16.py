#!/usr/bin/env python3
"""MID (decode 16) prologue stamps: tools/ly_moe_tl16.py's instrument with 128 stamps per CTA, slots 64..127 holding
every load and barrier from kernel entry to route() completion and the first weight wait.

    python3 tools/ly_moe_pro16.py --base muse_decode_layer/cand/ly_stk2.py --pin SHA256 --out FILE [--probe]

Slots 0..63 are ly_moe_tl16's (0 entry after griddepcontrol.wait, 1 route done after WS_JOIN, 2 first slot full).
Thread 0 (warp 0 lane 0): 64 entry before griddepcontrol.wait, 65 after the x L2 prefetch loop, 66 after the lut
loop, 67 the launch generation load returned, 68 route() start, 69 its topk load returned, 70 after CTA barrier 1,
71 after the count loop, 72 after barrier 2, 73 after the warp 0 scan, 74 after barrier 3, 75 after early_issue,
76 after the position loop, 77 after barrier 4 (route() end), 78 after the ss / counter zeroing, 79 before WS_JOIN,
80 / 81 warp 0's first x load_x_regs issued / returned.
FAST1 warp (tid NT - 32, lane 0): 84 after griddepcontrol.wait, 85 after its count zeroing, 86..89 count iteration
k's topk load returned (match done), 90 job scan done, 91 barrier inits done, 92 FAST1 TMA issues done, 93 / 94
after the x prefetch / lut loops.
Helper warp (lane 0): 96 entry, 97 after WS_JOIN, 98 after its ring issues.
--probe (-DOKC_TLP=1): before its WS_JOIN the helper times three dependent ld.global.cg loads on a phase B weight
page no prologue touches (CTA c: p.wBu + c * 2 MiB + 4 KiB): 100 first touch, 101 another line 64 KiB away in the
same 2 MiB region, 102 the first line again (L2 hit), in ns. Stamps change the instrumented launch only (p.dbg).
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "tools/moe_tiny_timeline_cta.py"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
MID_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EXTRA = [
    ("""  if (tid == 0) TLS(11, gtimer());
""", """  if (tid == 0) { TLS(11, gtimer()); TLS(120, clock64()); }
"""),
    ("#define TLS(k, v) do { if (TLX_ON) p.dbg[blockIdx.x * 128 + (k)] = (v); } while (0)\n",
     """#define TLS(k, v) do { if (TLX_ON) p.dbg[blockIdx.x * 128 + (k)] = (v); } while (0)
// stamp once v is in a register: the branch on v makes the timer read wait for the load that produced it
#define TLD(k, v) do { if (TLX_ON) { if ((int)(v) == 0x7f7f7f7f) asm volatile("trap;"); p.dbg[blockIdx.x * 128 + (k)] = clock64(); } } while (0)
#ifndef OKC_TLP
#define OKC_TLP 0
#endif
// stamp after a barrier: a volatile shared load cannot move above bar.sync, and the clock waits for the load
#define TLBR(k) do { if (TLX_ON) { const unsigned vv_ = *reinterpret_cast<volatile const unsigned*>(&S.lut[0]); TLD(k, vv_); } } while (0)
"""),
    ("""  DBG(0);
  // programmatic dependent launch: everything below reads""",
     """  DBG(0);
  if (tid == 0) { TLS(64, clock64()); TLS(122, gtimer()); }
  // programmatic dependent launch: everything below reads"""),
    ("""  if (tid == 0) TLS(0, gtimer());
""", """  if (tid == 0) TLS(0, gtimer());
  if (tid == NT - 32) TLS(84, clock64());
"""),
    ("""    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
""", """    S.fcnt[ln] = 0;
    __syncwarp();
    if (ln == 0) TLS(85, clock64());
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (ln == 0 && i0 < 128) TLD(86 + (i0 >> 5), peers);
"""),
    ("""    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
#endif
""", """    if (ln == 0) {
      TLS(90, clock64());
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
#endif
      TLS(91, clock64());
"""),
    ("""      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
""", """      TLS(92, clock64());
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
"""),
    ("""  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code""",
     """  if (tid == 0) TLS(65, clock64());
  if (tid == NT - 32) TLS(93, clock64());
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code"""),
    ("""    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
""", """    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) TLS(66, clock64());
  if (tid == NT - 32) TLS(94, clock64());
  if (tid == 0) {
"""),
    ("""    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
""", """    S.ssp = (gen & 1) ? p.ss1 : p.ss;
    TLD(67, gen);
  }
#if SKDYN
"""),
    ("""  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
""", """  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
  if (tid == 0) TLS(68, clock64());
"""),
    ("""    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
""", """    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  if (tid == 0) TLD(69, S.eid[0]);
  CTA_SYNC();
  if (tid == 0) TLBR(70);
"""),
    ("""    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
""", """    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  if (tid == 0) TLS(71, clock64());
  CTA_SYNC();
  if (tid == 0) TLBR(72);
  if (warp == 0) {
"""),
    ("""#endif
  }
  CTA_SYNC();
  early_issue(p, S);
""", """#endif
    if (lane == 0) TLS(73, clock64());
  }
  CTA_SYNC();
  if (tid == 0) TLBR(74);
  early_issue(p, S);
  if (tid == 0) TLS(75, clock64());
"""),
    ("""    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}
""", """    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  if (tid == 0) TLS(76, clock64());
  CTA_SYNC();
  if (tid == 0) TLBR(77);
}
"""),
    ("""    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
""", """    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
  if (tid == 0) TLS(78, clock64());
#if OKC_WS
"""),
    ("""    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
  if (tid == 0) TLS(1, gtimer());
""", """    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  if (tid == 0) TLS(79, clock64());
  WS_JOIN();
  if (tid == 0) TLS(1, gtimer());
"""),
    ("""  const int lane = TIDX & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
""", """  const int lane = TIDX & 31;
  if (lane == 0) TLS(96, clock64());
#if OKC_TLP
  if (TLX_ON && lane == 0) {
    const unsigned char* pa = p.wBu + ((size_t)blockIdx.x << 21) + 4096;
    unsigned v0, v1, v2;
    const unsigned long long t0 = clock64();
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v0) : "l"(pa) : "memory");
    if ((int)v0 == 0x7f7f7f7f) asm volatile("trap;");
    const unsigned long long t1 = clock64();
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v1) : "l"(pa + 65536) : "memory");
    if ((int)v1 == 0x7f7f7f7f) asm volatile("trap;");
    const unsigned long long t2 = clock64();
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v2) : "l"(pa) : "memory");
    if ((int)v2 == 0x7f7f7f7f) asm volatile("trap;");
    const unsigned long long t3 = clock64();
    TLS(100, t1 - t0); TLS(101, t2 - t1); TLS(102, t3 - t2);
  }
#endif
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  if (lane == 0) TLBR(97);
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  if (lane == 0) TLS(98, clock64());
"""),
    ("""        __syncwarp();
#if OKC_TLX
        if (tl) {
          const unsigned long long t1 = gtimer();
          tl_dep += t1 - tl_t0;
          if (tl_ndep == 0ull) { TLW(4, tl_t0); TLW(5, t1); }
          ++tl_ndep;
        }
#endif
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
""", """        __syncwarp();
#if OKC_TLX
        if (tl) {
          const unsigned long long t1 = gtimer();
          tl_dep += t1 - tl_t0;
          if (tl_ndep == 0ull) { TLW(4, tl_t0); TLW(5, t1); }
          ++tl_ndep;
        }
#endif
      }
#if OKC_TLX
      if (!PB && i == 0) TLW(80, clock64());
#endif
      load_x_regs<PB>(p, S, j, lane, warp, x0);
#if OKC_TLX
      if (!PB && i == 0 && tl && lane == 0) TLD(81, x0[0].x);
#endif
"""),
]

# --r2 (diagnostic): route() runs a second time right after the first (idempotent: it rebuilds the same shared tables
# from the same topk_ids), stamped 103 (second call start) and 99 (second call end); route()'s own slots 68..77 then
# hold the second call. A second call much shorter than the first means the first one waits on instruction fetch.
R2 = [("""  route(p, S);
#if OKC_WGS && OKC_WGSPR
""", """  route(p, S);
  if (TLX_ON) { if (tid == 0) TLS(103, clock64()); route(p, S); if (tid == 0) TLS(99, clock64()); }
#if OKC_WGS && OKC_WGSPR
""")]


def main():
    src = ROOT / arg("--base", "muse_decode_layer/cand/ly_stk2.py")
    pin = arg("--pin", "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846")
    dst = ROOT / arg("--out", "muse_decode_layer/cand/ly_stk2_pro16.py")
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        sys.exit(f"{src.name} is not {pin[:8]}")
    whole = raw.decode()
    assert whole.count(MARK) == 1
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    tool = STANDALONE.read_text()
    cuda = tool[tool.index("# ---------------------------------------------------------------- CUDA"):
                tool.index("# ---------------------------------------------------------------- host")]
    box = {"src": helper}

    def rep(old, new, count=1):
        n = box["src"].count(old)
        if n != count:
            sys.exit(f"helper0 anchor found {n} times, {count} expected: {old[:100]!r}")
        box["src"] = box["src"].replace(old, new)
    exec(cuda, {"rep": rep})
    rep("p.dbg[blockIdx.x * 64 + (k)]", "p.dbg[blockIdx.x * 128 + (k)]", 2)
    for old, new in EXTRA + (R2 if "--r2" in sys.argv else []):
        rep(old, new)
    if tail.count(MID_ADD) != 1:
        sys.exit("MID helper define anchor")
    defs = '"-DOKC_TLX=1", "-DOKC_TLP=1"' if "--probe" in sys.argv else '"-DOKC_TLX=1",'
    tail = tail.replace(MID_ADD, MID_ADD + f"MOE_MID_DEFINES += ({defs})\n")
    out = head + MARK + box["src"] + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
