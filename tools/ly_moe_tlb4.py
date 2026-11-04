#!/usr/bin/env python3
"""Decode 32 B4 timeline: writes muse_decode_layer/cand/ly_tlb4.py from ly_micro2.py (0bc2bd8b).

Per-CTA globaltimer stamps in the WGS moe_fused source (MOE_CUDA_SRC), compiled in only with -DOKC_TLB=<level>, which is
added to the B4 build (decode 32, 16 < M <= 32) alone; MAIN compiles the same text with OKC_TLB 0. The stamps run only
when a launch passes a fourth scratch tensor (p.dbg); the layer's own launches pass three. Fences, barriers, mbarrier
protocol, ring, MFIN counters and the final pass are unchanged.

Levels: 1 = stamps outside the consumers' stage loop only (kernel body, producers), and slots 24 + n
hold the weights producer's issue time of stage n (it issues stage n once the consumers released stage n - 3's codes);
2 = also the consumers' stage-end stamps in slots 24 + s (the producer stamps are off); 3 = also the consumers'
per-stage wait totals and the designated-finisher waits (slots 15 to 20, 22, 23). Levels 2 and 3 change the stage
loop's code (see the SASS reports), level 1 should not.

Slots (64 int64 per CTA):
  0 entry  1 routing done  2 phase A end  3 phase B end  4 first final item ready (MFIN wait done)  5 exit
  6 producers released (WS_JOIN)  7 phase A x gather done
  8 weights producer: first TMA issue   9 weights producer: first phase B TMA issue
  10 x producer: first phase A xready wait end   11 / 12 x producer: first phase B doneAs wait start / end
  13 x producer: phase B doneAs wait, total ns   14 x producer: phase A xready wait, total ns
  15 / 16 consumer tid 0, phase A: weights wait / x wait, total ns   17 / 18 the same in phase B
  19 / 20 designated-finisher wait (phase A / B), total ns
  21 stage ranges: A g0 | A g1 << 16 | B g0 << 32 | B g1 << 48 (a per-segment counter in wgs_phase moved the stage
     loop's descriptor math off the uniform datapath, 304 -> 386 instructions, so segments are counted on the host)
  22 first stage's weights arrived   23 first phase B stage's x arrived (consumer)
  24 + s: level 1 weights producer issue of stage s; level >= 2 consumer end of stage s, s < 40

    python3 tools/ly_moe_tlb4.py [--level 1|2|3] [--out muse_decode_layer/cand/ly_tlb4w.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
DST = ROOT / "muse_decode_layer/cand/ly_tlb4.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
B4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'

EDITS = [
    ('#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\n',
     '''#ifndef OKC_TLB
#define OKC_TLB 0
#endif
#if OKC_TLB
#define TLB_ON (p.dbg != nullptr)
#define TLS(k, v) do { if (TLB_ON) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)
#define DBG(k) do { if (TLB_ON && threadIdx.x == 0) p.dbg[blockIdx.x * 64 + (k)] = gtimer(); } while (0)
// lane 0 only, as predicated instructions (no divergent branch inside the stage loop)
__device__ __forceinline__ void tlb_st(unsigned long long* a, unsigned long long v, int lane) {
  asm volatile("{\\n.reg .pred q;\\nsetp.eq.s32 q, %2, 0;\\n@q st.global.u64 [%0], %1;\\n}\\n" ::"l"(a), "l"(v), "r"(lane) : "memory");
}
__device__ __forceinline__ void tlb_red(unsigned long long* a, unsigned long long v, int lane) {
  asm volatile("{\\n.reg .pred q;\\nsetp.eq.s32 q, %2, 0;\\n@q red.shared.add.u64 [%0], %1;\\n}\\n" ::"r"((uint32_t)__cvta_generic_to_shared(a)), "l"(v), "r"(lane) : "memory");
}
#else
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#endif
'''),
    ('  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n',
     '''  short pos[EMAX];   // entry -> pair position, -1 if not local
#if OKC_TLB
  unsigned long long tlb[6];
#endif
};
'''),
    ('    S.seq = 0;\n', '''    S.seq = 0;
#if OKC_TLB
    for (int k = 0; k < 6; ++k) S.tlb[k] = 0ull;
#endif
'''),
    ('''      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
''', '''      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
#if OKC_TLB
      if (TLB_ON && i == 0) TLS(8 + pb, gtimer());
#if OKC_TLB == 1
      if (TLB_ON && n < 40) TLS(24 + n, gtimer());
#endif
#endif
'''),
    ('''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }
''', '''#if OKC_TLB
        const unsigned long long ta = TLB_ON ? gtimer() : 0ull;
#endif
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }  // every row gathered
#if OKC_TLB
        if (TLB_ON && !pb) { const unsigned long long tb = gtimer(); twA += tb - ta; if (i == 0) TLS(10, tb); }
#endif
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
#if OKC_TLB
      const unsigned long long tc = TLB_ON ? gtimer() : 0ull;
#endif
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }
#if OKC_TLB
      if (TLB_ON && pb) { const unsigned long long td = gtimer(); twB += td - tc; if (i == 0) { TLS(11, tc); TLS(12, td); } }
#endif
'''),
    ('''  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
''', '''  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#if OKC_TLB
  unsigned long long twA = 0ull, twB = 0ull;
#endif
'''),
    ('''      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
''', '''      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
#if OKC_TLB
  TLS(13, twB);
  TLS(14, twA);
#endif
}
'''),
    ('''    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
''', '''    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
#if OKC_TLB >= 2
    const bool tl0 = TLB_ON && warp == 0;  // warp-uniform: a per-thread branch here moves the stage's address math off the uniform datapath
#endif
#if OKC_TLB >= 3
    const unsigned long long t0 = tl0 ? gtimer() : 0ull;
#endif
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
#if OKC_TLB >= 3
    const unsigned long long t1 = tl0 ? gtimer() : 0ull;
#endif
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
#if OKC_TLB >= 3
    if (tl0) {
      const unsigned long long t2 = gtimer();
      tlb_red(&S.tlb[2 * PB], t1 - t0, lane);
      tlb_red(&S.tlb[2 * PB + 1], t2 - t1, lane);
      if (seq == 0) tlb_st(p.dbg + blockIdx.x * 64 + 22, t1, lane);
      if (PB && seq == S.wgr[1] - S.wgr[0]) tlb_st(p.dbg + blockIdx.x * 64 + 23, t2, lane);
    }
#endif
'''),
    ('''    mbar_arrive1(S.bars + BX_E + xs);
    ++seq;
  };
''', '''    mbar_arrive1(S.bars + BX_E + xs);
    ++seq;
#if OKC_TLB >= 2
    if (tl0 && seq <= 40) tlb_st(p.dbg + blockIdx.x * 64 + 23 + seq, gtimer(), lane);
#endif
  };
'''),
    ('''  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
''', '''#if OKC_TLB >= 3
  const unsigned long long tf0 = (TLB_ON && tid == 0) ? gtimer() : 0ull;
#endif
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
#if OKC_TLB >= 3
  if (TLB_ON && tid == 0) S.tlb[4 + PB] += gtimer() - tf0;
#endif
'''),
    ('''    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
''', '''    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
#if OKC_TLB
    if (TLB_ON && tid == 0) {
      for (int k = 0; k < 6; ++k) TLS(15 + k, S.tlb[k]);
      TLS(21, (unsigned long long)S.wgr[0] | ((unsigned long long)S.wgr[1] << 16) | ((unsigned long long)S.wgr[2] << 32) |
              ((unsigned long long)S.wgr[3] << 48));
    }
#endif
'''),
    ('''          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
#if B4_EARLY_Y
''', '''          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
#if OKC_TLB
      if (it == (int)blockIdx.x) DBG(4);
#endif
#if B4_EARLY_Y
'''),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"ly_micro2.py is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    whole = raw.decode()
    assert whole.count(START) == 1 and whole.count(END) == 1
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    for old, new in EDITS:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:100]!r}")
        cu = cu.replace(old, new)
    if tail.count(B4) != 1:
        sys.exit("B4 define anchor")
    lvl = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 1
    tail = tail.replace(B4, B4[:-2] + f', "-DOKC_TLB={lvl}")')
    out = head + START + cu + END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
