"""Build muse_decode_layer/cand/ly_gb5.py from ly_stk4.py: MID build (moe_helper0 alt source) without the grid barrier.

usage: python3 tools/mk_ly_gb5.py <ly_stk4.py> <ly_gb5.py>
"""
import re
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
t = open(src_path).read()
head = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""
a0 = t.index(head) + len(head)
a1 = t.index('"""', a0)
alt = t[a0:a1]


def rep(s, old, new, where):
    n = s.count(old)
    assert n == 1, (where, n)
    return s.replace(old, new)


DB = "p.doneA + WSF_DB + S.mpar * JMAX"
alt = rep(alt, "constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;\n",
"""constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
#ifndef OKC_WSFIN
#define OKC_WSFIN 0
#endif
#if OKC_WSFIN && !(OKC_WS && OKC_TINY && OKC_FINPF && OKC_FINCOL && !OKC_WGS)
#error "OKC_WSFIN: MID build only (warp-specialized skinny, FINPF column final)"
#endif
// OKC_WSFIN words in the doneA block (1024 ints): doneB by MID parity at WSF_DB, the MID launch counter at WSF_PAR
constexpr int WSF_DB = 512, WSF_PAR = 1023;
static_assert(JMAX <= WSF_DB && WSF_DB + 2 * JMAX <= WSF_PAR, "doneA block layout");
""", "E1")
assert alt.count("\n  int gen;") == 1
m = re.search(r"\n  int gen;[^\n]*\n", alt)
alt = alt[:m.end()] + "#if OKC_WSFIN\n  int mpar;  // MID launch parity (doneB copy)\n#endif\n" + alt[m.end():]
WSI = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\n"
alt = rep(alt, WSI,
"""#if OKC_WSFIN
__device__ __forceinline__ int wsfin_job(const Smem& S, int q) {  // jobs are laid out in position order: jS ascends
  int lo = S.njF, hi = S.njF + S.njS;
  while (hi - lo > 1) { const int m = (lo + hi) >> 1; if (S.jS[m] <= q) lo = m; else hi = m; }
  return lo;
}
#endif
""" + WSI, "E3")
alt = rep(alt, """      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
""", """      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
#if OKC_WSFIN
    asm volatile("bar.sync %0, 32;\\n" ::"r"(9 + warp) : "memory");  // the warp's stores and ss adds, then one release
    if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(""" + DB + """ + j), "r"(1) : "memory");
#endif
  }
  seq = seq0 + nU;
""", "E4")
alt = rep(alt, "  for (int q = 0; q < Nh; ++q) {\n", "#if OKC_WSFIN\n  int pend = 0;\n#endif\n  for (int q = 0; q < Nh; ++q) {\n", "E5a")
alt = rep(alt, """        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
""", """        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
#if OKC_WSFIN
      ++pend;
      if (rb == RB_B - 1 || q == Nh - 1) {  // the helper's last unit of job j: the warp's stores, then one release
        asm volatile("bar.sync 8, 32;\\n" ::: "memory");
        if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(""" + DB + """ + j), "r"(pend) : "memory");
        pend = 0;
      }
#endif
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
#if OKC_WSFIN
  // the grid barrier's arrival without its wait: the last CTA past phase B (doneA's last reader anywhere) resets
  // doneA and advances the generation and the MID launch counter, which every CTA read at its start
  if (lane == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ctr + 4 + (S.gen & 1)), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) {
      for (int i = 0; i < JMAX; ++i) p.doneA[i] = 0;
      atomicAdd(p.ctr + 3, 1);
      atomicAdd(p.doneA + WSF_PAR, 1);
    }
  }
#endif
}
""", "E5b")
alt = rep(alt, "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n",
          "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n#if OKC_WSFIN\n"
          "    S.mpar = *reinterpret_cast<volatile const int*>(p.doneA + WSF_PAR) & 1;\n#endif\n", "E6")
alt = rep(alt, "    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n",
"""    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
#if OKC_WSFIN
    for (int i = blockIdx.x * NT + tid; i < JMAX; i += gridDim.x * NT) p.doneA[WSF_DB + (S.mpar ^ 1) * JMAX + i] = 0;
#endif
""", "E7")
b0 = "  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);\n"
b1 = "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n"
assert alt.count(b0) == 1 and alt.count(b1) == 1
i0 = alt.index(b0)
i1 = alt.index(b1)
assert i0 < i1
WAIT = """      if (q >= 0) {
        int* c = """ + DB + """ + wsfin_job(S, q);
        while (ld_acquire(c) < RB_B) __nanosleep(32);
      }
"""
alt = alt[:i0] + """#if OKC_WSFIN
  static_assert(!SKDYN, "OKC_WSFIN: ctr[0..1] (the dynamic pool) stay untouched only without the pool");
  // no grid barrier: wait for the phase-B releases of the jobs this CTA's first final item reads
  if (tid < 8 && hasp) {
    const int q = S.pos[tkp * 8 + tid];
""" + WAIT + """  }
  CTA_SYNC();
  DBG(4);
#else
""" + alt[i0:i1] + "#endif\n" + alt[i1:]
alt = rep(alt, """  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
""", """  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
#if OKC_WSFIN
    if (tid < 8) {
      const int q = S.pos[tk * 8 + tid];
""" + WAIT + """    }
    CTA_SYNC();
#endif
""", "E9")
t = t[:a0] + alt + t[a1:]
old = "'-DOKC_WSBLK=2', '-DOKC_FINPF=1')      # their mid-M build"
assert t.count(old) == 1
t = t.replace(old, "'-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_WSFIN=1')      # their mid-M build")
open(dst_path, "w").write(t)
print("ok")
