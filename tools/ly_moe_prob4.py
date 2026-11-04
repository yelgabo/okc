#!/usr/bin/env python3
"""B4 (decode 32) prologue stamps: tools/ly_moe_tlb4.py's level 1 instrument with 128 stamps per CTA, slots 64..127
holding every load and barrier from kernel entry to the gather, WS_JOIN and the first weight issue.

    python3 tools/ly_moe_prob4.py --base muse_decode_layer/cand/ly_stk2.py --pin SHA256 --out FILE [--probe]

Slots 0..63 are ly_moe_tlb4's (0 entry after griddepcontrol.wait, 1 route done, 7 gather done, 6 WS_JOIN, 8 first
weight issue, 10 x producer's first xready wait end). Thread 0: 64 entry before griddepcontrol.wait, 65 after the x
L2 prefetch loop, 66 after the lut loop, 67 after the mbarrier inits, 68 the launch generation atomic returned,
69 route() start, 70 its topk loads back, 71 after CTA barrier 1, 72 count loop, 73 barrier 2, 74 warp 0 scan,
75 barrier 3, 76 early_issue, 77 position loop, 78 barrier 4 (route() end), 79 after the parity zeroing, 80 jrow
scan, 81 the barrier after it, 82 first gather batch's loads back, 83 all batches stored, 84 after
fence.proxy.async.global, 85 the gather's CTA barrier, 86 the xready releases issued. Weights producer lane 0: 90
before WS_JOIN, 91 after it, 92 its stage range read. x producer lane 0: 94 entry, 95 after barrier 7, 96 after
wgs_ranges, 97 after WS_JOIN. --probe (-DOKC_TLP=1): the weights producer times three dependent ld.global.cg loads
before its WS_JOIN on a page no prologue touches (CTA c: p.wBs + c * 2 MiB + 4 KiB): 100 first touch, 101 a line
64 KiB away, 102 the first line again (ns). MAIN compiles the same text with OKC_TLB 0 (every stamp empty).
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ly_moe_tlb4 as tlb4  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


PROBE = """#if OKC_TLP
  if (TLB_ON && (threadIdx.x & 31) == 0) {
    const unsigned char* pa = p.wBs + ((size_t)blockIdx.x << 21) + 4096;
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
"""

EXTRA = [
    ("""  DBG(5);  // nothing to reset at exit""", """  if (tid == 0) TLS(120, clock64());
  DBG(5);  // nothing to reset at exit"""),
    ("#define TLS(k, v) do { if (TLB_ON) p.dbg[blockIdx.x * 128 + (k)] = (v); } while (0)\n",
     """#define TLS(k, v) do { if (TLB_ON) p.dbg[blockIdx.x * 128 + (k)] = (v); } while (0)
// stamp once v is in a register: the branch on v makes the timer read wait for the load that produced it
#define TLD(k, v) do { if (TLB_ON) { if ((int)(v) == 0x7f7f7f7f) asm volatile("trap;"); p.dbg[blockIdx.x * 128 + (k)] = clock64(); } } while (0)
#ifndef OKC_TLP
#define OKC_TLP 0
#endif
// stamp after a barrier: a volatile shared load cannot move above bar.sync, and the clock waits for the load
#define TLBR(k) do { if (TLB_ON) { const unsigned vv_ = *reinterpret_cast<volatile const unsigned*>(&S.lut[0]); TLD(k, vv_); } } while (0)
"""),
    ("#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\n#endif\n",
     """#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#define TLS(k, v) do { } while (0)
#define TLD(k, v) do { } while (0)
#define TLBR(k) do { } while (0)
#define OKC_TLP 0
#endif
"""),
    ("""  DBG(0);
  // programmatic dependent launch: everything below reads""",
     """  DBG(0);
  if (tid == 0) { TLS(64, clock64()); TLS(122, gtimer()); }
  // programmatic dependent launch: everything below reads"""),
    ("""  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code""",
     """  if (tid == 0) TLS(65, clock64());
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code"""),
    ("""    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
""", """    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) TLS(66, clock64());
  if (tid == 0) {
"""),
    ("""    S.preissued = 0;
#endif
    S.seq = 0;
""", """    S.preissued = 0;
#endif
    TLS(67, clock64());
    S.seq = 0;
"""),
    ("""    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
""", """    S.ssp = (gen & 1) ? p.ss1 : p.ss;
    TLD(68, gen);
  }
#if SKDYN
"""),
    ("""  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
""", """  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
  if (tid == 0) TLS(69, clock64());
"""),
    ("""    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
""", """    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  if (tid == 0) TLD(70, S.eid[0]);
  CTA_SYNC();
  if (tid == 0) TLBR(71);
"""),
    ("""    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
""", """    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  if (tid == 0) TLS(72, clock64());
  CTA_SYNC();
  if (tid == 0) TLBR(73);
  if (warp == 0) {
"""),
    ("""#endif
  }
  CTA_SYNC();
  early_issue(p, S);
""", """#endif
    if (lane == 0) TLS(74, clock64());
  }
  CTA_SYNC();
  if (tid == 0) TLBR(75);
  early_issue(p, S);
  if (tid == 0) TLS(76, clock64());
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
  if (tid == 0) TLS(77, clock64());
  CTA_SYNC();
  if (tid == 0) TLBR(78);
}
"""),
    ("""#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges""", """  if (tid == 0) TLS(79, clock64());
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges"""),
    ("""    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
""", """    if (tid == 0) S.jrow[nj] = carry;
    if (tid == 0) TLS(80, clock64());
  }
  CTA_SYNC();
  if (tid == 0) TLBR(81);
"""),
    ("""#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  asm volatile("fence.proxy.async.global;\\n" ::: "memory");
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
""", """#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
    if (tid == 0 && qb == q0) TLD(82, d[0] ? v[0].x : 0u);
  }
  if (tid == 0) TLS(83, clock64());
  asm volatile("fence.proxy.async.global;\\n" ::: "memory");
  if (tid == 0) TLS(84, clock64());
  CTA_SYNC();
  if (tid == 0) TLBR(85);
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
"""),
    ("""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
""", """      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
    TLS(86, clock64());
"""),
    ("""__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
""", """__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  if ((threadIdx.x & 31) == 0) TLS(90, clock64());
""" + PROBE + """  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  TLBR(91);
"""),
    ("""    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
""", """    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    if (pb == 0) TLD(92, g1 + sp);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
"""),
    ("""  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
""", """  if ((threadIdx.x & 31) == 0) TLS(94, clock64());
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
  if ((threadIdx.x & 31) == 0) TLBR(95);
  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)
  __syncwarp();
  if ((threadIdx.x & 31) == 0) TLS(96, clock64());
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  TLBR(97);
"""),
]

# --r2 (diagnostic): route() and wgs_ranges() each run a second time right after the first (both idempotent), stamped
# 103 / 99 (second route start / end) and 104 (second wgs_ranges end, after 96).
R2 = [("""  route(p, S);
#if OKC_WGS && OKC_WGSPR
""", """  route(p, S);
#if OKC_TLB
  if (TLB_ON) { if (tid == 0) TLS(103, clock64()); route(p, S); if (tid == 0) TLS(99, clock64()); }
#endif
#if OKC_WGS && OKC_WGSPR
"""), ("""  if ((threadIdx.x & 31) == 0) TLS(96, clock64());
""", """  if ((threadIdx.x & 31) == 0) TLS(96, clock64());
#if OKC_TLB
  if (TLB_ON) { wgs_ranges(p, S); __syncwarp(); if ((threadIdx.x & 31) == 0) TLS(104, clock64()); }
#endif
""")]


def main():
    src = ROOT / arg("--base", "muse_decode_layer/cand/ly_stk2.py")
    pin = arg("--pin", "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846")
    dst = ROOT / arg("--out", "muse_decode_layer/cand/ly_stk2_prob4.py")
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        sys.exit(f"{src.name} is not {pin[:8]}")
    whole = raw.decode()
    assert whole.count(tlb4.START) == 1 and whole.count(tlb4.END) == 1
    head, rest = whole.split(tlb4.START)
    cu, tail = rest.split(tlb4.END)
    for old, new in tlb4.EDITS:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"tlb4 anchor found {n} times: {old[:100]!r}")
        cu = cu.replace(old, new)
    n = cu.count("p.dbg[blockIdx.x * 64 + (k)]")
    if n != 2:
        sys.exit(f"stride anchor {n}")
    cu = cu.replace("p.dbg[blockIdx.x * 64 + (k)]", "p.dbg[blockIdx.x * 128 + (k)]")
    for old, new in EXTRA + (R2 if "--r2" in sys.argv else []):
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:100]!r}")
        cu = cu.replace(old, new)
    if tail.count(tlb4.B4) != 1:
        sys.exit("B4 define anchor")
    extra = ', "-DOKC_TLB=1", "-DOKC_TLP=1")' if "--probe" in sys.argv else ', "-DOKC_TLB=1")'
    tail = tail.replace(tlb4.B4, tlb4.B4[:-2] + extra)
    out = head + tlb4.START + cu + tlb4.END + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
