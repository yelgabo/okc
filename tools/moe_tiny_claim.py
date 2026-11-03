"""Build a Tiny / Tiny2 candidate from current_control_release.py (5df257af) whose last phase B units are claimed at run time.

  OKC_BCLM=k  the last P = min(G k, nB - G (NS + 1)) phase B units of a launch form a pool (nB: phase B units, G: CTAs).
              Every CTA keeps its static phase A range and a static phase B share of the other nB - P units (at least
              NS + 1 each), then thread 0 claims pool units one at a time from p.ctr[1] (zeroed after the grid barrier,
              as the release already does for its dynamic pool counters) as it refills ring slots. The claim for the
              next fill is issued one fill ahead, so its latency is hidden behind one unit. A CTA that finds the pool
              empty fills its slot with an end marker. P = 0 (for example T = 1 with two local experts) leaves the
              release's static split and loop.

The timeline in docs/runs-moe.md "MOE-TL-01" puts the phase B end spread at 4.0 us at T = 8 (46.8 / 48.4 / 50.8 us) with
balanced unit counts not closing it (MOE-BAL-01), so the spread is rate variance across CTAs, which claims absorb.

Ordering (no fence or barrier is removed or moved): thread 0 writes the claimed unit, or -1, to S.bq[slot] before the
slot's mbarrier arrive (PTX ISA: mbarrier.arrive defaults to .release at .cta scope, mbarrier.try_wait to .acquire at
.cta scope), then issues the bulk copy as the release does; an end marker is a plain arrive
with no transaction bytes. The unit of each partial buffer reaches the helper warp the same way, through S.bh[buffer]
and the existing wsrf arrive / wait pair. WSBL keeps the last OKC_WSBLK units' epilogues on compute warps 0..K-1: thread 0
knows the end position NS fills ahead, the compute warps learn it from S.bq one position ahead, and thread 0 sends the
helper an end marker at the first kept position. Weights, x / inter loads, doneA waits and epilogue math are the
release's.

    python3 tools/moe_tiny_claim.py <release.py> <out.py> [--k 2]
"""
import hashlib
import sys
from pathlib import Path

args = sys.argv[1:]
src_path, dst = Path(args[0]), Path(args[1])
k = int(args[args.index('--k') + 1]) if '--k' in args else 2
whole = src_path.read_text()
assert hashlib.sha256(whole.encode()).hexdigest().startswith('5df257af'), 'generator is pinned to the release 5df257af'
MARK = "_MA_ALT_SRCS = {'moe_helper0': r'''"
assert whole.count(MARK) == 1
head, src = whole.split(MARK)
head += MARK


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:120])
    src = src.replace(old, new)


rep('''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
''', '''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#ifndef OKC_BCLM
#define OKC_BCLM 0  // WS: the last min(G x OKC_BCLM, nB - G (NS + 1)) phase B units are claimed at run time
#endif
#if OKC_BCLM
static_assert(OKC_WS && OKC_TINY && !OKC_SPLIT && OKC_WSB && OKC_WSBL && OKC_WSBLK >= 1 && OKC_WSBLK <= 2 && OKC_NS == 3,
              "OKC_BCLM: WS tiny build, NS 3, one or two WSBL units");
#endif
''')

rep('''  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
''', '''  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#if OKC_BCLM
  int bq[8];                           // BCLM: claimed phase B unit per ring slot (-1: no unit left)
  int bh[8];                           // BCLM: phase B unit per partial buffer for the helper (-1: end)
  int wsP, wsBs;                       // BCLM: pool units; the pool's first unit (= static phase B units)
#endif
#endif
''')

rep('''    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
''', '''    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#if OKC_BCLM
    {
      const int lim = nB - G * (OKC_NS + 1), P0 = G * OKC_BCLM;
      const int P = lim <= 0 ? 0 : (P0 < lim ? P0 : lim);
      const int nBs = nB - P;
      S.wsr[2] = muldiv_g(nBs, c, G, rG); S.wsr[3] = muldiv_g(nBs, c + 1, G, rG);
      S.wsP = P; S.wsBs = nBs;
    }
#endif
  }
  WS_JOIN();
''')

rep('''#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
''', '''#if OKC_WS
  (void)cpf; (void)cBn;
#if OKC_BCLM
  if constexpr (PB) skinny_ring_ws_c(p, S, S.wsr[2], S.wsr[3], reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
  else
#endif
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
''')

NEW_FN = r'''#if OKC_BCLM
// BCLM phase B loop: the CTA's static phase B units, then (S.wsP > 0) pool units claimed by thread 0 at refill time.
// Without a pool it runs the release's loop. The unit body (wait, MMAs, partial handoff, CTA barrier) is the release's.
__device__ __forceinline__ void bclm_prefetch(const Params& p, const Smem& S, int v, int lane, float& lgs, uint32_t& lnw0, uint32_t& lnw1) {
  const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
  lgs = __ldg(p.gsB + S.jE[S.njF + v / RB_B]);
  lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
  lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
}
__device__ void skinny_ring_ws_c(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  const int nS = seq0 + nU;            // static positions: phase A units, then this CTA's static phase B units
  const bool dyn = S.wsP > 0;          // block-uniform; then nU >= NS + 1, so every pool fill happens in this call
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;
  uint32_t lnw0 = 0u, lnw1 = 0u;
  int vp1 = -1, vp2 = -1;              // units of the last two positions
  int pend = 0, qend = 0x7fffffff;     // thread 0: the claim issued one fill ahead; the end position once filled
  bool hend = false;                   // thread 0: the helper has its end marker
  int i = 0;
  for (;; ++i) {
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    int v;
    if (i < nU) {
      v = v0 + i;
    } else {
      if (!dyn) break;
      mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
      v = S.bq[sl];
      if (v < 0) break;
    }
    const int j = njF + v / RB_B;
    if (warp < OKC_WSBLK) {
      if (!dyn) {
        if (i == nU - 1 - warp) bclm_prefetch(p, S, v, lane, lgs, lnw0, lnw1);
      } else if (q + 1 >= nS && S.bq[(q + 1) % OKC_NS] < 0) {  // q + 1 was filled at iteration q - 2
        const int vk = warp == 0 ? v : vp1;
        if (vk >= 0) bclm_prefetch(p, S, vk, lane, lgs, lnw0, lnw1);
      }
    }
    vp2 = vp1;
    vp1 = v;
    if (j != curj) {
      curj = j;
      if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
      __syncwarp();
      load_x_regs<true>(p, S, j, lane, warp, x0);
    }
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      // WSBL: thread 0 knows the end position NS fills ahead, so it knows here whether unit q is one of the last K
      const bool kept = dyn ? q + OKC_WSBLK >= qend : i >= nU - OKC_WSBLK;
      const int qf = q + OKC_NS;  // the position that refills the slot just consumed
      if (qf < nS) {
        ws_issue(p, S, qf, seq0, S.wsr[0], S.wsr[2], ring);
        if (dyn && qf + 1 == nS) pend = atomicAdd(p.ctr + 1, 1);  // the first pool fill's claim, one fill ahead
      } else if (dyn && qf < qend) {
        const int u = pend;
        if (u < S.wsP) {
          const int vv = S.wsBs + u;
          S.bq[sl] = vv;                                   // ordered before the copy's expect_tx arrive (release.cta)
          ring_issue<true>(p, S, vv, ring + sl * UNIT_BYTES, S.bars + sl);
          pend = atomicAdd(p.ctr + 1, 1);
        } else {
          S.bq[sl] = -1;
          mbar_arrive1(S.bars + sl);                       // completes the phase with no bytes: the consumers' end marker
          qend = qf;
        }
      }
      if (!kept) { S.bh[bb] = v; mbar_arrive1(S.wsrf + bb); }
      else if (dyn && !hend) { S.bh[bb] = -1; mbar_arrive1(S.wsrf + bb); hend = true; }
    }
  }
  // the CTA's last units: epilogues here, not behind the helper's queue (i units ran in this call)
  if (warp < OKC_WSBLK && warp < i) {
    const int ul = i - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = warp == 0 ? vp1 : vp2;
    const int j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = fork_bf16(1.f + __uint_as_float(lnw0 << 16)), nw1 = fork_bf16(1.f + __uint_as_float(lnw1 << 16));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + i;
}
#endif

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {'''
rep('''
__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {''', '\n' + NEW_FN)

rep('''  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
''', '''#if OKC_BCLM
  const bool dyn = S.wsP > 0;  // the compute warps' thread 0 ends the queue with S.bh = -1 at the first WSBL unit
  const int Nh = dyn ? 0x7fffffff : N - min(OKC_WSBLK, b1 - b0);
#else
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
#endif
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
#if OKC_BCLM
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    int v = PB ? b0 + (q - nA) : a0 + q;
    if (dyn && PB) {
      v = S.bh[bb];
      if (v < 0) break;
    }
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
#else
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
#endif
''')

flag = "'-DOKC_BCLM=%d'" % k
for old, new in (("EXPERT_MOE_IDENTITIES = ('moe-control-release', ", "EXPERT_MOE_IDENTITIES = ('moe-tiny-bclm%d-01', " % k),
                 ("'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1')", "'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1', %s)" % flag),
                 ("'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0')", "'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0', %s)" % flag)):
    assert head.count(old) == 1, old
    head = head.replace(old, new)

dst.write_text(head + src)
print('wrote', dst, len(head + src), 'bytes', 'k', k)
