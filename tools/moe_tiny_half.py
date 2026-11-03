"""Build a Tiny / Tiny2 candidate from current_control_release.py (5df257af) that hides the phase A to phase B handoff.
Each mechanism is a compile-time switch on the tiny and tiny2 defines only (MID and B4 / MAIN bytes unchanged):

  OKC_HALF  a CTA's unit sequence is four chunks: phase A units of the first half of the jobs, phase A of the second
            half, then phase B of each half. The helper credits doneA at the end of each A chunk, so a B chunk's
            dependency (every CTA's A units of its jobs) is complete long before the CTA reaches it: the first half's
            while the second half's A units stream, the second half's during the first half's B units.
  OKC_XS    the next job's x rows (phase A: x; phase B: inter) are staged through smem by cp.async during the previous
            unit, once lane 0 has observed the job's doneA complete (phase B), so a job switch costs a few LDS instead
            of a dependent L2 fetch after the wait. Falls back to the release's wait-then-load path when the rows
            are not staged in time.
  OKC_TLX   the first worker's per-CTA globaltimer timeline (tools/moe_tiny_timeline.py), gated by a fourth scratch
            tensor, with the switch costs added; the timed path passes three tensors, so every stamp is dormant.

    python3 tools/moe_tiny_half.py <release.py> <out.py> [--no-half] [--no-xs] [--no-tlx]
"""
import hashlib
import sys
from pathlib import Path

args = sys.argv[1:]
src_path, dst = Path(args[0]), Path(args[1])
half = '--no-half' not in args
xs = '--no-xs' not in args
tlx = '--no-tlx' not in args
whole = src_path.read_text()
assert hashlib.sha256(whole.encode()).hexdigest().startswith('5df257af'), 'generator is pinned to the release 5df257af'
MARK = "_MA_ALT_SRCS = {'moe_helper0': r'''"
assert whole.count(MARK) == 1
head, src = whole.split(MARK)
head += MARK


def rep(old, new, count=1):
    """Edit the helper0 source and the host code after it; MOE_CUDA_SRC (B4 / MAIN) sits before the marker."""
    global src
    assert src.count(old) == count, (src.count(old), old[:120])
    src = src.replace(old, new)


# ---------------------------------------------------------------- defines and shared memory
rep('''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
''', '''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#ifndef OKC_HALF
#define OKC_HALF 0  // WS: A units of the first job half, A of the second, then B of each half (hides the doneA wait)
#endif
#ifndef OKC_XS
#define OKC_XS 0    // WS: the next job's x rows staged through smem by cp.async during the previous unit
#endif
#ifndef OKC_TLX
#define OKC_TLX 0
#endif
static_assert(!OKC_HALF || (OKC_WS && OKC_TINY && !OKC_SPLIT && OKC_WSB), "OKC_HALF: WS tiny build");
static_assert(!OKC_XS || OKC_HALF, "OKC_XS: needs the chunked sequence");
// OKC_TLX: per-CTA globaltimer timeline (64 stamps per CTA in p.dbg) when the host passes a fourth scratch tensor;
// without it (the timed path) every stamp is a uniform not-taken branch
#define TLX_ON (OKC_TLX && p.dbg != nullptr)
#define TLS(k, v) do { if (TLX_ON) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)
''')

rep('''  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
''', '''  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#if OKC_HALF
  int wsc[8];                          // HALF: the four chunks as [v0, v1) unit ranges (A half 0, A half 1, B half 0, B half 1)
  int wsN;                             // HALF: units in the sequence
#endif
#endif
''')

# ---------------------------------------------------------------- the WS ring: chunked sequence, staged x
START = 'template <bool PB>\n__device__ void skinny_ring_ws('
END = 'template <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split('
s0, s1 = src.index(START), src.index(END)
old_ws = src[s0:s1]
assert 'ws_helper' in old_ws and old_ws.count('atomicAdd(p.doneA + jj, b - a)') == 1

NEW_WS = r'''#if OKC_HALF
// HALF: chunk k of a CTA's sequence is a share of the A units of jobs [0, h) (k = 0), the A units of jobs [h, njS)
// (1), the B units of [0, h) (2) and of [h, njS) (3), h = njS / 2. Every CTA's chunk 0 ends before its chunk 1, so a
// job half's doneA is complete about half a phase before any CTA needs it. A chunk's share is the difference of the
// CTA's proportional splits of the cumulative unit count through that chunk and through the one before, so the
// shares tile each half and every CTA carries floor or ceil of the total over G (checked for G 100-136, njS 0-8)
__device__ __forceinline__ void ws_ranges(int njS, int c, int G, int (&r)[8]) {
  const float rG = __fdividef(1.f, (float)G);
  const int h = njS >> 1;
  const int n0 = h * RB_A, n1 = (njS - h) * RB_A, n2 = h * RB_B, n3 = (njS - h) * RB_B;
  const int C1 = n0, C2 = C1 + n1, C3 = C2 + n2, C4 = C3 + n3;
  const int s0 = 0, s1 = muldiv_g(C1, c, G, rG), s2 = muldiv_g(C2, c, G, rG), s3 = muldiv_g(C3, c, G, rG), s4 = muldiv_g(C4, c, G, rG);
  const int e1 = muldiv_g(C1, c + 1, G, rG), e2 = muldiv_g(C2, c + 1, G, rG), e3 = muldiv_g(C3, c + 1, G, rG), e4 = muldiv_g(C4, c + 1, G, rG);
  r[0] = s1 - s0; r[1] = e1;
  r[2] = n0 + (s2 - s1); r[3] = n0 + (e2 - e1);
  r[4] = s3 - s2; r[5] = e3 - e2;
  r[6] = n2 + (s4 - s3); r[7] = n2 + (e4 - e3);
}
__device__ __forceinline__ int ws_seq_unit(const Smem& S, int q, bool& PB) {  // sequence position -> (phase, unit)
  int k = 0;
  while (k < 3) {
    const int n = S.wsc[2 * k + 1] - S.wsc[2 * k];
    if (q < n) break;
    q -= n; ++k;
  }
  PB = k >= 2;
  return S.wsc[2 * k] + q;
}
__device__ __forceinline__ void ws_issue_h(const Params& p, Smem& S, int q, unsigned char* ring) {
  const int sl = q % OKC_NS;
  bool PB;
  const int v = ws_seq_unit(S, q, PB);
  if (PB) ring_issue<true>(p, S, v, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<false>(p, S, v, ring + sl * UNIT_BYTES, S.bars + sl);
}
#endif
#if OKC_XS
constexpr int XS_WARP = 8 * 2 * NCH * 4;  // staged x per warp, uint4: [row g][chunk c'][lane t], the x0 register layout
__device__ __forceinline__ int xs_key(int j, bool PB) { return j | (PB ? 1 << 16 : 0); }
// this warp's K slice of job j's rows into its staging area; rows past the job's tokens are zero-filled (src-size 0).
// cp.async (non-bulk) reads through the generic proxy, as the fat path's inter stage loads in this file already do
// after the same one-thread acquire on doneA, so no proxy fence is added (PTX ISA memory model, Proxies: the async
// proxy is cp.async.bulk / tensor copies)
__device__ __forceinline__ void xs_stage(const Params& p, const Smem& S, int j, bool PB, int lane, int warp, uint4* xsw) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const int r = g < cnt ? g : 0;
  const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + r) * ID : p.x + (size_t)(S.ent[st + r] >> 3) * HD;
  const uint4* srcp = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    cp_async16(xsw + ((g * NCH + c) * 2 + 0) * 4 + t, srcp + c * 8, g < cnt);
    cp_async16(xsw + ((g * NCH + c) * 2 + 1) * 4 + t, srcp + c * 8 + 1, g < cnt);
  }
  cp_commit();
}
#endif
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq
#if OKC_HALF
                               , int& xst, int jnext, bool PBnext, bool last, int chunk
#endif
                               ) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
#if !OKC_HALF
  constexpr bool last = true;
#endif
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
#if OKC_XS
  uint4* xsw = reinterpret_cast<uint4*>(ring + OKC_NS * UNIT_BYTES) + warp * XS_WARP;
#endif
#if OKC_TLX
  const bool tl = TLX_ON && warp == 0;
  unsigned long long tl_wait = 0ull, tl_buf = 0ull, tl_dep = 0ull, tl_ndep = 0ull, tl_t0 = 0ull;
#define TLW(k, v) do { if (tl && lane == 0) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)
#endif
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && last && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
#if OKC_XS
    // the next job of this CTA's sequence (the next unit's job when it differs; else, once this chunk has no later
    // job change, the next chunk's first): lane 0 polls its doneA now (phase B) and the warp stages its rows after
    // the unit's MMAs if it is complete. One staged job at a time, so two cp.async groups never target the area
    int jn = -1;
    bool PBn = PB;
    if (i + 1 < nU && njF + (v + 1) / rbs != j) jn = njF + (v + 1) / rbs;
    else if (njF + (v1 - 1) / rbs == j) { jn = jnext; PBn = PBnext; }
    const bool stg = jn >= 0 && (jn != j || PBn != PB) && xst != xs_key(jn, PBn);
    int rd = RB_A;
    if (stg && PBn && lane == 0) rd = ld_acquire(p.doneA + jn);
#endif
    if (j != curj) {
      curj = j;
#if OKC_TLX
      if (tl) tl_t0 = gtimer();
#endif
#if OKC_XS
      if (xst == xs_key(j, PB)) {
        cp_wait<0>();
        __syncwarp();
#pragma unroll
        for (int c = 0; c < 2 * NCH; ++c) x0[c] = xsw[((lane >> 2) * (2 * NCH) + c) * 4 + (lane & 3)];
        xst = -1;
      } else
#endif
      {
#if OKC_XS
        if (xst >= 0) { cp_wait<0>(); xst = -1; }  // (never by construction: a staged job that is not this one)
#endif
        if (PB) {
          if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
          __syncwarp();
        }
        load_x_regs<PB>(p, S, j, lane, warp, x0);
      }
#if OKC_TLX
      if (tl) {
        const unsigned long long t1 = gtimer();
        if (PB) {
          tl_dep += t1 - tl_t0;
          if (tl_ndep == 0ull) { TLW(4, tl_t0); TLW(5, t1); }
          ++tl_ndep;
        }
#if OKC_HALF
        if (i == 0 && chunk > 0) { TLW(44 + 2 * chunk, tl_t0); TLW(45 + 2 * chunk, t1); }
#endif
      }
#endif
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
#if OKC_TLX
    if (tl) tl_t0 = gtimer();
#endif
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
#if OKC_TLX
    if (tl) {
      const unsigned long long t1 = gtimer();
      tl_wait += t1 - tl_t0;
      if (!PB && q == 0) TLW(2, t1);
    }
#endif
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
#if OKC_XS
    if (stg) {
      rd = __shfl_sync(0xffffffffu, rd, 0);
      if (rd >= RB_A) {  // lane 0's acquire covers the warp through the warp barrier, as the release's own wait does
        __syncwarp();
        xs_stage(p, S, jn, PBn, lane, warp, xsw);
        xst = xs_key(jn, PBn);
      }
    }
#endif
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
#if OKC_TLX
    if (tl) tl_t0 = gtimer();
#endif
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
#if OKC_TLX
    if (tl) tl_buf += gtimer() - tl_t0;
#endif
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
#if OKC_HALF
      if (q + OKC_NS < S.wsN) ws_issue_h(p, S, q + OKC_NS, ring);
#else
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
#endif
      if (!(OKC_WSBL && PB && last && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
#if OKC_TLX
    if (tl) {
      const unsigned long long t = gtimer();
      if (q < 26) TLW(18 + q, t);
      if (i == nU - 1) TLW(PB ? 6 : 3, t);
    }
#endif
  }
  if (PB && last && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
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
#if OKC_TLX
  if (tl) {
    TLW(PB ? 13 : 12, tl_wait + (PB ? p.dbg[blockIdx.x * 64 + 13] : p.dbg[blockIdx.x * 64 + 12]));
    TLW(PB ? 15 : 14, tl_buf + (PB ? p.dbg[blockIdx.x * 64 + 15] : p.dbg[blockIdx.x * 64 + 14]));
    if (PB) {
      TLW(16, tl_dep + p.dbg[blockIdx.x * 64 + 16]);
#if OKC_HALF
      const unsigned long long nA = (unsigned long long)((S.wsc[1] - S.wsc[0]) + (S.wsc[3] - S.wsc[2]));
      const unsigned long long nB = (unsigned long long)((S.wsc[5] - S.wsc[4]) + (S.wsc[7] - S.wsc[6]));
#else
      const unsigned long long nA = (unsigned long long)(S.wsr[1] - S.wsr[0]), nB = (unsigned long long)(S.wsr[3] - S.wsr[2]);
#endif
      TLW(17, nA | (nB << 16) | ((unsigned long long)S.preissued << 32) | ((unsigned long long)S.njS << 40) | (tl_ndep << 48));
    }
  }
#undef TLW
#endif
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int njF = S.njF;
#if OKC_HALF
  const int N = S.wsN;
  int cv[4], cn[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) { cv[k] = S.wsc[2 * k]; cn[k] = S.wsc[2 * k + 1] - S.wsc[2 * k]; }
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue_h(p, S, q, ring);
  const int nBl = cn[3] > 0 ? cn[3] : cn[2];  // the last B chunk's units end with the compute warps' own epilogues
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, nBl) : N;
#else
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
#endif
  const int gg = lane >> 2, tt = lane & 3;
  for (int q = 0; q < Nh; ++q) {
#if OKC_HALF
    bool PB;
    const int v = ws_seq_unit(S, q, PB);
#else
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
#endif
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
#if OKC_HALF
      if (q == cn[0] - 1 || q == cn[0] + cn[1] - 1) {  // this CTA's phase A units of a job half are done: credit them
        const bool k1 = q >= cn[0];
        const int a0 = k1 ? cv[1] : cv[0], a1 = a0 + (k1 ? cn[1] : cn[0]);
#else
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
#endif
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
#if OKC_TLX
          if (TLX_ON) { if (p.dbg[blockIdx.x * 64 + 53] == 0ull) p.dbg[blockIdx.x * 64 + 53] = gtimer(); }
          TLS(7, gtimer());
#endif
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = fork_bf16(1.f + __bfloat162float(p.normw[n0])), nw1 = fork_bf16(1.f + __bfloat162float(p.normw[n0 + 8]));
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
  }
#if OKC_TLX
  if (lane == 0) TLS(8, gtimer());
#endif
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

'''
src = src[:s0] + NEW_WS + src[s1:]

# ---------------------------------------------------------------- FAST1: the first units of the chunked sequence
rep('''#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
''', '''#elif OKC_HALF
      // the same chunk table thread 0 derives after routing (njS = the touched-expert count): the first units of
      // chunk 0, then chunk 1 (both phase A; a CTA never has fewer than OKC_FAST1 A units when a job exists)
      int r1[8];
      ws_ranges(__popc(m), blockIdx.x, gridDim.x, r1);
      const int nc0 = r1[1] - r1[0];
      const int n = min(OKC_FAST1, nc0 + (r1[3] - r1[2]));
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
''')
rep('''#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
''', '''#elif OKC_HALF
      for (int i = 0; i < n; ++i) {
        const int v = i < nc0 ? r1[0] + i : r1[2] + (i - nc0);
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
''')

# ---------------------------------------------------------------- thread 0's chunk table, the phase calls
rep('''    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
''', '''    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#if OKC_HALF
    int r[8];
    ws_ranges(S.njS, c, G, r);
    int N = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) { S.wsc[2 * k] = r[2 * k]; S.wsc[2 * k + 1] = r[2 * k + 1]; N += r[2 * k + 1] - r[2 * k]; }
    S.wsN = N;
#endif
  }
  WS_JOIN();
#if OKC_TLX
  if (tid == 0) TLS(1, gtimer());
#endif
#endif
''')
rep('''#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
''', '''#elif OKC_HALF
  if (S.njF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int xst = -1;  // XS: the job staged in this warp's smem area (-1 none); carried across the chunks
    for (int k = 0; k < 4; ++k) {
      const int v0 = S.wsc[2 * k], v1 = S.wsc[2 * k + 1];
      if (v1 == v0) continue;
      int jnext = -1;  // the next non-empty chunk's first job: the staging target at this chunk's end
      bool PBnext = false;
      for (int kn = 3; kn > k; --kn)
        if (S.wsc[2 * kn + 1] > S.wsc[2 * kn]) { jnext = S.njF + S.wsc[2 * kn] / (kn >= 2 ? RB_B : RB_A); PBnext = kn >= 2; }
      if (k < 2) skinny_ring_ws<false>(p, S, v0, v1, ring, S.bars, S.seq, xst, jnext, PBnext, false, k);
      else skinny_ring_ws<true>(p, S, v0, v1, ring, S.bars, S.seq, xst, jnext, PBnext, jnext < 0, k);
    }
  }
  DBG(2);
  DBG(3);
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
''')

# ---------------------------------------------------------------- shared memory size: the staging area after the ring
rep('''  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
''', '''  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
#if OKC_XS
  xa += (size_t)okc::NW * okc::XS_WARP * 16;
#endif
''')

# ---------------------------------------------------------------- timeline stamps outside the ring (the first worker's)
rep('''  asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
''', '''  asm volatile("griddepcontrol.wait;\\n" ::: "memory");
  if (tid == 0) TLS(0, gtimer());
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
''')
rep('''  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
''', '''  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    TLS(9, gtimer());
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
    TLS(10, gtimer());
  }
  CTA_SYNC();
  DBG(4);
''')
rep('''  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}
''', '''  if (tid == 0) TLS(11, gtimer());
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}
''')

# ---------------------------------------------------------------- host: defines, identity, timeline prints
flags = (['-DOKC_HALF=1'] if half else []) + (['-DOKC_XS=1'] if xs else []) + (['-DOKC_TLX=1'] if tlx else [])
tail = ''.join(", '%s'" % f for f in flags)
ident = 'moe-tiny-%s-01' % ('-'.join(['half'] * half + ['xs'] * xs + ['tlx'] * tlx) or 'none')
for old, new in (("EXPERT_MOE_IDENTITIES = ('moe-control-release', ", "EXPERT_MOE_IDENTITIES = ('%s', " % ident),
                 ("'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1')", "'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1'%s)" % tail),
                 ("'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0')", "'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0'%s)" % tail)):
    assert head.count(old) == 1, old
    head = head.replace(old, new)

if tlx:
    rep('''def prepare(weights):
    return _expert_moe_prepare(weights, False)
''', '''def prepare(weights):
    state = _expert_moe_prepare(weights, False)
    state['_tl_done'] = {}
    for tokens in (1, 8):
        try:
            _tl_run(state, tokens, *_tl_synthetic(tokens, state['device']), 'prep')
        except Exception as exc:  # the timeline is a diagnostic: never fail prepare for it
            print('MOE_TL prep T=%d error %r' % (tokens, exc), flush=True)
    return state


TL_SLOTS = 64
TL_UNIT0 = 18


def _tl_synthetic(tokens, device):
    """Judge-like routing: skewed, this GPU the busiest. T=1 hits 2 local experts; T=8 hits 7 local experts over 20 routes, distinct per token."""
    g = torch.Generator(device='cpu').manual_seed(20261102 + tokens)
    x = (torch.randn(tokens, 4096, generator=g) * 0.5).to(torch.bfloat16).to(device)
    ids = torch.empty(tokens, 8, dtype=torch.int32)
    if tokens == 1:
        ids[0] = torch.tensor([3, 17, 40, 77, 101, 150, 200, 240], dtype=torch.int32)
    else:
        local = ([1, 4, 9], [1, 13], [1, 9, 20], [4, 13], [1, 25, 30], [13, 9], [1, 13, 25], [30, 4])
        rows = []
        for t in range(tokens):
            mine = local[t]
            rest = [int(v) for v in (32 + torch.randperm(224, generator=g)[:8 - len(mine)])]
            rows.append(sorted(mine + rest))
        ids = torch.tensor(rows, dtype=torch.int32)
    w = torch.softmax(torch.rand(tokens, 8, generator=g), 1).float()
    return x.contiguous(), ids.to(device).contiguous(), w.to(device).contiguous()


def _tl_run(state, tokens, x, ids, w, label, ref=None):
    """One extra launch of the same build with the timeline buffer as scratch[3]; prints a compact summary."""
    device = state['device']
    tl = torch.zeros(136 * TL_SLOTS, dtype=torch.int64, device=device)
    out = torch.empty(tokens, 4096, dtype=torch.bfloat16, device=device)
    ext = _fork_native_moe(tokens, state)
    torch.cuda.synchronize(device)
    ext.okc_run(state['ma_w'], state['ma_scratch'] + [tl], x, ids, w, out, 0, 32, state['nsm'])
    torch.cuda.synchronize(device)
    same = ''
    if ref is not None:
        same = ' same=%d maxdiff=%.3g' % (int(torch.equal(out, ref)), float((out.float() - ref.float()).abs().max()))
    _tl_summary(tl[: state['nsm'] * TL_SLOTS].view(state['nsm'], TL_SLOTS).cpu(), tokens, label + same)


def _tl_summary(s, tokens, label):
    import math
    n = s.shape[0]
    t0 = int(s[:, 0].min())
    us = lambda v: (v - t0) / 1000.0 if v > 0 else float('nan')
    cnt = s[:, 17]
    nA = (cnt & 0xFFFF).tolist()
    nB = ((cnt >> 16) & 0xFFFF).tolist()
    pre = int((cnt[0] >> 32) & 0xFF)
    njS = int((cnt[0] >> 40) & 0xFF)
    ndep = ((cnt >> 48) & 0xFFFF).tolist()
    col = lambda k: [us(int(v)) for v in s[:, k]]
    dur = lambda k: [int(v) / 1000.0 for v in s[:, k]]
    def st(vals):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return 'n/a'
        vals.sort()
        return '%.1f/%.1f/%.1f' % (vals[0], sum(vals) / len(vals), vals[-1])
    entry, route, first, aend = col(0), col(1), col(2), col(3)
    dep0, dep1, bend, credit, hdone, barr, brel, fin = col(4), col(5), col(6), col(7), col(8), col(9), col(10), col(11)
    credit0 = col(53)
    dts, dtsb = [], []
    for c in range(n):
        prev = None
        for q in range(26):
            v = int(s[c, TL_UNIT0 + q])
            if v == 0:
                break
            if prev is not None and q < nA[c]:
                dts.append((v - prev) / 1000.0)
            elif prev is not None and q > nA[c]:
                dtsb.append((v - prev) / 1000.0)
            prev = v
    dts.sort()
    rez = min((d for d in dts if d > 0), default=float('nan'))
    lines = []
    lines.append('MOE_TL %s T=%d grid=%d njS=%d pre=%d nA=%d..%d nB=%d..%d depwaits=%d..%d  (us min/mean/max over CTAs, from the first entry)'
                 % (label, tokens, n, njS, pre, min(nA), max(nA), min(nB), max(nB), min(ndep), max(ndep)))
    lines.append(' entry %s | route+ranges %s | first slot full %s' % (st(entry), st(route), st(first)))
    lines.append(' A end %s | A unit %s us (n=%d, min dt %.3f) | A wwait %s | A bufwait %s'
                 % (st(aend), st(dts) if dts else 'n/a', len(dts), rez, st(dur(12)), st(dur(14))))
    lines.append(' credit0 %s | credit %s | B0 switch start %s | end %s | B switch sum %s'
                 % (st(credit0), st(credit), st(dep0), st(dep1), st(dur(16))))
    lines.append(' B end %s | B unit %s us (n=%d) | B wwait %s | B bufwait %s | helper done %s'
                 % (st(bend), st(dtsb) if dtsb else 'n/a', len(dtsb), st(dur(13)), st(dur(15)), st(hdone)))
    sw = []
    for k in (1, 2, 3):
        a, b = col(44 + 2 * k), col(45 + 2 * k)
        sw.append('chunk%d %s' % (k, st([y - x for x, y in zip(a, b)])))
    lines.append(' switch cost (us) ' + ' | '.join(sw))
    bidle = [b - a for a, b in zip(barr, brel)]
    lines.append(' bar arrive %s | bar release %s | bar idle %s | final %s | exit %s'
                 % (st(barr), st(brel), st(bidle), st([b - a for a, b in zip(brel, fin)]), st(fin)))
    m = lambda vals: sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v)))
    lines.append(' mean CTA: ramp %.1f | A %.1f | A-end->B0 x ready %.1f (B switches %.1f) | B %.1f | B-end->bar %.1f | bar %.1f | final %.1f'
                 % (m(first), m(aend) - m(first), m(dep1) - m(aend), m(dur(16)), m(bend) - m(dep1), m(barr) - m(bend),
                    m(bidle), m(fin) - m(brel)))
    print('\\n'.join(lines), flush=True)
''')
    rep('''    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    return output
''', '''    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    if tokens in (1, 8) and not state.get('_tl_done', {}).get(tokens) and not torch.cuda.is_current_stream_capturing():
        state.setdefault('_tl_done', {})[tokens] = True
        try:
            _tl_run(state, tokens, data['hidden_states'], data['topk_ids'], data['topk_weights'], 'judge', output)
        except Exception as exc:
            print('MOE_TL judge T=%d error %r' % (tokens, exc), flush=True)
    return output
''')

dst.write_text(head + src)
print('wrote', dst, len(head + src), 'bytes', 'half', half, 'xs', xs, 'tlx', tlx, 'identity', ident)
