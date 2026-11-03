constexpr int UNIT_CODES = 32768, UNIT_BYTES = 36864;
constexpr int OWN_XP = 1032;
static_assert(NW == 4 && NTL == 160 && OKC_NS == 3 && OKC_SPLIT && OKC_WS && !OKC_WGS,
              "fixed full-K ownership selection");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile("{ .reg .pred p; OWN_WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra OWN_WAIT_%=; }"
               :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ float* ss_cur() {
  return reinterpret_cast<Smem*>(smem_raw)->ssp;
}
__device__ void early_issue(const Params&, Smem&) {}

__device__ void owned_producer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();
  if ((TIDX & 31) == 0) {
    int seq = 0;
    for (int pb = 0; pb < 2; ++pb) {
      const int blocks = pb ? 64 : 128;
      const int begin = S.wsr[pb * 2], end = S.wsr[pb * 2 + 1];
      for (int v = begin; v < end; ++v) {
        const int j = v / blocks, rb = v % blocks;
        for (int st = 0; st < 4; ++st, ++seq) {
          const int slot = seq % 3;
          if (seq >= 3) mbar_wait(S.wse + slot, ((seq / 3) & 1) ^ 1);
          const unsigned char* src = (pb ? p.wBu : p.wAu)
              + (((size_t)S.jE[j] * blocks + rb) * 4 + st) * UNIT_BYTES;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                       :: "r"(smem_u32(S.bars + slot)), "r"(UNIT_BYTES) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";"
                       :: "r"(smem_u32(ring + slot * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES),
                          "r"(smem_u32(S.bars + slot)) L2H_ARG : "memory");
        }
      }
    }
  }
  __syncwarp();
  WS_JOIN();
}

template <bool PB>
__device__ void owned_phase(const Params& p, Smem& S, unsigned char* ring, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = tid >> 5;
  const int begin = S.wsr[PB ? 2 : 0], end = S.wsr[PB ? 3 : 1];
  constexpr int blocks = PB ? 64 : 128;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(ring + 3 * UNIT_BYTES);
  for (int v = begin; v < end; ++v) {
    const int j = v / blocks, rb = v % blocks;
    const int cnt = S.jC[j], start = S.jS[j];
    if (PB) {
      if (lane == 0) while (ld_acquire(p.doneA + j) < 512) __nanosleep(64);
      __syncwarp();
    }
    float acc[4] = {}, acc2[4] = {}, acc3[4] = {}, acc4[4] = {};
    for (int stage = 0; stage < 4; ++stage, ++seq) {
      for (int i = tid; i < 8 * 128; i += 128) {
        const int row = i / 128, col = (i % 128) * 8;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (row < cnt) {
          const __nv_bfloat16* src = PB ? p.inter + (size_t)(start + row) * HD
              : p.x + (size_t)(S.ent[start + row] >> 3) * HD;
          x = *reinterpret_cast<const uint4*>(src + stage * 1024 + col);
        }
        *reinterpret_cast<uint4*>(xs + row * OWN_XP + col) = x;
      }
      CTA_SYNC();
      const int slot = seq % 3;
      mbar_wait(S.bars + slot, (seq / 3) & 1);
      const unsigned char* weights = ring + slot * UNIT_BYTES;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4 sc = *reinterpret_cast<const uint4*>(weights + UNIT_CODES + warp * 1024 + half * 512 + lane * 16);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const uint4 w = *reinterpret_cast<const uint4*>(weights + warp * 8192 + half * 4096 + c * 512 + lane * 16);
#if OKC_PRMTSC
          const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
          const uint32_t sg = SCDK(swd, (c & 1) * 2, 16u + p.zero), sg8 = SCDK(swd, (c & 1) * 2 + 1, 16u + p.zero);
#else
          const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
          const uint32_t sg = S.lut[sw & 255], sg8 = S.lut[(sw >> 8) & 255];
#endif
          const __nv_bfloat16* xp = xs + (lane >> 2) * OWN_XP + half * 512 + c * 64 + (lane & 3) * 16;
          const uint4 x0 = *reinterpret_cast<const uint4*>(xp), x1 = *reinterpret_cast<const uint4*>(xp + 8);
          const uint32_t b[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
          const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            uint32_t r0, r1, r2, r3;
            dq4(ww[k], r0, r1, r2, r3);
            mma_bf16(k == 0 ? acc : k == 1 ? acc2 : k == 2 ? acc3 : acc4,
                     bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * k], b[2 * k + 1]);
          }
        }
      }
      // The proxy fence orders generic weight reads before the async producer can overwrite this slot.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      CTA_SYNC();
      if (tid == 0) mbar_arrive1(S.wse + slot);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    const int row_block = rb * 4 + warp;
    if (!PB) {
      epi_A(p, start, cnt, 2 * (lane & 3), row_block * 8 + (lane >> 2), p.gsA[S.jE[j]] * TWO62, acc);
    } else {
      float sq[2];
      epi_B(p, start, cnt, 2 * (lane & 3), row_block * 16 + (lane >> 2), p.gsB[S.jE[j]] * TWO62, acc, sq);
      ss_reduce_add(p, start, cnt, 2 * (lane & 3), sq, lane >> 2);
    }
  }
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0 && begin < end) {
      for (int j = begin / 128; j <= (end - 1) / 128; ++j) {
        const int count = min(end, (j + 1) * 128) - max(begin, j * 128);
        atomicAdd(p.doneA + j, 4 * count);
      }
    }
  }
}

template <bool BIG>
__global__ void __launch_bounds__(160, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  unsigned char* ring = smem_raw + ((sizeof(Smem) + 127) / 128) * 128;
  const int tid = TIDX;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid >= 128) { owned_producer(p, S, ring); return; }
  if (tid == 0) {
    S.gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
    S.ssp = (S.gen & 1) ? p.ss1 : p.ss;
    for (int i = 0; i < 3; ++i) { mbar_init(S.bars + i, 1); mbar_init(S.wse + i, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += 128) {
    const int sign = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) b = 0xffffu;
    else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * .125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sign ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  route(p, S);
  float* other = (S.gen & 1) ? p.ss : p.ss1;
  for (int i = blockIdx.x * 128 + tid; i < EMAX; i += gridDim.x * 128) other[i] = 0.f;
  if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  if (tid == 0) {
    const float rG = __fdividef(1.f, (float)gridDim.x);
    for (int pb = 0; pb < 2; ++pb) {
      const int n = S.njS * (pb ? 64 : 128);
      S.wsr[pb * 2] = muldiv_g(n, blockIdx.x, gridDim.x, rG);
      S.wsr[pb * 2 + 1] = muldiv_g(n, blockIdx.x + 1, gridDim.x, rG);
    }
  }
  WS_JOIN();
  int seq = 0;
  owned_phase<false>(p, S, ring, seq);
  owned_phase<true>(p, S, ring, seq);
  __threadfence();
  WS_JOIN();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
