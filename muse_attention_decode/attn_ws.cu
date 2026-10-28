#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int GROUP = HQ / HKV;
constexpr int L = 8;
constexpr int ROWS = HQ * L;
constexpr int QKV_STRIDE = (2 * HQ + 2 * HKV) * D;
constexpr int BN = 64;
// One stage: keys of both heads in wgmma's K-major core-matrix order (head h at h * 4096), then
// values as [token][128 bytes] with 16-byte chunks XOR-swizzled by token.
constexpr int K_TILE = HKV * BN * D;
constexpr int STAGE = 2 * K_TILE;
constexpr int VH_HEAD = BN * D * 2;
constexpr int Q_HEAD = GROUP * L * D;
constexpr int PART = 2 + D;
constexpr int CONSUMERS = 256;
constexpr int THREADS = CONSUMERS + 32;
constexpr float EPS = 1e-5f;
constexpr float NEG_INF = -INFINITY;

template <int NS>
struct Smem {
    static constexpr int NEWT = NS * STAGE;
    static constexpr int VH = NEWT + STAGE;
    static constexpr int Q = VH + 2 * HKV * VH_HEAD;
    static constexpr int BAR = Q + HKV * Q_HEAD;
    static constexpr int BYTES = BAR + 2 * NS * 8;
};

struct Params {
    const __nv_bfloat16* qkv;
    const int* page_table;
    const int* seq_lens;
    const int64_t* cache_loc;
    const float* inv_freq;
    const float* k_scale;
    const float* v_scale;
    uint8_t* k_cache;
    uint8_t* v_cache;
    float* ws;
    unsigned long long* dbg;
    int pt_stride;
    int page_size;
    int window;
    int rope;
    int splits;
    float qk_scale;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_cp_async(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

// A wait that gives up after about a second and marks the CTA dead instead of hanging the run;
// a dead CTA skips every later wait, so a broken pipeline costs one second, not the 15-minute
// limit, and the dbg flag says which side starved.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, bool& dead,
                                          unsigned long long* dbg, int code) {
    if (dead) {
        return;
    }
    if (mbar_try(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    while (!mbar_try(bar, parity)) {
        if (clock64() - t0 > (1ll << 31)) {
            dead = true;
            if (dbg != nullptr) {
                atomicOr(dbg, static_cast<unsigned long long>(code));
            }
            return;
        }
    }
}

__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ uint64_t desc(const void* p, int lbo, int sbo) {
    const uint64_t addr = smem_u32(p);
    return ((addr & 0x3ffff) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16)
           | (static_cast<uint64_t>(sbo >> 4) << 32);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// Keeps the compiler from moving accumulator reads or writes across an async wgmma.
__device__ __forceinline__ void reg_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

#define WG_D32 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define WG_OUT32(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), \
    "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), \
    "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), \
    "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

// S (64 x 64 f32) = A (64 x 32 E4M3, smem) . B (64 tokens x 32 E4M3, smem, K-major).
template <int ACC>
__device__ __forceinline__ void wgmma_qk(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D32
                 ", %32, %33, p, 1, 1;\n}\n"
                 : WG_OUT32(d) : "l"(da), "l"(db), "n"(ACC));
}

// O (64 x 64 f32) += P (64 x 16 f16, registers) . V (16 tokens x 64 dims f16, smem, MN-major).
__device__ __forceinline__ void wgmma_pv(float (&d)[32], const uint32_t* a, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D32
                 ", {%32, %33, %34, %35}, %36, p, 1, 1, 1;\n}\n"
                 : WG_OUT32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t f16x2_from_fp8x2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// Byte offset of 16-byte chunk kc of row r in a K-major core-matrix tile with 4 chunks per row
// (64 bytes of E4M3): core (r / 8, kc) is 128 contiguous bytes, LBO 128, SBO 512.
__device__ __forceinline__ int kmaj(int r, int kc) {
    return ((r >> 3) * 4 + kc) * 128 + (r & 7) * 16;
}

__device__ __forceinline__ int vchunk(int t, int ch) {
    return t * 128 + ((ch ^ (t & 7)) << 4);
}

// One CTA per (split, request): warps 0-3 attend KV head 0, warps 4-7 KV head 1, each warpgroup
// covering 64 query rows (8 block tokens x 8 query heads, row 8j + g), and warp 8 loads K/V rows
// through an NS-stage ring guarded by mbarriers. No block-wide barrier inside the loop.
// MODE 0: the kernel. MODE 1: loads only (math warps wait for each stage and release it), to
// separate memory time from math. MODE 2: the kernel plus clock64 phase totals in dbg[1..].
template <int NS, int MODE>
__global__ void __launch_bounds__(THREADS, 1) attn_ws_kernel(const Params p) {
    constexpr bool STAMP = MODE == 2;
    long long acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    long long tclk = 0;
    auto lap = [&](int k) {
        if constexpr (STAMP) {
            const long long now = clock64();
            acc[k] += now - tclk;
            tclk = now;
        }
    };
    auto flush_stamps = [&](int base) {
        if constexpr (STAMP) {
            if ((threadIdx.x & 31) == 0) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    atomicAdd(p.dbg + 1 + base + k, static_cast<unsigned long long>(acc[k]));
                }
            }
        }
    };
    using S = Smem<NS>;
    extern __shared__ __align__(1024) uint8_t smem[];
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + S::BAR);
    const uint32_t full0 = smem_u32(bars);
    const uint32_t empty0 = smem_u32(bars + NS);

    const int seq = p.seq_lens[b];
    const int start = p.window ? max(0, seq - p.window + 1) : 0;
    int per = (seq - start + p.splits - 1) / p.splits;
    per = (per + BN - 1) / BN * BN;
    const int lo = min(seq, start + split * per);
    const int hi = min(seq, lo + per);
    const int ntiles = (hi - lo + BN - 1) / BN;
    const bool last_split = split == p.splits - 1;
    bool dead = false;

    if (tid == 0) {
        for (int s = 0; s < NS; ++s) {
            mbar_init(full0 + 8 * s, 32);
            mbar_init(empty0 + 8 * s, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == 8) {
        const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;
        const int psize = p.page_size;
        auto entry = [&](int tile, int t) {
            const int pos = lo + tile * BN + t;
            if (tile >= ntiles || pos >= hi) {
                return 0;
            }
            return __ldg(pt + (psize == 1 ? pos : pos / psize));
        };
        // Page entries run PF tiles ahead in registers, so a tile's cp.async never waits on
        // the dependent page-table load (job 418d8d492c074b88 lost 6x to that wait).
        constexpr int PF = 4;
        int pe[PF][2];
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            pe[u][0] = entry(u, lane);
            pe[u][1] = entry(u, lane + 32);
        }
        if constexpr (STAMP) {
            tclk = clock64();
        }
        const int rsub = lane >> 3;
        const int ck = lane & 7;
        for (int base = 0; base < ntiles; base += PF) {
#pragma unroll
            for (int u = 0; u < PF; ++u) {
                const int i = base + u;
                if (i >= ntiles) {
                    break;
                }
                const int s = i % NS;
                lap(2);
                if (i >= NS) {
                    mbar_wait(empty0 + 8 * s, ((i / NS) + 1) & 1, dead, p.dbg, 1);
                }
                lap(0);
                uint8_t* st = smem + s * STAGE;
                const int pos0 = lo + i * BN;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const int t = 4 * k + rsub;
                    const int e = __shfl_sync(0xffffffffu, k < 8 ? pe[u][0] : pe[u][1], t & 31);
                    const int pos = pos0 + t;
                    const bool valid = pos < hi;
                    size_t row = static_cast<size_t>(e);
                    if (psize != 1) {
                        row = row * psize + pos % psize;
                    }
                    const size_t off = row * (HKV * D) + ck * 16;
                    cp_async16(smem_u32(st + (ck >> 2) * (BN * D) + kmaj(t, ck & 3)),
                               valid ? p.k_cache + off : p.k_cache, valid);
                    cp_async16(smem_u32(st + K_TILE + vchunk(t, ck)),
                               valid ? p.v_cache + off : p.v_cache, valid);
                }
                mbar_arrive_cp_async(full0 + 8 * s);
                lap(1);
                pe[u][0] = entry(i + PF, lane);
                pe[u][1] = entry(i + PF, lane + 32);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        lap(3);
        flush_stamps(8);
        return;
    }

    if constexpr (STAMP) {
        tclk = clock64();
    }
    const int h = warp >> 2;
    const int wl = warp & 3;
    const int wtid = tid & 127;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int bar_id = 1 + h;
    const float k_scale = *p.k_scale;
    const float v_scale = *p.v_scale;
    uint8_t* qs = smem + S::Q + h * Q_HEAD;
    uint8_t* newt = smem + S::NEWT;

    // Query prep: two threads per row (row m = 8j + g'), 32 dims each, normalized, rotated and
    // rounded to E4M3 into the K-major A tile.
    {
        const int m = wtid >> 1;
        const int part = wtid & 1;
        const int j = m >> 3;
        const int qh = h * GROUP + (m & 7);
        const __nv_bfloat16* src =
            p.qkv + static_cast<size_t>(b * L + j) * QKV_STRIDE + qh * D + 32 * part;
        float x[32];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 w = reinterpret_cast<const uint4*>(src)[v];
            const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ws[e]));
                x[8 * v + 2 * e] = f.x;
                x[8 * v + 2 * e + 1] = f.y;
            }
        }
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            ss += x[e] * x[e];
        }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        const float r = rsqrtf(ss / D + EPS);
        const float pos = static_cast<float>(seq + j);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float a = x[2 * i] * r;
            float bb = x[2 * i + 1] * r;
            if (p.rope) {
                float sn, cs;
                sincosf(pos * p.inv_freq[16 * part + i], &sn, &cs);
                const float a2 = a * cs - bb * sn;
                bb = a * sn + bb * cs;
                a = a2;
            }
            x[2 * i] = a;
            x[2 * i + 1] = bb;
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t wv[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float* y = x + 16 * h2 + 4 * e;
                const uint32_t lo2 = __nv_cvt_float2_to_fp8x2(make_float2(y[0], y[1]),
                                                              __NV_SATFINITE, __NV_E4M3);
                const uint32_t hi2 = __nv_cvt_float2_to_fp8x2(make_float2(y[2], y[3]),
                                                              __NV_SATFINITE, __NV_E4M3);
                wv[e] = lo2 | (hi2 << 16);
            }
            *reinterpret_cast<uint4*>(qs + kmaj(m, 2 * part + h2)) =
                make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
    }

    // The last split also writes the block's new keys and values (KV head h) to the cache and
    // stages them in their own tile, attended after the loop. Each warpgroup zeroes and fills
    // only its head's bytes, so no cross-warpgroup barrier is needed.
    if (last_split) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = wtid + 128 * k;
            reinterpret_cast<uint4*>(newt + h * (BN * D))[i] = make_uint4(0, 0, 0, 0);
            const int t = i >> 2;
            *reinterpret_cast<uint4*>(newt + K_TILE + vchunk(t, h * 4 + (i & 3))) =
                make_uint4(0, 0, 0, 0);
        }
        named_sync(bar_id, 128);
        const float inv_k = 1.f / k_scale;
        const float inv_v = 1.f / v_scale;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int task = wl + 4 * k;
            const int j = task >> 1;
            const bool is_k = (task & 1) == 0;
            const size_t tok = static_cast<size_t>(b) * L + j;
            float2 x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                p.qkv + tok * QKV_STRIDE + (HQ + (is_k ? 0 : HKV) + h) * D)[lane]);
            float inv = inv_v;
            if (is_k) {
                const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
                x.x *= r;
                x.y *= r;
                if (p.rope) {
                    float sn, cs;
                    sincosf(static_cast<float>(seq + j) * p.inv_freq[lane], &sn, &cs);
                    x = make_float2(x.x * cs - x.y * sn, x.x * sn + x.y * cs);
                }
                inv = inv_k;
            }
            x.x *= inv;
            x.y *= inv;
            const uint16_t v8 = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
            uint8_t* dst = (is_k ? p.k_cache : p.v_cache) + p.cache_loc[tok] * (HKV * D);
            *reinterpret_cast<uint16_t*>(dst + h * D + 2 * lane) = v8;
            const int byte = 2 * lane;
            uint8_t* sp = is_k ? newt + h * (BN * D) + kmaj(j, byte >> 4) + (byte & 15)
                               : newt + K_TILE + vchunk(j, h * 4 + (byte >> 4)) + (byte & 15);
            *reinterpret_cast<uint16_t*>(sp) = v8;
        }
    }
    fence_async_smem();
    named_sync(bar_id, 128);

    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i] = 0.f;
    }
    float mrow[2] = {NEG_INF, NEG_INF};
    float lrow[2] = {0.f, 0.f};
    lap(7);
    const float qk_scale = p.qk_scale * k_scale;
    int qpos[2];
    int lo_row[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        qpos[hh] = seq + 2 * wl + hh;
        lo_row[hh] = p.window ? qpos[hh] - p.window + 1 : 0;
    }
    const uint64_t qd0 = desc(qs, 128, 512);
    const uint64_t qd1 = desc(qs + 256, 128, 512);

    // V conversion map: lane covers token 8kb + (lane & 7) and 8-dim group 2q + ((lane >> 3) &
    // 1), so a quarter warp's 16-byte f16 stores and a half warp's 8-byte E4M3 loads each hit
    // distinct banks.
    const int ctt = lane & 7;
    const int chalf = (lane >> 3) & 1;
    const int ckb = wl * 2 + (lane >> 4);
    const int ct = ckb * 8 + ctt;

    // One 64-token tile: S = Q.K^T (FP8 wgmma), the f16 values converted while it runs, online
    // softmax in registers, then O += P.V (f16 wgmma).
    auto tile = [&](const uint8_t* st, int it, int pos0, int lim, uint32_t empty_bar) {
        float s[32];
        wg_fence();
        const uint8_t* ks = st + h * (BN * D);
        wgmma_qk<0>(s, qd0, desc(ks, 128, 512));
        wgmma_qk<1>(s, qd1, desc(ks + 256, 128, 512));
        wg_commit();
        lap(1);
        // The value buffers alternate by tile: a warp's P.V wait does not formally cover the
        // other warps' share of the warpgroup's reads.
        uint8_t* vb = smem + S::VH + (h * 2 + (it & 1)) * VH_HEAD;
        const uint8_t* vs = st + K_TILE;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint2 w = *reinterpret_cast<const uint2*>(vs + vchunk(ct, h * 4 + q) + chalf * 8);
            const uint4 hv = make_uint4(f16x2_from_fp8x2(w.x & 0xffffu), f16x2_from_fp8x2(w.x >> 16),
                                        f16x2_from_fp8x2(w.y & 0xffffu), f16x2_from_fp8x2(w.y >> 16));
            *reinterpret_cast<uint4*>(vb + (ckb * 8 + 2 * q + chalf) * 128 + ctt * 16) = hv;
        }
        lap(2);
        fence_async_smem();
        named_sync(bar_id, 128);
        lap(3);
        wg_wait<0>();
        reg_fence(s);
        reg_fence(o);
        if (empty_bar != 0) {
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty_bar);
            }
        }
        const bool interior = pos0 + BN - 1 <= lim && (!p.window || pos0 >= seq + L - p.window);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int top = min(lim, qpos[hh]);
            float mx = NEG_INF;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    float v = s[4 * i + 2 * hh + k] * qk_scale;
                    if (!interior) {
                        const int pos = pos0 + 8 * i + 2 * c + k;
                        v = (pos >= lo_row[hh] && pos <= top) ? v : NEG_INF;
                    }
                    s[4 * i + 2 * hh + k] = v;
                    mx = fmaxf(mx, v);
                }
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(mrow[hh], mx);
            const float muse = mnew == NEG_INF ? 0.f : mnew;
            const float alpha = ex2(mrow[hh] - muse);
            mrow[hh] = mnew;
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float e = ex2(s[4 * i + 2 * hh + k] - muse);
                    s[4 * i + 2 * hh + k] = e;
                    sum += e;
                }
            }
            lrow[hh] = lrow[hh] * alpha + sum;
            if (__any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o[4 * i + 2 * hh] *= alpha;
                    o[4 * i + 2 * hh + 1] *= alpha;
                }
            }
        }
        lap(4);
        uint32_t pa[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            pa[k][0] = pack_f16(s[8 * k + 0], s[8 * k + 1]);
            pa[k][1] = pack_f16(s[8 * k + 2], s[8 * k + 3]);
            pa[k][2] = pack_f16(s[8 * k + 4], s[8 * k + 5]);
            pa[k][3] = pack_f16(s[8 * k + 6], s[8 * k + 7]);
        }
        reg_fence(o);
        wg_fence();
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            wgmma_pv(o, pa[k], desc(vb + k * 2048, 1024, 128));
        }
        wg_commit();
        // Leaving P.V in flight across the loop back-edge makes ptxas serialize every wgmma
        // in the kernel (C7514), so it retires here.
        wg_wait<0>();
        lap(5);
    };

    int it = 0;
    for (; it < ntiles; ++it) {
        const int s = it % NS;
        mbar_wait(full0 + 8 * s, (it / NS) & 1, dead, p.dbg, 2);
        lap(0);
        if constexpr (MODE == 1) {
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * s);
            }
            continue;
        }
        fence_async_smem();
        tile(smem + s * STAGE, it, lo + it * BN, hi - 1, empty0 + 8 * s);
    }
    if (MODE != 1 && last_split) {
        tile(newt, it, seq, seq + L - 1, 0);
    }
    wg_wait<0>();
    reg_fence(o);
    lap(6);

    float* ws = p.ws + (static_cast<size_t>(b) * p.splits + split) * ROWS * PART;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int j = 2 * wl + hh;
        const int row = j * HQ + h * GROUP + g;
        float* dst = ws + row * PART;
        float lsum = lrow[hh];
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
        if (c == 0) {
            dst[0] = mrow[hh];
            dst[1] = lsum;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            *reinterpret_cast<float2*>(dst + 2 + 8 * i + 2 * c) =
                make_float2(o[4 * i + 2 * hh], o[4 * i + 2 * hh + 1]);
        }
    }
    lap(6);
    flush_stamps(0);
}

__device__ __forceinline__ float2 load_gate(const __nv_bfloat16* qkv, size_t tok, int qh,
                                            int lane) {
    return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
        qkv + tok * QKV_STRIDE + (HQ + 2 * HKV + qh) * D)[lane]);
}

// Merges a row's split partials (lane k reads partial k first so the loads overlap) and applies
// the norm-gate: one warp per (token, query head).
__global__ void ws_combine_kernel(const float* __restrict__ ws,
                                  const __nv_bfloat16* __restrict__ qkv,
                                  const float* __restrict__ v_scale, __nv_bfloat16* __restrict__ out,
                                  int nparts) {
    const int tok = blockIdx.x;
    const int b = tok / L;
    const int j = tok % L;
    const int qh = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const size_t stride = static_cast<size_t>(ROWS) * PART;
    const float* base = ws + b * nparts * stride + (j * HQ + qh) * PART;
    const float2 gt = load_gate(qkv, tok, qh, lane);
    float mmax = NEG_INF;
    for (int s = lane; s < nparts; s += 32) {
        mmax = fmaxf(mmax, base[s * stride]);
    }
    mmax = warp_max(mmax);
    float lsum = 0.f;
    float2 acc = make_float2(0.f, 0.f);
    for (int s0 = 0; s0 < nparts; s0 += 32) {
        float w = 0.f;
        if (s0 + lane < nparts) {
            const float* q = base + (s0 + lane) * stride;
            w = q[0] == NEG_INF ? 0.f : exp2f(q[0] - mmax);
            lsum += w * q[1];
        }
        const int cnt = min(32, nparts - s0);
#pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const float wk = __shfl_sync(0xffffffffu, w, k);
            const float2 v = *reinterpret_cast<const float2*>(base + (s0 + k) * stride + 2 + 2 * lane);
            acc.x += wk * v.x;
            acc.y += wk * v.y;
        }
    }
    lsum = warp_sum(lsum);
    const float inv = *v_scale / lsum;
    acc.x *= inv;
    acc.y *= inv;
    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);
    acc.x = acc.x * r / (1.f + __expf(-gt.x));
    acc.y = acc.y * r / (1.f + __expf(-gt.y));
    reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(tok) * HQ * D + qh * D)[lane] =
        __float22bfloat162_rn(acc);
}

template <int NS, int MODE>
void launch_ws(const Params& p, int batch, cudaStream_t stream) {
    auto kern = attn_ws_kernel<NS, MODE>;
    constexpr int bytes = Smem<NS>::BYTES;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        configured = true;
    }
    kern<<<dim3(p.splits, batch), THREADS, bytes, stream>>>(p);
}

// INSTANTIATE: template void launch_ws<2, 0>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch_ws<4, 0>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch_ws<8, 0>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch_ws<8, 1>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch_ws<8, 2>(const Params&, int, cudaStream_t);

}  // namespace

// stages: 2, 4 or 8 (anything else runs 8); mode: see attn_ws_kernel (1 and 2 with 8 stages).
// dbg: int64 [17] or empty. dbg[0] bit 0: a loader wait timed out; bit 1: a math-warp wait
// timed out. With mode 2, dbg[1..8] sum math-warp cycles per phase (wait for data, Q.K issue,
// V convert, warpgroup barrier, softmax, P.V issue, tail, prologue) and dbg[9..12] loader
// cycles (wait for a free stage, cp.async issue, page-entry loads, drain), over all warps.
torch::Tensor attn_ws(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                      torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                      torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                      int64_t batch, int64_t page_size, int64_t window, bool rope,
                      double qk_scale, int64_t splits, int64_t stages, int64_t mode,
                      torch::Tensor dbg) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch, L, HQ * D}, qkv.options());
    auto ws = torch::empty({batch * splits * ROWS * PART}, qkv.options().dtype(torch::kFloat32));
    Params p;
    p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
    p.page_table = page_table.data_ptr<int>();
    p.seq_lens = seq_lens.data_ptr<int>();
    p.cache_loc = cache_loc.data_ptr<int64_t>();
    p.inv_freq = inv_freq.data_ptr<float>();
    p.k_scale = k_scale.data_ptr<float>();
    p.v_scale = v_scale.data_ptr<float>();
    p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
    p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
    p.ws = ws.data_ptr<float>();
    p.dbg = dbg.numel() ? reinterpret_cast<unsigned long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    p.pt_stride = page_table.size(1);
    p.page_size = page_size;
    p.window = window;
    p.rope = rope ? 1 : 0;
    p.splits = splits;
    p.qk_scale = static_cast<float>(qk_scale);
    if (mode == 1) {
        launch_ws<8, 1>(p, batch, stream);
    } else if (mode == 2) {
        launch_ws<8, 2>(p, batch, stream);
    } else if (stages == 2) {
        launch_ws<2, 0>(p, batch, stream);
    } else if (stages == 4) {
        launch_ws<4, 0>(p, batch, stream);
    } else {
        launch_ws<8, 0>(p, batch, stream);
    }
    ws_combine_kernel<<<batch * L, HQ * 32, 0, stream>>>(p.ws, p.qkv, p.v_scale,
                                                          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                          splits);
    return out;
}
