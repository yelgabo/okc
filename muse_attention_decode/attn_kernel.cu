#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int GROUP = HQ / HKV;
constexpr int QKV_STRIDE = (2 * HQ + 2 * HKV) * D;
constexpr int BN = 64;
constexpr int STAGES = 4;
constexpr int ROW_BYTES = HKV * D;
constexpr int STAGE_BYTES = 2 * BN * ROW_BYTES;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;
// One partial per row: m, l, then 64 unnormalized output dims.
constexpr int PART = 2 + D;
constexpr float EPS = 1e-5f;
constexpr float NEG_INF = -INFINITY;

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
    __nv_bfloat16* out;
    float* ws;
    int pt_stride;
    int page_size;
    int window;
    int rope;
    int splits;
    float qk_scale;
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

__device__ __forceinline__ uint32_t f16x2_from_fp8x2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Shared-memory layout of one stage: BN key rows then BN value rows, 128 bytes each (both KV
// heads). 16-byte chunks are XOR-swizzled by token so the fragment reads below hit distinct
// banks: key reads take 16 bytes from two tokens per quarter warp, value reads 8 bytes from
// four tokens per half warp.
__device__ __forceinline__ int k_chunk(int chunk, int tok) {
    return chunk ^ ((tok & 1) << 2);
}

__device__ __forceinline__ int v_chunk(int chunk, int tok) {
    return chunk ^ (((tok >> 1) & 3) << 1);
}

__device__ __forceinline__ size_t cache_row(const int* pt, int pos, int page_size) {
    if (page_size == 1) {
        return static_cast<size_t>(pt[pos]);
    }
    return static_cast<size_t>(pt[pos / page_size]) * page_size + pos % page_size;
}

// Rows of a CTA: r = j * HQ + qh for block token j and query head qh. A warp owns MT m16
// tiles of one KV head; tile row g < 8 is query head h * 8 + g. With L = 1 tile rows 8..15
// repeat rows 0..7 and are dropped; with L = 8 tile t of the head holds tokens 2t (rows
// 0..7) and 2t + 1 (rows 8..15).
template <int L, int MT, int TP>
struct Cfg {
    static constexpr int TILES_PER_HEAD = L == 1 ? 1 : GROUP * L / 16;
    static constexpr int MG = TILES_PER_HEAD / MT;
    static constexpr int WARPS = HKV * MG * TP;
    static constexpr int THREADS = WARPS * 32;
    static constexpr int ROWS = HQ * L;
    static constexpr int TOK_PER_THREAD = BN * 16 / THREADS;
    static constexpr int TOK_STEP = THREADS / 16;
    static constexpr int CHUNKS = BN / 16;
};

template <int L, int MT, int TP, bool CLUSTER>
__global__ void __launch_bounds__(Cfg<L, MT, TP>::THREADS)
attn_kernel(const Params p) {
    using C = Cfg<L, MT, TP>;
    extern __shared__ __align__(128) uint8_t smem[];
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int head = warp % HKV;
    const int mg = (warp / HKV) % C::MG;
    const int tp = warp / (HKV * C::MG);

    const int seq = p.seq_lens[b];
    const int start = p.window ? max(0, seq - p.window + 1) : 0;
    int per = (seq - start + p.splits - 1) / p.splits;
    per = (per + BN - 1) / BN * BN;
    const int lo = min(seq, start + split * per);
    const int hi = min(seq, lo + per);
    const int ntiles = (hi - lo + BN - 1) / BN;
    const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;

    // Loader role: chunk lc (0..7 key, 8..15 value) of tokens lt + i * TOK_STEP.
    const int lc = tid & 15;
    const int lt = tid >> 4;
    const uint8_t* lsrc = lc < 8 ? p.k_cache : p.v_cache;
    const int lchunk = lc & 7;

    auto issue = [&](int tile, const size_t* rows) {
        uint8_t* st = smem + (tile % STAGES) * STAGE_BYTES + (lc < 8 ? 0 : BN * ROW_BYTES);
        const int base = lo + tile * BN;
#pragma unroll
        for (int i = 0; i < C::TOK_PER_THREAD; ++i) {
            const int t = lt + i * C::TOK_STEP;
            const bool valid = base + t < hi;
            const int sw = lc < 8 ? k_chunk(lchunk, t) : v_chunk(lchunk, t);
            const uint8_t* src = valid ? lsrc + rows[i] * ROW_BYTES + lchunk * 16 : lsrc;
            cp_async16(smem_u32(st + t * ROW_BYTES + sw * 16), src, valid);
        }
    };
    auto fetch_rows = [&](int tile, size_t* rows) {
        const int base = lo + tile * BN;
#pragma unroll
        for (int i = 0; i < C::TOK_PER_THREAD; ++i) {
            const int pos = base + lt + i * C::TOK_STEP;
            rows[i] = (tile < ntiles && pos < hi) ? cache_row(pt, pos, p.page_size) : 0;
        }
    };

    size_t rows_a[C::TOK_PER_THREAD];
    size_t rows_b[C::TOK_PER_THREAD];
    size_t rows_c[C::TOK_PER_THREAD];
    static_assert(STAGES == 4, "prologue below issues three tiles");
    fetch_rows(0, rows_a);
    fetch_rows(1, rows_b);
    fetch_rows(2, rows_c);
    if (0 < ntiles) {
        issue(0, rows_a);
    }
    cp_commit();
    if (1 < ntiles) {
        issue(1, rows_b);
    }
    cp_commit();
    if (2 < ntiles) {
        issue(2, rows_c);
    }
    cp_commit();
    fetch_rows(3, rows_a);

    // Query fragments: rows g and g + 8 of each tile, dims 16c..16c+15, normalized, RoPE'd and
    // rounded to E4M3 as FA3's path does. Key dims are read in the same permuted order, so the
    // dot products are unchanged.
    uint4 qf[MT][2];
    int qpos[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int tile = mg * MT + mt;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int j = L == 1 ? 0 : 2 * tile + half;
            const int qh = head * GROUP + g;
            const int pos = seq + j;
            qpos[mt][half] = pos;
            const uint4* src = reinterpret_cast<const uint4*>(
                p.qkv + static_cast<size_t>(b * L + j) * QKV_STRIDE + qh * D + 16 * c);
            float x[16];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const uint4 w = src[v];
                const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv[e]));
                    x[v * 8 + 2 * e] = f.x;
                    x[v * 8 + 2 * e + 1] = f.y;
                }
            }
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                ss += x[e] * x[e];
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            const float r = rsqrtf(ss / D + EPS);
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                x[e] *= r;
            }
            if (p.rope) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float sn, cs;
                    sincosf(static_cast<float>(pos) * p.inv_freq[8 * c + i], &sn, &cs);
                    const float a = x[2 * i];
                    const float bb = x[2 * i + 1];
                    x[2 * i] = a * cs - bb * sn;
                    x[2 * i + 1] = a * sn + bb * cs;
                }
            }
            qf[mt][half] = make_uint4(fp8x4(x[0], x[1], x[2], x[3]), fp8x4(x[4], x[5], x[6], x[7]),
                                      fp8x4(x[8], x[9], x[10], x[11]),
                                      fp8x4(x[12], x[13], x[14], x[15]));
        }
    }
    const float qk_scale = p.qk_scale * *p.k_scale;

    float o[MT][8][4];
    float m[MT][2];
    float l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                o[mt][e][k] = 0.f;
            }
        }
        m[mt][0] = m[mt][1] = NEG_INF;
        l[mt][0] = l[mt][1] = 0.f;
    }

    // One 16-token chunk at smem stage st; tokens sit at positions pos0 + 0..15 and are visible
    // to row (mt, half) when lo_row <= pos <= lim (lim also caps at the row's own position).
    auto chunk = [&](const uint8_t* st, int ch, int pos0, int lim_all) {
        const uint8_t* ks = st;
        const uint8_t* vs = st + BN * ROW_BYTES;
        float s[MT][2][4];
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int t = ch * 16 + sub * 8 + g;
            const uint4 kf = *reinterpret_cast<const uint4*>(
                ks + t * ROW_BYTES + k_chunk(head * 4 + c, t) * 16);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float* acc = s[mt][sub];
                acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
                mma_fp8(acc, qf[mt][0].x, qf[mt][1].x, qf[mt][0].y, qf[mt][1].y, kf.x, kf.y);
                mma_fp8(acc, qf[mt][0].z, qf[mt][1].z, qf[mt][0].w, qf[mt][1].w, kf.z, kf.w);
            }
        }
        uint32_t pa[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int qp = qpos[mt][half];
                const int lo_row = p.window ? qp - p.window + 1 : 0;
                const int lim = min(lim_all, qp);
                float mx = NEG_INF;
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int pos = pos0 + ch * 16 + sub * 8 + 2 * c + k;
                        float v = s[mt][sub][2 * half + k] * qk_scale;
                        v = (pos >= lo_row && pos <= lim) ? v : NEG_INF;
                        s[mt][sub][2 * half + k] = v;
                        mx = fmaxf(mx, v);
                    }
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mnew = fmaxf(m[mt][half], mx);
                const float muse = mnew == NEG_INF ? 0.f : mnew;
                const float alpha = exp2f(m[mt][half] - muse);
                m[mt][half] = mnew;
                float sum = 0.f;
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const float e = exp2f(s[mt][sub][2 * half + k] - muse);
                        s[mt][sub][2 * half + k] = e;
                        sum += e;
                    }
                }
                l[mt][half] = l[mt][half] * alpha + sum;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o[mt][e][2 * half] *= alpha;
                    o[mt][e][2 * half + 1] *= alpha;
                }
            }
            pa[mt][0] = pack_f16(s[mt][0][0], s[mt][0][1]);
            pa[mt][1] = pack_f16(s[mt][0][2], s[mt][0][3]);
            pa[mt][2] = pack_f16(s[mt][1][0], s[mt][1][1]);
            pa[mt][3] = pack_f16(s[mt][1][2], s[mt][1][3]);
        }
        // Value fragments: tokens 2c, 2c+1, 2c+8, 2c+9 of the chunk, dims 8g..8g+7 of the head.
        // Output column g of n-tile e is head dim 8g + e.
        uint2 vt[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int t = ch * 16 + 2 * c + (k & 1) + (k >> 1) * 8;
            const int slot = head * 8 + g;
            vt[k] = *reinterpret_cast<const uint2*>(
                vs + t * ROW_BYTES + v_chunk(slot >> 1, t) * 16 + (slot & 1) * 8);
        }
#pragma unroll
        for (int hw = 0; hw < 2; ++hw) {
            const uint32_t x0 = hw ? vt[0].y : vt[0].x;
            const uint32_t x1 = hw ? vt[1].y : vt[1].x;
            const uint32_t x2 = hw ? vt[2].y : vt[2].x;
            const uint32_t x3 = hw ? vt[3].y : vt[3].x;
#pragma unroll
            for (int pr = 0; pr < 2; ++pr) {
                const uint32_t sel = pr ? 0x7362u : 0x5140u;
                const uint32_t w01 = __byte_perm(x0, x1, sel);
                const uint32_t w23 = __byte_perm(x2, x3, sel);
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int e = hw * 4 + pr * 2 + q;
                    const uint32_t b0 = f16x2_from_fp8x2(q ? w01 >> 16 : w01 & 0xffffu);
                    const uint32_t b1 = f16x2_from_fp8x2(q ? w23 >> 16 : w23 & 0xffffu);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_f16(o[mt][e], pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], b0, b1);
                    }
                }
            }
        }
    };

    for (int it = 0; it < ntiles; ++it) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < ntiles) {
            issue(it + STAGES - 1, rows_a);
        }
        cp_commit();
        fetch_rows(it + STAGES, rows_a);
        const uint8_t* st = smem + (it % STAGES) * STAGE_BYTES;
        const int pos0 = lo + it * BN;
#pragma unroll
        for (int ch = tp; ch < C::CHUNKS; ch += TP) {
            if (pos0 + ch * 16 < hi) {
                chunk(st, ch, pos0, hi - 1);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();

    // The last split also attends the block's own new keys and values, which it normalizes,
    // rotates, rounds to E4M3, writes to the cache and stages in smem.
    if (split == p.splits - 1) {
        uint8_t* st = smem;
        for (int i = tid; i < STAGE_BYTES / 16; i += C::THREADS) {
            reinterpret_cast<uint4*>(st)[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        const float inv_k = 1.f / *p.k_scale;
        const float inv_v = 1.f / *p.v_scale;
        for (int task = warp; task < L * HKV * 2; task += C::WARPS) {
            const int j = task / (HKV * 2);
            const int kh = (task / 2) % HKV;
            const bool is_k = (task & 1) == 0;
            const size_t tok = static_cast<size_t>(b) * L + j;
            const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
                p.qkv + tok * QKV_STRIDE + (HQ + (is_k ? 0 : HKV) + kh) * D);
            float2 x = __bfloat1622float2(src[lane]);
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
            const int byte = kh * D + 2 * lane;
            uint8_t* dst = is_k ? p.k_cache : p.v_cache;
            *reinterpret_cast<uint16_t*>(dst + p.cache_loc[tok] * ROW_BYTES + byte) = v8;
            const int sw = is_k ? k_chunk(byte >> 4, j) : v_chunk(byte >> 4, j);
            *reinterpret_cast<uint16_t*>(st + (is_k ? 0 : BN * ROW_BYTES) + j * ROW_BYTES
                                         + sw * 16 + (byte & 15)) = v8;
        }
        __syncthreads();
        if (tp == 0) {
            chunk(st, 0, seq, seq + L - 1);
        }
        __syncthreads();
    }

    // Partials: each warp's rows, m and l reduced over its quad, into smem [TP][ROWS][PART].
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int tile = mg * MT + mt;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (L == 1 && half == 1) {
                continue;
            }
            const int j = L == 1 ? 0 : 2 * tile + half;
            const int row = j * HQ + head * GROUP + g;
            float* dst = part + (tp * C::ROWS + row) * PART;
            float lsum = l[mt][half];
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
            if (c == 0) {
                dst[0] = m[mt][half];
                dst[1] = lsum;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                dst[2 + 8 * (2 * c) + e] = o[mt][e][2 * half];
                dst[2 + 8 * (2 * c + 1) + e] = o[mt][e][2 * half + 1];
            }
        }
    }

    if constexpr (!CLUSTER) {
        __syncthreads();
        float* ws = p.ws + (static_cast<size_t>(b) * p.splits + split) * TP * C::ROWS * PART;
        for (int i = tid; i < TP * C::ROWS * PART; i += C::THREADS) {
            ws[i] = part[i];
        }
        return;
    } else {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        const float v_scale = *p.v_scale;
        const int nwarps = p.splits * C::WARPS;
        for (int row = split * C::WARPS + warp; row < C::ROWS; row += nwarps) {
            float mmax = NEG_INF;
            for (int s = 0; s < p.splits; ++s) {
                const float* rp = cluster.map_shared_rank(part, s);
                for (int t = 0; t < TP; ++t) {
                    mmax = fmaxf(mmax, rp[(t * C::ROWS + row) * PART]);
                }
            }
            float lsum = 0.f;
            float2 acc = make_float2(0.f, 0.f);
            for (int s = 0; s < p.splits; ++s) {
                const float* rp = cluster.map_shared_rank(part, s);
                for (int t = 0; t < TP; ++t) {
                    const float* q = rp + (t * C::ROWS + row) * PART;
                    const float w = q[0] == NEG_INF ? 0.f : exp2f(q[0] - mmax);
                    lsum += w * q[1];
                    acc.x += w * q[2 + 2 * lane];
                    acc.y += w * q[3 + 2 * lane];
                }
            }
            const float inv = v_scale / lsum;
            acc.x *= inv;
            acc.y *= inv;
            const int j = row / HQ;
            const int qh = row % HQ;
            const size_t tok = static_cast<size_t>(b) * L + j;
            const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);
            const float2 gt = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                p.qkv + tok * QKV_STRIDE + (HQ + 2 * HKV + qh) * D)[lane]);
            acc.x = acc.x * r / (1.f + __expf(-gt.x));
            acc.y = acc.y * r / (1.f + __expf(-gt.y));
            reinterpret_cast<__nv_bfloat162*>(p.out + tok * HQ * D + qh * D)[lane] =
                __float22bfloat162_rn(acc);
        }
        cluster.sync();
    }
}

// Merges the partials in ws and applies the norm-gate: one warp per (token, query head).
__global__ void combine_kernel(const float* __restrict__ ws, const __nv_bfloat16* __restrict__ qkv,
                               const float* __restrict__ v_scale, __nv_bfloat16* __restrict__ out,
                               int L, int nparts) {
    const int tok = blockIdx.x;
    const int b = tok / L;
    const int j = tok % L;
    const int qh = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int rows = HQ * L;
    const int row = j * HQ + qh;
    const float* base = ws + static_cast<size_t>(b) * nparts * rows * PART + row * PART;
    float mmax = NEG_INF;
    for (int s = 0; s < nparts; ++s) {
        mmax = fmaxf(mmax, base[static_cast<size_t>(s) * rows * PART]);
    }
    float lsum = 0.f;
    float2 acc = make_float2(0.f, 0.f);
    for (int s = 0; s < nparts; ++s) {
        const float* q = base + static_cast<size_t>(s) * rows * PART;
        const float w = q[0] == NEG_INF ? 0.f : exp2f(q[0] - mmax);
        lsum += w * q[1];
        acc.x += w * q[2 + 2 * lane];
        acc.y += w * q[3 + 2 * lane];
    }
    const float inv = *v_scale / lsum;
    acc.x *= inv;
    acc.y *= inv;
    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);
    const float2 gt = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
        qkv + static_cast<size_t>(tok) * QKV_STRIDE + (HQ + 2 * HKV + qh) * D)[lane]);
    acc.x = acc.x * r / (1.f + __expf(-gt.x));
    acc.y = acc.y * r / (1.f + __expf(-gt.y));
    reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(tok) * HQ * D + qh * D)[lane] =
        __float22bfloat162_rn(acc);
}

template <int L, int MT, int TP, bool CLUSTER>
void launch(const Params& p, int batch, cudaStream_t stream) {
    using C = Cfg<L, MT, TP>;
    auto kern = attn_kernel<L, MT, TP, CLUSTER>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        if (CLUSTER) {
            cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        }
        configured = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.splits, batch);
    cfg.blockDim = dim3(C::THREADS);
    cfg.dynamicSmemBytes = SMEM_BYTES;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    cfg.numAttrs = 0;
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = p.splits;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, p);
    TORCH_CHECK(err == cudaSuccess, "attn_kernel launch: ", cudaGetErrorString(err));
}

// INSTANTIATE: template void launch<1, 1, 4, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<1, 1, 4, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, false>(const Params&, int, cudaStream_t);

}  // namespace

// variant: 0 = (L1: TP 4 | L8: MT 2 TP 1), 1 = L8 MT 1 TP 1
torch::Tensor attn_fused(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                         torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                         torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                         int64_t batch, int64_t q_len, int64_t page_size, int64_t window,
                         bool rope, double qk_scale, int64_t splits, bool cluster,
                         int64_t variant) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch, q_len, HQ * D}, qkv.options());
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
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.ws = nullptr;
    p.pt_stride = page_table.size(1);
    p.page_size = page_size;
    p.window = window;
    p.rope = rope ? 1 : 0;
    p.splits = splits;
    p.qk_scale = static_cast<float>(qk_scale);
    const int tp = q_len == 1 ? 4 : 1;
    torch::Tensor ws;
    if (!cluster) {
        ws = torch::empty({batch * splits * tp * HQ * q_len * PART},
                          qkv.options().dtype(torch::kFloat32));
        p.ws = ws.data_ptr<float>();
    }
    if (q_len == 1) {
        if (cluster) {
            launch<1, 1, 4, true>(p, batch, stream);
        } else {
            launch<1, 1, 4, false>(p, batch, stream);
        }
    } else if (cluster) {
        launch<8, 2, 1, true>(p, batch, stream);
    } else if (variant == 1) {
        launch<8, 1, 1, false>(p, batch, stream);
    } else {
        launch<8, 2, 1, false>(p, batch, stream);
    }
    if (!cluster) {
        combine_kernel<<<batch * q_len, HQ * 32, 0, stream>>>(
            p.ws, p.qkv, p.v_scale, p.out, q_len, splits * tp);
    }
    return out;
}
