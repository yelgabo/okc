"""M-C slice C-1 (archive/docs/plan-2026-10-31-m-c-t1-fusion.md, "C-1 revised before building" and
"C-1 implementation") on layer_m1rt51sc1.py: at one token the o projection and the attention
boundary run as one cooperative kernel with two grid barriers, in its own extension module.

What changes:
- _MC1_SRC / _MC1_CPP_SRC: the kernel (phase 1: the o rows in cgemv<64, 1>'s arithmetic, from
  the same 12-bit stream and escape table; barrier 1; phase 2: the boundary's residual add with
  its BF16 rounding of the normalized branch; barrier 2; the outputs), its wrapper, an
  occupancy query and a capture probe. Built on its own thread from _SKP_GEMV_SRC's device
  helpers (GEMV_DEVICE_ONLY) plus _MC1_SRC; no other extension's source changes.
- _mc1_* functions: setup (repacks o for cgemv, probes the cooperative launch under capture,
  sizes the grid from the occupancy), self-test against the two-kernel path, the fresh-input
  replay gate at 1x1 sliding and full, the stable pick (preferred "off"), and the enable step
  prepare calls after the M1 tuner.
- kernel_fn at one token runs _mc1 when state["mc1_on"]; SUMMARY prints the C-1 lines.

Static checks: the C-1 module's source has no griddepcontrol and its launch sets only
cudaLaunchAttributeCooperative; every source block that was in IN is unchanged in OUT.

    python3 tools/dl_mc_c1_patch.py IN.py OUT.py "HEADER LINE 2"
"""
import re
import sys

MC1_SRC = r'''
# M-C slice C-1 (archive/docs/plan-2026-10-31-m-c-t1-fusion.md, "C-1 implementation"): at one token the
# o projection and the attention boundary as one cooperative launch. Its own module, built from
# _SKP_GEMV_SRC's device helpers (cgemv's chunk arithmetic, unchanged) and this block.
_MC1_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <vector>

namespace {
namespace mc1 {

constexpr int H = 8192;
constexpr int KO = 1024;
constexpr int HALF = H / 2;
// cgemv<64, 1>'s CTA, two compute warps and the escape warp, is one unit here; a CTA holds U.
constexpr int NTH = 64;
constexpr int UT = NTH + 32;
constexpr int U = 8;
constexpr int NT = U * UT;
constexpr int GROUPS = H / 4;
// Phase 2 keeps a CTA's columns in its first warp, at most 32 float4 groups.
constexpr int GMIN = (GROUPS + 31) / 32;
constexpr int GMAX = 1024;
constexpr long long kSpinLimit = 1ll << 31;
constexpr int kGen = 0;
constexpr int kDiag = 1;
constexpr int kBar1 = 2;
constexpr int kBar2 = 4;
constexpr int kCtrInts = 8;
static_assert(16 * NTH == KO, "one 16-k chunk per compute thread, as cgemv<64, 1>");
static_assert(U + 1 <= 16, "a named barrier per unit, 0 stays __syncthreads");

struct Params {
    const uint8_t* w;
    const int2* table;
    const __nv_bfloat16* x;
    const float* stream;
    const float* alpha;
    const float* beta;
    const __nv_bfloat16* w_next;
    // Written by this launch and read back by other CTAs: no __restrict__, read with ld.cg.
    __nv_bfloat16* branch;
    float* part;
    int* ctr;
    float* stream_out;
    __nv_bfloat16* next;
    float* router;
    int pmax;
    unsigned base4;
    float eps_post;
    float eps_next;
};

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ bool spin_failed(long long t0, int* ctr, int code) {
    if (clock64() - t0 < kSpinLimit) {
        return false;
    }
    atomicOr(ctr + kDiag, code);
    return true;
}

__device__ __forceinline__ void bar_unit(int id) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(UT) : "memory");
}

__device__ __forceinline__ float bf16r(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// The fused MoE's barrier (_MOE1_SRC): the arrival counter of this launch's parity was zeroed
// by the previous launch, so graph replays need no host reset. Every CTA's stores before the
// first __syncthreads() are ordered before every CTA's loads after the second one; the PTX
// argument is in the plan's "C-1 implementation" section. A wait past kSpinLimit cycles marks
// ctr[kDiag] and gives up (the output is then wrong; the self-test and the gate read the flag).
__device__ __forceinline__ void grid_barrier(int* ctr, int slot, int G, bool bump, int code) {
    __syncthreads();
    if (threadIdx.x == 0) {
        int* bc = ctr + slot;
        __threadfence();
        const int old = atomicAdd(bc, 1);
        if (bump && old == G - 1) {
            // The arrival atomics are relaxed; this fence makes the bump an acquire of every
            // CTA's arrival, so a reader of the new generation is ordered after all of them
            // (independent review of C-1, 2026-10-31).
            __threadfence();
            atomicAdd(ctr + kGen, 1);
        }
        if (ld_acquire(bc) < G) {
            const long long t0 = clock64();
            while (ld_acquire(bc) < G) {
                __nanosleep(64);
                if (spin_failed(t0, ctr, code)) {
                    break;
                }
            }
        }
    }
    __syncthreads();
}

// cgemv_body's escape warp for row pair c, unchanged but for c and the output slot.
__device__ __forceinline__ void escape_pair(const Params& p, int c, int lane, float (*red)[3]) {
    const int2* tb = p.table + (size_t)c * p.pmax;
    int2 ent = __ldg(tb + min(lane, p.pmax - 1));
    const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);
    const unsigned base = p.base4 & 0xFFu;
    float corr0 = 0.f, corr1 = 0.f;
    for (int e = lane; e <= count; e += 32) {
        const int row = ent.x >> 16, kk = ent.x & 0xFFFF;
        const unsigned bits = (unsigned)ent.y;
        const bool live = e >= 1;
        ent = __ldg(tb + min(e + 32, p.pmax - 1));
        if (live) {
            const int off = min(max((int)((bits >> 8) & 0x7Fu) - (int)base, 0), EXP_WINDOW - 1);
            const unsigned dec = (bits & 0x80FFu) | ((base + off) << 8);
            const float delta = __fsub_rn(__uint_as_float(bits << 16), __uint_as_float(dec << 16));
            const float xv = __uint_as_float(
                (unsigned)__ldg(reinterpret_cast<const unsigned short*>(p.x) + kk) << 16);
            const float d = __fmul_rn(delta, xv);
            corr0 = __fadd_rn(corr0, row ? 0.f : d);
            corr1 = __fadd_rn(corr1, row ? d : 0.f);
        }
    }
    corr0 = warp_sum(corr0);
    corr1 = warp_sum(corr1);
    if (lane == 0) {
        red[0][NTH / 32] = corr0;
        red[1][NTH / 32] = corr1;
    }
}

// cgemv_body's final sum for row pair c (the same order: zero, compute warp 0, warp 1, the
// escapes), the BF16 store, and the post-attention norm's sum of squares of the stored values.
__device__ __forceinline__ float finish_pair(const float (*red)[3], __nv_bfloat16* branch, int c,
                                             float ssq) {
    float r0 = 0.f, r1 = 0.f;
#pragma unroll
    for (int w = 0; w <= NTH / 32; ++w) {
        r0 += red[0][w];
        r1 += red[1][w];
    }
    const unsigned v = pack_bf16x2(r0, r1);
    *reinterpret_cast<unsigned*>(branch + 2 * c) = v;
    const float f0 = lo_f(v);
    const float f1 = hi_f(v);
    ssq += f0 * f0;
    ssq += f1 * f1;
    return ssq;
}

__device__ __forceinline__ void ld_pair(const Params& p, int c, int t, uint4& l0, uint4& l1,
                                        uint2& q0, uint2& q1) {
    const uint8_t* lp = p.w + (size_t)c * 2 * KO;
    const uint8_t* cp = p.w + (size_t)HALF * 2 * KO + (size_t)c * KO;
    l0 = ld_stream(reinterpret_cast<const uint4*>(lp) + t);
    l1 = ld_stream(reinterpret_cast<const uint4*>(lp + KO) + t);
    q0 = __ldg(reinterpret_cast<const uint2*>(cp) + t);
    q1 = __ldg(reinterpret_cast<const uint2*>(cp + KO / 2) + t);
}

// Phase 1: unit u of CTA b takes row pairs b * U + u + i * G * U, one per round. Two pairs in
// flight per unit spilled at the 40 registers two CTAs per SM allow. Phase 2: CTA b takes
// float4 groups [b * GROUPS / G, (b + 1) * GROUPS / G) of the 8192 columns in its first warp
// and computes for them what
// attn_boundary_kernel computes per element (the same expressions), with the two norms' sums
// taken over every CTA's partial in index order. No PDL wait or trigger: the launch is plain
// and cooperative, so it starts after its predecessor completes, and nothing launched after
// it can start before every CTA has exited.
__global__ void __launch_bounds__(NT, 2) mc1_kernel(const Params p) {
    __shared__ float red[U][2][NTH / 32 + 1];
    __shared__ float uss[U];
    __shared__ __align__(16) float s_stream[32][4];
    __shared__ __align__(16) float s_alpha[32][4];
    __shared__ __align__(16) float s_beta[32][4];
    __shared__ __align__(8) __nv_bfloat16 s_wn[32][4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int u = tid / UT;
    const int ut = tid - u * UT;
    const int wu = ut >> 5;

    // The generation picks this launch's barrier counters; the last CTA through barrier 2
    // bumps it, after every CTA read it here. The other parity's counters were this launch's
    // predecessor's, complete in stream order, and are the next launch's.
    int gen = 0;
    if (tid == 0) {
        gen = *reinterpret_cast<volatile int*>(p.ctr + kGen);
        if (cta == 0) {
            p.ctr[kBar1 + ((gen + 1) & 1)] = 0;
            p.ctr[kBar2 + ((gen + 1) & 1)] = 0;
        }
    }

    const int g0 = cta * GROUPS / G;
    const int ng = (cta + 1) * GROUPS / G - g0;
    const int col = 4 * (g0 + lane);
    if (warp == 0 && lane < ng) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem_u32(s_stream[lane])), "l"(p.stream + col) : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem_u32(s_alpha[lane])), "l"(p.alpha + col) : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem_u32(s_beta[lane])), "l"(p.beta + col) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem_u32(s_wn[lane])), "l"(p.w_next + col) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    float ssq = 0.f;
    uint4 xa = make_uint4(0, 0, 0, 0);
    uint4 xb = make_uint4(0, 0, 0, 0);
    if (wu < 2) {
        const uint4* xp = reinterpret_cast<const uint4*>(p.x) + 2 * ut;
        xa = __ldg(xp);
        xb = __ldg(xp + 1);
    }
    for (int c = cta * U + u; c < HALF; c += G * U) {
        if (wu < 2) {
            uint4 l0, l1;
            uint2 q0, q1;
            ld_pair(p, c, ut, l0, l1, q0, q1);
            float a0 = 0.f, a1 = 0.f;
            chunk_fma(l0, q0.x, q0.y, xa, xb, p.base4, a0);
            chunk_fma(l1, q1.x, q1.y, xa, xb, p.base4, a1);
            a0 = warp_sum(a0);
            a1 = warp_sum(a1);
            if (lane == 0) {
                red[u][0][wu] = a0;
                red[u][1][wu] = a1;
            }
        } else {
            escape_pair(p, c, lane, red[u]);
        }
        bar_unit(1 + u);
        if (ut == 0) {
            ssq = finish_pair(red[u], p.branch, c, ssq);
        }
        // The next round overwrites red[u].
        bar_unit(1 + u);
    }
    if (ut == 0) {
        uss[u] = ssq;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < U; ++i) {
            s += uss[i];
        }
        p.part[cta] = s;
    }
    grid_barrier(p.ctr, kBar1 + (gen & 1), G, false, 1);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (warp == 0) {
        float t1 = 0.f;
        for (int i = lane; i < G; i += 32) {
            t1 += __ldcg(p.part + i);
        }
        t1 = warp_sum(t1);
        const float rb = rsqrtf(t1 / H + p.eps_post);
        asm volatile("cp.async.wait_all;" ::: "memory");
        float ss2 = 0.f;
        if (lane < ng) {
            const uint2 bv = __ldcg(reinterpret_cast<const uint2*>(p.branch + col));
            const float fa[4] = {lo_f(bv.x), hi_f(bv.x), lo_f(bv.y), hi_f(bv.y)};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[k] = s_alpha[lane][k] * s_stream[lane][k] + s_beta[lane][k] * bf16r(fa[k] * rb);
                ss2 += s[k] * s[k];
            }
        }
        ss2 = warp_sum(ss2);
        if (lane == 0) {
            p.part[GMAX + cta] = ss2;
        }
    }
    grid_barrier(p.ctr, kBar2 + (gen & 1), G, true, 2);

    if (warp == 0) {
        float t2 = 0.f;
        for (int i = lane; i < G; i += 32) {
            t2 += __ldcg(p.part + GMAX + i);
        }
        t2 = warp_sum(t2);
        const float rn = rsqrtf(t2 / H + p.eps_next);
        if (lane < ng) {
            float o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                o[k] = s[k] * rn * __bfloat162float(s_wn[lane][k]);
            }
            *reinterpret_cast<float4*>(p.stream_out + col) = make_float4(s[0], s[1], s[2], s[3]);
            const __nv_bfloat162 h0 = __floats2bfloat162_rn(o[0], o[1]);
            const __nv_bfloat162 h1 = __floats2bfloat162_rn(o[2], o[3]);
            *reinterpret_cast<uint2*>(p.next + col) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                           *reinterpret_cast<const uint32_t*>(&h1));
            *reinterpret_cast<float4*>(p.router + col) = make_float4(o[0], o[1], o[2], o[3]);
        }
    }
}

__global__ void __launch_bounds__(NT, 2) mc1_probe_kernel(int* d) {
    if (threadIdx.x == 0) {
        atomicAdd(d, 1);
    }
}

}  // namespace mc1
}  // namespace

// [resident CTAs per SM, SMs, registers per thread, local bytes per thread, max grid].
std::vector<int64_t> mc1_occupancy() {
    using namespace mc1;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mc1_kernel, NT, 0));
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, mc1_kernel));
    return {per_sm, sms, fa.numRegs, static_cast<int64_t>(fa.localSizeBytes), GMAX};
}

// 1 when a cooperative launch of `grid` CTAs of the kernel's block size captures into a graph
// behind a plain launch, instantiates and replays on this driver (attn_one_probe_coop's method,
// without the PDL attribute). Errors are cleared.
int64_t mc1_probe_coop(int64_t grid) {
    using namespace mc1;
    int* d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    bool ok = cudaMemsetAsync(d, 0, sizeof(int), s) == cudaSuccess;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ok = ok && cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        mc1_probe_kernel<<<1, NT, 0, s>>>(d);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(static_cast<unsigned>(grid));
        cfg.blockDim = dim3(NT);
        cfg.stream = s;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeCooperative;
        attr[0].val.cooperative = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        ok = cudaLaunchKernelEx(&cfg, mc1_probe_kernel, d) == cudaSuccess;
        ok = (cudaStreamEndCapture(s, &graph) == cudaSuccess) && ok;
    }
    if (ok) {
        ok = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    if (ok) {
        ok = cudaGraphLaunch(exec, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess;
    }
    int h = 0;
    if (ok) {
        ok = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess
             && h == 1 + grid;
    }
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(s);
    cudaFree(d);
    cudaGetLastError();
    return ok ? 1 : 0;
}

// x: the attention output, BF16 [1, 1024]; stream: FP32 [1, 8192]. Returns the boundary's
// (next input, stream, router copy) and the BF16 branch (the o projection's output).
std::vector<torch::Tensor> mc1_fwd(torch::Tensor w, torch::Tensor table, int64_t pmax,
                                   int64_t base4, torch::Tensor x, torch::Tensor stream,
                                   torch::Tensor alpha, torch::Tensor beta, torch::Tensor w_next,
                                   double eps_post, double eps_next, torch::Tensor ctr,
                                   torch::Tensor part, int64_t grid) {
    using namespace mc1;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kByte && w.is_contiguous()
                && w.numel() == static_cast<int64_t>(H) * KO * 3 / 2,
                "mc1_fwd: w must be o's cgemv stream, uint8 [4096, 64, 48]");
    TORCH_CHECK(table.scalar_type() == torch::kInt32 && table.is_contiguous()
                && table.dim() == 3 && table.size(0) == HALF && table.size(1) == pmax
                && table.size(2) == 2 && pmax >= 2,
                "mc1_fwd: table must be int32 [4096, pmax, 2]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == KO,
                "mc1_fwd: x must be a contiguous BF16 [1, 1024]");
    TORCH_CHECK(stream.scalar_type() == torch::kFloat32 && stream.is_contiguous()
                && stream.numel() == H, "mc1_fwd: stream must be a contiguous FP32 [1, 8192]");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32
                && w_next.scalar_type() == torch::kBFloat16 && alpha.is_contiguous()
                && beta.is_contiguous() && w_next.is_contiguous() && alpha.numel() == H
                && beta.numel() == H && w_next.numel() == H, "mc1_fwd: boundary weights");
    TORCH_CHECK(ctr.scalar_type() == torch::kInt32 && ctr.is_contiguous()
                && ctr.numel() >= kCtrInts && part.scalar_type() == torch::kFloat32
                && part.is_contiguous() && part.numel() >= 2 * GMAX, "mc1_fwd: workspace");
    TORCH_CHECK(grid >= GMIN && grid <= GMAX, "mc1_fwd: grid ", grid, " outside [", GMIN, ", ",
                GMAX, "]");
    for (const auto* t : {&x, &stream, &alpha, &beta, &w_next, &w}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "mc1_fwd: alignment");
    }
    auto branch = torch::empty({1, H}, x.options());
    auto next = torch::empty({1, H}, x.options());
    auto stream_out = torch::empty({1, H}, stream.options());
    auto router = torch::empty({1, H}, stream.options());
    Params p;
    p.w = w.data_ptr<uint8_t>();
    p.table = reinterpret_cast<const int2*>(table.data_ptr<int32_t>());
    p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    p.stream = stream.data_ptr<float>();
    p.alpha = alpha.data_ptr<float>();
    p.beta = beta.data_ptr<float>();
    p.w_next = reinterpret_cast<const __nv_bfloat16*>(w_next.data_ptr());
    p.branch = reinterpret_cast<__nv_bfloat16*>(branch.data_ptr());
    p.part = part.data_ptr<float>();
    p.ctr = ctr.data_ptr<int>();
    p.stream_out = stream_out.data_ptr<float>();
    p.next = reinterpret_cast<__nv_bfloat16*>(next.data_ptr());
    p.router = router.data_ptr<float>();
    p.pmax = static_cast<int>(pmax);
    p.base4 = static_cast<unsigned>(base4);
    p.eps_post = static_cast<float>(eps_post);
    p.eps_next = static_cast<float>(eps_next);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(grid));
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    // The grid barriers rest on this: a PDL attribute would let the launch start while its
    // predecessor still holds SMs (plan, "C-1 implementation").
    TORCH_CHECK(cfg.numAttrs == 1 && cfg.attrs[0].id == cudaLaunchAttributeCooperative,
                "mc1_fwd: the launch must carry the cooperative attribute only");
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mc1_kernel, p));
    return {next, stream_out, router, branch};
}
"""

_MC1_CPP_SRC = r"""
#include <torch/extension.h>
std::vector<int64_t> mc1_occupancy();
int64_t mc1_probe_coop(int64_t grid);
std::vector<torch::Tensor> mc1_fwd(torch::Tensor w, torch::Tensor table, int64_t pmax,
                                   int64_t base4, torch::Tensor x, torch::Tensor stream,
                                   torch::Tensor alpha, torch::Tensor beta, torch::Tensor w_next,
                                   double eps_post, double eps_next, torch::Tensor ctr,
                                   torch::Tensor part, int64_t grid);
"""


def _mc1_sources():
    return ["#define GEMV_DEVICE_ONLY 1\n" + _SKP_GEMV_SRC, _MC1_SRC]


def _mc1_load():
    return load_inline(
        name="okc_decode_layer_mc1",
        cpp_sources=_MC1_CPP_SRC,
        cuda_sources=_mc1_sources(),
        functions=["mc1_fwd", "mc1_occupancy", "mc1_probe_coop"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )


def _mc1_ext():
    return _bg_ext("mc1", _mc1_load)


# The kernel's limits (mc1::GMIN, mc1::GMAX) and its counter layout: [0] generation, [1] spin
# timeout flags, [2 + parity] barrier 1, [4 + parity] barrier 2.
MC1_GMIN = 64
MC1_GMAX = 1024
MC1_CTR_INTS = 8
# Self-test: inputs, and per call the most branch mantissas whose stream elements may differ
# and the most BF16 next-input elements allowed to differ by one ulp elsewhere. A 1-ulp change
# of the first norm's scale can flip the BF16 rounding of f * rb, which depends only on f's
# 8-bit mantissa, so it flips every element with that mantissa at once (about 64 of 8192;
# tools/dl_mc_c1_cpu_check.py: up to 64 elements of one mantissa, within the flip bound).
MC1_TEST_CASES = 8
MC1_FLIP_CLASSES = 2
MC1_NEXT_FLIPS = 8
MC1_ULPS = 8
MC1_STRESS_REPLAYS = 2000
MC1_PREF = "off"


def _mc1_setup(state):
    """Repacks o for cgemv (the pick at m = 1 must be cgemv, so the branch can match it
    bitwise), probes the cooperative launch under graph capture at the grid the occupancy
    allows, and allocates the zeroed counters and partials. Raises when any of it fails."""
    assert not torch.cuda.is_current_stream_capturing(), "mc1 setup inside a capture"
    pick = state["skp_pick"].get(("o", 1))
    if pick is None or pick[0] != "cgemv":
        raise RuntimeError(f"o pick at m=1 is {None if pick is None else pick[0]}, not cgemv")
    w = state["dense"]["o"]
    if tuple(w.shape) != (HIDDEN, 1024) or cgemv_threads(*w.shape) != 64:
        raise RuntimeError(f"o is {tuple(w.shape)}, cgemv threads {cgemv_threads(*w.shape)}")
    cg = pack_cgemv(w.contiguous())
    if cg is None:
        raise RuntimeError("o past the escape rule")
    ext = _mc1_ext()
    per_sm, sms, regs, local, gmax = [int(v) for v in ext.mc1_occupancy()]
    grid = min(per_sm * sms, gmax)
    if grid < MC1_GMIN:
        raise RuntimeError(f"cooperative grid {grid} ({per_sm} per SM x {sms}) < {MC1_GMIN}")
    if not ext.mc1_probe_coop(grid):
        raise RuntimeError(f"cooperative launch of {grid} CTAs refused under capture")
    dev = w.device
    state["mc1"] = {"stream": cg["stream"], "table": cg["table"], "pmax": cg["pmax"],
                    "base4": cg["base4"], "grid": grid,
                    "ctr": torch.zeros(MC1_CTR_INTS, dtype=torch.int32, device=dev),
                    "part": torch.zeros(2 * MC1_GMAX, dtype=torch.float32, device=dev)}
    state["mc1_report"] = [f"grid {grid} ({per_sm}/SM x {sms}) x 768 threads, regs {regs}, "
                           f"local {local} B, o escapes {cg['escapes']} pmax {cg['pmax']}"]


def _mc1(x, stream, state):
    """(next input, stream, router copy, branch) at one token from the attention output."""
    m, abnd = state["mc1"], state["abnd"]
    return _mc1_ext().mc1_fwd(m["stream"], m["table"], m["pmax"], m["base4"], x.contiguous(),
                              stream.contiguous(), abnd["alpha"], abnd["beta"], abnd["w_next"],
                              abnd["eps_post"], abnd["eps_next"], m["ctr"], m["part"], m["grid"])


def _mc1_check(state, launches=None):
    """None when the counters are clean: no spin timeout, this generation's counters zero and
    the previous one's full; with `launches`, the generation advanced by exactly that many since
    state["mc1_gen0"]. Else the reason."""
    ctr = state["mc1"]["ctr"].tolist()
    gen, grid = ctr[0], state["mc1"]["grid"]
    if ctr[1]:
        return f"spin timeout flags {ctr[1]}"
    if ctr[2 + (gen & 1)] or ctr[4 + (gen & 1)]:
        return f"counters of generation {gen} not zero: {ctr}"
    if gen and (ctr[2 + ((gen + 1) & 1)] != grid or ctr[4 + ((gen + 1) & 1)] != grid):
        return f"counters of generation {gen - 1} not {grid}: {ctr}"
    if launches is not None and gen - state["mc1_gen0"] != launches:
        return f"generation advanced {gen - state['mc1_gen0']}, launches {launches}"
    return None


def _mc1_ulp(t):
    """FP32 ulp of each element (2^-149 at zero)."""
    a = t.abs().clamp(min=2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(a)) - 23)


def _mc1_compare(got, ref, abnd):
    """Per call: (stream flips, next flips, router rel L2 off the flips, worst ulps off the
    flips) or raises. The stream may differ only where the normalized branch's BF16 rounding
    flipped: on elements of at most MC1_FLIP_CLASSES branch mantissas, each by at most beta x
    2^-7 of the normalized branch plus MC1_ULPS FP32 ulps. Off those elements the stream is
    bitwise equal, the router copy within MC1_ULPS ulps and relative L2 1e-6, and the next input
    differs by one BF16 ulp on at most MC1_NEXT_FLIPS elements."""
    nxt, out, router, br = got
    r_next, r_out, r_router, r_branch = ref
    if not torch.equal(br, r_branch):
        raise RuntimeError(f"branch differs from the o projection on "
                           f"{int((br != r_branch).sum())} elements")
    if not all(bool(torch.isfinite(t.float()).all()) for t in (nxt, out, router)):
        raise RuntimeError("non-finite output")
    d = (out - r_out).abs().flatten()
    flip = d > 0
    f = r_branch.float().flatten()
    rb = torch.rsqrt(f.double().pow(2).mean() + abnd["eps_post"]).float()
    lim = abnd["beta"].abs() * (f * rb).abs() * 2.0 ** -7 + MC1_ULPS * _mc1_ulp(r_out.flatten())
    mant = r_branch.flatten().view(torch.int16)[flip] & 0x7F
    classes = int(torch.unique(mant).numel())
    if classes > MC1_FLIP_CLASSES or bool((d[flip] > lim[flip]).any()):
        raise RuntimeError(f"stream: {int(flip.sum())} elements of {classes} branch mantissas "
                           f"differ, worst {float((d / lim).max()):.2f} of the flip bound")
    keep = ~flip
    rd = (router - r_router).abs().flatten()
    ulps = float((rd[keep] / _mc1_ulp(r_router.flatten()[keep])).max()) if keep.any() else 0.0
    rel = float(rd[keep].norm() / r_router.flatten()[keep].norm().clamp(min=1e-30))
    if not (ulps <= MC1_ULPS and rel < 1e-6):
        raise RuntimeError(f"router copy: {ulps:.1f} ulps, rel {rel:.2e} off the flips")
    nb = nxt.view(torch.int16).flatten().int()
    rbits = r_next.view(torch.int16).flatten().int()
    nd = (nb != rbits) & keep
    if int(nd.sum()) > MC1_NEXT_FLIPS or bool(((nb - rbits).abs()[nd] > 1).any()):
        raise RuntimeError(f"next input: {int(nd.sum())} elements differ off the flips, "
                           f"largest {int((nb - rbits).abs()[nd].max())} BF16 ulps")
    return int(flip.sum()), int(nd.sum()), rel, ulps


def _mc1_self_test(state):
    """The fused kernel against the two-kernel path (the cgemv o pick, then our 8-CTA boundary,
    whose per-element arithmetic every geometry shares) on MC1_TEST_CASES inputs at several
    scales: the BF16 branch bitwise, and the outputs as _mc1_compare allows. Then the counters
    (no spin timeout, the generation advanced once per call). Sets state["mc1_ok"]."""
    state["mc1_ok"] = False
    dev = state["dense"]["o"].device
    gen = torch.Generator(device=dev).manual_seed(5150)
    state["mc1_gen0"] = int(state["mc1"]["ctr"][0])
    flips = [0, 0]
    worst = [0.0, 0.0]
    for case in range(MC1_TEST_CASES):
        xs = (0.05, 0.5, 2.0, 0.01)[case % 4]
        ss = (3.0, 1.0, 30.0, 0.3)[case // 2 % 4]
        x = (torch.randn((1, 1024), generator=gen, device=dev) * xs).to(torch.bfloat16)
        if case == MC1_TEST_CASES - 1:
            x.zero_()
            x[0, 17] = 64.0
        stream = torch.randn((1, HIDDEN), generator=gen, device=dev) * ss
        branch = _proj(x, "o", HIDDEN, state)
        ref = list(_abnd(stream.clone(), branch, True, state, ABND_GEOS[0])) + [branch]
        got = _mc1(x, stream, state)
        torch.cuda.synchronize()
        sf, nf, rel, ulps = _mc1_compare(got, ref, state["abnd"])
        flips[0] += sf
        flips[1] += nf
        worst = [max(worst[0], rel), max(worst[1], ulps)]
    msg = _mc1_check(state, MC1_TEST_CASES)
    if msg:
        raise RuntimeError(f"[workspace] mc1 counters after the self-test: {msg}")
    state["mc1_report"].append(
        f"selftest {MC1_TEST_CASES} cases: branch bitwise; stream flips {flips[0]}, next flips "
        f"{flips[1]}, router rel {worst[0]:.1e} worst {worst[1]:.1f} ulps off the flips")
    state["mc1_ok"] = True


def _mc1_fresh(state, data, on, bank, refs, replays):
    """kernel_fn with C-1 on or off, captured with a device counter that copies the next of the
    bank's (stream, attn_input) pairs into the captured inputs before each replay; each replay
    is compared on the device with that pair's C-1-off eager reference (_gate_add). With C-1
    on, the counters must show one launch per call (two warm-up calls and the replays)."""
    dev = data["stream"].device
    bank_s, bank_a = bank
    acc = _gate_acc(2, dev)
    sel = torch.zeros(1, dtype=torch.long, device=dev)
    saved = state["mc1_on"]
    state["mc1_on"] = on

    def run():
        sel.add_(1)
        i = sel % bank_s.shape[0]
        data["stream"].copy_(bank_s.index_select(0, i)[0])
        data["attn_input"].copy_(bank_a.index_select(0, i)[0])
        out = kernel_fn(data, state)
        _gate_add(acc, out[:2], refs[:2], i, ids=(out[2], refs[2]))

    try:
        state["mc1_gen0"] = int(state["mc1"]["ctr"][0])
        graph = _graph_build(run)
        _gate_zero(acc)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        del graph
    finally:
        state["mc1_on"] = saved
    msg = _mc1_check(state, (2 + replays) if on else 0)
    return acc, msg


def _mc1_gate(state):
    """The fresh-input gate at 1x1 sliding and full, as _m1_stress runs it: five (stream,
    attn_input) pairs, each run eagerly with C-1 off as its reference; C-1 off (the control) and
    on (the candidate) captured and replayed MC1_STRESS_REPLAYS times each, every replay on the
    next pair and checked on the device; _gate_verdict decides, and the counters must be clean.
    Then the negative controls on an eager C-1-on output of pair 0. Raises on any failure."""
    lines = []
    saved = state["mc1_on"]
    try:
        for layer in ("sliding", "full"):
            data = _timeline_case(state, 1, 1, layer, 9800)
            label = f"1x1{layer[0]}"
            dev = data["stream"].device
            gen = torch.Generator(device=dev).manual_seed(778)
            bank_s = torch.randn((5, 1, HIDDEN), generator=gen, device=dev)
            bank_a = (bank_s * torch.rsqrt(bank_s.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                      ).to(torch.bfloat16)
            state["mc1_on"] = False
            hs, ss, ids = [], [], []
            for k in range(5):
                data["stream"].copy_(bank_s[k])
                data["attn_input"].copy_(bank_a[k])
                out = kernel_fn(data, state)
                hs.append(out[0].clone())
                ss.append(out[1].clone())
                ids.append(out[2].clone())
            refs = [_gate_refs(hs), _gate_refs(ss), torch.stack(ids)]
            ctrl, _ = _mc1_fresh(state, data, False, (bank_s, bank_a), refs, MC1_STRESS_REPLAYS)
            cand, msg = _mc1_fresh(state, data, True, (bank_s, bank_a), refs, MC1_STRESS_REPLAYS)
            ok, text, classes = _gate_verdict(cand, ctrl)
            if msg:
                ok = False
                classes = sorted(set(classes) | {"workspace"})
                text += f" FAIL counters {msg}"
            state["mc1_on"] = True
            data["stream"].copy_(bank_s[0])
            data["attn_input"].copy_(bank_a[0])
            out = [t.clone() for t in kernel_fn(data, state)[:2]]
            state["mc1_on"] = False
            negs = []
            for k, name in enumerate(("hidden", "stream")):
                one = _gate_acc(2, dev)
                for c in ("rel", "el"):
                    one[c].copy_(ctrl[c][k].expand(2))
                one["cnt"].copy_(ctrl["cnt"])
                neg, misses = _gate_negctl(out[k].reshape(out[k].shape[0], -1), refs[k], 0, 1,
                                           one, 61 + 7 * k, M1_NEG_ELEM)
                negs.append(f"{name} {neg}")
                if misses:
                    ok = False
                    classes = sorted(set(classes) | {"sensitivity"})
                    text += f" FAIL negative controls {name} {misses}"
            state.setdefault("gate_negctl", {})[f"mc1 {label}"] = "; ".join(negs)
            lines.append(f"{label} {'ok' if ok else 'FAIL'} {text}")
            if not ok:
                raise RuntimeError(f"[{','.join(classes)}] C-1 fresh-input gate: " + lines[-1])
    finally:
        state["mc1_on"] = saved
    return lines


def _mc1_tune(state):
    """The whole step (kernel_fn) at 1x1 sliding and full with C-1 off and on, captured once
    each, STABLE_ROUNDS rotated rounds of cold-L2 medians (_m1_rounds); per round the load
    46 x sliding + 16 x full over 62; _stable_pick with MC1_PREF. Sets state["mc1_on"]."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    datas = {layer: _timeline_case(state, 1, 1, layer, 9900) for layer in ("sliding", "full")}
    graphs = {}
    saved = state["mc1_on"]
    try:
        for mode in ("off", "on"):
            state["mc1_on"] = mode == "on"
            for layer, data in datas.items():
                graphs[(mode, layer)] = _graph_build(lambda d=data: kernel_fn(d, state))
    finally:
        state["mc1_on"] = saved
    per = _m1_rounds(graphs, flush)
    del graphs
    med = {k: sorted(v)[len(v) // 2] for k, v in per.items()}
    rounds = {mode: [(46 * s + 16 * f) / 62 for s, f in zip(per[(mode, "sliding")],
                                                           per[(mode, "full")])]
              for mode in ("off", "on")}
    best, note = _stable_pick(rounds, MC1_PREF)
    state.setdefault("stable_report", {})["mc1 T=1"] = note
    state["mc1_on"] = best == "on"
    state["mc1_report"].append("T=1 step s/f " + " ".join(
        f"{m} {med[(m, 'sliding')]:.1f}/{med[(m, 'full')]:.1f}" for m in ("off", "on"))
        + f" -> {best}")


def _mc1_enable(state):
    """Setup, self-test, gate, then the stable pick; any failure leaves C-1 off."""
    state["mc1_on"] = False
    state["mc1_report"] = []
    try:
        _mc1_setup(state)
        _mc1_self_test(state)
        for line in _mc1_gate(state):
            state["mc1_report"].append("gate " + line)
        _mc1_tune(state)
    except Exception:
        state["mc1_ok"] = False
        state["mc1_on"] = False
        raise


'''

REPS = [
    # The switch, next to M1's.
    ('''PORT_SK_PDL = True
''',
     '''PORT_SK_PDL = True
# M-C slice C-1: at one token, the o projection and the attention boundary as one cooperative
# kernel (_mc1_*), behind its self-test, fresh-input gate and stable pick (preferred off).
PORT_MC1 = True
'''),
    # The extension builds on its thread with the others.
    ('''    if PORT_PROJ_SK:
        _bg_start("skp", _skp_load)
''',
     '''    if PORT_PROJ_SK:
        _bg_start("skp", _skp_load)
    if PORT_PROJ_SK and PORT_MC1:
        _bg_start("mc1", _mc1_load)
'''),
    ('''        "m1_ok": False,
''',
     '''        "m1_ok": False,
        "mc1_ok": False,
        "mc1_on": False,
'''),
    # After the M1 tuner: the o pick, the boundary and M1's policy are final.
    ('''        _try("M1 sk pdl tuner", lambda: _m1_enable(state))
''',
     '''        _try("M1 sk pdl tuner", lambda: _m1_enable(state))
    if PORT_MC1 and state["skp_ok"] and state["abnd_ours"]:
        _try("mc1", lambda: _mc1_enable(state))
'''),
    ('''    attn_output = _proj(o, "o", HIDDEN, state)  # all-reduce: identity

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    if state["abnd_ours"]:
        ffn_input, stream, router_input = _abnd(data["stream"], attn_output, not split_router,
                                                state, _abnd_geo(tokens, state))
    else:
        ffn_input, stream, router_input = state["attn_boundary"](
            data["stream"], attn_output, branch_dtype=torch.bfloat16,
            router_dtype=None if split_router else torch.float32)
''',
     '''    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    if tokens == 1 and state.get("mc1_on"):
        ffn_input, stream, router_input, _ = _mc1(o, data["stream"], state)
    else:
        attn_output = _proj(o, "o", HIDDEN, state)  # all-reduce: identity
        if state["abnd_ours"]:
            ffn_input, stream, router_input = _abnd(data["stream"], attn_output,
                                                    not split_router, state,
                                                    _abnd_geo(tokens, state))
        else:
            ffn_input, stream, router_input = state["attn_boundary"](
                data["stream"], attn_output, branch_dtype=torch.bfloat16,
                router_dtype=None if split_router else torch.float32)
'''),
    ('''           "skp": _SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC + _SKP_CPP_SRC}''',
     '''           "skp": _SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC + _SKP_CPP_SRC,
           "mc1": "".join(_mc1_sources()) + _MC1_CPP_SRC}'''),
    ('''    for item in state.get("m1_report", []):
        lines.append("m1 " + item)
''',
     '''    for item in state.get("m1_report", []):
        lines.append("m1 " + item)
    lines.append(f"mc1 ok={int(bool(state.get('mc1_ok')))} on={int(bool(state.get('mc1_on')))}"
                 + (f" timeout flags {int(state['mc1']['ctr'][1])}" if state.get("mc1") else ""))
    for item in state.get("mc1_report", []):
        lines.append("mc1 " + item)
'''),
]

ANCHOR = '''def _try(name, fn):
'''


def _blocks(text):
    return dict(re.findall(r'^(_\w+_SRC) = r"""(.*?)"""', text, re.S | re.M))


def _static_checks(text):
    blocks = _blocks(text)
    mc1 = blocks["_MC1_SRC"] + blocks["_SKP_GEMV_SRC"]
    if "griddepcontrol" in mc1:
        raise SystemExit("the C-1 module's source must not contain griddepcontrol")
    if "Programmatic" in blocks["_MC1_SRC"] or "programmatic" in blocks["_MC1_SRC"]:
        raise SystemExit("the C-1 launch must not carry the PDL attribute")
    ids = re.findall(r"attr\[\d\]\.id = (\w+);", blocks["_MC1_SRC"])
    if not ids or set(ids) != {"cudaLaunchAttributeCooperative"}:
        raise SystemExit(f"the C-1 launches must set only the cooperative attribute: {ids}")
    if chr(0x2014) in text:
        raise SystemExit("em-dash in the output")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, out, header = sys.argv[1:]
    text = open(src).read()
    if "_MC1_SRC" in text:
        raise SystemExit("IN already has C-1")
    before = _blocks(text)
    for old, new in REPS:
        if text.count(old) != 1:
            raise SystemExit(f"not where expected: {old[:70]!r}")
        text = text.replace(old, new)
    if text.count(ANCHOR) != 1:
        raise SystemExit("anchor _try not found once")
    text = text.replace(ANCHOR, MC1_SRC.lstrip("\n") + ANCHOR)
    after = _blocks(text)
    changed = [k for k, v in before.items() if after.get(k) != v]
    if changed:
        raise SystemExit(f"source blocks changed: {changed}")
    _static_checks(text)
    lines = text.split("\n")
    if lines[0] != "# okc: problem=muse_decode_layer":
        raise SystemExit("line 1 is not the okc header")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
