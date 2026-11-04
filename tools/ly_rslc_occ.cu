// Which resource limits slice CTAs beside a running TINY MoE CTA (layer lane, "Decode 1 register slice").
//
// An occupier kernel takes the TINY moe_fused footprint measured on ly_v1un (tools/ly_micro_sass.py, tiny build:
// 288 threads, REG 145, SHARED 1024 static, okc_smem_bytes() = 214784 dynamic), one CTA per SM, and spins for
// OCC_US. A probe kernel is launched on a second stream right after; each probe config differs in warps, registers
// or shared memory. Each probe CTA records entry time, %smid and every warp's %warpid, then spins for longer than the
// occupier, so a probe CTA whose entry precedes the occupier's first exit was resident beside it. Printed per probe
// config: probe CTAs resident per SM (min / max over the 132 SMs), and the warp-slot residues mod 4 of the occupier
// and of the resident probe warps, which is how the four sub-partitions are told apart (hypothesis: SMSP = %warpid % 4).
//
//   nvcc -O3 -std=c++17 -arch=sm_90a -Xptxas -v -o ly_rslc_occ tools/ly_rslc_occ.cu && flock gpu.lock ./ly_rslc_occ
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned v; asm volatile("mov.u32 %0, %%smid;" : "=r"(v)); return v; }
__device__ __forceinline__ unsigned warpid() { unsigned v; asm volatile("mov.u32 %0, %%warpid;" : "=r"(v)); return v; }

constexpr int OCC_THREADS = 288, OCC_DYN = 214784;
constexpr int MAXW = 32;

// rec per CTA: [0] entry ns, [1] exit ns, [2] smid, [3] nwarps, [4..4+MAXW) warpid per warp
constexpr int REC = 4 + MAXW;

template <int NLIVE>
__device__ __forceinline__ void hold_live(const unsigned* __restrict__ src, unsigned long long until, unsigned* sink) {
    unsigned v[NLIVE];
#pragma unroll
    for (int i = 0; i < NLIVE; ++i) v[i] = src[(threadIdx.x * 7 + i * 131) & 4095];
    while (gt() < until) {
#pragma unroll
        for (int i = 0; i < NLIVE; ++i) asm volatile("" : "+r"(v[i]));
    }
    unsigned s = 0;
#pragma unroll
    for (int i = 0; i < NLIVE; ++i) s ^= v[i] * (i + 1);
    if (s == 0x9e3779b9u) *sink = s;
}

__global__ void __launch_bounds__(OCC_THREADS, 1)
occ_kernel(const unsigned* __restrict__ src, unsigned* sink, unsigned long long* rec, int spin_ns) {
    extern __shared__ uint8_t dyn[];
    __shared__ uint8_t st[1024];
    unsigned long long t0 = gt();
    unsigned long long* r = rec + (size_t)blockIdx.x * REC;
    if ((threadIdx.x & 31) == 0) r[4 + (threadIdx.x >> 5)] = warpid();
    if (threadIdx.x == 0) { r[0] = t0; r[2] = smid(); r[3] = blockDim.x / 32; }
    st[threadIdx.x] = (uint8_t)threadIdx.x;
    dyn[threadIdx.x * 700] = (uint8_t)threadIdx.x;
    __syncthreads();
    hold_live<132>(src, t0 + spin_ns, sink);
    if (st[(threadIdx.x + 1) % OCC_THREADS] == 7 && dyn[5] == 3) *sink = 1;
    __syncthreads();
    if (threadIdx.x == 0) r[1] = gt();
}

template <int W, int R, int NLIVE, int SMEM>
__global__ void __maxnreg__(R)
probe_kernel(const unsigned* __restrict__ src, unsigned* sink, unsigned long long* rec, int spin_ns) {
    __shared__ uint8_t st[SMEM];
    unsigned long long t0 = gt();
    unsigned long long* r = rec + (size_t)blockIdx.x * REC;
    if ((threadIdx.x & 31) == 0) r[4 + (threadIdx.x >> 5)] = warpid();
    if (threadIdx.x == 0) { r[0] = t0; r[2] = smid(); r[3] = W; }
    st[threadIdx.x % SMEM] = (uint8_t)threadIdx.x;
    __syncthreads();
    hold_live<NLIVE>(src, t0 + spin_ns, sink);
    if (st[(threadIdx.x + 1) % SMEM] == 7) *sink = 2;
    if (threadIdx.x == 0) r[1] = gt();
}

struct Probe { const char* name; void (*k)(const unsigned*, unsigned*, unsigned long long*, int); int W; int R; int smem; };

template <int W, int R, int NLIVE, int SMEM>
Probe mk(const char* n) { return Probe{n, probe_kernel<W, R, NLIVE, SMEM>, W, R, SMEM}; }

int main() {
    int nsm = 0;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaFuncSetAttribute(occ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_DYN));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, occ_kernel));
    printf("occ: regs %d static smem %zu dyn %d threads %d\n", fa.numRegs, fa.sharedSizeBytes, OCC_DYN, OCC_THREADS);
    std::vector<Probe> probes = {
        mk<5, 40, 26, 64>("cg40 shape: 5 warps, 40 regs"),
        mk<5, 32, 18, 64>("cg32 shape: 5 warps, 32 regs"),
        mk<5, 24, 10, 64>("5 warps, 24 regs"),
        mk<4, 56, 42, 64>("4 warps, 56 regs"),
        mk<4, 64, 50, 64>("4 warps, 64 regs"),
        mk<8, 40, 26, 64>("8 warps, 40 regs"),
        mk<8, 48, 34, 64>("8 warps, 48 regs"),
        mk<12, 40, 26, 64>("12 warps, 40 regs"),
        mk<13, 40, 26, 64>("13 warps, 40 regs"),
        mk<16, 40, 26, 64>("16 warps, 40 regs"),
        mk<17, 40, 26, 64>("17 warps, 40 regs (544 threads)"),
        mk<17, 32, 18, 64>("17 warps, 32 regs (544 threads)"),
        mk<5, 24, 10, 8192>("5 warps, 24 regs, 8 KB smem"),
        mk<5, 24, 10, 12288>("5 warps, 24 regs, 12 KB smem"),
        mk<5, 24, 10, 16384>("5 warps, 24 regs, 16 KB smem"),
        mk<3, 56, 42, 64>("3 warps, 56 regs"),
        mk<6, 56, 42, 64>("6 warps, 56 regs"),
        mk<7, 56, 42, 64>("7 warps, 56 regs"),
        mk<7, 48, 34, 64>("7 warps, 48 regs"),
        mk<7, 40, 26, 64>("7 warps, 40 regs"),
        mk<11, 24, 10, 64>("11 warps, 24 regs"),
        mk<7, 56, 42, 14336>("7 warps, 56 regs, 14 KB smem"),
        mk<7, 56, 42, 15360>("7 warps, 56 regs, 15 KB smem"),
    };
    unsigned *src, *sink;
    unsigned long long *rec_o, *rec_p;
    const int PG = 132 * 8;
    CK(cudaMalloc(&src, 4096 * 4));
    CK(cudaMemset(src, 1, 4096 * 4));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMalloc(&rec_o, (size_t)nsm * REC * 8));
    CK(cudaMalloc(&rec_p, (size_t)PG * REC * 8));
    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    for (auto& p : probes) {
        cudaFuncAttributes pa;
        CK(cudaFuncGetAttributes(&pa, p.k));
        int alone = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&alone, p.k, p.W * 32, 0));
        for (int rep = 0; rep < 2; ++rep) {
            CK(cudaMemset(rec_o, 0, (size_t)nsm * REC * 8));
            CK(cudaMemset(rec_p, 0, (size_t)PG * REC * 8));
            CK(cudaDeviceSynchronize());
            occ_kernel<<<nsm, OCC_THREADS, OCC_DYN, s0>>>(src, sink, rec_o, 40000);
            p.k<<<PG, p.W * 32, 0, s1>>>(src, sink, rec_p, 60000);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<unsigned long long> ro((size_t)nsm * REC), rp((size_t)PG * REC);
            CK(cudaMemcpy(ro.data(), rec_o, ro.size() * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rp.data(), rec_p, rp.size() * 8, cudaMemcpyDeviceToHost));
            unsigned long long occ_first_exit = ~0ull, occ_last_entry = 0;
            std::vector<int> occ_res(4, 0);
            for (int b = 0; b < nsm; ++b) {
                occ_first_exit = std::min(occ_first_exit, ro[b * REC + 1]);
                occ_last_entry = std::max(occ_last_entry, ro[b * REC + 0]);
                if (b == 0) for (int w = 0; w < 9; ++w) occ_res[ro[b * REC + 4 + w] % 4]++;
            }
            std::vector<int> per_sm(nsm, 0);
            std::vector<int> pres(4, 0);
            int resident = 0, sm0_cta = -1;
            for (int b = 0; b < PG; ++b) {
                const unsigned long long* r = &rp[(size_t)b * REC];
                if (r[0] && r[0] < occ_first_exit) {
                    per_sm[r[2]]++;
                    resident++;
                    if (r[2] == ro[2] && sm0_cta < 0) sm0_cta = b;
                }
            }
            if (sm0_cta >= 0) {
                for (int b = 0; b < PG; ++b) {
                    const unsigned long long* r = &rp[(size_t)b * REC];
                    if (r[0] && r[0] < occ_first_exit && r[2] == ro[2])
                        for (int w = 0; w < p.W; ++w) pres[r[4 + w] % 4]++;
                }
            }
            int mn = *std::min_element(per_sm.begin(), per_sm.end()), mx = *std::max_element(per_sm.begin(), per_sm.end());
            printf("PROBE %-34s regs %3d smem %5zu alone/SM %2d | beside occ: %4d CTAs, per SM min %d max %d | "
                   "occ warps per slot%%4 %d %d %d %d, probe warps on that SM %d %d %d %d | occ entry spread %.2f us\n",
                   p.name, pa.numRegs, pa.sharedSizeBytes, alone, resident, mn, mx,
                   occ_res[0], occ_res[1], occ_res[2], occ_res[3], pres[0], pres[1], pres[2], pres[3],
                   (occ_last_entry - ro[0]) / 1e3);
        }
    }
    return 0;
}
