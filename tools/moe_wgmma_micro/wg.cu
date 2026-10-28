// Micro test: which element of the stage loop makes ptxas serialize the wgmmas.
#include <cstdint>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t make_desc(const void* ptr, int lbo, int sbo) {
    const uint64_t addr = smem_u32(ptr);
    return ((addr & 0x3ffffu) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16)
           | (static_cast<uint64_t>(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void wgmma_n24(float (&d)[32], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
template <int SD>
__device__ __forceinline__ void wgmma_n24i(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(SD));
}

__device__ __forceinline__ unsigned bmul2(unsigned a, unsigned s) {
    const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&s));
    return *reinterpret_cast<const unsigned*>(&r);
}
__device__ __forceinline__ void dequant(unsigned w, unsigned sg, unsigned sg8, unsigned (&a)[4]) {
    constexpr unsigned kM = 0x81C081C0u;
    a[0] = bmul2(w & kM, sg);
    a[1] = bmul2((w << 6) & kM, sg8);
    a[2] = bmul2(__funnelshift_r(w, w, 6) & kM, sg);
    a[3] = bmul2(((w >> 4) & 0x00400040u) | ((w << 4) & 0x81808180u), sg8);
}
__device__ __forceinline__ void a_pin(unsigned (&a)[2][4]) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(a[i][k]) :: "memory");
}
__device__ __forceinline__ void acc_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 12; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

// MODE bits: 1 = runtime scale_d (first ? 0 : 1) else immediate 1 with a zeroed acc;
//            2 = runtime outer stage loop (unroll 1) else a single stage;
//            4 = pin A fragments after each wait;
//            8 = wait_group 0 at the end of each stage.
template <int MODE>
__global__ void __launch_bounds__(576, 1) k(const unsigned* __restrict__ words, float* out, int nst) {
    extern __shared__ __align__(128) unsigned char smem[];
    float acc[32];
    unsigned af[2][2][4];
    const int tid = threadIdx.x;
    const unsigned* cp = words + tid * 16;
    bool first = true;
    if (!(MODE & 1)) {
#pragma unroll
        for (int i = 0; i < 12; ++i) acc[i] = 0.f;
    }
    acc_fence(acc);
    const int nstages = (MODE & 2) ? nst : 1;
#pragma unroll 1
    for (int st = 0; st < nstages; ++st) {
        unsigned cw[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) cw[i] = cp[st * 16 * 576 + i];
        const uint64_t d0 = make_desc(smem + (st % 3) * 32768, 128, 4096);
#pragma unroll
        for (int s = 0; s < 16; s += 2) {
            const int b = (s >> 1) & 1;
            dequant(cw[s], 0x40004000u + s, 0x40004001u, af[b][0]);
            dequant(cw[s + 1], 0x40004000u + s, 0x40004002u, af[b][1]);
            wg_fence();
            if (MODE & 1) {
                wgmma_n24(acc, af[b][0], d0 + s * 16, first ? 0 : 1);
            } else {
                wgmma_n24i<1>(acc, af[b][0], d0 + s * 16);
            }
            wgmma_n24i<1>(acc, af[b][1], d0 + (s + 1) * 16);
            first = false;
            wg_commit();
            wg_wait<1>();
            if (MODE & 4) a_pin(af[b ^ 1]);
        }
        if (MODE & 8) wg_wait<0>();
    }
    wg_wait<0>();
    acc_fence(acc);
#pragma unroll
    for (int i = 0; i < 12; ++i) out[tid * 12 + i] = acc[i];
}

template __global__ void k<0>(const unsigned*, float*, int);
template __global__ void k<1>(const unsigned*, float*, int);
template __global__ void k<2>(const unsigned*, float*, int);
template __global__ void k<3>(const unsigned*, float*, int);
template __global__ void k<6>(const unsigned*, float*, int);
template __global__ void k<7>(const unsigned*, float*, int);
template __global__ void k<10>(const unsigned*, float*, int);
template __global__ void k<15>(const unsigned*, float*, int);
