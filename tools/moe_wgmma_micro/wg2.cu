// Micro test 2: the fused kernel's consumer structure (switch over N variants sharing one
// accumulator array, outer segment loops, epilogue reads) to find what serializes wgmma.
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

__device__ __forceinline__ void wgmma_n8(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
__device__ __forceinline__ void wgmma_n24(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
__device__ __forceinline__ void wgmma_n64(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NTT>
__device__ __forceinline__ void wgmma(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    if constexpr (NTT == 1) { wgmma_n8(d, a, desc); }
    if constexpr (NTT == 3) { wgmma_n24(d, a, desc); }
    if constexpr (NTT == 8) { wgmma_n64(d, a, desc); }
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
template <int NTT>
__device__ __forceinline__ void acc_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

// MODE bits: 1 = switch over three N variants (else only N=24); 2 = outer segment loops;
//            4 = a_pin; 8 = epilogue with per-token predicates and a fin[] copy path.
template <int NTT, int MODE>
__device__ __forceinline__ void run_segment(const unsigned* __restrict__ words, float* out, float (&acc)[32],
                                            unsigned char* smem, int st_b, int st_e, int ntok, bool partial) {
    const int tid = threadIdx.x;
    unsigned af[2][2][4];
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) acc[i] = 0.f;
    acc_fence<NTT>(acc);
#pragma unroll 1
    for (int st = st_b; st < st_e; ++st) {
        unsigned cw[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) cw[i] = words[(st * 576 + tid) * 16 + i];
        const uint64_t d0 = make_desc(smem + (st % 3) * 32768, 128, 4096);
#pragma unroll
        for (int s = 0; s < 16; s += 2) {
            const int b = (s >> 1) & 1;
            dequant(cw[s], 0x40004000u + s, 0x40004001u, af[b][0]);
            dequant(cw[s + 1], 0x40004000u + s, 0x40004002u, af[b][1]);
            wg_fence();
            wgmma<NTT>(acc, af[b][0], d0 + s * 16);
            wgmma<NTT>(acc, af[b][1], d0 + (s + 1) * 16);
            wg_commit();
            wg_wait<1>();
            if (MODE & 4) a_pin(af[b ^ 1]);
        }
    }
    wg_wait<0>();
    if (MODE & 4) { a_pin(af[0]); a_pin(af[1]); }
    acc_fence<NTT>(acc);
    if (MODE & 8) {
        if (partial) {
            float fin[32];
#pragma unroll
            for (int i = 0; i < 4 * NTT; ++i) fin[i] = __ldcg(out + tid * 32 + i) + 1.f;
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt) {
                if (8 * nt + 2 * (tid & 3) < ntok) out[tid * 32 + nt] = fin[4 * nt] * fin[4 * nt + 2];
                if (8 * nt + 2 * (tid & 3) + 1 < ntok) out[tid * 32 + nt + 8] = fin[4 * nt + 1] * fin[4 * nt + 3];
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt) {
                if (8 * nt + 2 * (tid & 3) < ntok) out[tid * 32 + nt] = acc[4 * nt] * acc[4 * nt + 2];
                if (8 * nt + 2 * (tid & 3) + 1 < ntok) out[tid * 32 + nt + 8] = acc[4 * nt + 1] * acc[4 * nt + 3];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4 * NTT; ++i) out[tid * 32 + i] = acc[i];
    }
}

template <int MODE>
__global__ void __launch_bounds__(576, 1) k(const unsigned* __restrict__ words, float* out, const int* info, int nseg) {
    extern __shared__ __align__(128) unsigned char smem[];
    float acc[32];
    const int loops = (MODE & 2) ? nseg : 1;
#pragma unroll 1
    for (int i = 0; i < loops; ++i) {
        const int ntt = (MODE & 1) ? info[4 * i] : 3;
        const int st_b = info[4 * i + 1], st_e = info[4 * i + 2], ntok = info[4 * i + 3];
        const bool partial = st_b != 0;
        switch (ntt) {
            case 1: run_segment<1, MODE>(words, out, acc, smem, st_b, st_e, ntok, partial); break;
            case 3: run_segment<3, MODE>(words, out, acc, smem, st_b, st_e, ntok, partial); break;
            default: run_segment<8, MODE>(words, out, acc, smem, st_b, st_e, ntok, partial); break;
        }
    }
}

template __global__ void k<0>(const unsigned*, float*, const int*, int);
template __global__ void k<1>(const unsigned*, float*, const int*, int);
template __global__ void k<2>(const unsigned*, float*, const int*, int);
template __global__ void k<3>(const unsigned*, float*, const int*, int);
template __global__ void k<7>(const unsigned*, float*, const int*, int);
template __global__ void k<8>(const unsigned*, float*, const int*, int);
template __global__ void k<11>(const unsigned*, float*, const int*, int);
template __global__ void k<15>(const unsigned*, float*, const int*, int);
