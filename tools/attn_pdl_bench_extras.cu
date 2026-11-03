// Producer / consumer / audit / hog kernels for tools/attn_make_pdl_bench.py: the decode layer's qkv -> attention ->
// o edges at T = 8 rebuilt around the layer's own attention build, so each edge's drift can be measured element by
// element. Host entry points take raw addresses and the stream as int64 (no torch headers: nvcc builds this alone).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace pdlb {

__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gdc_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int ld_cg_s32(const int* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cg_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

// Stands in for the qkv sk2 GEMM: streams w_bytes_per_cta of a weight buffer, triggers its dependents after the
// stream (sk2 triggers after its last weight stage), then writes the qkv rows. fresh: rows from src1 on odd replays.
__global__ void producer(const uint4* w, long long wv_per_cta, const uint4* src0, const uint4* src1, uint4* qkv,
                         long long n_vec, const int* ctr, int fresh, int trigger, unsigned* sink) {
  extern __shared__ uint4 pad_smem[];
  unsigned acc = 0;
  const uint4* wp = w + (long long)blockIdx.x * wv_per_cta;
  for (long long i = threadIdx.x; i < wv_per_cta; i += blockDim.x) {
    const uint4 v = ld_cg_v4(wp + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (trigger) gdc_trigger();
  const uint4* src = (fresh && (ld_cg_s32(ctr) & 1)) ? src1 : src0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += (long long)gridDim.x * blockDim.x)
    qkv[i] = ld_cg_v4(src + i);
  if (acc == 0x9e3779b9u) {   // never true for the random weights; keeps the stream's loads live
    pad_smem[threadIdx.x] = make_uint4(acc, 0, 0, 0);
    sink[0] = pad_smem[threadIdx.x ^ 1].x;
  }
}

__device__ __forceinline__ int bf_ord(unsigned short b) { return (b & 0x8000) ? -(int)(b & 0x7fff) : (int)b; }
__device__ __forceinline__ bool bf_nan(unsigned short b) { return (b & 0x7f80) == 0x7f80 && (b & 0x7f); }

// Stands in for the o sk2 GEMM: weight stream before the wait (sk2 issues N stages before griddepcontrol.wait), then
// reads attention's output with ld.global.cg and compares it with the PDL-off reference of the same input parity.
// Per element: replays that differed, max BF16 ulp distance, max |diff| (float bits); per replay: differing elements
// and max ulp. seen keeps the values this kernel read, for the serialized audit.
__global__ void consumer(const uint4* w, long long wv_per_cta, const uint4* out, const uint4* ref0, const uint4* ref1,
                         long long n_vec, const int* ctr, int fresh, uint4* seen, unsigned* cnt, unsigned* maxulp,
                         unsigned* maxabs, unsigned* rep_nd, unsigned* rep_ulp, int nrep, unsigned* sink) {
  unsigned acc = 0;
  const uint4* wp = w + (long long)blockIdx.x * wv_per_cta;
  for (long long i = threadIdx.x; i < wv_per_cta; i += blockDim.x) {
    const uint4 v = ld_cg_v4(wp + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  gdc_wait();
  const int c = ld_cg_s32(ctr);
  const uint4* ref = (fresh && (c & 1)) ? ref1 : ref0;
  const int rep = c % nrep;
  unsigned nd = 0, mu = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += (long long)gridDim.x * blockDim.x) {
    const uint4 v = ld_cg_v4(out + i);
    const uint4 r = ld_cg_v4(ref + i);
    seen[i] = v;
    const unsigned* vw = &v.x;
    const unsigned* rw = &r.x;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned short a = (unsigned short)(vw[e >> 1] >> (16 * (e & 1)));
      const unsigned short b = (unsigned short)(rw[e >> 1] >> (16 * (e & 1)));
      if (a != b) {
        const long long idx = i * 8 + e;
        unsigned d = (bf_nan(a) || bf_nan(b)) ? 65535u : (unsigned)abs(bf_ord(a) - bf_ord(b));
        const float fa = __uint_as_float((unsigned)a << 16), fb = __uint_as_float((unsigned)b << 16);
        const float ad = fabsf(fa - fb);
        atomicAdd(cnt + idx, 1u);
        atomicMax(maxulp + idx, d);
        atomicMax(maxabs + idx, (bf_nan(a) || bf_nan(b)) ? 0x7f800000u : __float_as_uint(ad));
        ++nd;
        mu = max(mu, d);
      }
    }
  }
  if (nd) {
    atomicAdd(rep_nd + rep, nd);
    atomicMax(rep_ulp + rep, mu);
  }
  if (acc == 0x9e3779b9u) sink[1] = acc;
}

// Serialized after the consumer: any word the consumer read that differs from the final output means the consumer
// read before attention's writes were visible (mechanism c). The last thread bumps the replay counter.
__global__ void audit(const uint4* out, const uint4* seen, long long n_vec, unsigned* bad, int* ctr) {
  unsigned nb = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += (long long)gridDim.x * blockDim.x) {
    const uint4 a = ld_cg_v4(out + i), b = ld_cg_v4(seen + i);
    nb += (a.x != b.x) + (a.y != b.y) + (a.z != b.z) + (a.w != b.w);
  }
  if (nb) atomicAdd(bad, nb);
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(ctr, 1);
}

// Occupies one SM per CTA (large dynamic smem keeps attention CTAs off it) for base_ns plus a per-replay jitter, on a
// side stream concurrent with a plain attention launch: staggers attention CTA starts without any PDL edge.
__global__ void hog(const int* ctr, long long base_ns, long long jitter_ns, unsigned* sink) {
  extern __shared__ uint4 pad_smem[];
  const int c = ld_cg_s32(ctr);
  const long long ns = base_ns + (jitter_ns ? ((long long)((unsigned)c * 2654435761u + blockIdx.x * 40503u) % jitter_ns) : 0);
  const unsigned long long t0 = gtime();
  while ((long long)(gtime() - t0) < ns) __nanosleep(200);
  if (threadIdx.x == 0 && ns < 0) sink[2] = pad_smem[0].x;
}

}  // namespace

static void launch_pdl(const void* fn, dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl, void** args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelExC(&cfg, fn, args);
}

int64_t pdlb_producer(int64_t w, int64_t wv_per_cta, int64_t src0, int64_t src1, int64_t qkv, int64_t n_vec,
                      int64_t ctr, int64_t fresh, int64_t trigger, int64_t sink, int64_t grid, int64_t smem,
                      int64_t stream) {
  static int init_smem = -1;
  if (init_smem != (int)smem) {
    cudaFuncSetAttribute(pdlb::producer, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    init_smem = (int)smem;
  }
  const uint4* a0 = (const uint4*)w;
  long long a1 = wv_per_cta;
  const uint4* a2 = (const uint4*)src0;
  const uint4* a3 = (const uint4*)src1;
  uint4* a4 = (uint4*)qkv;
  long long a5 = n_vec;
  const int* a6 = (const int*)ctr;
  int a7 = (int)fresh, a8 = (int)trigger;
  unsigned* a9 = (unsigned*)sink;
  void* args[] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9};
  launch_pdl((const void*)pdlb::producer, dim3((unsigned)grid), dim3(256), (size_t)smem, (cudaStream_t)stream, false, args);
  return (int64_t)cudaGetLastError();
}

int64_t pdlb_consumer(int64_t w, int64_t wv_per_cta, int64_t out, int64_t ref0, int64_t ref1, int64_t n_vec,
                      int64_t ctr, int64_t fresh, int64_t seen, int64_t cnt, int64_t maxulp, int64_t maxabs,
                      int64_t rep_nd, int64_t rep_ulp, int64_t nrep, int64_t sink, int64_t grid, int64_t pdl,
                      int64_t stream) {
  const uint4* a0 = (const uint4*)w;
  long long a1 = wv_per_cta;
  const uint4* a2 = (const uint4*)out;
  const uint4* a3 = (const uint4*)ref0;
  const uint4* a4 = (const uint4*)ref1;
  long long a5 = n_vec;
  const int* a6 = (const int*)ctr;
  int a7 = (int)fresh;
  uint4* a8 = (uint4*)seen;
  unsigned *a9 = (unsigned*)cnt, *a10 = (unsigned*)maxulp, *a11 = (unsigned*)maxabs, *a12 = (unsigned*)rep_nd,
           *a13 = (unsigned*)rep_ulp;
  int a14 = (int)nrep;
  unsigned* a15 = (unsigned*)sink;
  void* args[] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15};
  launch_pdl((const void*)pdlb::consumer, dim3((unsigned)grid), dim3(256), 0, (cudaStream_t)stream, pdl != 0, args);
  return (int64_t)cudaGetLastError();
}

int64_t pdlb_audit(int64_t out, int64_t seen, int64_t n_vec, int64_t bad, int64_t ctr, int64_t stream) {
  const uint4* a0 = (const uint4*)out;
  const uint4* a1 = (const uint4*)seen;
  long long a2 = n_vec;
  unsigned* a3 = (unsigned*)bad;
  int* a4 = (int*)ctr;
  void* args[] = {&a0, &a1, &a2, &a3, &a4};
  launch_pdl((const void*)pdlb::audit, dim3(16), dim3(256), 0, (cudaStream_t)stream, false, args);
  return (int64_t)cudaGetLastError();
}

int64_t pdlb_hog(int64_t ctr, int64_t base_ns, int64_t jitter_ns, int64_t sink, int64_t grid, int64_t smem,
                 int64_t stream) {
  static int init_smem = -1;
  if (init_smem != (int)smem) {
    cudaFuncSetAttribute(pdlb::hog, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    init_smem = (int)smem;
  }
  const int* a0 = (const int*)ctr;
  long long a1 = base_ns, a2 = jitter_ns;
  unsigned* a3 = (unsigned*)sink;
  void* args[] = {&a0, &a1, &a2, &a3};
  launch_pdl((const void*)pdlb::hog, dim3((unsigned)grid), dim3(128), (size_t)smem, (cudaStream_t)stream, false, args);
  return (int64_t)cudaGetLastError();
}
