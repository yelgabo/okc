// One-CTA schedule prologue: topk_ids / topk_weights -> ClSchedule (cl_schedule.h), same bytes as cl_schedule.py.
// Loaded with torch.utils.cpp_extension.load_inline (cl/tests/test_schedule.py shows the call).
// Everything before griddepcontrol.wait touches only shared memory, so under PDL it overlaps the producer.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cl_schedule.h"

#define CL_PRO_THREADS 256
#define CL_PRO_WARPS (CL_PRO_THREADS / 32)
#define CL_PRO_MAX_T 288
#define CL_PRO_NCH ((CL_PRO_MAX_T * 8 + CL_PRO_THREADS - 1) / CL_PRO_THREADS)

__device__ __forceinline__ unsigned long long cl_gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// stamps (nullable, for the bench): [0] kernel entry, [1] after griddepcontrol.wait, [2] struct stored.
__global__ void __launch_bounds__(CL_PRO_THREADS, 1)
cl_prologue_kernel(const int* __restrict__ topk_ids, const float* __restrict__ topk_w, int T, int first, int count,
                   int policy, int target, int n_clusters, ClSchedule* __restrict__ out,
                   unsigned long long* __restrict__ stamps) {
  __shared__ __align__(16) ClSchedule s;
  // Per (chunk, warp, expert): pair count in that warp's 32 entries, then rewritten as the exclusive prefix
  // over (chunk, warp) in flat order, which is the rank base that keeps tok_list in (token, slot) order.
  __shared__ int wbase[CL_PRO_NCH][CL_PRO_WARPS][CL_MAX_EXPERTS];
  __shared__ int eoff[CL_MAX_EXPERTS];
  __shared__ int overflow;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned FULL = 0xffffffffu;
  if (stamps && tid == 0) stamps[0] = cl_gtime();

  const int npairs = T * 8;
  const int nch = (npairs + CL_PRO_THREADS - 1) / CL_PRO_THREADS;
  int* sw = reinterpret_cast<int*>(&s);
  for (int i = tid; i < CL_SCHED_WORDS; i += CL_PRO_THREADS) sw[i] = 0;
  int* wb = &wbase[0][0][0];
  for (int i = tid; i < nch * CL_PRO_WARPS * CL_MAX_EXPERTS; i += CL_PRO_THREADS) wb[i] = 0;
  __syncthreads();

  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (stamps && tid == 0) stamps[1] = cl_gtime();

  int e_r[CL_PRO_NCH], rank_r[CL_PRO_NCH];
  float g_r[CL_PRO_NCH];
#pragma unroll
  for (int c = 0; c < CL_PRO_NCH; ++c) {
    e_r[c] = -1;
    g_r[c] = 0.f;
    rank_r[c] = 0;
    if (c < nch) {
      const int idx = c * CL_PRO_THREADS + tid;
      int e = -1;
      if (idx < npairs) {
        e = __ldg(topk_ids + idx) - first;
        if ((unsigned)e >= (unsigned)count) e = -1;
        else g_r[c] = __ldg(topk_w + idx);
      }
      const unsigned m = __match_any_sync(FULL, e);
      rank_r[c] = __popc(m & ((1u << lane) - 1u));
      if (e >= 0 && rank_r[c] == 0) wbase[c][warp][e] = __popc(m);
      e_r[c] = e;
    }
  }
  __syncthreads();

  if (warp == 0) {
    const int e = lane;
    int run = 0;
    for (int c = 0; c < nch; ++c)
#pragma unroll
      for (int w = 0; w < CL_PRO_WARPS; ++w) {
        const int v = wbase[c][w][e];
        wbase[c][w][e] = run;
        run += v;
      }
    const int cnt = run;
    int inc = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(FULL, inc, o);
      if (lane >= o) inc += v;
    }
    const int total = __shfl_sync(FULL, inc, 31);
    const int off = inc - cnt;
    eoff[e] = off;
    const bool routed = cnt > 0;
    const unsigned rm = __ballot_sync(FULL, routed);
    const int d = __popc(rm);
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(FULL, cnt, j);
      if (((rm >> j) & 1u) && (cj > cnt || (cj == cnt && j < lane))) ++rank;
    }
    int S = 0;
    if (routed) {
      if (policy == CL_POLICY_S1) S = 1;
      else if (policy == CL_POLICY_S2) S = 2;
      else {
        const int base = min(CL_N_CHUNKS, max(1, target / d));
        const int extra = (base * d < target && base < CL_N_CHUNKS) ? target - base * d : 0;
        S = base + (rank < extra ? 1 : 0);
      }
    }
    int sinc = S;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(FULL, sinc, o);
      if (lane >= o) sinc += v;
    }
    const int nsl = __shfl_sync(FULL, sinc, 31);
    const bool ovf = total > CL_MAX_PAIRS || nsl > CL_MAX_SLICES;
    if (lane == 0) {
      overflow = ovf;
      s.n_slices = ovf ? -1 : nsl;
    }
    if (!ovf) {
      s.expert_nslices[e] = S;
      if (S > 0) {
        const int q = CL_N_CHUNKS / S, rem = CL_N_CHUNKS % S;
        for (int j = 0, i = sinc - S; j < S; ++j, ++i) {
          s.slice_expert[i] = e;
          s.slice_col0[i] = CL_COL_CHUNK * (j * q + min(j, rem));
          s.slice_ncol[i] = CL_COL_CHUNK * (q + (j < rem ? 1 : 0));
          s.slice_tok0[i] = off;
          s.slice_ntok[i] = cnt;
        }
      }
    }
    if (lane == 0) s.next_slice = n_clusters;
  }
  __syncthreads();

  if (!overflow) {
#pragma unroll
    for (int c = 0; c < CL_PRO_NCH; ++c) {
      const int e = e_r[c];
      if (c < nch && e >= 0) {
        const int pos = eoff[e] + wbase[c][warp][e] + rank_r[c];
        s.tok_list[pos] = (c * CL_PRO_THREADS + tid) >> 3;
        s.tok_gate[pos] = g_r[c];
      }
    }
  }
  __syncthreads();

  const int2* src = reinterpret_cast<const int2*>(&s);
  int2* dst = reinterpret_cast<int2*>(out);
  for (int i = tid; i < CL_SCHED_WORDS / 2; i += CL_PRO_THREADS) dst[i] = src[i];
  if (stamps) {
    __syncthreads();
    if (tid == 0) stamps[2] = cl_gtime();
  }
}

// Bench producer: triggers its dependents at entry, spins ns nanoseconds, then publishes the routing the
// prologue reads. stamps[3] is taken after the copy.
__global__ void cl_dummy_producer(const int* __restrict__ src_ids, const float* __restrict__ src_w,
                                  int* __restrict__ dst_ids, float* __restrict__ dst_w, int n, long long ns,
                                  unsigned long long* __restrict__ stamps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const unsigned long long t0 = cl_gtime();
  while ((long long)(cl_gtime() - t0) < ns) {
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    dst_ids[i] = src_ids[i];
    dst_w[i] = src_w[i];
  }
  if (stamps) {
    __syncthreads();
    if (threadIdx.x == 0) stamps[3] = cl_gtime();
  }
}

__global__ void cl_empty_kernel() {}

static unsigned long long* opt_stamps(const torch::Tensor& t) {
  return t.numel() ? reinterpret_cast<unsigned long long*>(t.data_ptr<int64_t>()) : nullptr;
}

void cl_prologue(torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, int64_t first, int64_t count,
                 int64_t policy, int64_t target, int64_t n_clusters, bool pdl, torch::Tensor stamps) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 && topk_ids.is_contiguous());
  TORCH_CHECK(topk_w.is_cuda() && topk_w.scalar_type() == torch::kFloat32 && topk_w.is_contiguous());
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(1) == 8 && topk_w.sizes() == topk_ids.sizes());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.numel() == CL_SCHED_WORDS);
  TORCH_CHECK(topk_ids.size(0) <= CL_PRO_MAX_T, "T above CL_PRO_MAX_T");
  TORCH_CHECK(count > 0 && count <= CL_MAX_EXPERTS);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(CL_PRO_THREADS);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, cl_prologue_kernel, (const int*)topk_ids.data_ptr<int>(),
                                    (const float*)topk_w.data_ptr<float>(), (int)topk_ids.size(0), (int)first,
                                    (int)count, (int)policy, (int)target, (int)n_clusters,
                                    reinterpret_cast<ClSchedule*>(out.data_ptr<int>()), opt_stamps(stamps)));
}

void cl_producer(torch::Tensor src_ids, torch::Tensor src_w, torch::Tensor dst_ids, torch::Tensor dst_w, int64_t ns,
                 torch::Tensor stamps) {
  cl_dummy_producer<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      src_ids.data_ptr<int>(), src_w.data_ptr<float>(), dst_ids.data_ptr<int>(), dst_w.data_ptr<float>(),
      (int)src_ids.numel(), (long long)ns, opt_stamps(stamps));
  C10_CUDA_CHECK(cudaGetLastError());
}

void cl_empty() {
  cl_empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_CHECK(cudaGetLastError());
}
