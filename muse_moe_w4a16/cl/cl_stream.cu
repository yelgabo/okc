// CL-0.1 weight stream skeleton (docs/plans/2026-11-04-moe-cluster-rewrite.md, task 0.1).
// One thread-block cluster streams one expert slice's units (36,864 B each, the helper0
// packing: 512 w13 units then 256 w2 units per expert) through a cp.async.bulk ring into
// shared memory; consumer warps XOR every word of each unit and release the slot. Units are
// claimed from a counter in the leader CTA's shared memory over DSMEM (mode 0) or assigned
// round-robin by CTA rank (mode 1). With n_clusters > 0 a fixed number of clusters claims
// slices from a global counter (the plan's D2 dynamic slice claiming). Built as a torch extension by cl_stream_bench.py.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

constexpr int kUnitBytes = 36864;
constexpr int kW13Units = 512;
constexpr int kW2Units = 256;
constexpr int kMaxDepth = 8;

struct StreamParams {
  const uint8_t* w13;          // [32, 512, 36864]
  const uint8_t* w2;           // [32, 256, 36864]
  const int* slice_expert;     // [n_slices] local expert of each cluster
  const int* slice_u0;         // [n_slices] first unit in the expert's 768-unit space
  const int* slice_u1;         // [n_slices] one past the last unit
  unsigned long long* stamps;  // [grid, 4]: entry, first unit ready, consumers done, units taken | smid << 32
  unsigned* sink;              // [grid * blockDim]
  int* slice_ctr;              // global slice claim counter, zero on entry (persist mode)
  int n_slices;
  int persist;                 // 1: launched clusters claim slices from slice_ctr until none are left
  int depth;
  int mode;
  int evict_first;
  int consume;                 // 0: one word per unit is XORed (the task's discard); 1: every word
};

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  unsigned ok;
  do {
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  } while (!ok);
}

__device__ __forceinline__ void bulk_load(void* dst, const void* src, unsigned bytes, uint64_t* bar,
                                          uint64_t policy, bool hint) {
  if (hint)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1], %2, [%3], %4;"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
  else
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

template <int C>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(512, 1) cl_stream_kernel(StreamParams p) {
  extern __shared__ __align__(1024) uint8_t ring[];
  __shared__ __align__(8) uint64_t full[kMaxDepth];
  __shared__ __align__(8) uint64_t empty[kMaxDepth];
  __shared__ int slot_unit[kMaxDepth];
  __shared__ unsigned claim_ctr;
  __shared__ int cur_slice;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n_cons_warps = (blockDim.x >> 5) - 1;
  const int D = p.depth;
  unsigned long long* st = p.stamps + 4ull * blockIdx.x;
  const int* leader_slice = cluster.map_shared_rank(&cur_slice, 0);

  if (threadIdx.x == 0) {
    st[0] = gtimer();
    for (int s = 0; s < D; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], n_cons_warps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }

  // Ring iteration counters run on across slices so the mbarrier parities stay in step.
  int it = 0;
  unsigned long long taken = 0;
  uint4 acc = make_uint4(0, 0, 0, 0);
  bool first_ready = true;
  for (int round = 0;; ++round) {
    if (rank == 0 && threadIdx.x == 0) {
      int sl;
      if (p.persist) sl = atomicAdd(p.slice_ctr, 1);
      else sl = round == 0 ? static_cast<int>(blockIdx.x / C) : p.n_slices;
      cur_slice = sl;
      if (sl < p.n_slices) claim_ctr = p.slice_u0[sl];
    }
    // The leader's counter and slice id must be set before any CTA reads or claims from them
    // and must not change until every CTA has stopped claiming (the sync at the loop's end).
    cluster.sync();
    const int slice = *leader_slice;
    if (slice >= p.n_slices) break;
    const int expert = p.slice_expert[slice];
    const int u0 = p.slice_u0[slice], u1 = p.slice_u1[slice];

    if (warp == 0) {
      if (lane == 0) {
        unsigned* leader_ctr = cluster.map_shared_rank(&claim_ctr, 0);
        uint64_t pol = 0;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
        int next_static = u0 + rank;
        for (;; ++it) {
          int u;
          if (p.mode == 0) {
            u = static_cast<int>(atomicAdd(leader_ctr, 1u));
          } else {
            u = next_static;
            next_static += C;
          }
          const int s = it % D;
          if (it >= D) mbar_wait(&empty[s], ((it / D) - 1) & 1);
          if (u >= u1) {
            slot_unit[s] = -1;
            mbar_arrive(&full[s]);
            ++it;
            break;
          }
          slot_unit[s] = u;
          const uint8_t* src = u < kW13Units
              ? p.w13 + (static_cast<size_t>(expert) * kW13Units + u) * kUnitBytes
              : p.w2 + (static_cast<size_t>(expert) * kW2Units + (u - kW13Units)) * kUnitBytes;
          mbar_expect_tx(&full[s], kUnitBytes);
          bulk_load(ring + static_cast<size_t>(s) * kUnitBytes, src, kUnitBytes, &full[s], pol,
                    p.evict_first != 0);
          ++taken;
        }
      }
      // cluster.sync() is barrier.cluster.*.aligned: the whole warp must reach it together.
      __syncwarp();
    } else {
      const int ct = threadIdx.x - 32;
      const int nct = n_cons_warps * 32;
      for (;; ++it) {
        const int s = it % D;
        mbar_wait(&full[s], (it / D) & 1);
        if (first_ready && ct == 0) st[1] = gtimer();
        first_ready = false;
        if (slot_unit[s] < 0) {
          ++it;
          break;
        }
        const uint4* v = reinterpret_cast<const uint4*>(ring + static_cast<size_t>(s) * kUnitBytes);
        if (p.consume) {
          // Unrolled so the smem reads overlap; rolled, this loop took about 0.8 us per unit and
          // throttled the ring to 46 GB/s per SM.
#pragma unroll 6
          for (int j = ct; j < kUnitBytes / 16; j += nct) {
            const uint4 w = v[j];
            acc.x ^= w.x; acc.y ^= w.y; acc.z ^= w.z; acc.w ^= w.w;
          }
        } else if (ct == 0) {
          acc.x ^= reinterpret_cast<const unsigned*>(v)[(slot_unit[s] * 977) % (kUnitBytes / 4)];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
      }
    }
    cluster.sync();
  }
  if (warp == 0 && lane == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    st[3] = taken | (static_cast<unsigned long long>(smid) << 32);
  }
  if (warp != 0) {
    p.sink[blockIdx.x * blockDim.x + threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (threadIdx.x == 32) st[2] = gtimer();
  }
  // No CTA may exit while another can still read its cur_slice over DSMEM.
  cluster.sync();
}

template <int C>
static void set_attrs(int smem) {
  auto fn = cl_stream_kernel<C>;
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
  if (C > 8)
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess);
}

template <int C>
static void launch_c(const StreamParams& p, int n_clusters, int threads, int smem, cudaStream_t stream) {
  set_attrs<C>(smem);
  cl_stream_kernel<C><<<n_clusters * C, threads, smem, stream>>>(p);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "cl_stream launch: ", cudaGetErrorString(e));
}

template <int C>
static int occupancy_c(int threads, int smem, int grid_clusters) {
  set_attrs<C>(smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_clusters * C);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = C;
  a[0].val.clusterDim.y = 1;
  a[0].val.clusterDim.z = 1;
  cfg.attrs = a;
  cfg.numAttrs = 1;
  int n = -1;
  const cudaError_t e = cudaOccupancyMaxActiveClusters(&n, cl_stream_kernel<C>, &cfg);
  TORCH_CHECK(e == cudaSuccess, "cudaOccupancyMaxActiveClusters: ", cudaGetErrorString(e));
  return n;
}

void launch(torch::Tensor w13, torch::Tensor w2, torch::Tensor slice_expert, torch::Tensor slice_u0,
            torch::Tensor slice_u1, torch::Tensor stamps, torch::Tensor sink, torch::Tensor slice_ctr,
            int64_t n_clusters, int64_t C, int64_t depth, int64_t cons_warps, int64_t mode, int64_t evict_first,
            int64_t smem_pad, int64_t consume) {
  TORCH_CHECK(depth >= 1 && depth <= kMaxDepth);
  StreamParams p;
  p.w13 = w13.data_ptr<uint8_t>();
  p.w2 = w2.data_ptr<uint8_t>();
  p.slice_expert = slice_expert.data_ptr<int>();
  p.slice_u0 = slice_u0.data_ptr<int>();
  p.slice_u1 = slice_u1.data_ptr<int>();
  p.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  p.sink = reinterpret_cast<unsigned*>(sink.data_ptr<int>());
  p.depth = static_cast<int>(depth);
  p.mode = static_cast<int>(mode);
  p.evict_first = static_cast<int>(evict_first);
  p.consume = static_cast<int>(consume);
  const int n_slices = static_cast<int>(slice_expert.size(0));
  p.slice_ctr = slice_ctr.data_ptr<int>();
  p.n_slices = n_slices;
  p.persist = n_clusters > 0 ? 1 : 0;
  const int grid_clusters = n_clusters > 0 ? static_cast<int>(n_clusters) : n_slices;
  const int threads = 32 * static_cast<int>(1 + cons_warps);
  const int smem = static_cast<int>(depth * kUnitBytes + smem_pad);
  TORCH_CHECK(stamps.numel() >= 4ll * grid_clusters * C && sink.numel() >= 1ll * grid_clusters * C * threads);
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  switch (C) {
    case 1: launch_c<1>(p, grid_clusters, threads, smem, s); break;
    case 2: launch_c<2>(p, grid_clusters, threads, smem, s); break;
    case 4: launch_c<4>(p, grid_clusters, threads, smem, s); break;
    case 8: launch_c<8>(p, grid_clusters, threads, smem, s); break;
    case 16: launch_c<16>(p, grid_clusters, threads, smem, s); break;
    default: TORCH_CHECK(false, "C must be 1, 2, 4, 8 or 16");
  }
}

int64_t occupancy(int64_t C, int64_t threads, int64_t smem, int64_t grid_clusters) {
  switch (C) {
    case 1: return occupancy_c<1>(threads, smem, grid_clusters);
    case 2: return occupancy_c<2>(threads, smem, grid_clusters);
    case 4: return occupancy_c<4>(threads, smem, grid_clusters);
    case 8: return occupancy_c<8>(threads, smem, grid_clusters);
    case 16: return occupancy_c<16>(threads, smem, grid_clusters);
  }
  TORCH_CHECK(false, "C must be 1, 2, 4, 8 or 16");
  return -1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
  m.def("occupancy", &occupancy);
  m.attr("UNIT_BYTES") = kUnitBytes;
}
