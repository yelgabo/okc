// Does one launch carrying both cudaLaunchAttributeCooperative and
// cudaLaunchAttributeProgrammaticStreamSerialization work on this box, eager and under graph capture?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA_ERR %s at %s:%d\n", cudaGetErrorString(e_), #x, __LINE__); fflush(stdout); _exit(2); } } while (0)

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
extern __shared__ char dsm[];

__global__ void prim(int* data, unsigned long long* tend, int base_ns, int step_ns, int magic) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned long long t0 = gtime(), lim = (unsigned long long)base_ns + (unsigned long long)step_ns * blockIdx.x;
    while (gtime() - t0 < lim) {}
    data[blockIdx.x] = magic;
    __threadfence();
    atomicMax(tend, gtime());
  }
}

// stats: 0 invalid grid, 1 data mismatches, 2 barrier errors, 3 CTAs that ran
__global__ void sec(const int* data, int nprim, int magic, unsigned long long* tstart, int* stats, unsigned* bar, int rounds) {
  if (threadIdx.x == 0) { dsm[0] = 1; atomicMin(tstart, gtime()); }
  cg::grid_group g = cg::this_grid();
  const bool valid = g.is_valid();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    atomicAdd(&stats[3], 1);
    if (!valid) atomicAdd(&stats[0], 1);
    for (int i = 0; i < nprim; i++) if (((volatile const int*)data)[i] != magic) atomicAdd(&stats[1], 1);
  }
  if (!valid) return;
  for (int r = 0; r < rounds; r++) {
    g.sync();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned target = (unsigned)(r + 1) * gridDim.x;
      __threadfence();
      atomicAdd(bar, 1u);
      while (atomicAdd(bar, 0u) < target) {}
      __threadfence();
      if (atomicAdd(bar, 0u) > (unsigned)rounds * gridDim.x) atomicAdd(&stats[2], 1);
    }
    __syncthreads();
  }
}

struct Buf { int* data; unsigned long long* tend; unsigned long long* tstart; int* stats; unsigned* bar; };

static void wait_or_die(cudaStream_t st, const char* what) {
  cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)); CK(cudaEventRecord(ev, st));
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    cudaError_t q = cudaEventQuery(ev);
    if (q == cudaSuccess) break;
    if (q != cudaErrorNotReady) { printf("CUDA_ERR %s during %s\n", cudaGetErrorString(q), what); fflush(stdout); _exit(2); }
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) { printf("HANG during %s\n", what); fflush(stdout); _exit(3); }
    usleep(100);
  }
  CK(cudaEventDestroy(ev));
}

// mode 0: cudaLaunchCooperativeKernel (the base file's path); 1: Ex cooperative+programmatic;
// 2: Ex programmatic only (must report invalid grid, proving the check sees a non-cooperative launch); 3: Ex cooperative only
static cudaError_t enqueue(int mode, cudaStream_t st, const Buf& b, int pg, int psm, int sg, int ssm, int nt, int magic, int rounds) {
  CK(cudaMemsetAsync(b.data, 0, 4096 * sizeof(int), st));
  CK(cudaMemsetAsync(b.tend, 0, 8, st));
  CK(cudaMemsetAsync(b.tstart, 0xff, 8, st));
  CK(cudaMemsetAsync(b.stats, 0, 4 * sizeof(int), st));
  CK(cudaMemsetAsync(b.bar, 0, 4, st));
  prim<<<pg, 128, psm, st>>>(b.data, b.tend, 20000, 300, magic);
  cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return e;
  if (mode == 0) {
    int np = pg; void* args[] = {(void*)&b.data, &np, &magic, (void*)&b.tstart, (void*)&b.stats, (void*)&b.bar, &rounds};
    return cudaLaunchCooperativeKernel((const void*)sec, dim3(sg), dim3(nt), args, ssm, st);
  }
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(sg); cfg.blockDim = dim3(nt); cfg.dynamicSmemBytes = ssm; cfg.stream = st;
  cudaLaunchAttribute at[2]; int na = 0;
  if (mode == 1 || mode == 3) { at[na].id = cudaLaunchAttributeCooperative; at[na].val.cooperative = 1; na++; }
  if (mode == 1 || mode == 2) { at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[na].val.programmaticStreamSerializationAllowed = 1; na++; }
  cfg.attrs = at; cfg.numAttrs = na;
  return cudaLaunchKernelEx(&cfg, sec, (const int*)b.data, pg, magic, b.tstart, b.stats, b.bar, rounds);
}

static int probe(const Buf& b, cudaStream_t st, int SM, int SSM, int NT) {
  // 1) a cooperative launch one CTA larger than co-residency allows must be refused, with and without the PDL attribute
  for (int mode = 1; mode <= 3; mode += 2) {
    cudaError_t e = enqueue(mode, st, b, 8, 0, SM + 1, SSM, NT, 7, 1);
    printf("PROBE oversize grid %d mode=%d launch=%s\n", SM + 1, mode, cudaGetErrorString(e));
    cudaGetLastError(); CK(cudaStreamSynchronize(st));
  }
  // 2) captured graph: edge type into the secondary node and its cooperative attribute
  cudaGraph_t gr; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  CK(enqueue(1, st, b, 8, 0, SM, SSM, NT, 7, 1));
  CK(cudaStreamEndCapture(st, &gr));
  size_t ne = 0; CK(cudaGraphGetEdges(gr, nullptr, nullptr, nullptr, &ne));
  cudaGraphNode_t fr[64], to[64]; cudaGraphEdgeData ed[64]; CK(cudaGraphGetEdges(gr, fr, to, ed, &ne));
  for (size_t i = 0; i < ne; i++) {
    cudaGraphNodeType tf, tt; cudaGraphNodeGetType(fr[i], &tf); cudaGraphNodeGetType(to[i], &tt);
    printf("PROBE edge %zu from_type=%d to_type=%d edge_type=%d from_port=%d to_port=%d\n", i, (int)tf, (int)tt,
           (int)ed[i].type, (int)ed[i].from_port, (int)ed[i].to_port);
    if (tt == cudaGraphNodeTypeKernel && tf == cudaGraphNodeTypeKernel) {
      cudaLaunchAttributeValue v; memset(&v, 0, sizeof v);
      cudaError_t e = cudaGraphKernelNodeGetAttribute(to[i], cudaLaunchAttributeCooperative, &v);
      printf("PROBE secondary node cooperative=%d (%s)\n", v.cooperative, cudaGetErrorString(e));
    }
  }
  printf("PROBE edge types: default=%d programmatic=%d ports: programmatic=1 launch_completion=2\n", (int)cudaGraphDependencyTypeDefault,
         (int)cudaGraphDependencyTypeProgrammatic);
  return 0;
}

int main(int argc, char** argv) {
  int dev = 0; CK(cudaSetDevice(dev)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("device %s sm_%d%d SMs %d coopLaunch %d driver %d runtime %d smemOptin %zu\n", p.name, p.major, p.minor,
         p.multiProcessorCount, p.cooperativeLaunch, drv, rt, p.sharedMemPerBlockOptin);
  const int SM = p.multiProcessorCount, NT = 384, SSM = 200 * 1024, ROUNDS = 6, ITERS = 40;
  CK(cudaFuncSetAttribute(sec, cudaFuncAttributeMaxDynamicSharedMemorySize, SSM));
  CK(cudaFuncSetAttribute(prim, cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sec, NT, SSM));
  printf("sec occupancy per SM %d (grid %d)\n", occ, SM);
  Buf b; CK(cudaMalloc(&b.data, 4096 * sizeof(int))); CK(cudaMalloc(&b.tend, 8)); CK(cudaMalloc(&b.tstart, 8));
  CK(cudaMalloc(&b.stats, 16)); CK(cudaMalloc(&b.bar, 4));
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (argc > 1) return probe(b, st, SM, SSM, NT);
  struct Cfg { int pg, psm; const char* name; } cfgs[] = {{8, 0, "route-like 8 CTAs no smem"}, {132, 120 * 1024, "132 CTAs 120KB"}, {264, 120 * 1024, "264 CTAs 120KB (2 waves)"}};
  for (auto& c : cfgs) for (int mode = 0; mode < 4; mode++) for (int graph = 0; graph < 2; graph++) {
    cudaGraphExec_t ge = nullptr; cudaError_t le = cudaSuccess;
    if (graph) {
      cudaGraph_t gr; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      le = enqueue(mode, st, b, c.pg, c.psm, SM, SSM, NT, 7, ROUNDS);
      cudaError_t ce = cudaStreamEndCapture(st, &gr);
      if (le == cudaSuccess && ce != cudaSuccess) le = ce;
      if (le == cudaSuccess) { le = cudaGraphInstantiate(&ge, gr, 0); }
      cudaGetLastError();
    }
    int inval = 0, mism = 0, berr = 0, ran = 0, overl = 0; double lead_sum = 0, lead_max = -1e30;
    for (int it = 0; it < ITERS && le == cudaSuccess; it++) {
      if (graph) le = cudaGraphLaunch(ge, st); else le = enqueue(mode, st, b, c.pg, c.psm, SM, SSM, NT, 7, ROUNDS);
      if (le != cudaSuccess) break;
      wait_or_die(st, c.name);
      int s[4]; unsigned long long te, ts;
      CK(cudaMemcpy(s, b.stats, 16, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&te, b.tend, 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&ts, b.tstart, 8, cudaMemcpyDeviceToHost));
      inval += s[0]; mism += s[1]; berr += s[2]; ran += s[3];
      double lead = ((double)te - (double)ts) / 1000.0;
      if (ts < te) overl++;
      lead_sum += lead; if (lead > lead_max) lead_max = lead;
    }
    printf("RESULT cfg='%s' mode=%d graph=%d launch=%s iters=%d ctas_ran=%d invalid_grid=%d data_mismatch=%d barrier_err=%d "
           "overlapped_iters=%d sec_start_before_prim_end_us mean=%.2f max=%.2f\n",
           c.name, mode, graph, cudaGetErrorString(le), ITERS, ran, inval, mism, berr, overl, lead_sum / ITERS, lead_max);
    fflush(stdout);
    if (ge) cudaGraphExecDestroy(ge);
    cudaGetLastError();
  }
  printf("ALL_DONE\n");
  return 0;
}
