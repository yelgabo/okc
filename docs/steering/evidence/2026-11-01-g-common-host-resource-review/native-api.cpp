#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
namespace okc {constexpr int NTL=288;template<bool B>void moe_fused(){};}
static size_t okc_smem_bytes(){return 155520;}

std::vector<int64_t> qual_native_resources() {
  std::vector<int64_t> out;const size_t shared=okc_smem_bytes();
  int dev=0;cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDevice(&dev)==cudaSuccess && cudaGetDeviceProperties(&prop,dev)==cudaSuccess,
      "GQUAL expert properties");
  for(int big=0;big<2;++big) {
    auto kernel=big?okc::moe_fused<true>:okc::moe_fused<false>;
    cudaFuncAttributes a{};int resident=0;
    TORCH_CHECK(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared)==cudaSuccess &&
        cudaFuncGetAttributes(&a,(const void*)kernel)==cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,shared)==cudaSuccess,
        "GQUAL expert selected resources");
    const int64_t row[]={a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,
        (int64_t)shared,okc::NTL,a.maxThreadsPerBlock,resident,prop.multiProcessorCount,
        prop.cooperativeLaunch,(int64_t)resident*prop.multiProcessorCount};
    out.insert(out.end(),row,row+10);
  }
  return out;
}
