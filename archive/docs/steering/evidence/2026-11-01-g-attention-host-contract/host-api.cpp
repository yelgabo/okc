#include <cuda_runtime.h>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
#define OKC_FLATCOOP1C 0
struct Params{int flat,coop;};
template<int L,bool F>struct Cfg{static constexpr int NTX=192;};
using KernFn=void(*)(Params);
template<int L>void checked_launch(Params p,Params q,bool cl,bool pdl,dim3 grid,int sm,cudaStream_t stream,KernFn kern){
  // OFF-prototype common progress repair: capacity alone is not cooperative admission.
  const bool waiting_finalize = q.coop && ((!p.flat && !cl) ||
      (p.flat && (L == 8 || OKC_FLATCOOP1C)));
  if (waiting_finalize) {
    TORCH_CHECK(!pdl && !cl, "ATT7_PROGRESS requires sequential non-cluster launch");
    int device=0, resident=0; cudaDeviceProp prop{}; cudaFuncAttributes attrs{};
    TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess &&
                cudaGetDeviceProperties(&prop,device)==cudaSuccess,
                "ATT7_PROGRESS device properties");
    TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch &&
                prop.multiProcessorCount>0, "ATT7_PROGRESS SM90 cooperative support");
    const int threads=p.flat ? Cfg<L,true>::NTX : Cfg<L,false>::NTX;
    TORCH_CHECK(cudaFuncSetAttribute((const void*)kern,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess &&
                cudaFuncGetAttributes(&attrs,(const void*)kern)==cudaSuccess,
                "ATT7_PROGRESS selected function attributes");
    const unsigned long long blocks=(unsigned long long)grid.x*grid.y*grid.z;
    TORCH_CHECK(grid.x>0 && grid.y>0 && grid.z>0 && threads>0 &&
                threads<=attrs.maxThreadsPerBlock && sm>=0 &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kern,threads,sm)==cudaSuccess &&
                resident>0 && blocks<=(unsigned long long)resident*prop.multiProcessorCount,
                "ATT7_PROGRESS selected function admission");
    void* args[]={&q};
    const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kern,grid,
        dim3((unsigned)threads,1,1),args,sm,stream);
    TORCH_CHECK(launch==cudaSuccess,"ATT7_PROGRESS launch ",cudaGetErrorString(launch));
    return;
  }

}
template void checked_launch<1>(Params,Params,bool,bool,dim3,int,cudaStream_t,KernFn);
template void checked_launch<8>(Params,Params,bool,bool,dim3,int,cudaStream_t,KernFn);
