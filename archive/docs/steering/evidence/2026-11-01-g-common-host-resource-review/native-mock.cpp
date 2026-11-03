#include <vector>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
namespace okc {constexpr int NTL=288;template<bool B>void moe_fused(){};}
static size_t okc_smem_bytes(){return 155520;}

constexpr int cudaSuccess=0,cudaFuncAttributeMaxDynamicSharedMemorySize=1;
struct cudaDeviceProp{int multiProcessorCount=132,cooperativeLaunch=1;};
struct cudaFuncAttributes{int numRegs=147;size_t localSizeBytes=8,sharedSizeBytes=0;int maxThreadsPerBlock=1024;};
int fail=0,calls=0;
int cudaGetDevice(int*p){++calls;*p=0;return fail==1;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){++calls;*p={};return fail==2;}
template<class K>int cudaFuncSetAttribute(K,int,int size){++calls;if(size!=155520)return 99;return fail==3;}
template<class K>int cudaFuncGetAttributes(cudaFuncAttributes*a,K){++calls;*a={};return fail==4;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*p,K,int nt,size_t sm){++calls;*p=2;if(nt!=288||sm!=155520)return 99;return fail==5;}

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

#include <iostream>
int main(){auto r=qual_native_resources();if(r.size()!=20||calls!=8)return 1;
for(int i:{0,10})if(r[i]!=147||r[i+1]!=8||r[i+2]!=0||r[i+3]!=155520||r[i+4]!=288||r[i+5]!=1024||r[i+6]!=2||r[i+7]!=132||r[i+8]!=1||r[i+9]!=264)return 2;
for(int f=1;f<=5;++f){fail=f;bool caught=false;try{qual_native_resources();}catch(...){caught=true;}if(!caught)return 10+f;}
std::cout<<"PASS two10fieldrows nonzero-local-preserved fivefatalAPIs\n";
}
