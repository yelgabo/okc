
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
namespace moe1 {struct Params{};struct SmemMain {char a[228736];};constexpr int kNTM=576;template<int MODE>void moe_main_kernel(Params){}}
int prepared=0;
namespace one {struct Params{};template<int L,int MODE>void attn_one_kernel(Params){};template<int L,int MODE>struct Instance {static constexpr int NT=256,BYTES=90704;static int prepare(){++prepared;return 264;}};}
struct cudaFuncAttributes {int numRegs;size_t localSizeBytes,sharedSizeBytes;int maxThreadsPerBlock;};
enum {cudaFuncAttributeMaxDynamicSharedMemorySize,cudaDevAttrMultiProcessorCount,cudaDevAttrCooperativeLaunch};
int fail_api=0,seen_smem=0,seen_threads=0;
template<class K>bool is_moe(K k){return (const void*)k==(const void*)&moe1::moe_main_kernel<51>;}
template<class K>int cudaFuncSetAttribute(K k,int a,int n){seen_smem=n;return fail_api==1?1:0;}
template<class K>int cudaFuncGetAttributes(cudaFuncAttributes*a,K k){*a={is_moe(k)?96:82,0,0,1024};return fail_api==2?1:0;}
int cudaGetDevice(int*p){*p=0;return fail_api==3?1:0;}
int cudaDeviceGetAttribute(int*p,int a,int){if(a==cudaDevAttrMultiProcessorCount){*p=132;return fail_api==4?1:0;}*p=1;return fail_api==5?1:0;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*p,K k,int threads,int bytes){seen_threads=threads;if(bytes!=(is_moe(k)?228736:90704))return 99;*p=is_moe(k)?1:2;return fail_api==6?1:0;}
#define C10_CUDA_CHECK(x) do{if((x)!=0)throw std::runtime_error("CUDA_FATAL");}while(0)

std::vector<int64_t> qual_moe_resources() {
    auto kernel=moe1::moe_main_kernel<51>;
    const int bytes=sizeof(moe1::SmemMain),threads=moe1::kNTM;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,bytes));
    return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,threads,
            a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
}


template<int MODE> void qual_one_append(std::vector<int64_t>& out) {
    using I=one::Instance<8,MODE>;I::prepare();auto kernel=one::attn_one_kernel<8,MODE>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,I::BYTES));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,I::NT,I::BYTES));
    const int64_t row[]={a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,I::BYTES,
                         I::NT,a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
    out.insert(out.end(),row,row+10);
}
std::vector<int64_t> qual_one_resources() {
    std::vector<int64_t> out;qual_one_append<3>(out);qual_one_append<51>(out);return out;
}

int main(){
 auto m=qual_moe_resources();if(m.size()!=10||m[0]!=96||m[3]!=228736||m[4]!=576||m[6]!=1||m[7]!=132||m[8]!=1||m[9]!=132)return 1;
 auto o=qual_one_resources();if(o.size()!=20||prepared!=2)return 2;
 for(int off:{0,10})if(o[off+3]!=90704||o[off+4]!=256||o[off+6]!=2||o[off+7]!=132||o[off+8]!=1||o[off+9]!=264)return 3;
 int fatal=0;
 for(int mode=0;mode<2;++mode)for(int f=1;f<=6;++f){fail_api=f;try{if(mode)qual_one_resources();else qual_moe_resources();return 4;}catch(const std::runtime_error&){++fatal;}}
 if(fatal!=12)return 5;std::cout<<"QUAL_RESOURCE_HOST_PASS3vectors12fatalAPIbranches\n";
}
