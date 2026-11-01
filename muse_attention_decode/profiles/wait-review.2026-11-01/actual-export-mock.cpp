#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
using cudaError_t=int;constexpr int cudaSuccess=0;bool fail=false;
struct cudaFuncAttributes {int numRegs=80;int localSizeBytes=0;};
template<int L,int M>void attn_one_kernel(){}
template<int L,int M>struct Instance {static constexpr int BYTES=107104;static constexpr int NT=384;static inline int per_sm=2;static int prepare(){return 264;}};
template<class T>int cudaFuncGetAttributes(cudaFuncAttributes* a,T){*a={};return fail?1:0;}
#define TORCH_CHECK(x,msg) do {if(!(x))throw std::runtime_error(msg);}while(0)
#define ONE_INSTANCES(X) X(1,49) X(8,51)
std::vector<int64_t> attn_wait_resources(){
 std::vector<int64_t> out;
#define WR(L,M) {Instance<L,M>::prepare();cudaFuncAttributes a{};TORCH_CHECK(cudaFuncGetAttributes(&a,attn_one_kernel<L,M>)==cudaSuccess,"wait attrs"); for(int64_t v:{int64_t(L),int64_t(M),int64_t(a.numRegs),int64_t(a.localSizeBytes),int64_t(Instance<L,M>::BYTES),int64_t(Instance<L,M>::NT),int64_t(Instance<L,M>::per_sm)})out.push_back(v);}
 ONE_INSTANCES(WR)
#undef WR
 return out;
}

int main(){auto r=attn_wait_resources();if(r.size()!=14||r[0]!=1||r[1]!=49||r[7]!=8||r[8]!=51||r[2]!=80||r[3]!=0||r[4]!=107104||r[5]!=384||r[6]!=2)return 1;fail=true;try{attn_wait_resources();return 2;}catch(const std::runtime_error&){}std::cout<<"WAIT_EXPORT_ACTUAL_CPP_PASS 2rows7fields CUDAerrorfatal\n";}
