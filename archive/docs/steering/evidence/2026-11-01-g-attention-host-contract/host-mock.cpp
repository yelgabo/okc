
#include <stdexcept>
#include <iostream>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
using cudaError_t=int;const int cudaSuccess=0,cudaFuncAttributeMaxDynamicSharedMemorySize=1;
struct cudaDeviceProp{int major=9,minor=0,cooperativeLaunch=1,multiProcessorCount=132;};
struct cudaFuncAttributes{int maxThreadsPerBlock=1024;};
struct Params{int flat=0,coop=1;};template<int L,bool F>struct Cfg{static constexpr int NTX=F?192:128;};
using KernFn=void(*)(Params);void actual(Params){};
int fail=0,calls=0,fallback=0,occupancy=2,wrong=0,shared_seen=-1;unsigned blocks_seen=0;void* qexpected=nullptr;
cudaDeviceProp properties;
int cudaGetDevice(int*d){*d=0;return fail==1?1:0;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){*p=properties;return fail==2?1:0;}
int cudaFuncSetAttribute(const void* k,int,int sm){if(k!=(const void*)actual)wrong++;shared_seen=sm;return fail==3?1:0;}
int cudaFuncGetAttributes(cudaFuncAttributes*a,const void* k){if(k!=(const void*)actual)wrong++;a->maxThreadsPerBlock=fail==4?1:1024;return fail==5?1:0;}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*r,KernFn k,int,int){if(k!=actual)wrong++;*r=occupancy;return fail==6?1:0;}
int cudaLaunchCooperativeKernel(const void*k,dim3 g,dim3 b,void**args,int sm,int stream){calls++;blocks_seen=g.x*g.y*g.z;if(k!=(const void*)actual||args[0]!=qexpected||b.y!=1||b.z!=1||sm!=shared_seen||stream!=17)wrong++;return fail==7?1:0;}
const char*cudaGetErrorString(int){return "mock";}
#define OKC_FLATCOOP1C 0

template<int L>void launch(Params p,Params&q,bool cl,bool pdl,dim3 grid,int sm){KernFn kern=actual;int stream=17;q=p;
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

fallback++;
}
void reset(){fail=0;calls=0;fallback=0;occupancy=2;wrong=0;properties={};}
int main(){Params p,q;qexpected=&q;
reset();launch<8>(p,q,false,false,dim3(132),107104);if(calls!=1||fallback||wrong||blocks_seen!=132)return 1;
reset();p.flat=1;launch<8>(p,q,false,false,dim3(264),107104);if(calls!=1||wrong)return 2;
reset();launch<1>(p,q,false,false,dim3(264),1000);if(calls||fallback!=1)return 3;
p.flat=0;p.coop=0;reset();launch<8>(p,q,false,false,dim3(132),1000);if(calls||fallback!=1)return 4;
p.coop=1;
for(int f=1;f<=7;f++){reset();fail=f;bool rejected=false;try{launch<8>(p,q,false,false,dim3(132),1000);}catch(...){rejected=true;}if(!rejected)return 10+f;if(f!=7&&calls)return 20+f;}
for(int f=0;f<6;f++){reset();bool pd=false;dim3 grid(132);if(f==0)properties.cooperativeLaunch=0;if(f==1)properties.major=8;if(f==2)occupancy=0;if(f==3)grid=dim3(265);if(f==4)grid=dim3(0);if(f==5)pd=true;bool rejected=false;try{launch<8>(p,q,false,pd,grid,1000);}catch(...){rejected=true;}if(!rejected||calls)return 30+f;}
std::cout<<"HOST_MOCK_PASS 2cooperative 2ordinary 13rejects\n";
}
