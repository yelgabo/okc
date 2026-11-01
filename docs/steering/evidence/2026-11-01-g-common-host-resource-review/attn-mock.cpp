
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
using cudaStream_t=int;constexpr int cudaSuccess=0,cudaLaunchAttributeClusterDimension=1;
struct cudaDeviceProp{int major=9,minor=0,cooperativeLaunch=1,multiProcessorCount=132;};
struct cudaFuncAttributes{int numRegs=100,localSizeBytes=0,sharedSizeBytes=0,maxThreadsPerBlock=1024,maxDynamicSharedSizeBytes=232448;};
struct cudaLaunchAttribute{int id;struct{struct{unsigned x,y,z;}clusterDim;}val;};
struct cudaLaunchConfig_t{dim3 gridDim,blockDim;int dynamicSmemBytes;cudaStream_t stream;cudaLaunchAttribute*attrs;int numAttrs;};
struct P{int B=1,flat=0,coop=1,resident=0,unev=0,n_splits=132,cs=1,ncl=1,window=2048;};
using Kernel=void(*)(const P);void actual(const P){};

int calls=0,fail=0;
int cudaDriverGetVersion(int*x){*x=12080;return fail==6;}
int cudaRuntimeGetVersion(int*x){*x=12080;return fail==7;}
int cudaGetDevice(int*x){calls++;*x=0;return fail==1;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){calls++;*p={};return fail==2;}
int cudaFuncGetAttributes(cudaFuncAttributes*a,const void*){calls++;*a={};return fail==3;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*x,K,int,int){calls++;*x=2;return fail==4;}
template<class K>int cudaOccupancyMaxActiveClusters(int*x,K,const cudaLaunchConfig_t*cfg){calls++;*x=14;if(cfg->attrs[0].val.clusterDim.x!=2)return 1;return fail==5;}

// Prepare-only, per-calling-thread witness. Normal inference leaves this disabled.
#include <vector>
static thread_local bool g_qual_att7_enabled=false;
static thread_local std::vector<int64_t> g_qual_att7_rows;
void qual_att7_begin() {
  TORCH_CHECK(!g_qual_att7_enabled,"GQUAL nested attention audit");
  g_qual_att7_rows.clear();g_qual_att7_enabled=true;
}
std::vector<int64_t> qual_att7_end() {
  g_qual_att7_enabled=false;
  std::vector<int64_t> out;out.swap(g_qual_att7_rows);return out;
}
std::vector<int64_t> qual_att7_versions() {
  int driver=0,runtime=0;
  TORCH_CHECK(cudaDriverGetVersion(&driver)==cudaSuccess &&
      cudaRuntimeGetVersion(&runtime)==cudaSuccess,"GQUAL CUDA identity");
#if defined(__CUDACC_VER_MAJOR__)
  return {driver,runtime,__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__};
#else
  return {driver,runtime,0,0,0};  // host-adapter compiler identity unavailable
#endif
}
template<class Kernel,class P>
static void qual_att7_observe(Kernel kern,const P& q,dim3 grid,int threads,int shared,
    int L,bool waiting,bool cluster,bool pdl,cudaStream_t stream,
    const Kernel* kn,const Kernel* kn1,const Kernel* ksp,const Kernel* ksp1) {
  if (!g_qual_att7_enabled) return;  // no CUDA API or write on the normal path
  int tag=q.flat?100:-1;
  if (!q.flat) {
    for(int i=0;i<4;++i)if(kern==kn[i])tag=i;
    for(int i=0;i<4;++i)if(kern==kn1[i])tag=10+i;
    for(int i=0;i<4;++i)if(kern==ksp[i])tag=20+i;
    for(int i=0;i<4;++i)if(kern==ksp1[i])tag=30+i;
  }
  TORCH_CHECK(tag>=0 && !pdl,"GQUAL selected attention policy");
  int device=0,resident=0,clusters=0;cudaDeviceProp prop{};cudaFuncAttributes a{};
  TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess &&
      cudaGetDeviceProperties(&prop,device)==cudaSuccess,"GQUAL attention properties");
  TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch &&
      prop.multiProcessorCount>0,"GQUAL attention SM90 capability");
  TORCH_CHECK(cudaFuncGetAttributes(&a,(const void*)kern)==cudaSuccess &&
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kern,threads,shared)==cudaSuccess,
      "GQUAL selected attention attributes");
  const unsigned long long blocks=(unsigned long long)grid.x*grid.y*grid.z;
  TORCH_CHECK(grid.x && grid.y && grid.z && threads>0 && threads<=a.maxThreadsPerBlock &&
      shared>=0 && shared<=a.maxDynamicSharedSizeBytes && resident>0,
      "GQUAL selected attention resources");
  if(waiting)TORCH_CHECK(!cluster && blocks<=(unsigned long long)resident*prop.multiProcessorCount,
      "GQUAL attention cooperative admission");
  if(cluster) {
    cudaLaunchConfig_t cfg{};cfg.gridDim=grid;cfg.blockDim=dim3((unsigned)threads,1,1);
    cfg.dynamicSmemBytes=shared;cfg.stream=stream;
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x=q.cs;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
    cfg.attrs=&attr;cfg.numAttrs=1;
    TORCH_CHECK(q.cs>1 && grid.x%(unsigned)q.cs==0 &&
        cudaOccupancyMaxActiveClusters(&clusters,kern,&cfg)==cudaSuccess && clusters>0,
        "GQUAL attention cluster admission");
  }
  const int64_t row[]={1,L,q.B,q.flat,q.coop,q.resident,q.unev,q.n_splits,q.cs,q.ncl,q.window,
      grid.x,grid.y,grid.z,threads,shared,a.numRegs,(int64_t)a.localSizeBytes,
      (int64_t)a.sharedSizeBytes,a.maxThreadsPerBlock,a.maxDynamicSharedSizeBytes,
      resident,prop.multiProcessorCount,prop.cooperativeLaunch,waiting?1:cluster?2:0,tag,clusters};
  g_qual_att7_rows.insert(g_qual_att7_rows.end(),row,row+27);
}

int main(){P q;Kernel kn[4]={nullptr,nullptr,nullptr,actual},other[4]={};
qual_att7_observe(actual,q,dim3(132),128,1000,1,true,false,false,17,kn,other,other,other);
if(calls||!qual_att7_end().empty())return 1;
qual_att7_begin();qual_att7_observe(actual,q,dim3(132),128,1000,1,true,false,false,17,kn,other,other,other);
auto row=qual_att7_end();if(row.size()!=27||row[24]!=1||row[25]!=3||row[21]!=2||g_qual_att7_enabled)return 2;
q.cs=2;qual_att7_begin();qual_att7_observe(actual,q,dim3(132),128,1000,8,false,true,false,17,kn,other,other,other);
row=qual_att7_end();if(row.size()!=27||row[24]!=2||row[26]!=14)return 3;
for(int f=1;f<=5;++f){fail=f;qual_att7_begin();bool caught=false;try{
qual_att7_observe(actual,q,dim3(132),128,1000,8,false,true,false,17,kn,other,other,other);
}catch(...){caught=true;}if(!caught||!qual_att7_end().empty()||g_qual_att7_enabled)return 10+f;}
fail=0;qual_att7_begin();bool caught=false;try{qual_att7_begin();}catch(...){caught=true;}
if(!caught)return 20;qual_att7_end();
qual_att7_begin();caught=false;try{
qual_att7_observe(actual,q,dim3(265),128,1000,1,true,false,false,17,kn,other,other,other);
}catch(...){caught=true;}if(!caught||!qual_att7_end().empty())return 21;
std::cout<<"GQUAL_ATT7_HOST_PASS disabled0calls cooperative cluster 7rejects\n";
}
