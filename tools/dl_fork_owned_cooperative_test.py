"""Emit CPU C++ mocks to exercise the exact optional host launch helper."""
from pathlib import Path
import sys
from dl_fork_owned_cooperative import LAUNCH

STUB=r'''
#include <stdexcept>
#include <cassert>
#define TORCH_CHECK(x,...) do{if(!(x)) throw std::runtime_error("guard");}while(0)
#define C10_CUDA_CHECK(x) TORCH_CHECK((x)==0)
struct Params{int M=128,nloc=32,pdl=0;};
struct cudaDeviceProp{int major=9,minor=0,cooperativeLaunch=1,multiProcessorCount=132;};
struct dim3{int x;explicit dim3(int v):x(v){}};
using cudaStream_t=void*;
constexpr int kEmax=4096,kTopK=8,kLocalMax=32,cudaFuncAttributeMaxDynamicSharedMemorySize=0;
cudaDeviceProp props;int resident_blocks=1,attr_status=0,launch_status=0,calls=0;
int cudaGetDevice(int* d){*d=0;return 0;}
int cudaGetDeviceProperties(cudaDeviceProp* p,int){*p=props;return 0;}
template<typename K>int cudaFuncSetAttribute(K,int,int){return attr_status;}
template<typename K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* r,K,int,int){*r=resident_blocks;return 0;}
int cudaLaunchCooperativeKernel(const void*,dim3 g,dim3 b,void** a,int,void*){
  assert(g.x==132&&b.x==288);assert(static_cast<Params*>(a[0])->M==128);calls++;return launch_status;
}
void kernel(Params){}
'''
MAIN=r'''
int main(){
 Params p;auto run=[&](){moe1_launch(&kernel,132,288,65536,nullptr,p);};
 run();assert(calls==1);
 auto rejects=[&](){bool failed=false;try{run();}catch(const std::runtime_error&){failed=true;}assert(failed);};
 resident_blocks=0;rejects();resident_blocks=1;assert(calls==1);
 p.pdl=1;rejects();p.pdl=0;
 props.major=8;rejects();props.major=9;
 props.cooperativeLaunch=0;rejects();props.cooperativeLaunch=1;
 p.M=513;rejects();p.M=128;
 attr_status=1;rejects();attr_status=0;
 launch_status=1;rejects();assert(calls==2);
}
'''
Path(sys.argv[1]).write_text(STUB+'\n'+LAUNCH+'\n'+MAIN,encoding='utf-8')
