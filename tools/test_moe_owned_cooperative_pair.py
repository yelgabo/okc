import ast
from pathlib import Path
import subprocess
import unittest

from moe_owned_cooperative_pair import MAIN,SKINNY,cooperative_source
from moe_multiply_decode import patch

ROOT=Path(__file__).resolve().parents[1]


class CooperativeTests(unittest.TestCase):
    def test_only_two_host_wrapper_bodies_change(self):
        src=(ROOT/'muse_moe_w4a16/submission.py').read_text(encoding='utf-8')
        for original in (src,patch(src)):
            changed=cooperative_source(original)
            for anchor,new in (('template <int MODE>\nvoid launch_main(',MAIN),('template <int SPLIT>\nvoid launch(',SKINNY)):
                a=original.index(anchor);brace=original.index('{',a);z=brace+1;depth=1
                while depth:depth+=(original[z]=='{')-(original[z]=='}');z+=1
                changed=changed.replace(new,original[a:z],1)
            self.assertEqual(original,changed)
        self.assertNotIn('cudaLaunchKernelEx',MAIN+SKINNY)
        self.assertNotIn('<<<',MAIN+SKINNY)
        self.assertIn('p.M<=kEmax/kTopK',MAIN)

    def test_actual_host_wrappers_with_cuda_api_mocks(self):
        cpp=r'''
#include <stdexcept>
#include <iostream>
struct Params {int M,lo,nloc;}; constexpr int kEmax=4096,kTopK=8,kNTM=576,kNTL=288;
struct SmemMain {char a[228736];}; struct Smem {char a[150000];};
using cudaStream_t=int;
struct dim3 {int x; explicit dim3(int x):x(x){}};
struct cudaDeviceProp {int major=9,minor=0,cooperativeLaunch=1,multiProcessorCount=132;};
cudaDeviceProp actual; int res=1,attrError=0,launchError=0,calls=0,seenGrid=0,seenBlock=0,seenSmem=0,seenM=0,seenStream=0;
enum {cudaFuncAttributeMaxDynamicSharedMemorySize};
int cudaGetDevice(int* x){*x=0;return 0;}
int cudaGetDeviceProperties(cudaDeviceProp* p,int){*p=actual;return 0;}
template<class T>int cudaFuncSetAttribute(T,int,int){return attrError;}
template<class T>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* r,T,int,int){*r=res;return 0;}
int cudaLaunchCooperativeKernel(const void*,dim3 grid,dim3 block,void** args,int smem,int stream){
 ++calls;seenGrid=grid.x;seenBlock=block.x;seenSmem=smem;seenM=((Params*)args[0])->M;seenStream=stream;return launchError;}
#define C10_CUDA_CHECK(x) do {if((x)!=0)throw std::runtime_error("CUDA_FATAL");}while(0)
#define TORCH_CHECK(x,msg) do {if(!(x))throw std::runtime_error(msg);}while(0)
template<int T>void moe_main_kernel(Params){}
template<int T>void moe_fused_kernel(Params){}
'''+MAIN+'\n'+SKINNY+r'''
int main(){
 Params p{257,0,32};launch_main<47>(p,132,73);
 if(calls!=1||seenGrid!=132||seenBlock!=576||seenSmem!=228736||seenM!=257||seenStream!=73)return 1;
 p.M=512;launch_main<47>(p,132,73);p.M=16;launch<1>(p,132,41);
 if(calls!=3||seenBlock!=288||seenSmem!=150000||seenM!=16||seenStream!=41)return 2;
 int rejected=0;
 for(int i=0;i<10;++i){
  actual={};res=1;attrError=launchError=0;p={128,0,32};int grid=132;
  if(i==0)actual.major=8;if(i==1)actual.cooperativeLaunch=0;if(i==2)grid=0;if(i==3)grid=133;
  if(i==4)p.M=513;if(i==5)p.lo=32;if(i==6)p.nloc=31;if(i==7)res=0;if(i==8)attrError=1;if(i==9)launchError=1;
  try{launch_main<47>(p,grid,3);}catch(const std::runtime_error&){++rejected;}
 }
 if(rejected!=10)return 3;
 std::cout<<"OWNED_COOP_HOST_PASS two wrappers, M257/M512, ten rejected/fatal controls\n";
}
'''
        folder=ROOT/'.build/today-moe';folder.mkdir(parents=True,exist_ok=True)
        (folder/'multiply_coop_host.cpp').write_text(cpp,encoding='utf-8')
        linux='/mnt/c/'+str(ROOT).replace('\\','/')[3:]
        zig='/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig'
        subprocess.run(['wsl','-d','Ubuntu','--exec',zig,'c++','-std=c++17','-O1',linux+'/.build/today-moe/multiply_coop_host.cpp','-o',linux+'/.build/today-moe/multiply_coop_host'],check=True,timeout=180)
        run=subprocess.run(['wsl','-d','Ubuntu','--exec',linux+'/.build/today-moe/multiply_coop_host'],check=True,text=True,capture_output=True)
        print(run.stdout.strip())


if __name__=='__main__':unittest.main()
