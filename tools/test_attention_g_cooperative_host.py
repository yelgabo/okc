"""CPU source/schema and host-adapter tests; CUDA APIs are mocks, no GPU."""
import ast,hashlib,os
from pathlib import Path
import unittest
from attention_g_cooperative_host import build,REPAIR,PIN
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))

MOCK=r'''
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
@@REPAIR@@
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
'''
API_SOURCE=r'''#include <cuda_runtime.h>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
#define OKC_FLATCOOP1C 0
struct Params{int flat,coop;};
template<int L,bool F>struct Cfg{static constexpr int NTX=192;};
using KernFn=void(*)(Params);
template<int L>void checked_launch(Params p,Params q,bool cl,bool pdl,dim3 grid,int sm,cudaStream_t stream,KernFn kern){
@@REPAIR@@
}
template void checked_launch<1>(Params,Params,bool,bool,dim3,int,cudaStream_t,KernFn);
template void checked_launch<8>(Params,Params,bool,bool,dim3,int,cudaStream_t,KernFn);
'''

class HostRepair(unittest.TestCase):
    def test_source(self):
        source=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_bytes().decode()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),PIN)
        self.assertEqual(build(source),source)
        new=build(source,True)
        self.assertEqual(ast.dump(ast.parse(new.replace(REPAIR,''))),ast.dump(ast.parse(source)))
        self.assertEqual(new.count(REPAIR),1)
        self.assertIn('cudaLaunchCooperativeKernel',new)
        self.assertNotIn('programmaticStreamSerializationAllowed = 1;',new)
        with self.assertRaises(AssertionError):build(source+'\n',True)
    def test_selected_semantics(self):
        # Exactly the host predicate, including nonjoining flat L1 and clusters.
        pred=lambda coop,flat,L,cl:coop and ((not flat and not cl)or(flat and L==8))
        self.assertTrue(pred(1,0,1,0));self.assertTrue(pred(1,0,8,0));self.assertTrue(pred(1,1,8,0))
        self.assertFalse(pred(1,1,1,0));self.assertFalse(pred(0,0,8,0));self.assertFalse(pred(1,0,8,1))
    def test_mock_fixture_emission(self):
        self.assertEqual(MOCK.count('@@REPAIR@@'),1)
        self.assertIn('args[]={&q}',REPAIR)

if __name__=='__main__':
    import argparse
    if '--emit-api' in __import__('sys').argv:
        Path(__import__('sys').argv[-1]).write_text(API_SOURCE.replace('@@REPAIR@@',REPAIR),encoding='utf8')
    elif '--emit-mock' in __import__('sys').argv:
        dest=Path(__import__('sys').argv[-1]);dest.write_text(MOCK.replace('@@REPAIR@@',REPAIR),encoding='utf8')
    else:unittest.main()
