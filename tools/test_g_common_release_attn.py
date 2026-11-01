"""CPU source and C++ API witnesses; no CUDA execution/link qualification."""
import argparse
import ast
import os
from pathlib import Path
import unittest
from g_common_release_attn import AUDIT,patch_cuda

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
PARAMS='''struct P{int B=1,flat=0,coop=1,resident=0,unev=0,n_splits=132,cs=1,ncl=1,window=2048;};
using Kernel=void(*)(const P);void actual(const P){};
'''
API=r'''
#include <cuda_runtime.h>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
@@PARAMS@@
@@AUDIT@@
template void qual_att7_observe<Kernel,P>(Kernel,const P&,dim3,int,int,int,bool,bool,bool,
    cudaStream_t,const Kernel*,const Kernel*,const Kernel*,const Kernel*);
'''


def actual_routes_api(root):
    import sys
    sys.path.insert(0,str(root/'tools'))
    try:
        from g_integration_common import build
        base=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
        source,_=build(base,True)
    finally:sys.path.pop(0)
    value=ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                               and getattr(n.targets[0],'id','')=='ATT7_CUDA_SRC'))
    value=patch_cuda(value)
    a=value.index('  // OFF-prototype common progress repair:')
    z=value.index('\ntypedef CUresult',a)
    tail=value[a:z]
    ordinary='    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);'
    assert tail.count(ordinary)==1
    # CPU compiler cannot parse unchanged <<<>>>. This sole disclosed adapter
    # is outside the new audit code; all new launch/occupancy APIs remain exact.
    tail=tail.replace(ordinary,'    void* qa[]={&q};cudaLaunchKernel((const void*)kern,grid,dim3(Cfg<L,false>::NTX),qa,sm,stream);')
    adapter='''\n#define OKC_FLATCOOP1C 0
#define OKC_CLSM 0
template<int L,bool F>struct Cfg{static constexpr int NTX=F?192:128;};
template<int L>void actual_routes(P p,dim3 grid,bool cl,bool pdl,Kernel kern,cudaStream_t stream,int sm,
    Kernel* kn,Kernel* kn1,Kernel* ksp,Kernel* ksp1){P q=p;int cs=q.cs;
'''+tail+'''
template void actual_routes<1>(P,dim3,bool,bool,Kernel,cudaStream_t,int,Kernel*,Kernel*,Kernel*,Kernel*);
template void actual_routes<8>(P,dim3,bool,bool,Kernel,cudaStream_t,int,Kernel*,Kernel*,Kernel*,Kernel*);
'''
    return API.replace('@@PARAMS@@',PARAMS).replace('@@AUDIT@@',AUDIT)+adapter
MOCK=r'''
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
@@PARAMS@@
int calls=0,fail=0;
int cudaDriverGetVersion(int*x){*x=12080;return fail==6;}
int cudaRuntimeGetVersion(int*x){*x=12080;return fail==7;}
int cudaGetDevice(int*x){calls++;*x=0;return fail==1;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){calls++;*p={};return fail==2;}
int cudaFuncGetAttributes(cudaFuncAttributes*a,const void*){calls++;*a={};return fail==3;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*x,K,int,int){calls++;*x=2;return fail==4;}
template<class K>int cudaOccupancyMaxActiveClusters(int*x,K,const cudaLaunchConfig_t*cfg){calls++;*x=14;if(cfg->attrs[0].val.clusterDim.x!=2)return 1;return fail==5;}
@@AUDIT@@
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
for(int f=6;f<=7;++f){fail=f;caught=false;try{qual_att7_versions();}catch(...){caught=true;}if(!caught)return 30+f;}
fail=0;auto versions=qual_att7_versions();if(versions.size()!=5||versions[0]!=12080||versions[1]!=12080)return 40;
std::cout<<"GQUAL_ATT7_HOST_PASS disabled0calls cooperative cluster 9rejects\n";
}
'''


class Source(unittest.TestCase):
    def test_actual_host_only_delta_and_success_sites(self):
        import sys
        sys.path.insert(0,str(ROOT/'tools'))
        try:
            from g_integration_common import build
            original=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
            source,_=build(original,True)
        finally:sys.path.pop(0)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='ATT7_CUDA_SRC')
        before=ast.literal_eval(node.value);after=patch_cuda(before)
        anchor='template <int L>\nvoid launch('
        self.assertEqual(before[:before.index(anchor)],after[:after.index(AUDIT)])
        self.assertEqual(after.count('if (g_qual_att7_enabled) qual_att7_observe'),3)
        self.assertIn('const cudaError_t checked=cudaLaunchKernelEx',after)
        self.assertIn('cudaGetLastError()==cudaSuccess',after)
        self.assertIn('if (!g_qual_att7_enabled) return;',AUDIT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--emit-api',type=Path);parser.add_argument('--emit-mock',type=Path);parser.add_argument('--emit-routes',type=Path)
    args,rest=parser.parse_known_args()
    if args.emit_api or args.emit_mock or args.emit_routes:
        path=args.emit_api or args.emit_mock or args.emit_routes;path.parent.mkdir(parents=True,exist_ok=True)
        text=actual_routes_api(ROOT) if args.emit_routes else (API if args.emit_api else MOCK).replace('@@PARAMS@@',PARAMS).replace('@@AUDIT@@',AUDIT)
        path.write_text(text,encoding='utf-8')
    else:unittest.main(argv=[__file__,*rest])
