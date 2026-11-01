"""Independent CPU/header gate for exact G native getter; never CUDA execution."""
import argparse, ast, hashlib, json, os, subprocess, sys
from pathlib import Path

ZIG='/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig'
CUDA='/root/.cache/okc-cuda128/packages/nvidia'


def run(owner,out):
    source=(owner/'tools/g_common_release.py').read_text(encoding='utf8')
    getter=next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='MOE_RESOURCE')
    out.mkdir(parents=True,exist_ok=True)
    common='''#include <vector>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("reject");}while(0)
namespace okc {constexpr int NTL=288;template<bool B>void moe_fused(){};}
static size_t okc_smem_bytes(){return 155520;}
'''
    api='#include <cuda_runtime.h>\n'+common+getter
    (out/'native-api.cpp').write_text(api,encoding='utf8')
    commands=[]
    def invoke(cmd):
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=60,
                         env=dict(os.environ,OKC_REPO_ROOT=str(owner.parent/'okc')))
        commands.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,p.stderr
        return p.stdout
    invoke([ZIG,'c++','-std=c++17','-c',str(out/'native-api.cpp'),'-o',str(out/'native-api.o'),
            '-I'+CUDA+'/cuda_runtime/include','-I'+CUDA+'/cuda_nvcc/include'])
    mock=common+'''
constexpr int cudaSuccess=0,cudaFuncAttributeMaxDynamicSharedMemorySize=1;
struct cudaDeviceProp{int multiProcessorCount=132,cooperativeLaunch=1;};
struct cudaFuncAttributes{int numRegs=147;size_t localSizeBytes=8,sharedSizeBytes=0;int maxThreadsPerBlock=1024;};
int fail=0,calls=0;
int cudaGetDevice(int*p){++calls;*p=0;return fail==1;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){++calls;*p={};return fail==2;}
template<class K>int cudaFuncSetAttribute(K,int,int size){++calls;if(size!=155520)return 99;return fail==3;}
template<class K>int cudaFuncGetAttributes(cudaFuncAttributes*a,K){++calls;*a={};return fail==4;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*p,K,int nt,size_t sm){++calls;*p=2;if(nt!=288||sm!=155520)return 99;return fail==5;}
'''+getter+'''
#include <iostream>
int main(){auto r=qual_native_resources();if(r.size()!=20||calls!=8)return 1;
for(int i:{0,10})if(r[i]!=147||r[i+1]!=8||r[i+2]!=0||r[i+3]!=155520||r[i+4]!=288||r[i+5]!=1024||r[i+6]!=2||r[i+7]!=132||r[i+8]!=1||r[i+9]!=264)return 2;
for(int f=1;f<=5;++f){fail=f;bool caught=false;try{qual_native_resources();}catch(...){caught=true;}if(!caught)return 10+f;}
std::cout<<"PASS two10fieldrows nonzero-local-preserved fivefatalAPIs\\n";
}
'''
    (out/'native-mock.cpp').write_text(mock,encoding='utf8')
    invoke([ZIG,'c++','-std=c++17','-O1',str(out/'native-mock.cpp'),'-o',str(out/'native-mock')])
    invoke([str(out/'native-mock')])
    # Peer-provided tests emit the exact new ATT7 audit and actual resolved
    # launch branches. Their chevron-only adapter is disclosed in that source.
    emitter=owner/'tools/test_g_common_release_attn.py'
    for mode in ('api','routes','mock'):
        cpp=out/('attn-'+mode+'.cpp')
        invoke([sys.executable,str(emitter),'--emit-'+mode,str(cpp)])
        cmd=[ZIG,'c++','-std=c++17',str(cpp),'-o',str(out/('attn-'+mode+('.o' if mode!='mock' else '')))]
        if mode!='mock':
            cmd+=['-c','-I'+CUDA+'/cuda_runtime/include','-I'+CUDA+'/cuda_nvcc/include']
        invoke(cmd)
        if mode=='mock':invoke([str(out/'attn-mock')])
    data=dict(owner_builder_sha256=hashlib.sha256(source.encode()).hexdigest(),
              getter_sha256=hashlib.sha256(getter.encode()).hexdigest(),commands=commands,
              attention_audit_sha256=hashlib.sha256((owner/'tools/g_common_release_attn.py').read_bytes()).hexdigest(),
              adapter='Only okc kernel declarations/NTL/shared-byte stubs and TORCH_CHECK macro; exact getter unchanged.',
              scope='Real CUDA12.8 host-header compile-only and CPU API mocks; no Torch link, device or GPU execution.')
    (out/'native-review.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print(json.dumps(data,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('owner',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.owner,a.out)
