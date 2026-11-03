"""Actual getter C++ CPU mocks and exact oracle/source scope; no CUDA runtime."""
import ast,json,os,subprocess,tempfile,unittest
from pathlib import Path
from prod_size_qualify import MOE_GETTER,ONE_GETTER,helpers

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[2]/'okc'))
ZIG=Path('/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig')

class Qualifier(unittest.TestCase):
 def test_oracle_preserves_arithmetic_thresholds_only_adds_finite_guards(self):
  text,manifest=helpers(ROOT);nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  original=nodes['_ab_validate'];changed=nodes['_qual_oracle'];changed.name='_ab_validate'
  bound=next(n for n in ast.walk(changed) if isinstance(n,ast.FunctionDef) and n.name=='bound');self.assertIsInstance(bound.body.pop(0),ast.Assert)
  rel=next(n for n in ast.walk(changed) if isinstance(n,ast.FunctionDef) and n.name=='rel');self.assertIsInstance(rel.body[1],ast.Assert)
  rel.body=[ast.Return(value=rel.body[0].value)]
  self.assertEqual(ast.dump(original),ast.dump(changed))
  self.assertEqual(manifest['helper_source_sha256'],'54881e3c51e58af323021e439e8ea3dd8c63bb2db7b7dbb184ca60c22adc9e4a')
 def test_actual_resource_getter_cpp_normal_and_all_api_errors_fatal(self):
  if not ZIG.exists():self.skipTest('isolated Zig C++ unavailable')
  prefix=r'''
#include <vector>
#include <cstdint>
#include <cassert>
#include <stdexcept>
namespace moe1 {template<int M> void moe_main_kernel(){} constexpr int kNTM=576;struct SmemMain{char x[228736];};}
namespace one {template<int L,int M> void attn_one_kernel(){} template<int L,int M>struct Instance{
static constexpr int NT=M==3?256:384,BYTES=M==3?60000:110000;static void prepare(){};};}
using Kernel=void(*)();int failure=0,calls=0,current=0;
struct cudaFuncAttributes{int numRegs;std::size_t localSizeBytes,sharedSizeBytes;int maxThreadsPerBlock;};
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=1,cudaDevAttrMultiProcessorCount=2,cudaDevAttrCooperativeLaunch=3;
int api(){return ++calls==failure?1:0;}void check(int code){if(code)throw std::runtime_error("CUDA failure");}
#define C10_CUDA_CHECK(x) check(x)
int cudaFuncSetAttribute(Kernel k,int,int bytes){current=k==&moe1::moe_main_kernel<51>?0:k==&one::attn_one_kernel<8,3>?1:2;assert(bytes==(current==0?228736:current==1?60000:110000));return api();}
int cudaFuncGetAttributes(cudaFuncAttributes*a,Kernel){a->numRegs=current==0?96:current==1?114:80;a->localSizeBytes=0;a->sharedSizeBytes=0;a->maxThreadsPerBlock=current==0?576:current==1?256:384;return api();}
int cudaGetDevice(int*d){*d=0;return api();}
int cudaDeviceGetAttribute(int*out,int field,int){*out=field==cudaDevAttrMultiProcessorCount?132:1;return api();}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*out,Kernel,int threads,int bytes){assert(threads==(current==0?576:current==1?256:384));assert(bytes==(current==0?228736:current==1?60000:110000));*out=current==0?1:2;return api();}
'''
  suffix=r'''
int main(){auto m=qual_moe_resources();assert(m.size()==10&&m[0]==96&&m[3]==228736&&m[4]==576&&m[9]==132);
auto o=qual_one_resources();assert(o.size()==20&&o[0]==114&&o[4]==256&&o[9]==264&&o[10]==80&&o[14]==384&&o[19]==264);
for(int i=1;i<=6;++i){calls=0;failure=i;bool threw=false;try{qual_moe_resources();}catch(const std::runtime_error&){threw=true;}assert(threw);}
for(int i=1;i<=12;++i){calls=0;failure=i;bool threw=false;try{qual_one_resources();}catch(const std::runtime_error&){threw=true;}assert(threw);}
}
'''
  with tempfile.TemporaryDirectory(prefix='qual-getters-') as folder:
   p=Path(folder);(p/'test.cpp').write_text(prefix+MOE_GETTER+ONE_GETTER+suffix)
   compiled=subprocess.run([str(ZIG),'c++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True)
   self.assertEqual(compiled.returncode,0,compiled.stderr)
   run=subprocess.run([str(p/'test')],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)

if __name__=='__main__':unittest.main()
