"""Source scope and actual reused launcher C++ CPU mock; no CUDA execution."""
import ast,os,subprocess,sys,unittest
from pathlib import Path
from verify_owned_progress import patch,literal,native_function,cooperative_launcher
from verify_owned_progress_compose import compose
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
PACKED=Path(os.environ['OKC_PACKED_OWNED'])
class Tests(unittest.TestCase):
 def test_scope_pdl_and_source_epoch(self):
  old=PACKED.read_text();new,m=patch(old,ROOT)
  a,b=ast.parse(old),ast.parse(new)
  # Reverse the two Python edits and one native literal replacement exactly by AST.
  for n in b.body:
   if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_MOE1_SRC':n.value=ast.Constant(literal(old,'_MOE1_SRC'))
   if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn':
    c=next(v for v in ast.walk(n) if isinstance(v,ast.Call) and getattr(v.func,'id','')=='_moe1_run');self.assertIs(c.args[-1].value,False);c.args[-1]=ast.Constant(True)
   if isinstance(n,ast.FunctionDef) and n.name=='_moe1_load':
    for v in ast.walk(n):
     if isinstance(v,ast.keyword) and v.arg=='name':self.assertEqual(v.value.value,'okc_decode_layer_moe1_coop_trap_v1');v.value=ast.Constant('okc_decode_layer_moe1')
  self.assertEqual(ast.dump(a),ast.dump(b));self.assertEqual(m['output_bytes']-m['input_bytes'],770)
  d=literal(new,'_MOE1_SRC');self.assertIn('asm volatile("trap;" ::: "memory");',native_function(d,'__device__ __forceinline__ bool spin_failed('))
  with self.assertRaises(AssertionError):patch(new,ROOT)
  with self.assertRaisesRegex(AssertionError,'Python source epoch'):patch(old+'\n',ROOT)
 def test_composition_changes_only_owned_and_private_pin(self):
  package_dir=Path(os.environ['OKC_PROGRESS_PACKAGES'])
  for name in ('hybrid-mask4.py','hybrid-mask15.py','private4.py'):
   # For composed files, reverse their owned patch via the corresponding input.
   changed=(package_dir/name).read_text()
   if name=='private4.py':
    sys.path.insert(0,str(ROOT/'tools'))
    try:
     import dl_verify_pair
     original=dl_verify_pair.build(ROOT)[0]
    finally:sys.path.pop(0)
   else:original=(PACKED.parent/name).read_text()
   generated,m=compose(original,ROOT);self.assertEqual(generated,changed)
   a,b=ast.parse(original),ast.parse(changed)
   for node in b.body:
    if isinstance(node,ast.Assign) and getattr(node.targets[0],'id','') in ('_VERIFY_HYBRID_SOURCE_OWNED','_PAIR_SOURCE_owned','_PAIR_PINS'):
     old=next(n for n in a.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==node.targets[0].id);node.value=old.value
   self.assertEqual(ast.dump(a),ast.dump(b));self.assertLess(m['package_output_bytes'],1000000)
 def test_reused_launcher_and_timeout_actual_cpp(self):
  source=PACKED.read_text();new,m=patch(source,ROOT);launch=cooperative_launcher(ROOT)
  fail=native_function(literal(new,'_MOE1_SRC'),'__device__ __forceinline__ bool spin_failed(')
  # Only the trap instruction is replaced by a host exception to exercise its branch.
  fail=fail.replace('asm volatile("trap;" ::: "memory");','throw Trap{};')
  cpp=r'''
#include <stdexcept>
#include <iostream>
#define __device__
#define __forceinline__ inline
struct Params {int M,nloc,pdl;};constexpr int kEmax=4096,kTopK=8,kLocalMax=32;
constexpr long long kSpinLimit=1ll<<31;long long ticks=0;
long long clock64(){return ticks;}int atomicOr(int* p,int v){int old=*p;*p|=v;return old;}struct Trap{};
using cudaStream_t=int;struct dim3{int x;explicit dim3(int a):x(a){}};
struct cudaDeviceProp{int major=9,minor=0,cooperativeLaunch=1,multiProcessorCount=132;}prop;
int resident=1,devErr=0,propErr=0,attrErr=0,occErr=0,launchErr=0,calls=0,seenM=0,seenGrid=0,seenSmem=0,seenStream=0;
enum{cudaFuncAttributeMaxDynamicSharedMemorySize};
int cudaGetDevice(int*p){*p=0;return devErr;}int cudaGetDeviceProperties(cudaDeviceProp*p,int){*p=prop;return propErr;}
template<class K>int cudaFuncSetAttribute(K,int,int){return attrErr;}
template<class K>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*p,K,int,int){*p=resident;return occErr;}
int cudaLaunchCooperativeKernel(const void*,dim3 g,dim3,void**a,int smem,int stream){++calls;seenM=((Params*)a[0])->M;seenGrid=g.x;seenSmem=smem;seenStream=stream;return launchErr;}
#define C10_CUDA_CHECK(x) do{if((x)!=0)throw std::runtime_error("CUDA_FATAL");}while(0)
#define TORCH_CHECK(x,msg) do{if(!(x))throw std::runtime_error(msg);}while(0)
void kernel(Params){}
'''+launch+'\n'+fail+r'''
int main(){
 Params p{128,32,0};moe1_launch(kernel,132,576,228736,73,p);
 if(calls!=1||seenM!=128||seenGrid!=132||seenSmem!=228736||seenStream!=73)return 1;
 p.M=256;moe1_launch(kernel,132,576,228736,73,p);
 int rejected=0;
 for(int i=0;i<16;++i){prop={};resident=1;devErr=propErr=attrErr=occErr=launchErr=0;p={128,32,0};int grid=132,block=576,smem=228736;
  if(i==0)p.pdl=1;if(i==1)prop.major=8;if(i==2)prop.cooperativeLaunch=0;if(i==3)grid=0;if(i==4)grid=133;if(i==5)block=0;if(i==6)smem=-1;
  if(i==7)p.M=0;if(i==8)p.M=513;if(i==9)p.nloc=0;if(i==10)p.nloc=33;if(i==11)resident=0;if(i==12)devErr=1;if(i==13)propErr=1;if(i==14)attrErr=1;if(i==15)occErr=1;
  int old=calls;try{moe1_launch(kernel,grid,block,smem,73,p);}catch(const std::runtime_error&){++rejected;}if(calls!=old)return 2;
 }
 prop={};resident=1;devErr=propErr=attrErr=occErr=0;launchErr=1;p={128,32,0};try{moe1_launch(kernel,132,576,228736,73,p);return 3;}catch(const std::runtime_error&){}
 int ctr[16]={};ticks=100+kSpinLimit-1;if(spin_failed(100,ctr,4)||ctr[8])return 4;
 ticks=100+kSpinLimit;try{spin_failed(100,ctr,4);return 5;}catch(const Trap&){}if(ctr[8]!=4||rejected!=16)return 6;
 std::cout<<"VERIFY_OWNED_COOP_HOST_PASS 2 valid,16 prelaunch rejects,1 fatal launch,timeout trap branch\n";
}
'''
  out=Path(__file__).resolve().parents[1]/'.build/today-moe/verify-owned-progress';out.mkdir(parents=True,exist_ok=True);(out/'host.cpp').write_text(cpp)
  compiler=Path('/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig')
  if sys.platform!='linux':self.skipTest('Run actual C++ mock under WSL persistent compiler')
  subprocess.run([str(compiler),'c++','-std=c++17','-O1',str(out/'host.cpp'),'-o',str(out/'host')],check=True,timeout=60)
  result=subprocess.run([str(out/'host')],check=True,text=True,capture_output=True);print(result.stdout.strip())
if __name__=='__main__':unittest.main()
