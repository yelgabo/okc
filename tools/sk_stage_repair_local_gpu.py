"""Local RTX bit-only execution of exact stage-local SXS helper, never a SK timing test.
Requires CUDA PyTorch context, NVRTC12.8 and CUDA driver; no host C++ compiler.
"""
import argparse,ast,ctypes as C,hashlib,json,time
from pathlib import Path
import torch
PIN='b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e'
HELPER_PIN='48a651aa0138c31e0f83a9eb1f7667513ec33b442efeb0cd71270af259332149'
def ck(rc,label):
 if rc:raise RuntimeError(label+' status '+str(rc))
def run(a):
 begin=time.monotonic();a.output.mkdir(parents=True,exist_ok=True)
 raw=a.artifact.read_bytes();artifact_hash=hashlib.sha256(raw).hexdigest();assert artifact_hash==PIN
 t=ast.parse(raw.decode());cu=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SXS_CUDA')
 # Exact stage helper slice, including the shared-u16 accessor it calls.
 start=cu.index('__device__ __forceinline__ unsigned sxs_lds16(');end=cu.index('// Eight 12-bit weights ->',start)
 helper=cu[start:end];assert hashlib.sha256(helper.encode()).hexdigest()==HELPER_PIN,'EXACT_B726_HELPER_REQUIRED'
 wrapper=r'''
extern "C" __global__ void probe(const uint32_t* input,uint32_t* output,
 const uint32_t* headers,const uint16_t* records,int units){
 int u=blockIdx.x,lane=threadIdx.x;if(u>=units)return;
 __shared__ __align__(16) uint16_t arena[544];
 for(int i=lane;i<544;i+=32)arena[i]=records[u*544+i];
 __syncthreads();
 unsigned long long address;
 asm volatile("cvta.to.shared.u64 %0, %1;":"=l"(address):"l"(arena));
 uint32_t b[8];
 #pragma unroll
 for(int r=0;r<8;++r)b[r]=input[(u*32+lane)*8+r];
 sxs_repair(b,headers[u],(uint32_t)address,lane);
 #pragma unroll
 for(int r=0;r<8;++r)output[(u*32+lane)*8+r]=b[r];
}
'''
 device='typedef unsigned int uint32_t;typedef unsigned short uint16_t;\n'+helper+wrapper
 (a.output/'device.cu').write_text(device,encoding='utf8')
 lib=C.CDLL(str(a.nvrtc));lib.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
 lib.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
 for name in ('nvrtcGetProgramLogSize','nvrtcGetCUBINSize'):getattr(lib,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
 for name in ('nvrtcGetProgramLog','nvrtcGetCUBIN'):getattr(lib,name).argtypes=[C.c_void_p,C.c_void_p]
 prog=C.c_void_p();ck(lib.nvrtcCreateProgram(C.byref(prog),device.encode(),b'sx_exact_helper.cu',0,None,None),'create')
 opts=['--gpu-architecture=sm_120','--std=c++17','--generate-line-info'];argv=(C.c_char_p*len(opts))(*(o.encode() for o in opts))
 rc=lib.nvrtcCompileProgram(prog,len(opts),argv);sz=C.c_size_t();ck(lib.nvrtcGetProgramLogSize(prog,C.byref(sz)),'logsize');buf=C.create_string_buffer(sz.value);ck(lib.nvrtcGetProgramLog(prog,buf),'log');(a.output/'nvrtc.log').write_bytes(buf.value);ck(rc,'compile')
 ck(lib.nvrtcGetCUBINSize(prog,C.byref(sz)),'cubinsize');cubin=C.create_string_buffer(sz.value);ck(lib.nvrtcGetCUBIN(prog,cubin),'cubin');(a.output/'device.cubin').write_bytes(cubin.raw)
 torch.cuda.init();context_anchor=torch.empty(1,device='cuda');assert torch.cuda.get_device_capability()==(12,0)
 drv=C.CDLL('libcuda.so.1');drv.cuInit.argtypes=[C.c_uint];ck(drv.cuInit(0),'cuInit')
 drv.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];drv.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
 drv.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
 module=C.c_void_p();ck(drv.cuModuleLoadData(C.byref(module),cubin),'module');fn=C.c_void_p();ck(drv.cuModuleGetFunction(C.byref(fn),module,b'probe'),'function')
 units=132;capacity=units*544
 inp=torch.empty(units*32*8,dtype=torch.int32,device='cuda');out=torch.empty_like(inp);headers=torch.empty(units,dtype=torch.int32,device='cuda');records=torch.empty(capacity,dtype=torch.int16,device='cuda')
 def launch():
  vals=[C.c_uint64(x.data_ptr()) for x in (inp,out,headers,records)]+[C.c_int(units)]
  params=(C.c_void_p*len(vals))(*(C.cast(C.byref(v),C.c_void_p) for v in vals))
  ck(drv.cuLaunchKernel(fn,units,1,1,32,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'launch')
 def bank(seed):
  bits=torch.arange(units*512,dtype=torch.int64).reshape(units,32,8,2)%65536
  # Change each replay's BF16 exponent targets/record values and keep full bit coverage.
  bits[:128]=torch.roll(bits[:128].reshape(-1),seed*7919).reshape(128,32,8,2)
  bits[128]=0x3f80 # explicit zero-record unit at base63
  bits[129]=0x0001 # explicit maximum512 record unit
  bits[130]=0x8000;bits[130,0,0,0]=0x3f80 #511records, mixed sign/half
  bits[131]=0x3f80;bits[131,31,7,1]=(0x7f7f if seed%2 else 0x0001) #last-lane/reg/high-half singleton
  base=63;e=(bits>>8)&127
  # Operator precedence kept explicit for independent CPU reference below.
  decoded=(bits&0x80ff)|(((e-base).clamp(0,7)+base)<<8)
  patch=torch.full((units,544),0xa55a,dtype=torch.int64);hs=[];counts=[]
  for u in range(units):
   coords=(bits[u]!=decoded[u]).nonzero();start=1+(u%17);counts.append(len(coords));assert start+len(coords)<=544
   for rank,(lane,r,h) in enumerate(coords.tolist()):
    x=int(((bits[u,lane,r,h]^decoded[u,lane,r,h])>>8)&127)
    patch[u,start+rank]=lane|(r<<5)|(h<<8)|(x<<9)
   hs.append((start<<10)|len(coords))
  assert counts[128:]==[0,512,511,1]
  words=lambda x:(x[...,0]|(x[...,1]<<16)).reshape(-1).to(torch.int32)
  rr=patch.reshape(-1)
  return words(decoded),words(bits),torch.tensor(hs,dtype=torch.int64).to(torch.int32),rr.to(torch.int16),counts
 checks=0;banks=[]
 for seed in (0,1,2):
  assert time.monotonic()-begin<240,'LOCAL_PROBE_BUDGET'
  x,ref,h,r,counts=bank(seed);inp.copy_(x);headers.copy_(h);records.copy_(r);out.fill_(0x12345678);launch();torch.cuda.synchronize();assert torch.equal(out.cpu(),ref),'EAGER_BITS';checks+=1;banks.append((x,ref,h,r))
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  out.fill_(0x12345678);launch()
 for rep in range(20):
  assert time.monotonic()-begin<240,'LOCAL_PROBE_BUDGET'
  x,ref,h,r=banks[rep%3];inp.copy_(x);headers.copy_(h);records.copy_(r);graph.replay();torch.cuda.synchronize();assert torch.equal(out.cpu(),ref),'REPLAY_BITS';checks+=1
 result={'artifact_sha256':artifact_hash,'reviewed_helper_owner_artifact':PIN,'helper_sha256':hashlib.sha256(helper.encode()).hexdigest(),'device_sha256':hashlib.sha256(device.encode()).hexdigest(),'cubin_sha256':hashlib.sha256(cubin.raw).hexdigest(),'nvrtc':str(a.nvrtc),'options':opts,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability()),'units_per_launch':units,'unique_BF16_patterns_per_bank':65536,'explicit_patch_counts':[0,512,511,1],'nonzero_local_prefix_range':[1,17],'shared_uint16_capacity_per_CTA':544,'padding_poison':hex(0xa55a),'eager_banks':3,'fresh_changed_record_graph_replays':20,'exact_uint32_checks':checks,'passed':True,'elapsed_setup_validation_seconds':time.monotonic()-begin,'scope':'BIT OPS ONLY; generic shared BIT OPS ONLY; not async TMA/ring protocol, full SK, Hopper, projection math, latency/performance or H200 qualification'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('output',type=Path);p.add_argument('--nvrtc',type=Path,default=Path('/root/.cache/okc-cuda128/packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'));run(p.parse_args())
