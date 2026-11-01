"""Generate an actual lossless pair7 GEMV trial; control judged path is unchanged.

python3 tools/proj_make_bench_pair7.py OUT.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
__device__ float psum(float v) { for(int s=16;s;s>>=1) v+=__shfl_down_sync(0xffffffff,v,s); return v; }
template<int NT> __global__ void p7(const unsigned char* raw,const unsigned char* codes,
 const int* lut,const int4* tab,int pm,const __nv_bfloat16* x,__nv_bfloat16* out,int K) {
 __shared__ float red[2][NT/32+1]; int t=threadIdx.x, lane=t&31, warp=t>>5,c=blockIdx.x;
 float a=0,b=0;
 if(t<NT) {
  for(int row=0;row<2;row++) {
   float acc=0;
   for(int j=t;j<K/16;j+=NT) {
    uint4 lo=reinterpret_cast<const uint4*>(raw+(size_t)(2*c+row)*K)[j];
    uint4 xa=reinterpret_cast<const uint4*>(x)[2*j], xb=reinterpret_cast<const uint4*>(x)[2*j+1];
    const unsigned char* lp=reinterpret_cast<const unsigned char*>(&lo);
    const unsigned short* xp0=reinterpret_cast<const unsigned short*>(&xa);
    const unsigned short* xp1=reinterpret_cast<const unsigned short*>(&xb);
    const unsigned char* cp=codes+(size_t)(2*c+row)*(K/16)*7+(j/32)*224+(j&31);
    unsigned long long bits=0;
    #pragma unroll
    for(int q=0;q<7;q++) bits|=(unsigned long long)cp[q*32]<<(8*q);
    #pragma unroll
    for(int p=0;p<8;p++) {
     unsigned ee=__ldg(lut+((bits>>(7*p))&127));
     #pragma unroll
     for(int h=0;h<2;h++) {
      int k=2*p+h; unsigned low=lp[k];
      unsigned w=(low&127)|((low&128)<<8)|(((ee>>(8*h))&255)<<7);
      unsigned xv=k<8?xp0[k]:xp1[k-8];
      acc=__fmaf_rn(__uint_as_float(w<<16),__uint_as_float(xv<<16),acc);
     }
    }
   }
   acc=psum(acc); if(lane==0) red[row][warp]=acc;
  }
 } else {
  const int4* tb=tab+(size_t)c*pm; int count=tb[0].x;
  for(int i=lane+1;i<=count;i+=32) {
   int4 e=tb[i]; float delta=__fsub_rn(__uint_as_float((unsigned)e.y<<16),__uint_as_float((unsigned)e.z<<16));
   unsigned xv=__ldg(reinterpret_cast<const unsigned short*>(x)+(e.x&65535));
   float d=__fmul_rn(delta,__uint_as_float(xv<<16));
   if(e.x>>16) b=__fadd_rn(b,d); else a=__fadd_rn(a,d);
  }
  a=psum(a);b=psum(b);if(lane==0){red[0][NT/32]=a;red[1][NT/32]=b;}
 }
 __syncthreads();
 if(t==0){float aa=0,bb=0;for(int w=0;w<=NT/32;w++){aa+=red[0][w];bb+=red[1][w];}out[2*c]=__float2bfloat16_rn(aa);out[2*c+1]=__float2bfloat16_rn(bb);}
}
torch::Tensor p7_fwd(torch::Tensor raw,torch::Tensor codes,torch::Tensor lut,torch::Tensor tab,torch::Tensor x,int64_t nt){
 int N=raw.size(0),K=raw.size(1),pm=tab.size(1);auto out=torch::empty({1,N},x.options());
 #define GO(T) p7<T><<<N/2,T+32,0,at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<unsigned char>(),codes.data_ptr<unsigned char>(),lut.data_ptr<int>(),reinterpret_cast<const int4*>(tab.data_ptr<int>()),pm,reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),K)
 if(nt==64){GO(64);}else if(nt==128){GO(128);}else if(nt==256){GO(256);}else{GO(512);}
 return out;
}
'''

VECTOR_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
__device__ float psum(float v){for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);return v;}
__device__ __forceinline__ float lf(unsigned x){return __uint_as_float(x<<16);}
__device__ __forceinline__ float hf(unsigned x){return __uint_as_float(x&0xffff0000u);}
__device__ __forceinline__ void f4(unsigned l,unsigned h,unsigned x0,unsigned x1,float& a){
 unsigned p=__byte_perm(l,h,0x5140),q=__byte_perm(l,h,0x7362);
 a=fmaf(lf(p),lf(x0),a);a=fmaf(hf(p),hf(x0),a);a=fmaf(lf(q),lf(x1),a);a=fmaf(hf(q),hf(x1),a);
}
__device__ __forceinline__ void p7fma(uint4 raw,unsigned long long c,uint4 xa,uint4 xb,const unsigned* lut,float& acc){
 unsigned l[4]={0,0,0,0},h[4]={0,0,0,0};const unsigned short* r=reinterpret_cast<const unsigned short*>(&raw);
 #pragma unroll
 for(int p=0;p<8;p++){
  unsigned z=lut[(c>>(7*p))&127],rp=r[p];
  l[p/2]|=((rp&0x7f7f)|(z&65535))<<((p&1)*16);
  h[p/2]|=((rp&0x8080)|(z>>16))<<((p&1)*16);
 }
 f4(l[0],h[0],xa.x,xa.y,acc);f4(l[1],h[1],xa.z,xa.w,acc);
 f4(l[2],h[2],xb.x,xb.y,acc);f4(l[3],h[3],xb.z,xb.w,acc);
}
template<int NT,int CPT> __global__ void p7(const unsigned char* raw,const unsigned char* codes,
 const int* lut,const int4* tab,int pm,const __nv_bfloat16* x,__nv_bfloat16* out,int K){
 __shared__ float red[2][NT/32+1];__shared__ unsigned sl[128];
 int t=threadIdx.x,lane=t&31,warp=t>>5,c=blockIdx.x;
 for(int i=t;i<128;i+=NT+32)sl[i]=lut[i];__syncthreads();
 if(t<NT){
  uint4 l0[CPT],l1[CPT],xa[CPT],xb[CPT];unsigned long long q0[CPT],q1[CPT];
  #pragma unroll
  for(int i=0;i<CPT;i++){
   int j=i*NT+t;l0[i]=reinterpret_cast<const uint4*>(raw+(size_t)(2*c)*K)[j];
   l1[i]=reinterpret_cast<const uint4*>(raw+(size_t)(2*c+1)*K)[j];
   const unsigned char* p=codes+(size_t)(2*c)*(K/16)*7+(j/32)*224+(j&31);
   q0[i]=0;q1[i]=0;
   #pragma unroll
   for(int b=0;b<7;b++){q0[i]|=(unsigned long long)p[b*32]<<(8*b);q1[i]|=(unsigned long long)p[(K/16)*7+b*32]<<(8*b);}
  }
  #pragma unroll
  for(int i=0;i<CPT;i++){int j=i*NT+t;xa[i]=reinterpret_cast<const uint4*>(x)[2*j];xb[i]=reinterpret_cast<const uint4*>(x)[2*j+1];}
  float a=0,b=0;
  #pragma unroll
  for(int i=0;i<CPT;i++){p7fma(l0[i],q0[i],xa[i],xb[i],sl,a);p7fma(l1[i],q1[i],xa[i],xb[i],sl,b);}
  a=psum(a);b=psum(b);if(lane==0){red[0][warp]=a;red[1][warp]=b;}
 }else{
  const int4* tb=tab+(size_t)c*pm;int count=tb[0].x;float a=0,b=0;
  for(int i=lane+1;i<=count;i+=32){int4 e=tb[i];float d=__fsub_rn(lf((unsigned)e.y),lf((unsigned)e.z));
   unsigned xv=__ldg(reinterpret_cast<const unsigned short*>(x)+(e.x&65535));d=__fmul_rn(d,lf(xv));
   if(e.x>>16)b=__fadd_rn(b,d);else a=__fadd_rn(a,d);}
  a=psum(a);b=psum(b);if(lane==0){red[0][NT/32]=a;red[1][NT/32]=b;}
 }
 __syncthreads();if(t==0){float a=0,b=0;for(int i=0;i<=NT/32;i++){a+=red[0][i];b+=red[1][i];}out[2*c]=__float2bfloat16_rn(a);out[2*c+1]=__float2bfloat16_rn(b);}
}
torch::Tensor p7_fwd(torch::Tensor raw,torch::Tensor codes,torch::Tensor lut,torch::Tensor tab,torch::Tensor x,int64_t nt){
 int N=raw.size(0),K=raw.size(1),pm=tab.size(1);auto out=torch::empty({1,N},x.options());
 #define GO(T,C) p7<T,C><<<N/2,T+32,0,at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<unsigned char>(),codes.data_ptr<unsigned char>(),lut.data_ptr<int>(),reinterpret_cast<const int4*>(tab.data_ptr<int>()),pm,reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),K)
 if(nt==64){GO(64,1);}else if(nt==128){GO(128,2);}else if(nt==256){GO(256,2);}else{GO(512,1);}
 return out;
}
template<int N,int C> std::vector<int64_t> pinfo(){cudaFuncAttributes a{};TORCH_CHECK(cudaFuncGetAttributes(&a,p7<N,C>)==cudaSuccess,"p7 attributes");int b=0;TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,p7<N,C>,N+32,0)==cudaSuccess,"p7 occupancy");return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,b};}
std::vector<int64_t> p7_info(int64_t nt){if(nt==64)return pinfo<64,1>();if(nt==128)return pinfo<128,2>();if(nt==256)return pinfo<256,2>();return pinfo<512,1>();}
'''

RUN = r'''
# Actual pair7 packed trial; all packing is weight-only and outside timed calls.
import random as _p7random
import statistics as _p7stats
_P7_LINES=[]
_P7_VECTOR=@@VECTOR@@
def _p7print(s):
    _P7_LINES.append(s);print(s,flush=True)
def _p7pack(w):
    n,k=w.shape;b=w.view(torch.int16).to(torch.int32)&65535;e=(b>>7)&255
    pairs=e.reshape(n,k//2,2);key=pairs[:,:,0]+256*pairs[:,:,1]
    hist=torch.bincount(key.flatten().long(),minlength=65536)
    lut=hist.argsort(descending=True)[:128].to(torch.int32)
    mapping=torch.empty(65536,dtype=torch.int64,device=w.device)
    le0=lut&255;le1=lut>>8
    for start in range(0,65536,4096):
        v=torch.arange(start,start+4096,device=w.device)
        matches=((v[:,None]&255)==le0).int()+((v[:,None]>>8)==le1).int()
        mapping[start:start+4096]=matches.argmax(1)
    mapping[lut.long()]=torch.arange(128,device=w.device)
    idx=mapping[key.long()];dec=lut[idx].long()
    de=torch.stack([dec&255,dec>>8],-1).reshape(n,k)
    low=((b&127)|((b>>8)&128)).to(torch.uint8).contiguous()
    decoded=(low.int()&127)|((low.int()&128)<<8)|(de.int()<<7)
    esc=decoded!=b
    ci,ri,ki=esc.reshape(n//2,2,k).nonzero().unbind(1)
    count=torch.bincount(ci,minlength=n//2);pm=int(count.max())+1
    pos=torch.arange(len(ci),device=w.device)-(count.cumsum(0)-count)[ci]+1
    tab=torch.zeros((n//2,pm,4),dtype=torch.int32,device=w.device)
    tab[:,0,0]=count.int();tab[ci,pos,0]=((ri<<16)|ki).int()
    tab[ci,pos,1]=b.reshape(n//2,2,k)[ci,ri,ki]
    tab[ci,pos,2]=decoded.reshape(n//2,2,k)[ci,ri,ki]
    groups=idx.reshape(n,k//16,8);packed=torch.zeros((n,k//16),dtype=torch.int64,device=w.device)
    for j in range(8):packed|=groups[:,:,j]<<(7*j)
    cp=torch.stack([(packed>>(8*j))&255 for j in range(7)],-1).reshape(n,k//512,32,7).transpose(2,3).contiguous().byte()
    # Decode stored bytes, not idx, for an exact weight roundtrip gate.
    rec=cp.transpose(2,3).reshape(n,k//16,7).long();rr=torch.zeros_like(packed)
    for j in range(7):rr|=rec[:,:,j]<<(8*j)
    ii=torch.stack([(rr>>(7*j))&127 for j in range(8)],-1).reshape(n,k//2)
    dd=lut[ii].long();ee=torch.stack([dd&255,dd>>8],-1).reshape(n,k)
    restored=(low.int()&127)|((low.int()&128)<<8)|(ee.int()<<7)
    restored.reshape(n//2,2,k)[ci,ri,ki]=tab[ci,pos,1]
    assert torch.equal(restored,b),'pair7 stored-byte roundtrip failed'
    if _P7_VECTOR:
        vl=((lut&1)<<7)|(((lut>>8)&1)<<15)|(((lut&255)>>1)<<16)|((lut>>9)<<24)
        inv=(((vl>>16)&127)*2+((vl>>7)&1))|((((vl>>24)&127)*2+((vl>>15)&1))<<8)
        assert torch.equal(inv,lut),'vector LUT roundtrip failed'
        lut=vl.int()
    return low,cp,lut.contiguous(),tab
def _p7bench(weights):
    from torch.utils.cpp_extension import load_inline
    decl='torch::Tensor p7_fwd(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);'
    if _P7_VECTOR:decl+='std::vector<int64_t> p7_info(int64_t);'
    ext=load_inline(name='okc_owned_pair7_vector' if _P7_VECTOR else 'okc_owned_pair7_trial',cpp_sources=decl,cuda_sources=@@CUDA@@,functions=['p7_fwd']+(['p7_info'] if _P7_VECTOR else []),extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    ctrl=_build_ext();dev=weights['pooler'].device;gen=torch.Generator(device=dev).manual_seed(61031)
    flush=torch.empty(256*1024*1024,dtype=torch.uint8,device=dev)
    for name in ('qkv_gate','o','pooler','unpooler'):
        w=weights[name];w=w[:1024] if name=='unpooler' else w
        pack=_p7pack(w);nt=cgemv_threads(*w.shape);old=pack_cgemv(w)
        if old is None:raise RuntimeError('control pack unavailable')
        if _P7_VECTOR:_p7print(f'P7RES {name} nt={nt} regs/local/shared/blocks={ext.p7_info(nt)}')
        x=torch.randn((1,w.shape[1]),generator=gen,device=dev).to(torch.bfloat16)
        def prod():
            out=torch.empty((1,w.shape[0]),dtype=torch.bfloat16,device=dev)
            ctrl.cgemv_fwd(old['stream'],old['table'],old['pmax'],old['base4'],x,out,nt);return out
        def trial():return ext.p7_fwd(*pack,x,nt)
        for probe in (False,True):
            x.normal_(generator=gen)
            if probe:x.zero_();x[0,17]=2
            ref=x.double()@w.double().T
            trials=[('prod',prod),('p7',trial)]
            if _P7_VECTOR and name=='pooler':trials.append(('p7nt256',lambda:ext.p7_fwd(*pack,x,256)))
            for nm,fn in trials:
                out=fn();err=float(torch.linalg.vector_norm(out.double()-ref)/torch.linalg.vector_norm(ref))
                assert err<0.005,(name,nm,err)
            _p7print(f'P7CHECK {name} probe={probe} PASS')
        x.normal_(generator=gen);graphs={}
        variants=[('prod',prod),('null',prod),('p7',trial)]
        if _P7_VECTOR and name=='pooler':
            variants.append(('p7nt256',lambda:ext.p7_fwd(*pack,x,256)))
            _p7print(f'P7RES pooler nt=256 regs/local/shared/blocks={ext.p7_info(256)}')
        for nm,fn in variants:
            fn();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            graphs[nm]=g
        times={nm:[] for nm in graphs};rng=_p7random.Random(61031)
        for rd in range(100):
            order=list(graphs);rng.shuffle(order)
            for nm in order:
                flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                a.record();graphs[nm].replay();z.record();z.synchronize();times[nm].append(1000*a.elapsed_time(z))
        delta={nm:_p7stats.median(a-b for a,b in zip(v,times['prod'])) for nm,v in times.items()}
        _p7print(f'P7 {name} deltas_us={delta} bytes={sum(t.numel()*t.element_size() for t in pack)} oldbytes={old["stream"].numel()+old["table"].numel()*4}')
_p7original_prepare=prepare
def prepare(weights):
    _p7bench(weights)
    state=_p7original_prepare(weights)
    for line in _P7_LINES:print(line,flush=True)
    return state
'''

def cpu_gate():
    # Every BF16 bit pattern, both lanes of a pair, arbitrary exponent LUT fallback.
    for b in range(65536):
        raw=(b&127)|((b>>8)&128);e=(b>>7)&255
        dec=(raw&127)|((raw&128)<<8)|(e<<7)
        assert dec==b
        assert ((raw&127)|((e&1)<<7)|(((e>>1)|(raw&128))<<8))==b
    vals=[(j*37)&127 for j in range(8)]
    packed=sum(v<<(7*j) for j,v in enumerate(vals))
    data=packed.to_bytes(7,'little');restored=int.from_bytes(data,'little')
    assert [(restored>>(7*j))&127 for j in range(8)]==vals
    # Exercise the real pair lookup, stored 7-byte words, and sparse escapes over
    # all BF16 patterns in both pair positions (not merely the raw-byte identity).
    from collections import Counter
    originals=[(b,(b*4051+17)&65535) for b in range(65536)]
    keys=[((a>>7)&255)|(((b>>7)&255)<<8) for a,b in originals]
    lut=[k for k,_ in Counter(keys).most_common(128)]
    indices={k:i for i,k in enumerate(lut)}
    for key in set(keys)-indices.keys():
        indices[key]=max(range(128),key=lambda i:
            int((lut[i]&255)==(key&255))+int((lut[i]>>8)==(key>>8)))
    codes=[indices[k] for k in keys]
    stored=b''.join(sum(codes[c+j]<<(7*j) for j in range(8)).to_bytes(7,'little')
                    for c in range(0,len(codes),8))
    escapes={}
    raw=[]
    for i,(a,b) in enumerate(originals):
        for h,v in enumerate((a,b)):
            low=(v&127)|((v>>8)&128);raw.append(low)
            decoded=(low&127)|((low&128)<<8)|(((lut[codes[i]]>>(8*h))&255)<<7)
            if decoded!=v:escapes[2*i+h]=v
    for i,(a,b) in enumerate(originals):
        word=int.from_bytes(stored[(i//8)*7:(i//8+1)*7],'little')
        ee=lut[(word>>(7*(i%8)))&127]
        for h,v in enumerate((a,b)):
            low=raw[2*i+h]
            decoded=(low&127)|((low&128)<<8)|(((ee>>(8*h))&255)<<7)
            assert escapes.get(2*i+h,decoded)==v

if __name__=='__main__':
    cpu_gate()
    src=(ROOT/'muse_bf16_proj/submission.py').read_text()
    vector='--vector' in sys.argv
    Path(sys.argv[1]).write_text(src+RUN.replace('@@CUDA@@',repr(VECTOR_CUDA if vector else CUDA)).replace('@@VECTOR@@',repr(vector)))
    print('CPU all-BF16/56-bit packing gate PASS; wrote',sys.argv[1])
