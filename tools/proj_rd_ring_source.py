"""Offline source/ownership gates for one rd16 cp.async ring hypothesis.

Emits CUDA only; does not dispatch a GPU job or change production.
"""
from pathlib import Path
import ast
import sys

ROOT=Path(__file__).resolve().parents[1]
HELPER=r'''
__device__ __forceinline__ void rr_copy(void* dst,const void* src){
 unsigned d=(unsigned)__cvta_generic_to_shared(dst);
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(d),"l"(src):"memory");
}
__device__ __forceinline__ void rr_commit(){asm volatile("cp.async.commit_group;":::"memory");}
'''
PIPE=r'''
    static_assert(VB==0 && KU==8 && MT==2 && WR==2 && WK==16,"bounded rd16 ring only");
    extern __shared__ __align__(16) uint8_t rr_smem[];
    // Every lane writes and reads its own three 16-byte spans. No cross-thread reader.
#pragma unroll
    for(int p=0;p<2;++p){
        uint8_t* dst=rr_smem+((p*NWARP+warp)*1536)+lane*16;
        const uint8_t* src=wp+(size_t)p*ustride;
        rr_copy(dst,src);rr_copy(dst+512,src+512);rr_copy(dst+1024,src+1024);rr_commit();
    }
#pragma unroll
    for(int s=0;s<KU;++s){
        // Two groups until s=6. At s=7 only the final group remains.
        if(s+1<KU)asm volatile("cp.async.wait_group 1;":::"memory");
        else asm volatile("cp.async.wait_group 0;":::"memory");
        __syncwarp();
        const uint8_t* src=rr_smem+(((s%2)*NWARP+warp)*1536)+lane*16;
        const uint4 l0=*reinterpret_cast<const uint4*>(src);
        const uint4 l1=*reinterpret_cast<const uint4*>(src+512);
        const uint4 cw=*reinterpret_cast<const uint4*>(src+1024);
        // Source-language reads finish before this lane overwrites its own slot.
        // No other lane reads those spans, so a CTA barrier is not required here.
        if(s+2<KU){
            uint8_t* dst=rr_smem+(((s%2)*NWARP+warp)*1536)+lane*16;
            const uint8_t* next=wp+(size_t)(s+2)*ustride;
            rr_copy(dst,next);rr_copy(dst+512,next+512);rr_copy(dst+1024,next+1024);rr_commit();
        }
'''
HOST=r'''
} // namespace
std::vector<int64_t> rr_info(bool ring,bool memory){
 cudaFuncAttributes a{};int blocks=0;const void* f;
 if(ring)f=memory?(const void*)rr_ring<2,2,16,8,1,0,true>:(const void*)rr_ring<2,2,16,8,1,0,false>;
 else f=memory?(const void*)rr_old<2,2,16,8,1,0,true>:(const void*)rr_old<2,2,16,8,1,0,false>;
 int dynamic=ring?98304:0;
 if(ring)C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,f));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,1024,dynamic));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,dynamic,blocks};
}
void rr_fwd(at::Tensor ws,at::Tensor esc,at::Tensor x,at::Tensor out,int64_t base,bool ring,bool memory){
 TORCH_CHECK(x.is_contiguous() && x.size(1)==8192 && x.size(0)>=1 && x.size(0)<=16,"rr x bounds");
 TORCH_CHECK(out.size(1)==4096 && out.size(0)==x.size(0),"rr out bounds");
 auto s=at::cuda::getCurrentCUDAStream();auto w=ws.data_ptr<uint8_t>();
 auto e=reinterpret_cast<const uint32_t*>(esc.data_ptr<int>());
 auto a=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());auto o=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
 int m=x.size(0);
 #define RRGO(F,MEM,SM) F<2,2,16,8,1,0,MEM><<<128,1024,SM,s>>>(w,e,a,o,m,4096,8192,(uint32_t)base)
 if(ring){if(memory){RRGO(rr_ring,true,98304);}else{RRGO(rr_ring,false,98304);}}
 else{if(memory){RRGO(rr_old,true,0);}else{RRGO(rr_old,false,0);}}
 #undef RRGO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

def literal(source,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))

def generate(source):
    sk=literal(source,'_SK_SRC');rd=literal(source,'_RD_SRC')
    helpers=sk[sk.index('__device__ __forceinline__ void mma_bf16'):sk.index('template <int MT, int TOKR')]
    rd=rd[:rd.index('#ifndef SK_DEVICE_ONLY')]
    rd=rd.replace('int VB = 0>','int VB = 0, bool MEMORY = false>',1)
    mma='                mma_bf16(acc[nt], av, xw[nt][2 * b], xw[nt][2 * b + 1]);'
    assert rd.count(mma)==1
    rd=rd.replace(mma,'''                if constexpr(MEMORY){
                    asm volatile(""::"r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),
                        "r"(xw[nt][2*b]),"r"(xw[nt][2*b+1]):"memory");
                }else mma_bf16(acc[nt], av, xw[nt][2 * b], xw[nt][2 * b + 1]);''')
    start=rd.index('    // A ring of PF + 1 units in registers;')
    end=rd.index('        const int k0 = ',start)
    old=rd.replace('rd_kernel(', 'rr_old(',1)
    ring=(rd[:start]+PIPE+rd[end:]).replace('rd_kernel(', 'rr_ring(',1)
    # Every math/escape/output statement after obtaining raw words is source identical.
    assert old[old.index('        const int k0 = '):]==ring[ring.index('        const int k0 = '):]
    assert old.count('__syncthreads();')==ring.count('__syncthreads();')
    prefix='#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\nnamespace {\n'
    # Each fragment was originally in an anonymous namespace; strip that one opening.
    old=old.replace('namespace {','',1)
    ring=ring[ring.index('template <int MT, int WR, int WK, int KU, int PF, int VB = 0, bool MEMORY = false>'):]
    return (prefix+helpers+HELPER+old+ring+HOST).replace('RD_ESCN','32')

def cpu_gate():
    # Track actual per-thread async group issue/completion and private slot lifetime.
    for warp in range(32):
        pending=[0,1];slots={0:0,1:1}
        for s in range(8):
            keep=1 if s<7 else 0
            completed=[]
            while len(pending)>keep:completed.append(pending.pop(0))
            assert s in completed
            assert slots[s%2]==s
            if s+2<8:slots[s%2]=s+2;pending.append(s+2)
        assert not pending
    # Three planes partition each stage/warp buffer without collisions or misalignment.
    spans=[]
    for stage in range(2):
        for warp in range(32):
            for lane in range(32):
                for plane in range(3):
                    off=(stage*32+warp)*1536+lane*16+plane*512
                    assert off%16==0 and 0<=off and off+16<=98304
                    spans.extend(range(off,off+16))
    assert sorted(spans)==list(range(98304))
    # Global units cover each full row group once, including last output-row group.
    addresses=[]
    for block in range(128):
        for wr in range(2):
            for wk in range(16):
                for step in range(8):
                    addresses.append(((wk*8+step)*256+block*2+wr)*1536)
    assert sorted(addresses)==list(range(0,256*128*1536,1536))
    for m in (1,7,15,16):
        written=[]
        for block in range(128):
            for i in range(2*2*32):
                ln=i&31;nt=(i>>5)%2;wr=i//64
                row=(block*2+wr)*16+(ln>>2);t=nt*8+2*(ln&3)
                for token in (t,t+1):
                    if token<m:written.extend(((token,row),(token,row+8)))
        assert sorted(written)==[(t,n) for t in range(m) for n in range(4096)]
    assert 2*32*1536+32*2*32*16==131072
    print('CPU async-group, private-span, full global coverage and shared bounds PASS')

if __name__=='__main__':
    cpu_gate();source=(ROOT/'muse_bf16_proj/submission.py').read_text()
    Path(sys.argv[1]).write_text(generate(source))
    print('Wrote offline CUDA; unchanged escape/math/reduction source gate PASS')
