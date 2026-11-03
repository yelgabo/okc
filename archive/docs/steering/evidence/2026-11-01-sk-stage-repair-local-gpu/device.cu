typedef unsigned int uint32_t;typedef unsigned short uint16_t;
__device__ __forceinline__ unsigned sxs_lds16(uint32_t address) {
    unsigned value;
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}
// Same warp-uniform record order/masks as the global variant. Every record
// access is now part of the credited ring stage, and precedes its empty arrive.
__device__ __forceinline__ void sxs_repair(uint32_t (&b)[8],uint32_t h,
                                          uint32_t arena,int lane) {
    const unsigned count=h&1023u,start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=sxs_lds16(arena+2*(start+i));
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane?x:0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r?mask:0u;
    }
}


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
