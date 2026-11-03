typedef unsigned int uint32_t;typedef unsigned short uint16_t;
__device__ __forceinline__ uint32_t sx_lds32(uint32_t address) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(address) : "memory");
    return v;
}
// Header and all packed operands have been copied from the ring before release.
// Only immutable global records are read here. Every lane executes the same
// count/loop; lane ownership masks XOR without divergent MMA participation.
__device__ __forceinline__ void sx_repair(uint32_t (&b)[8], uint32_t h,
                                         const uint16_t* records, int lane) {
    const unsigned count=h&1023u, start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=records[start+i];
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane ? x : 0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r ? mask : 0u;
    }
}


extern "C" __global__ void probe(const uint32_t* input,uint32_t* output,
 const uint32_t* headers,const uint16_t* records,int units){
 int u=blockIdx.x,lane=threadIdx.x;if(u>=units)return;
 uint32_t b[8];
 #pragma unroll
 for(int r=0;r<8;++r)b[r]=input[(u*32+lane)*8+r];
 sx_repair(b,headers[u],records,lane);
 #pragma unroll
 for(int r=0;r<8;++r)output[(u*32+lane)*8+r]=b[r];
}
