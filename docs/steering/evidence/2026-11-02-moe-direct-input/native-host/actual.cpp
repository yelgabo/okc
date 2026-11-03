#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
constexpr int HD=4096;
struct uint4 { unsigned x,y,z,w; };
uint4 make_uint4(unsigned x,unsigned y,unsigned z,unsigned w){return {x,y,z,w};}
struct { int x; } threadIdx;
struct Params { const uint16_t* x; };
struct Smem { int jC[4]{},jS[4]{},ent[2048]{}; };
unsigned __ldg(const unsigned* p){return *p;}
void __syncwarp(){}
void wgs_direct_x(const Params& p,const Smem& S,unsigned char* dst,int j,int st,int npad) {
  const int lane = threadIdx.x & 31;
  uint4* out = reinterpret_cast<uint4*>(dst);
#pragma unroll 1
  for (int rb = 0; rb < npad; rb += 8) {
    const int r = rb + (lane & 7);
    const bool valid = r < S.jC[j];
    const int token = valid ? (S.ent[S.jS[j] + r] >> 3) : 0;
    const unsigned* src = reinterpret_cast<const unsigned*>(p.x + (size_t)token * HD + st * 256);
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int G = z * 4 + (lane >> 3);
      const int b = 8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1);
      const uint4 value = valid ? make_uint4(__ldg(src + b),__ldg(src + b + 32),__ldg(src + b + 64),__ldg(src + b + 96)) : make_uint4(0,0,0,0);
      out[(((r >> 3) * 32 + G) * 8) + (r & 7)] = value;
    }
  }
}

int main(){
  std::vector<unsigned> data(257*2048);
  for(size_t i=0;i<data.size();++i)data[i]=unsigned(i*1664525u+1013904223u);
  Params p{reinterpret_cast<const uint16_t*>(data.data())};
  const uint4 guard{0xdeadbeef,0xcafebabe,0x12345678,0xabcdef01};
  size_t words=0;
  for(int count=0;count<=64;++count){
    Smem s;const int j=3;s.jC[j]=count;s.jS[j]=7;
    for(int r=0;r<count;++r)s.ent[7+r]=(((r*13+r/3)%257)<<3)|(r&7);
    const int padded=(count+7)/8*8,ntt=padded/8;
    for(int stage=0;stage<16;++stage){
      std::vector<uint4> out(padded*32+2,guard);
      for(int lane=0;lane<32;++lane){threadIdx.x=512+32+lane;wgs_direct_x(p,s,reinterpret_cast<unsigned char*>(out.data()+1),j,stage,padded);}
      if(std::memcmp(&out.front(),&guard,16)||std::memcmp(&out.back(),&guard,16))return 2;
      for(int r=0;r<padded;++r)for(int g=0;g<32;++g){
        const int original=(((stage*ntt+r/8)*32+g)*8+r%8)-stage*padded*32;
        uint4 expected{};
        if(r<count){
          const size_t pos=(s.ent[7+r]>>3)*2048+stage*128+8*((g>>1)&3)+2*(g>>3)+(g&1);
          expected=make_uint4(data[pos],data[pos+32],data[pos+64],data[pos+96]);
        }
        if(std::memcmp(&out[1+original],&expected,16))return 1;
        words+=4;
      }
    }
  }
  std::cout<<"1040 count/stage cases, "<<words<<" uint32 values, exact and guarded\n";
}
