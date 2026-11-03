#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
constexpr int HD=4096;
struct __nv_bfloat16 { uint16_t bits; };
struct __nv_bfloat162 { __nv_bfloat16 x,y; };
struct float2 { float x,y; };
float2 make_float2(float x,float y){return {x,y};}
float frombits(unsigned raw){float out;std::memcpy(&out,&raw,4);return out;}
unsigned bits(float value){unsigned out;std::memcpy(&out,&value,4);return out;}
float2 __bfloat1622float2(__nv_bfloat162 value){return {frombits(unsigned(value.x.bits)<<16),frombits(unsigned(value.y.bits)<<16)};}
float fork_bf16(float value){unsigned raw=bits(value);raw=(raw+0x7fff+((raw>>16)&1))&0xffff0000u;return frombits(raw);}
float __fmul_rn(float left,float right){volatile float value=left*right;return value;}
unsigned __ldg(const unsigned* ptr){return *ptr;}
unsigned __ldcg(const unsigned* ptr){return *ptr;}
struct Params { float* ybuf; const __nv_bfloat16* normw; };
#define OKC_YFACTOR 1
float2 wgs_factor_pair(uint32_t packed,const __nv_bfloat16* weights) {
  const float2 down = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
  const uint32_t word = __ldg(reinterpret_cast<const uint32_t*>(weights));
  const float2 raw = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word));
  return make_float2(fork_bf16(__fmul_rn(down.x,fork_bf16(1.f + raw.x))),__fmul_rn(down.y,fork_bf16(1.f + raw.y)));
}
float2 wgs_factor_y2(const Params& p,int row,int col) {
  const uint32_t word = __ldcg(reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)row * HD + col));
  return wgs_factor_pair(word,p.normw + col);
}

int main(){
  const unsigned ws[]={0,0x8000,0x3c00,0x3b80,0xbf80,0xbf81,0xbf7f,0x3f80,0x3f81,0x7f7f,0xff7f,1,0x8001,0x0080,0x8080,0x4000};
  size_t products=0;
  for(unsigned raw=0;raw<65536;++raw){
    if((raw&0x7f80)==0x7f80)continue;
    const unsigned packed=raw|((raw^0x8000)<<16);
    for(unsigned w:ws){
      __nv_bfloat16 weights[2]={{uint16_t(w)},{uint16_t(w^0x8000)}};
      const float2 got=wgs_factor_pair(packed,weights);
      const float first=frombits(raw<<16),second=frombits((raw^0x8000)<<16);
      const float gx=fork_bf16(1.f+frombits(w<<16)),gy=fork_bf16(1.f+frombits((w^0x8000)<<16));
      const float ex=__fmul_rn(first,gx),ey=__fmul_rn(second,gy);
      if(bits(got.x)!=bits(ex)||bits(got.y)!=bits(ey))return 1;
      products+=2;
    }
  }
  std::vector<__nv_bfloat16> weights(HD),buffer(4*HD);
  for(int c=0;c<HD;++c)weights[c].bits=uint16_t(ws[c%16]);
  for(int i=0;i<4*HD;++i)buffer[i].bits=uint16_t(0x3e00+(i%512));
  Params p{reinterpret_cast<float*>(buffer.data()),weights.data()};
  for(int row=0;row<4;++row)for(int col=0;col<HD;col+=2){
    const float2 got=wgs_factor_y2(p,row,col);
    for(int h=0;h<2;++h){
      const float down=frombits(unsigned(buffer[row*HD+col+h].bits)<<16);
      const float gamma=fork_bf16(1.f+frombits(unsigned(weights[col+h].bits)<<16));
      if(bits(h?got.y:got.x)!=bits(__fmul_rn(down,gamma)))return 2;
    }
  }
  std::cout<<products<<" products and8192 paired row/column loads, exact\n";
}
