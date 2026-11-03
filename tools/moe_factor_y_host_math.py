"""Execute actual native reconstruction code with CPU BF16 bit operations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from moe_factor_y import HELPER

PREFIX='''#include <cstdint>
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
'''
MAIN='''
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
  std::cout<<products<<" products and8192 paired row/column loads, exact\\n";
}
'''


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    helper=HELPER.replace('__device__ __forceinline__ ','')
    expression='__fmul_rn(down.x,fork_bf16(1.f + raw.x))'
    assert helper.count(expression)==1
    results=[]
    for label,code in [('actual',helper),('injected-post-gamma-round',helper.replace(expression,'fork_bf16('+expression+')'))]:
        source=out/(label+'.cpp');source.write_text(PREFIX+code+MAIN);binary=out/(label+'.bin')
        command=['clang++','-O2','-std=c++17','-ffp-contract=off','-fno-strict-aliasing',str(source),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,text=True);(out/(label+'.compile.txt')).write_text(compiled.stdout+compiled.stderr)
        assert compiled.returncode==0,compiled.stderr
        run=subprocess.run([str(binary)],capture_output=True,text=True);(out/(label+'.run.txt')).write_text(run.stdout+run.stderr)
        assert run.returncode==(0 if label=='actual' else 1),(label,run.returncode)
        results.append(dict(case=label,compile_exit=compiled.returncode,run_exit=run.returncode,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    summary=dict(helper_sha256=hashlib.sha256(HELPER.encode()).hexdigest(),results=results,
                 scope='Actual native helper under CPU bit-conversion and float-multiply stubs. No CUDA execution or H200 arithmetic claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().out),indent=2))
