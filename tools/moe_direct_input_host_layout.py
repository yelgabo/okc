"""Execute the actual native address code on a CPU, excluding CUDA synchronization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from moe_direct_input import HELPER

PREFIX='''#include <cstdint>
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
'''
MAIN='''
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
  std::cout<<"1040 count/stage cases, "<<words<<" uint32 values, exact and guarded\\n";
}
'''


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    helper=HELPER.replace('__device__ __forceinline__ ','')
    fence='  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n'
    assert helper.count(fence)==1
    helper=helper.replace(fence,'')
    barrier='  asm volatile("bar.sync 8, 32;\\n" ::: "memory");\n'
    assert helper.count(barrier)==1
    helper=helper.replace(barrier,'')
    records=[]
    for key,code in [('actual',helper),('injected-row-swap',helper.replace('(r & 7)]','((r & 7) ^ 1)]'))]:
        assert key=='actual' or code!=helper
        cpp=out/(key+'.cpp');cpp.write_text(PREFIX+code+MAIN)
        binary=out/(key+'.bin')
        command=['clang++','-O2','-std=c++17','-fno-strict-aliasing',str(cpp),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,text=True)
        (out/(key+'.compile.txt')).write_text(compiled.stdout+compiled.stderr)
        assert compiled.returncode==0,compiled.stderr
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        (out/(key+'.run.txt')).write_text(result.stdout+result.stderr)
        assert result.returncode==(0 if key=='actual' else 1),(key,result.returncode)
        records.append(dict(case=key,compile_exit=compiled.returncode,run_exit=result.returncode,source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest()))
    summary=dict(native_helper_sha256=hashlib.sha256(HELPER.encode()).hexdigest(),records=records,
                 scope='Actual native index/load/store code under CPU types; CUDA fences/synchronization excluded. No GPU execution or timing claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().out),indent=2))
