"""Exact generic pair-codebook alternative for the selected WGS decoder."""
import argparse
import json
from pathlib import Path
import struct

from cuda_issue_groups import generate,compile_pair


def table():
    grid=(0.,.5,1.,1.5,2.,3.,4.,6.)
    result=[]
    for sf in range(64):
        scale=(1+(sf&7)/8)*2.**((sf>>3)+1-62)
        for pair in range(256):
            values=[]
            for code in (pair&15,pair>>4):
                value=grid[code&7]*scale
                if code&8:value=-value
                bits=struct.unpack('<I',struct.pack('<f',value))[0]
                values.append(((bits+0x7fff+((bits>>16)&1))>>16)&0xffff)
            word=values[0]|values[1]<<16
            result.append(word if word<2**31 else word-2**32)
    return result


def pack(codes):
    return sum(code<<(4*i) for i,code in enumerate(codes))


def transform(native):
    old='''          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);'''
    new='''#if OKC_SC6W
          const uint32_t sgp = sgs[js];
          const uint32_t lo = (sgp & 0x03f0u) << 4;
          const uint32_t hi = ((sgp >> 16) & 0x03f0u) << 4;
          af[ci & 1][jj][0] = pair_lookup(p.lut,lo | (q & 255u));
          af[ci & 1][jj][1] = pair_lookup(p.lut,hi | ((q >> 8) & 255u));
          af[ci & 1][jj][2] = pair_lookup(p.lut,lo | ((q >> 16) & 255u));
          af[ci & 1][jj][3] = pair_lookup(p.lut,hi | (q >> 24));
#else
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
#endif'''
    assert native.count(old)==1
    native=native.replace(old,new)
    anchor='template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg('
    helper='''__device__ __forceinline__ uint32_t pair_lookup(const uint32_t* table,uint32_t index) {
  uint32_t value;
  asm volatile("{ .reg .u64 a; mad.wide.u32 a, %1, 4, %2; ld.global.nc.b32 %0, [a+1024]; }"
               :"=r"(value):"r"(index),"l"(table));
  return value;
}
'''
    assert native.count(anchor)==1
    return native.replace(anchor,helper+anchor),[(old,new,1),(anchor,helper+anchor,1)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    print(json.dumps(generate(args.out,transform),indent=2),flush=True)
    if args.compile:compile_pair(args.out)
