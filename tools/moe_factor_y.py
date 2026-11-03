"""Store the existing BF16 down result and reconstruct its exact FP32 gamma product."""
import argparse
import json
from pathlib import Path
import struct

from cuda_issue_groups import generate,compile_pair


def reconstruct(down,weight):
    value=struct.unpack('<f',struct.pack('<I',down<<16))[0]
    raw=struct.unpack('<f',struct.pack('<I',weight<<16))[0]
    try:bits=struct.unpack('<I',struct.pack('<f',1.+raw))[0]
    except OverflowError:bits=0xff800000 if raw<0 else 0x7f800000
    gamma=((bits+0x7fff+((bits>>16)&1))>>16)&0xffff
    factor=struct.unpack('<f',struct.pack('<I',gamma<<16))[0]
    try:return struct.unpack('<I',struct.pack('<f',value*factor))[0]
    except OverflowError:return 0xff800000 if value*factor<0 else 0x7f800000


HELPER='''#define OKC_YFACTOR 1
__device__ __forceinline__ float2 wgs_factor_pair(uint32_t packed,const __nv_bfloat16* weights) {
  const float2 down = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
  const uint32_t word = __ldg(reinterpret_cast<const uint32_t*>(weights));
  const float2 raw = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word));
  return make_float2(__fmul_rn(down.x,fork_bf16(1.f + raw.x)),__fmul_rn(down.y,fork_bf16(1.f + raw.y)));
}
__device__ __forceinline__ float2 wgs_factor_y2(const Params& p,int row,int col) {
  const uint32_t word = __ldcg(reinterpret_cast<const uint32_t*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)row * HD + col));
  return wgs_factor_pair(word,p.normw + col);
}
'''


def transform(native):
    anchor='struct WgsPend { int kind, u, slot, nst; };'
    store='''#if OKC_YBF
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0), __float2bfloat16(fork_bf16(d4[h] * gsc) * w0));
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8), __float2bfloat16(fork_bf16(d4[2 + h] * gsc) * w1));'''
    raw='''#if OKC_YFACTOR
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0), __float2bfloat16(d4[h] * gsc));
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8), __float2bfloat16(d4[2 + h] * gsc));
#elif OKC_YBF'''+store[len('#if OKC_YBF'):]
    load='''#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);'''
    factored='''#if OKC_YFACTOR
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = wgs_factor_pair(yr.x,p.normw + tid * 8), y1 = wgs_factor_pair(yr.y,p.normw + tid * 8 + 2);
          const float2 y2 = wgs_factor_pair(yr.z,p.normw + tid * 8 + 4), y3 = wgs_factor_pair(yr.w,p.normw + tid * 8 + 6);
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#elif OKC_YBF'''+load[len('#if OKC_YBF'):]
    b4='__ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col))'
    edits=[(anchor,HELPER+anchor,1),(store,raw,1),(load,factored,2),(b4,'wgs_factor_y2(p,q,col)',2)]
    for old,new,count in edits:
        assert native.count(old)==count,(old[:60],native.count(old))
        native=native.replace(old,new)
    return native,edits


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    print(json.dumps(generate(args.out,transform),indent=2),flush=True)
    if args.compile:compile_pair(args.out)
