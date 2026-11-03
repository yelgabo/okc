"""Eliminate global phase-A input materialization in the selected WGS engine."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import generate,compile_pair


def addresses(tokens,stage,lane,rb,z):
    row=rb+(lane&7);group=4*z+(lane>>3)
    destination=((row//8)*32+group)*8+row%8
    if row>=len(tokens):return destination,None
    base=tokens[row]*2048+stage*128+8*((group>>1)&3)+2*(group>>3)+(group&1)
    return destination,tuple(base+32*h for h in range(4))


def layout(tokens,stage):
    return dict(addresses(tokens,stage,lane,rb,z)
                for lane in range(32) for rb in range(0,(len(tokens)+7)//8*8,8) for z in range(8))


HELPER='''__device__ __forceinline__ void wgs_direct_x(const Params& p,const Smem& S,unsigned char* dst,int j,int st,int npad) {
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
  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
  asm volatile("bar.sync 8, 32;\\n" ::: "memory");
}
'''


def transform(native):
    start=native.rfind('__device__',0,native.index('void wgs_xproducer('))
    end=native.index('\n}',start)+2
    old=native[start:end];new=old
    def once(left,right):
        nonlocal new
        assert new.count(left)==1,left
        new=new.replace(left,right)
    once("  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the\n  // stage ranges while the consumers gather phase A's x\n",'')
    once('  if ((threadIdx.x & 31) != 0) return;','  const int lane = threadIdx.x & 31;')
    once('  for (int pb = 0; pb < 2; ++pb) {','  for (int pb = 0; pb < 2; ++pb) {\n    if (pb && lane != 0) return;')
    once('        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }  // every row gathered\n','')
    block='''      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");'''
    once(block,'      if (pb) {\n'+block+'\n      } else {\n        wgs_direct_x(p,S,xring + xs * WGS_XS,j,st,npad);\n        if (lane == 0) mbar_arrive1(bar);\n      }')
    new=HELPER+new
    gather='''  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);'''
    edits=[(old,new,1),(gather,'  DBG(7);',1)]
    for before,after,count in edits:
        assert native.count(before)==count
        native=native.replace(before,after)
    return native,edits


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    print(json.dumps(generate(args.out,transform),indent=2),flush=True)
    if args.compile:compile_pair(args.out)
