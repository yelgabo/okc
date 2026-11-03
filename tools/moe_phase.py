"""Sparse boundary observations on the selected, unchanged WGS algorithm."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import generate, compile_pair

FIELDS = 24
HELPERS = r'''
template<int FIELD>
__device__ __forceinline__ void ph_mark(const Params& p) {
  if (threadIdx.x == 0) {
    asm volatile("{ .reg .u64 t; mov.u64 t, %%globaltimer; st.global.u64 [%0], t; }"
                 ::"l"(p.dbg + blockIdx.x * 24 + FIELD):"memory");
  }
}
template<int FIRST>
__device__ __forceinline__ void ph_wg(const Params& p) {
  if ((threadIdx.x & 127) == 0) {
    asm volatile("{ .reg .u64 t; mov.u64 t, %%globaltimer; st.global.u64 [%0], t; }"
                 ::"l"(p.dbg + blockIdx.x * 24 + FIRST + (threadIdx.x >> 7)):"memory");
  }
}
__device__ __forceinline__ void ph_init(const Params& p) {
  if (threadIdx.x == 0) {
    for (int i=14;i<18;++i) p.dbg[blockIdx.x * 24+i]=0;
    unsigned sm; asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    p.dbg[blockIdx.x * 24+18]=sm;
  }
  ph_mark<0>(p);
}
__device__ __forceinline__ void ph_meta(const Params& p,const Smem& S) {
  if (threadIdx.x == 0) {
    p.dbg[blockIdx.x * 24+19]=S.njF+S.njS;
    for(int i=0;i<4;++i)p.dbg[blockIdx.x * 24+20+i]=S.wgr[i];
  }
  ph_mark<2>(p);
}
'''


def transform(native):
    edits = []

    def once(old, new):
        nonlocal native
        assert native.count(old) == 1, old
        native = native.replace(old, new)
        edits.append((old, new, 1))

    once('__device__ void wgs_wproducer(', HELPERS+'\n__device__ void wgs_wproducer(')
    once('  asm volatile("griddepcontrol.wait;\\n" ::: "memory");',
         '  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n  ph_init(p);')
    once('  DBG(1);', '  DBG(1);\n  ph_mark<1>(p);')
    once('  DBG(6);', '  DBG(6);\n  ph_meta(p, S);')
    once('    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);',
         '    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    ph_wg<3>(p);')
    once('    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);',
         '    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    ph_wg<7>(p);')
    once('    int* bc = p.ctr + 4 + (S.gen & 1);',
         '    ph_mark<11>(p);\n    int* bc = p.ctr + 4 + (S.gen & 1);')
    once('  DBG(4);\n  // phases A and B', '  DBG(4);\n  ph_mark<12>(p);\n  // phases A and B')
    once('  DBG(5);  // nothing', '  ph_mark<13>(p);\n  DBG(5);  // nothing')
    anchor='      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }'
    once(anchor,'      if (pb && i == 0) ph_mark<14>(p);\n'+anchor+'\n      if (pb && i == 0) ph_mark<15>(p);')
    anchor='      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n}'
    once(anchor,'      if (pb && i == 0) ph_mark<17>(p);\n      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n    if (!pb) ph_mark<16>(p);\n  }\n}')
    return native, edits


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    print(json.dumps(generate(args.out,transform),indent=2),flush=True)
    if args.compile:compile_pair(args.out)
