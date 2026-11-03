"""Build whole-unit phase-B claiming against the selected WGS engine."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, generate, compile_pair, sha

QUEUE = r'''
__device__ __forceinline__ void moe_bqueue(const Params& p, Smem& S, unsigned char* ring, int role, int seq) {
  static_assert(!OKC_MFIN, "queue reset requires the retained grid barrier");
  const int lane = threadIdx.x & 31;
  const int total = (S.njF + S.njS) * 16;
  for (;;) {
    WS_JOIN();
    if (threadIdx.x == 0) S.wgr[2] = atomicAdd(p.ctr + 1, 1);
    WS_JOIN();
    const int unit = S.wgr[2];
    if (unit >= total) break;
    if (role == 0) {
      WgsPend pending{0, 0, 0, 0};
      switch ((S.jC[unit >> 4] + 7) >> 3) {
        case 1: wgs_seg<true, 1>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 2: wgs_seg<true, 2>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 3: wgs_seg<true, 3>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 4: wgs_seg<true, 4>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 5: wgs_seg<true, 5>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 6: wgs_seg<true, 6>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        case 7: wgs_seg<true, 7>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
        default: wgs_seg<true, 8>(p, S, ring, unit, 0, 16, -1, seq, pending); break;
      }
    } else if (lane == 0) {
      const int job = unit >> 4, ub = unit & 15;
      const int npad = ((S.jC[job] + 7) >> 3) << 3;
      for (int st = 0; st < 16; ++st, ++seq) {
        if (role == 1) {
          const int ws = seq % WGS_NS;
          if (seq >= WGS_NS) wgs_wait(S.bars + BW_E + ws, ((seq / WGS_NS) & 1u) ^ 1u, 7100 + seq);
          if (seq >= 2 && p.M <= 64) {
            const int pn = seq - 2;
            wgs_wait(S.bars + BW_F + pn % WGS_NS, (pn / WGS_NS) & 1u, 7200 + seq);
          }
          uint64_t* bar = S.bars + BW_F + ws;
          const unsigned char* src = p.wBs + (size_t)(((S.jE[job] * 16 + ub) * 16 + st)) * WGS_STG;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                       ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
        } else {
          const int xs = seq % WGS_NX;
          if (seq >= WGS_NX) wgs_wait(S.bars + BX_E + xs, ((seq / WGS_NX) & 1u) ^ 1u, 7300 + seq);
          while (ld_acquire(mf_da(p, S) + job * 16 + st) < 8) __nanosleep(32);
          uint64_t* bar = S.bars + BX_F + xs;
          const uint32_t bytes = (uint32_t)npad * 512u;
          const __nv_bfloat16* src = p.xB + (size_t)S.jrow[job] * HD + (size_t)st * npad * 256;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(smem_u32(ring + WGS_NS * UNIT_BYTES + xs * WGS_XS)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
        }
      }
    }
  }
}
'''


def function(source, name):
    start = source.index('__device__ void ' + name + '(')
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def transform(native):
    edits = []

    def replace(old, new):
        nonlocal native
        if native.count(old) != 1:
            raise ValueError('MOE_BQUEUE_SOURCE_DRIFT: ' + old[:80])
        native = native.replace(old, new)
        edits.append((old, new, 1))

    for name, role in [('wgs_wproducer', 1), ('wgs_xproducer', 2)]:
        old = function(native, name)
        new = old.replace('  if ((threadIdx.x & 31) != 0) return;',
                          '  if ((threadIdx.x & 31) == 0) {')
        assert new != old and new.count('pb < 2') == 1
        new = new.replace('pb < 2', 'pb < 1')
        new = new[:-1] + '  }\n  moe_bqueue(p, S, ring, ' + str(role) + ', S.wgr[1] - S.wgr[0]);\n}'
        replace(old, new)
    replace('__device__ void wgs_wproducer(',
            '__device__ __forceinline__ void moe_bqueue(const Params& p, Smem& S, unsigned char* ring, int role, int seq);\n__device__ void wgs_wproducer(')
    replace('// cost-weighted stage ranges:', QUEUE + '\n// cost-weighted stage ranges:')
    replace('    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);',
            '    moe_bqueue(p, S, ring, 0, seq);')
    return native, edits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.out, transform), indent=2), flush=True)
    if args.compile:
        compile_pair(args.out)
