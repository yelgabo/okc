"""Check whether constant scales permit a stable shared-pointer load to be hoisted."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import compile_pair, generate
from cuda_scale_constant import transform as scale_transform


def transform(native):
    native, edits = scale_transform(native)
    old = '''// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}'''
    new = '''__device__ __forceinline__ float* ss_cur() {
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;
}'''
    if native.count(old) != 1 or native.count('S.ssp = (gen & 1) ? p.ss1 : p.ss;') != 1:
        raise ValueError('Stable sum-of-squares pointer source drift')
    return native.replace(old, new), edits + [(old, new, 1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.out, transform), indent=2), flush=True)
    compile_pair(args.out, arms=('candidate',))
