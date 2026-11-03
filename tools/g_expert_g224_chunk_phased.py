"""One fixed chunk-phased gather lifetime treatment; scratch/offline only."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

import g_expert_g224_prefix as parent

ROOT = Path(__file__).resolve().parents[1]
PARENT_TOOL_SHA = "114ed80ae87b485ade24a6db2dab6fbab7ab8ff2c4cbc512ca58c4bb7f799e1a"
CONTROL_SHA = "ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f"
PARENT_VARIANT_SHA = "bd22962e25f94abc3c96aa31ebd594f08c63ec7881252dba12ca37590ab21b93"
OLD_BODY = parent.GATHER.rstrip()
NEW_BODY = r'''__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  static_assert(NT * 8 * 2 == HD, "G224 two chunks per compute thread");
  constexpr int GB = OKC_GATHB;
#pragma unroll 1
  for (int chunk = 0; chunk < 2; ++chunk) {
    int j = 0;
    for (int qb = q0; qb < q1; qb += GB) {
      uint4 v[GB];
      uint4* d[GB];
#pragma unroll
      for (int k = 0; k < GB; ++k) {
        const int q = qb + k;
        if (q < q1) { while (S.jrow[j + 1] <= q) ++j; }
        d[k] = nullptr;
        if (q < q1) {
          const int r = q - S.jrow[j], cnt = S.jC[j];
          if (r < cnt) {
            const int ntt = (cnt + 7) >> 3, column = tid + chunk * NT;
            d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD)
                + ((((column >> 5) * ntt + (r >> 3)) * 32 + (column & 31)) * 8 + (r & 7));
            const int group = column & 31;
            const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (column >> 5) * 256)
                + (8 * ((group >> 1) & 3) + 2 * (group >> 3) + (group & 1));
            v[k] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));
          }
        }
      }
#pragma unroll
      for (int k = 0; k < GB; ++k)
        if (d[k]) *d[k] = v[k];
    }
  }
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
  CTA_SYNC();
  if (tid == 0) {
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int end = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(end - q) : "memory");
      q = end;
    }
  }
}'''


def body(source):
    start = source.index("__device__ void wgs_gather(")
    opening = source.index("{", start)
    depth, cursor = 1, opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def transform(control):
    assert parent.sha(control) == CONTROL_SHA
    assert body(control) == OLD_BODY
    changed = parent.once(control, OLD_BODY, NEW_BODY)
    assert parent.once(changed, NEW_BODY, OLD_BODY) == control
    assert control.replace(OLD_BODY, "<GATHER>") == changed.replace(NEW_BODY, "<GATHER>")
    return changed


def appendix(changed):
    return ('\nimport hashlib as _g224_chunk_hash\n'
            + 'assert _g224_chunk_hash.sha256(_G224_CUDA.encode()).hexdigest() == ' + repr(CONTROL_SHA) + '\n'
            + '_G224_CHUNK_OLD = ' + repr(OLD_BODY) + '\n'
            + '_G224_CHUNK_NEW = ' + repr(NEW_BODY) + '\n'
            + 'assert _G224_CUDA.count(_G224_CHUNK_OLD) == 1\n'
            + '_G224_CUDA = _G224_CUDA.replace(_G224_CHUNK_OLD, _G224_CHUNK_NEW)\n'
            + 'assert _g224_chunk_hash.sha256(_G224_CUDA.encode()).hexdigest() == ' + repr(parent.sha(changed)) + '\n')


def generate(root, out):
    assert parent.sha((root / 'tools/g_expert_g224_prefix.py').read_text(encoding='utf8')) == PARENT_TOOL_SHA
    control_variant, control, control_manifest = parent.generate(root, out / 'control')
    assert parent.sha(control_variant) == PARENT_VARIANT_SHA
    changed = transform(control)
    suffix = appendix(changed)
    variant = control_variant + suffix
    ast.parse(variant)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'variant.py').write_text(variant, encoding='utf8', newline='\n')
    (out / 'selected.cu').write_text(changed, encoding='utf8', newline='\n')
    (out / 'gather.diff').write_text(''.join(difflib.unified_diff(control.splitlines(True), changed.splitlines(True), fromfile='ed082519-control', tofile='chunk-phased-treatment')), encoding='utf8', newline='\n')
    manifest = dict(control=control_manifest, control_native_sha256=CONTROL_SHA,
                    native_sha256=parent.sha(changed), normalized_outside_gather_sha256=parent.sha(changed.replace(NEW_BODY, '<GATHER>')),
                    variant_sha256=parent.sha(variant), variant_bytes=len(variant.encode()), parent_prefix_unchanged=variant.startswith(control_variant),
                    append_only=True, exact_reverse=True, batch=8, chunk_phases=2, gpu_execution=False, judge_dispatch=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8', newline='\n')
    return variant, changed, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(generate(ROOT, options.out)[2], indent=2))
