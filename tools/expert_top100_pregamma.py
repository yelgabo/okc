"""One fixed, scratch-only pre-gamma WGS transport prototype. No GPU imports."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import types

from prod_size_compile_g import bindings
from g_native_patch_ladder import flags

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
HELPERS = r'''
// Scratch prototype: transport the required BF16 down result, not rounded gamma*y.
// Keep the FP32 allocation/fixup offsets and the original global publication chain.
static_assert(OKC_YBF == 0, "pre-gamma transport requires FP32 product semantics");
__device__ __forceinline__ float2 expert_top100_y2(const Params& p, int q, int col) {
  const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)q * HD + col;
  const __nv_bfloat162 packed = __ldcg(reinterpret_cast<const __nv_bfloat162*>(src));
  const float2 v = __bfloat1622float2(packed);
  const float g0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + col)));
  const float g1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + col + 1)));
  // Explicit FP32 products prevent contraction through the former global-store boundary.
  return make_float2(__fmul_rn(v.x, g0), __fmul_rn(v.y, g1));
}
__device__ __forceinline__ float4 expert_top100_y4(const Params& p, int q, int col) {
  const float2 a = expert_top100_y2(p, q, col), b = expert_top100_y2(p, q, col + 2);
  return make_float4(a.x, a.y, b.x, b.y);
}
'''


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def transform(native):
    edits = []
    def replace(old, new, count=1):
        nonlocal native
        assert native.count(old) == count, (old, native.count(old), count)
        native = native.replace(old, new)
        edits.append((old, new, count))
    for index, column, gamma in [('h', 'n0', 'w0'), ('2 + h', 'n0 + 8', 'w1')]:
        replace(f'st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + {column}), fork_bf16(d4[{index}] * gsc) * {gamma});',
                f'st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + {column}), __float2bfloat16(fork_bf16(d4[{index}] * gsc)));')
    # B4 early and ordinary reads; B4 partial sum-of-squares math remains untouched.
    replace('__ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col))',
            'expert_top100_y2(p, q, col)', 2)
    # Both MAIN passes reconstruct identical per-column FP32 values.
    replace('const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);',
            'const float4 a = expert_top100_y4(p, qv[sl], tid * 8), b = expert_top100_y4(p, qv[sl], tid * 8 + 4);', 2)
    # Insert outside the epilogue template, after the YBF macro definition is in scope.
    replace('template <bool PB>\n__device__ __forceinline__ void wgs_flush(', HELPERS + '\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush(')
    return native, edits


def capture(source):
    env = bindings(source)
    env.update(flags(source))
    calls = []
    class Extension:
        def okc_set_pdl(self, value):
            assert value == 0
    def load_inline(**kwargs):
        calls.append(kwargs)
        return Extension()
    env.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
               load_inline=load_inline, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False, _RouterExt=lambda v: v)
    for name in ('_main_defines', '_ma_ext', '_fork_native_moe'):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(ast.get_source_segment(source, node), env)
    state = dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})
    routes = []
    for tokens in (1, 8, 16, 32, 128, 256):
        ext = env['_fork_native_moe'](tokens, state)
        key = next(k for k, v in state['ma_ext'].items() if v is ext)
        routes.append(dict(tokens=tokens, key=key))
    return calls, routes


def generate(out):
    base = BASE.read_text(encoding='utf8')
    assert sha(base) == PIN
    old = bindings(base)['MOE_CUDA_SRC']
    native, edits = transform(old)
    assert base.count(old) == 1
    variant = base.replace(old, native)
    ast.parse(variant)
    old_calls, routes = capture(base)
    new_calls, new_routes = capture(variant)
    assert routes == new_routes
    selection = []
    for a, b in zip(old_calls, new_calls, strict=True):
        assert a['extra_cuda_cflags'] == b['extra_cuda_cflags']
        assert a['cpp_sources'] == b['cpp_sources']
        changed = a['cuda_sources'] != b['cuda_sources']
        assert changed == ('-DOKC_WGS=1' in b['extra_cuda_cflags'])
        if changed:
            assert a['cuda_sources'] == old and b['cuda_sources'] == native
        selection.append(dict(flags=b['extra_cuda_cflags'], changed=changed,
                              control_sha256=sha(a['cuda_sources']), candidate_sha256=sha(b['cuda_sources'])))
    restored = native
    for a, b, count in reversed(edits):
        assert restored.count(b) == count
        restored = restored.replace(b, a)
    assert restored == old
    # All synchronization/atomics and host tail retained exactly, not merely counted.
    assert old[old.index('#include <torch/extension.h>'):] == native[native.index('#include <torch/extension.h>'):]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidate.py').write_bytes(variant.encode())
    (out / 'control.cu').write_bytes(old.encode())
    (out / 'candidate.cu').write_bytes(native.encode())
    (out / 'native.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), native.splitlines(True), fromfile='control', tofile='candidate')))
    manifest = dict(base_sha256=PIN, candidate_sha256=sha(variant), candidate_bytes=len(variant.encode()),
                    native_control_sha256=sha(old), native_candidate_sha256=sha(native),
                    routes=routes, selection=selection, exact_reverse=True, gpu=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(generate(parser.parse_args().out), indent=2))
