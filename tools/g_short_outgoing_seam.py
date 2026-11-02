"""Private outgoing-only expert-to-slice8 seam source builder; never dispatches."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PIN = "4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280"
MARKER = "\n# PRIVATE OUTGOING SEAM ONLY; NOT GPU QUALIFIED; NEVER SUBMIT.\n"
SUPPORT = r'''
_OS_PRIOR_PREPARE = prepare
_OS_PRIOR_GEMM = _bfp_gemm
_OS_PRIOR_RUN = bfp__run_sk2
_OS_ENABLED = @@ENABLED@@

def _os_native_sources():
    cfg = next(cfg for cfg in bfp_SK2_CONFIGS if cfg[0] == 'slice8_r192')
    assert cfg[1:] == ('unpooler', 1024, 8, 1, 1, 8, 2, 2, 1)
    _, stages = bfp__sk2_geom(cfg, 8)
    source = bfp_SK2_SRC
    boundary = source.index('static void launch_sk2(')
    device = source[:boundary]
    host = source[boundary:]
    old = 'const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr)'
    assert host.count(old) == 1
    host = host.replace(old, 'int seam_pdl = 0, ' + old)
    assert host.count('const int pdl = g_sk2_pdl;') == 1
    host = host.replace('const int pdl = g_sk2_pdl;', 'const int pdl = seam_pdl;')
    assert host.count('programmaticStreamSerializationAllowed = 0;') == 1
    host = host.replace('programmaticStreamSerializationAllowed = 0;', 'programmaticStreamSerializationAllowed = 1;')
    old = 'int64_t tokr, int64_t dg, at::Tensor dbgt)'
    assert host.count(old) == 1
    host = host.replace(old, 'int64_t tokr, int64_t dg, at::Tensor dbgt, int64_t seam_pdl)')
    host = host.replace('void sk2(at::Tensor', 'void seam_sk2(at::Tensor')
    dispatch = (f'TORCH_CHECK(M == 1 || M == 8, "outgoing seam tokens"); '
                f'TORCH_CHECK(N == 1024 && K == 4096 && tokr == 8 && dg == 0, "outgoing seam shape"); '
                f'TORCH_CHECK(seam_pdl == 0 || seam_pdl == 1073741826, "outgoing seam policy"); '
                f'launch_sk2<1, 8, 1, 1, 8, 2, {stages}, true, false, 2, 1, __nv_bfloat16>'
                f'(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg, (int)seam_pdl); return;')
    source = device + host.replace('@@SK2_DISPATCH@@', dispatch).replace('@@SK2F_DISPATCH@@', '')
    cpp = bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]
    cpp = cpp.replace('void sk2(', 'void seam_sk2(').replace('torch::Tensor dbgt);', 'torch::Tensor dbgt, int64_t seam_pdl);')
    shared = bfp_CUDA_SRC.replace('@@MBAR@@', bfp_MBAR_HELPERS).replace('@@BODY@@', '')
    return shared + bfp_MAP_SRC + source, '#include <torch/extension.h>\n' + cpp

def _os_install(state):
    import hashlib
    source, cpp = _os_native_sources()
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
    flags = ['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}'] + ([f'-DSK2_DELAY_ESC={bfp__SK2_DELAY}'] if bfp__SK2_DELAY else [])
    tag = hashlib.sha256((source + cpp + repr(flags)).encode()).hexdigest()[:16]
    extension = load_inline(name='private_outgoing_slice8_' + tag, cpp_sources=cpp,
        cuda_sources=source, functions=['seam_sk2'], no_implicit_headers=True,
        extra_cuda_cflags=flags, verbose=False)
    scoped = dict(state['bfp'])
    scoped['_os_launch'] = (extension, 1073741826 if _OS_ENABLED else 0)
    state['_os_unpool_bfp'] = scoped
    return state

def prepare(weights):
    return _os_install(_OS_PRIOR_PREPARE(weights))

def _bfp_gemm(st, name, x, m, n, k):
    if name != 'un' or m not in (1, 8):
        return _OS_PRIOR_GEMM(st, name, x, m, n, k)
    assert (n, k) == (1024, 4096), 'outgoing seam shape'
    scoped = st['_os_unpool_bfp']
    choice = scoped.get('choice', {}).get(('unpooler', m)) or bfp__nearby_choice('unpooler', m, scoped) or bfp__static_choice('unpooler', m, scoped)
    assert choice == ('sk2',), 'outgoing seam route'
    return bfp_kernel_fn({'x': x, 'config': {'layer': 'unpooler', 'm': m, 'n': n, 'k': k}}, scoped)

def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    if '_os_launch' not in state:
        return _OS_PRIOR_RUN(x, layer, m, n, k, state, dg, dbg)
    assert layer == 'unpooler' and m in (1, 8) and (n, k, dg) == (1024, 4096, 0)
    name = bfp__sk2_name(layer, m)
    assert name == 'slice8_r192', 'outgoing seam selection'
    ci, stream, tab, pmax, b4, n_rows = state['sk2'][name]
    assert n_rows == 1024
    extension, policy = state['_os_launch']
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    extension.seam_sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, 8, dg,
                       state['dummy'] if dbg is None else dbg, policy)
    return out
'''


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def build(source, enabled=False):
    assert sha(source) == PIN, "exact raw common OFF required"
    result = source + MARKER + SUPPORT.replace('@@ENABLED@@', repr(bool(enabled)))
    ast.parse(result)
    assert result.split(MARKER)[0] == source
    return result, {
        "input_sha256": PIN, "source_sha256": sha(result), "bytes": len(result.encode()),
        "enabled": bool(enabled), "policy": 1073741826 if enabled else 0,
        "reverse_exact": True, "native_primary_unchanged": True,
        "secondary_device_body_unchanged": True, "secondary_host_adapter_changed": True,
        "runtime_graph_support": "NOT_RUN", "GPU_qualified": False,
        "compiled_host": False, "dispatch_enabled": False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--enable', action='store_true')
    args = parser.parse_args()
    result, manifest = build(args.source.read_text(encoding='utf8'), args.enable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.encode())
    args.output.with_suffix('.manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps(manifest))
