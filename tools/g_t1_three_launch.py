"""One offline T1-only three-launch expert prototype on exact public607."""

import argparse
import ast
import difflib
import hashlib
import json
import re
import time
import types
from pathlib import Path

from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
BASE_SHA = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-worker-t1-three-launch'
HELPERS = ('bmul2', 'mma_bf16', 'dq4', 'fork_bf16', 'probe_q')
CPP = '''void t1_three_run(std::vector<torch::Tensor> weights, std::vector<torch::Tensor> scratch,
    torch::Tensor input, torch::Tensor ids, torch::Tensor route_weights, torch::Tensor output);'''
FLAGS = ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3',
         '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_EPI4=0',
         '-DOKC_FAST1=2', '-DOKC_L2H=0', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1',
         '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1',
         '-DOKC_PRMTSC=1', '-DOKC_DQ7=1']

APPEND = '''

_T1_THREE_CUDA = {native!r}
_T1_THREE_CPP = {cpp!r}
_T1_THREE_FLAGS = {flags!r}

class _T1ThreeAdapter:
    def __init__(self, original, module, scratch):
        self.original, self.module, self.scratch = original, module, scratch

    def __deepcopy__(self, memo):
        import copy
        result = type(self)(self.original, self.module, copy.deepcopy(self.scratch, memo))
        memo[id(self)] = result
        return result

    def okc_run(self, weights, scratch, x, ids, tw, out, lo, nloc, grid):
        if (lo == 0 and nloc == 32 and tuple(x.shape) == (1, 4096)
                and tuple(ids.shape) == (1, 8) and tuple(tw.shape) == (1, 8)
                and tuple(out.shape) == (1, 4096)
                and x.dtype == out.dtype == torch.bfloat16 and ids.dtype == torch.int32
                and tw.dtype == torch.float32
                and x.is_contiguous() and ids.is_contiguous() and tw.is_contiguous() and out.is_contiguous()):
            return self.module.t1_three_run(weights, self.scratch, x, ids, tw, out)
        return self.original.okc_run(weights, scratch, x, ids, tw, out, lo, nloc, grid)

_t1_three_previous_moe = _moe
def _moe(M=None, state=None):
    if M == 1 and state is not None and state.get('_t1_three_adapter') is not None:
        return state['_t1_three_adapter']
    return _t1_three_previous_moe(M, state)

_t1_three_previous_prepare = prepare
def prepare(weights):
    state = _t1_three_previous_prepare(weights)
    if (state.get('_fork_native_eligible', False)
            and tuple(state.get('ma_sfx', ())) == ('-DOKC_PRMTSC=1',)
            and len(state.get('ma_w', ())) >= 12):
        module = _RouterExt(load_inline(name='t1_three_launch_v1', cpp_sources=_T1_THREE_CPP,
            cuda_sources=_T1_THREE_CUDA, functions=['t1_three_run'], extra_cflags=['-O3', '-std=c++20'],
            extra_cuda_cflags=_T1_THREE_FLAGS, verbose=False))
        device = weights['w13'].device
        scratch = [torch.empty((8, 4096), dtype=torch.bfloat16, device=device),
                   torch.empty((8, 4096), dtype=torch.float32, device=device),
                   torch.empty((8, 256), dtype=torch.float32, device=device)]
        original = _t1_three_previous_moe(1, state)
        state['_t1_three_adapter'] = _T1ThreeAdapter(original, module, scratch)
    return state
'''


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def cuda_function(source, name):
    match = re.search(r'^__device__[^\n]*\b' + name + r'\(', source, re.M)
    if match is None:
        raise ValueError('missing actual selected helper: ' + name)
    opening = source.index('{', match.start())
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == '{') - (source[cursor] == '}')
        cursor += 1
    return source[match.start():cursor]


def native_source(original):
    helper = bindings(original)['_MA_ALT_SRCS']['moe_helper0']
    selected = '\n\n'.join(cuda_function(helper, name) for name in HELPERS)
    template = (ROOT / 'tools/t1_three_launch.cu').read_bytes().decode()
    if template.count('@@SELECTED_HELPERS@@') != 1:
        raise ValueError('helper boundary')
    return template.replace('@@SELECTED_HELPERS@@', selected)


def appendix(original):
    return APPEND.format(native=native_source(original), cpp=CPP, flags=FLAGS)


def build(original):
    if sha(original) != BASE_SHA or '\r' in original:
        raise ValueError('not exact LF ba9')
    return original + appendix(original)


def native_contract(native):
    required = (
        'const int slot = blockIdx.y;',
        'if (!local_route(params, slot)) return;',
        'return expert >= params.local_start && expert < params.local_start + params.local_count;',
        'const int expert = params.ids[slot] - params.local_start;',
        'const float gate = fork_bf16(gate_or_first * global_scale);',
        'const float up = fork_bf16(up_or_second * global_scale);',
        'SILU_DIV(gate, 1.f + __expf(-gate)) * up',
        'fork_bf16(1.f + __bfloat162float(params.norm[column]))',
        'fork_bf16(gate_or_first * global_scale) * gamma_first',
        'fork_bf16(up_or_second * global_scale) * gamma_second',
        'params.partials[(size_t)slot * 256 + row_block] = square;',
        'params.partials[slot * 256 + thread]',
        'rsqrtf(square_sum * (1.f / 4096) + 1e-8f)',
        'const float weight = params.route_weights[slot] * inverse[slot];',
        'combined += weight * params.expert_y[slot * 4096 + column];',
        '__float2bfloat16_rn(combined)',
        'acc[component] += acc2[component] + acc3[component] + acc4[component];',
    )
    if any(fragment not in native for fragment in required):
        raise ValueError('lost T1 mapping/rounding/per-expert normalization contract')
    launches = ('gate_up<<<dim3(512, 8), 256, 0, stream>>>',
                'down<<<dim3(256, 8), 256, 0, stream>>>',
                'norm_combine<<<1, 256, 0, stream>>>')
    if native.count('<<<') != 3 or any(native.count(launch) != 1 for launch in launches):
        raise ValueError('not exactly three selected ordinary launches')
    if [native.index(launch) for launch in launches] != sorted(native.index(launch) for launch in launches):
        raise ValueError('phase order')
    if native.count('for (int slot = 0; slot < 8; ++slot)') != 2:
        raise ValueError('complete norm and ordered slots')
    return True


def validate(source, original):
    if source != build(original) or source[:-len(appendix(original))] != original:
        raise ValueError('not exact append-only T1 transform')
    if '\r' in source or not source.endswith('\n') or len(source.encode()) >= 1000000:
        raise ValueError('physical representation')
    ast.parse(source)
    native = native_source(original)
    native_contract(native)
    return dict(base_sha256=BASE_SHA, source_sha256=sha(source), bytes=len(source.encode()),
        native_sha256=sha(native), cpp_sha256=sha(CPP), exact_reverse=True, scope='T1 only',
        scratch_bytes=204800, selected_math_helpers={name:sha(cuda_function(bindings(original)['_MA_ALT_SRCS']['moe_helper0'], name)) for name in HELPERS},
        existing_factory_flags_unchanged=True, CUDA_numerics='NOT RUN', target_qualified=False,
        performance_qualified=False, launches=3, extra_launches=2)


def generate(out):
    original = BASE.read_bytes().decode()
    source = build(original)
    manifest = validate(source, original)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('candidate.py',source),('selected.cu',native_source(original)),
                       ('binding.cpp',CPP),('append.py.txt',appendix(original)),
                       ('source.diff',''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='ba9.py',tofile='T1-three.py'))),
                       ('generation.json',json.dumps(manifest,indent=2)+'\n')]:
        (out/name).write_bytes(data.encode())
    return manifest


def compile_fixed(out):
    from test_g_timed_silu_table import capture
    generated = generate(out/'generated')
    calls, routes = capture(BASE.read_bytes().decode())
    if calls[0]['extra_cuda_cflags'] != FLAGS:
        raise ValueError('selected T1 flag mismatch')
    driver_path = ROOT/'.build/research-gamma13/artifacts/adapted-driver.py'
    driver = driver_path.read_bytes().decode()
    if sha(driver) != 'e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c':
        raise ValueError('fixed installed recipe identity')
    old = "expressions = [b'&okc::moe_fused<false>', b'&okc::moe_fused<true>']"
    new = "expressions = [b'&okc_t1::gate_up', b'&okc_t1::down', b'&okc_t1::norm_combine']"
    if driver.count(old) != 1:
        raise ValueError('fixed entry adapter')
    adapted = driver.replace(old,new)
    namespace = dict(__name__='_t1_three_fixed_secondary13')
    exec(compile(adapted,str(driver_path),'exec'),namespace)
    folder = out/'fixed-compile'
    record = dict(started_epoch=time.time(), generation=generated,
        scope='ONE NVRTC13/PTXAS13 sm90a device-prefix compilation; no host build/link, Driver, GPU or challenge CLI',
        driver_sha256=sha(driver), adapted_driver_sha256=sha(adapted), actual_flags=FLAGS,
        actual_routes=routes, numerical_validation='NOT RUN', target_qualified=False)
    out.mkdir(parents=True,exist_ok=True)
    (out/'adapted-driver.py.txt').write_bytes(adapted.encode())
    (out/'compile-result.json').write_text(json.dumps(record,indent=2)+'\n')
    args = types.SimpleNamespace(source=out/'generated/selected.cu',out=folder,
        cache=Path('/root/.cache/okc-cuda130-88'),define=[flag for flag in FLAGS if flag.startswith('-D')],
        sc6=1,forward_declare_wgs=False)
    try:
        namespace['compile_source'](args)
        record['compiler_manifest'] = json.loads((folder/'manifest.json').read_text())
        record['decision'] = 'DEVICE_COMPILE_PASS_NOT_TARGET_QUALIFIED'
    except Exception as error:
        record['decision'] = 'STOP_FIXED_COMPILE_FAILURE'
        record['error'] = repr(error)
        raise
    finally:
        record['finished_epoch'] = time.time()
        (out/'compile-result.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=EVIDENCE)
    parser.add_argument('--compile',action='store_true')
    args = parser.parse_args()
    result = compile_fixed(args.out) if args.compile else generate(args.out/'generated')
    print(json.dumps(result,indent=2))
