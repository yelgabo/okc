"""Generate and compile one shape-specific WGMMA grouping experiment."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from prod_size_compile_g import bindings

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py'
BASE_PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
NATIVE_PIN = 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71'


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def transform(native):
    edits = [
        ('constexpr int GS = OKC_WGSGS;',
         'constexpr int GS = NTT <= 4 ? 4 : OKC_WGSGS;\n'
         '  constexpr int FLUSH_CI = ((OKC_WGSFCI + 1) * OKC_WGSGS + GS - 1) / GS - 1;'),
        ('if (FIRST && ci == OKC_WGSFCI && pd.kind)',
         'if (FIRST && ci == FLUSH_CI && pd.kind)'),
    ]
    for old, new in edits:
        if native.count(old) != 1:
            raise ValueError('WGS source drift: ' + old)
        native = native.replace(old, new)
    return native, [(old, new, 1) for old, new in edits]


def generate(out, treatment=transform):
    base = BASE.read_text()
    assert sha(base) == BASE_PIN
    control = bindings(base)['MOE_CUDA_SRC']
    assert sha(control) == NATIVE_PIN
    candidate, edits = treatment(control)
    restored = candidate
    for old, new, count in reversed(edits):
        assert restored.count(new) == count
        restored = restored.replace(new, old)
    assert restored == control
    boundary = '#include <torch/extension.h>'
    assert control[control.index(boundary):] == candidate[candidate.index(boundary):]
    out.mkdir(parents=True, exist_ok=True)
    for arm, source in [('control', control), ('candidate', candidate)]:
        (out / (arm + '.cu')).write_text(source)
        device = source[:source.index(boundary)]
        anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase('
        declaration = '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
        assert device.count(anchor) == 1
        device = device.replace(anchor, declaration + anchor)
        device += '\nvoid compile_entries(okc::Params p) {\n'
        device += 'okc::moe_fused<false><<<1, okc::NTL>>>(p);\n'
        device += 'okc::moe_fused<true><<<1, okc::NTL>>>(p);\n}\n'
        (out / (arm + '-device.cu')).write_text(device)
    (out / 'native.diff').write_text(''.join(difflib.unified_diff(
        control.splitlines(True), candidate.splitlines(True), fromfile='control', tofile='candidate')))
    manifest = dict(base_sha256=BASE_PIN, control_native_sha256=NATIVE_PIN,
                    candidate_native_sha256=sha(candidate), edits=edits, exact_reverse=True,
                    adapter='Remove host tail, add declaration-only wgs_cta_of adapter and two compile-only launch references',
                    gpu=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def compile_pair(out, arms=('control', 'candidate')):
    flags = ['-O3', '-std=c++17', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-Xptxas=-v',
             '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_FNST=4', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1',
             '-DOKC_SC6W=1', '-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1']
    results = []
    for key in ('b4', 'main'):
        for arm in arms:
            stem = key + '-' + arm
            actual = flags + (['-DOKC_B4FIN=1'] if key == 'b4' else [])
            command = ['docker', 'run', '--rm', '--platform', 'linux/arm64', '-v', str(out.resolve()) + ':/w',
                       'nvidia/cuda:12.8.1-devel-ubuntu24.04', 'nvcc', *actual, '-c', '/w/' + arm + '-device.cu',
                       '-o', '/w/' + stem + '.o']
            result = subprocess.run(command, capture_output=True, text=True)
            (out / (stem + '.log')).write_text(result.stdout + result.stderr)
            results.append(dict(key=key, arm=arm, flags=actual, exit_code=result.returncode))
            print(key, arm, result.returncode, flush=True)
            result.check_returncode()
            command = ['docker', 'run', '--rm', '--platform', 'linux/arm64', '-v', str(out.resolve()) + ':/w',
                       'nvidia/cuda:12.8.1-devel-ubuntu24.04', 'cuobjdump', '-sass', '/w/' + stem + '.o']
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            (out / (stem + '.sass')).write_text(result.stdout)
    (out / 'compile.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.out), indent=2), flush=True)
    if args.compile:
        compile_pair(args.out)
