"""One private current-helper comparison: three versus four weight-ring slots."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import expert_top100_moe_adapter as adapter
from cuda_issue_groups import BASE_PIN, NATIVE_PIN, ROOT, sha
from cuda_issue_groups_harness import assignment
from expert_top100_moe_harness import ADAPTER_PIN, REFERENCE_PIN, NVFP4_PIN, extract
from expert_top100_pregamma import capture
from prod_size_compile_g import bindings

HELPER_PIN = 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'
OLD_DISPATCH = """    if arm and key in ('b4', 'main'):
        defines = tuple(defines) + ('@expert_top100_pregamma',)"""
NEW_DISPATCH = """    if arm and key in ('tiny', 'tiny2', 'mid'):
        defines = tuple(defines) + ('-DOKC_NS=4',)"""


def contract_source():
    source = (ROOT / 'tools/expert_top100_moe_harness_contract.py').read_text()
    edits = (
        ('if row[1] >= 32 and any(', 'if row[1] < 32 and any('),
        ('if row[1] < 32:\n            drift.append', 'if row[1] >= 32:\n            drift.append'),
        ("row[0] == seed and row[2] == dist]", "row[0] == seed and row[2] == dist and row[1] < 32]"),
        (' / 6 for index', ' / 3 for index'),
    )
    for old, new in edits:
        assert source.count(old) == (2 if old == ' / 6 for index' else 1)
        source = source.replace(old, new)
    anchor = "    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),"
    assert source.count(anchor) == 1
    source = source.replace(anchor, """    unresolved_drift = [row[:3] for row in rows if row[1] >= 32
                        and any(abs(math.log(ratio)) > max(math.log(1.01), 2 * max(row[6]))
                                for ratio in row[5])]
    return dict(component_pass=not regressions and not unresolved_drift
                and all(row['passed'] for row in summaries), unresolved_drift=unresolved_drift,""")
    return source


def build():
    prefix, unused = adapter.build()
    assert sha(prefix) == ADAPTER_PIN
    assert prefix.count(OLD_DISPATCH) == 1
    prefix = prefix.replace(OLD_DISPATCH, NEW_DISPATCH)
    prefix = assignment(prefix, '_EXPERT_PAIR_EDITS', ())
    prefix = assignment(prefix, 'EXPERT_MOE_IDENTITIES',
                        ('gopt_skinny_depth', BASE_PIN, HELPER_PIN, 'OKC_NS=3:4'))
    values = bindings(prefix)
    assert sha(values['MOE_CUDA_SRC']) == NATIVE_PIN
    assert sha(values['_MA_ALT_SRCS']['moe_helper0']) == HELPER_PIN
    calls, routes = capture(prefix)
    treatment_calls, treatment_routes = capture(prefix.replace(
        "arm = state.get('_expert_pair_arm', False)", 'arm = True'))
    assert routes == treatment_routes
    assert len(calls) == len(treatment_calls) == 5
    selected = []
    for route, control, candidate in zip(routes[:5], calls, treatment_calls):
        affected = route['key'] in ('tiny', 'tiny2', 'mid')
        assert control['cuda_sources'] == candidate['cuda_sources']
        assert control['cpp_sources'] == candidate['cpp_sources']
        without_depth = lambda call: [flag for flag in call['extra_cuda_cflags']
                                      if not flag.startswith('-DOKC_NS=')]
        assert without_depth(control) == without_depth(candidate)
        depth = [flag for flag in candidate['extra_cuda_cflags'] if flag.startswith('-DOKC_NS=')]
        assert depth == ['-DOKC_NS=' + ('4' if affected else '3')]
        if not affected:
            assert control == candidate
        selected.append(dict(key=route['key'], affected=affected,
                             control_flags=control['extra_cuda_cflags'],
                             candidate_flags=candidate['extra_cuda_cflags'],
                             cuda_sha256=sha(control['cuda_sources'])))
    unique = calls + [call for call in treatment_calls if call not in calls]
    assert len(unique) == 8
    descriptors = []
    for call in unique:
        item = dict(call)
        for key in ('cpp_sources', 'cuda_sources'):
            item[key] = sha(item[key])
        descriptors.append(item)
    reference = (ROOT / 'muse_moe_w4a16/reference.py').read_text()
    nvfp4 = (ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
    assert sha(reference) == REFERENCE_PIN and sha(nvfp4) == NVFP4_PIN
    route_source = '\n\n'.join((
        'D_HIDDEN = D_EXPERT = 4096\nN_EXPERTS = 256\nROUTED_PER_TOKEN = 8\nPOPULARITY = .03\nLOCAL = {"ep8": (0, 32), "full": (0, 256)}',
        extract(nvfp4, 'outlier_channels'), extract(reference, 'generate_input')))
    contract = contract_source()
    runtime = (ROOT / 'tools/expert_top100_moe_harness_runtime.py').read_text()
    identity = sha(prefix + route_source + contract + runtime + json.dumps(descriptors, sort_keys=True))
    source = prefix + '\n' + route_source + '\n' + contract + '\n'
    source += f'H_RUNTIME_ID = {identity!r}\nH_HOLD = False\nH_BUILD_CALLS = {descriptors!r}\n' + runtime
    assert len(source.encode()) < 1_000_000
    ast.parse(source)
    manifest = dict(experiment='GOPT-PIPE-01', source_sha256=sha(source), bytes=len(source.encode()),
                    runtime_id=identity, base_sha256=BASE_PIN, helper_sha256=HELPER_PIN,
                    native_sha256=NATIVE_PIN, official_arm='candidate2', hold=False,
                    compiler_calls=descriptors, selection=selected, routes=routes,
                    contract_sha256=sha(contract), runtime_sha256=sha(runtime),
                    adapter_sha256=ADAPTER_PIN, reference_sha256=REFERENCE_PIN, nvfp4_sha256=NVFP4_PIN,
                    public_submission=False, target_qualified=False,
                    gate='T1/T8/T16 geomean >=1% in all four crosses, each bank/distribution, above twice duplicate null; no affected >=1% regression; unchanged T32/T128/T256 drift <= max(1%, twice row duplicate null)')
    return source, manifest, calls[:3], treatment_calls[:3]


def compile_pairs(out, controls, candidates):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for key, control, candidate in zip(('tiny', 'tiny2', 'mid'), controls, candidates):
        for arm, call in (('control', control), ('candidate', candidate)):
            native = call['cuda_sources']
            boundary = '#include <torch/extension.h>'
            device = native[:native.index(boundary)]
            anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase('
            assert device.count(anchor) == 1
            device = device.replace(anchor, '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n' + anchor)
            device += '\nvoid compile_entries(okc::Params p) {\n'
            device += 'okc::moe_fused<false><<<1, okc::NTL>>>(p);\n'
            device += 'okc::moe_fused<true><<<1, okc::NTL>>>(p);\n}\n'
            device += '''#include <cstdio>
int main() {
  const size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
  const size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES;
  const size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  std::printf("{\\"smem_base\\":%zu,\\"dynamic_smem\\":%zu,\\"threads\\":%d,\\"slot_bytes\\":%d}\\n",
              base, base + (xa > xb ? xa : xb), okc::NTL, okc::UNIT_BYTES);
}
'''
            stem = key + '-' + arm
            (out / (stem + '.cu')).write_text(device)
            flags = call['extra_cuda_cflags'] + ['-std=c++17', '-Xptxas=-v,--warn-on-spills']
            docker = ['docker', 'run', '--rm', '--platform', 'linux/arm64', '--entrypoint', '',
                      '-v', str(out.resolve()) + ':/w', 'nvidia/cuda:12.8.1-devel-ubuntu24.04']
            command = docker + ['nvcc', *flags, '/w/' + stem + '.cu', '-o', '/w/' + stem]
            result = subprocess.run(command, capture_output=True, text=True)
            (out / (stem + '.log')).write_text(result.stdout + result.stderr)
            print(stem, 'compiler exit', result.returncode, flush=True)
            result.check_returncode()
            result = subprocess.run(docker + ['/w/' + stem], capture_output=True, text=True, check=True)
            resources = json.loads(result.stdout)
            result = subprocess.run(docker + ['cuobjdump', '-res-usage', '-sass', '/w/' + stem],
                                    capture_output=True, text=True, check=True)
            (out / (stem + '.sass')).write_text(result.stdout)
            records.append(dict(key=key, arm=arm, source_sha256=sha(native), flags=flags,
                                exit_code=0, **resources))
            (out / 'compile.json').write_text(json.dumps(records, indent=2) + '\n')
            print(stem, resources, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile-out', type=Path)
    args = parser.parse_args()
    source, manifest, controls, candidates = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('experiment', 'source_sha256', 'bytes', 'runtime_id', 'gate')}, indent=2), flush=True)
    if args.compile_out:
        compile_pairs(args.compile_out, controls, candidates)
